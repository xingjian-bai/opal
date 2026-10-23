"""Mailbox: for GPU machines where no coding agent is installed or logged in (cluster nodes, shared servers).

Training side:  opal.wrap(..., backend="mailbox:/path/on/the/gpu/machine")
Agent side:     opal broker /path/on/the/gpu/machine --ssh user@gpu-host --backend claude

Each run gets its own subfolder <box>/<run id>/ (created 0700). The training side writes request-0001.json, ... and a
small latest.json naming the newest request, then waits (in its background thread; training goes on) for the matching
response. Every request carries the run id and an attempt id (new for each process start), and a response counts only
if it echoes both, so answers left over from another run or from before a crash are never applied. When the run stops
waiting (no broker, or the timeout), an answer that still comes before its next request is applied; a later one is not,
and the next message says so. The broker, on a laptop or login node, polls latest.json over ssh (or a shared
filesystem), runs a local backend on a local copy of the workspace, and writes the response back. All writes are atomic.

A request carries the programs and the files that are new since the previous request, plus a manifest of every other
workspace file (inbox/, reference/, SYSTEM.md, the agent's own notes); each such file is also kept in <run>/files/, so
a broker that lacks some (a new --workdir, another machine) fetches them from there, and drops inbox/ and reference/
files the run no longer has. A response carries the programs and the agent's own files (NOTES.md, scratch/), which the
training side keeps, so notes survive a change of broker. A broker claims a request (claim-NNNN.json, created
exclusively) before it starts the agent, so two brokers never both answer it, and runs the agent for at most the time
the request says the agent was told it has; when it stops the agent sooner (a shorter --timeout, or a request that
waited long), the message it gives the agent starts with the limit that holds.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import re
import shlex
import socket
import subprocess
import threading
import time
import traceback
import uuid
from pathlib import Path, PurePosixPath

from ..misc.taskfacts import duration
from . import unavailable
from .base import Backend, error_tail
from ._privacy import redact_secrets

_PROGRAM = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\.py")  # program files, as in programs.py
AGENT_FILE_MAX = 256 * 1024  # bytes per agent file sent back with a response
AGENT_FILES_MAX = 2 * 1024 * 1024  # bytes of agent files per response
CLAIM_STALE = 120  # seconds after which a claim its broker no longer refreshes may be taken over
ANSWER_MARGIN = 60  # seconds the broker keeps in hand to write an answer before the run stops waiting
IDLE_SWITCH = 10  # seconds a broker not pinned to a run (--run) stays on an idle run while another one is waiting
BEAT_SECONDS = 30  # between a broker's heartbeats
HEARTBEAT_STALE = 90  # seconds after which a heartbeat no longer counts


def _atomic_write(path: Path, data: bytes) -> None:
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def _sha(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", "replace")).hexdigest()


def _safe_rel(rel: str) -> bool:
    """Workspace paths a request may carry: top-level program files, SYSTEM.md, files under inbox/ or reference/, and
    the agent's own files."""
    p = PurePosixPath(rel)
    if p.is_absolute() or ".." in p.parts or not p.parts:
        return False
    if len(p.parts) == 1 and (_PROGRAM.fullmatch(p.name) or p.name == "SYSTEM.md"):
        return True
    if p.parts[0] in ("inbox", "reference"):
        return not any(x.startswith(".") for x in p.parts)
    return _agent_rel(rel)


def _agent_rel(rel: str) -> bool:
    """The agent's own files, which a response may carry back: top-level files that are not programs (NOTES.md) and
    files under scratch/. No hidden files, no __pycache__."""
    p = PurePosixPath(rel)
    if p.is_absolute() or ".." in p.parts or not p.parts or any(x.startswith(".") for x in p.parts):
        return False
    if "__pycache__" in p.parts or p.parts[0] in ("inbox", "reference"):
        return False
    if len(p.parts) == 1:
        return not _PROGRAM.fullmatch(p.name) and p.name != "SYSTEM.md"
    return p.parts[0] == "scratch"


def _inside(ws: Path, rel: str):
    """ws/rel if no part of the way below ws is a symbolic link (an agent's program can make links in its folder), else
    None."""
    ws = Path(ws)
    cur = ws
    for part in PurePosixPath(rel).parts:
        cur = cur / part
        if cur.is_symlink():
            return None
    real_ws = os.path.realpath(ws)
    if not os.path.realpath(cur).startswith(real_ws.rstrip(os.sep) + os.sep):
        return None
    return cur


def _write_in(ws: Path, rel: str, text: str) -> bool:
    f = _inside(ws, rel)
    if f is None:
        return False
    f.parent.mkdir(parents=True, exist_ok=True)
    if _inside(ws, rel) is None:  # a link made between the check and the mkdir
        return False
    f.write_text(text)
    return True


def _files(ws: Path, keep):
    """{rel: path} of the regular files below ws for which keep(rel) holds, never through a symbolic link."""
    out = {}
    ws = Path(ws)
    for root, dirs, names in os.walk(ws):  # os.walk does not follow links into folders
        dirs[:] = sorted(d for d in dirs if not os.path.islink(os.path.join(root, d)) and d != "__pycache__")
        for n in sorted(names):
            p = Path(root) / n
            if p.is_symlink() or not p.is_file():
                continue
            rel = p.relative_to(ws).as_posix()
            if keep(rel):
                out[rel] = p
    return out


def _agent_files(ws: Path, known: dict, log=None) -> dict:
    """The agent's own files that differ from `known` ({rel: sha}), capped in size, as {rel: text}."""
    out, total = {}, 0
    for rel, p in _files(ws, _agent_rel).items():
        try:
            if p.stat().st_size > AGENT_FILE_MAX:
                raise ValueError("too large")
            text = p.read_bytes().decode("utf-8")
        except (OSError, ValueError):
            if log:
                log(f"broker: not sent back (binary, unreadable or over {AGENT_FILE_MAX // 1024} KB): {rel}")
            continue
        if known.get(rel) == _sha(text):
            continue
        if total + len(text) > AGENT_FILES_MAX:
            if log:
                log(f"broker: not sent back (over {AGENT_FILES_MAX // 1024**2} MB of agent files): {rel}")
            continue
        out[rel], total = text, total + len(text)
    return out


class MailboxBackend(Backend):
    name = "mailbox"

    def __init__(self, spec, poll=2.0, timeout=6 * 3600, broker_wait=120, unavailable_timeout=None):
        self.box = Path(spec.split(":", 1)[1]).expanduser()
        self.poll, self.timeout, self.broker_wait = poll, timeout, broker_wait
        # no broker for broker_wait seconds, or a broker whose agent is unavailable: the request is kept and waited for
        # (backends/unavailable.py), for at most unavailable_timeout seconds when given
        self.unavailable_timeout = unavailable_timeout
        self.training_waits = None  # set by the run before each decision (the wait line)
        self.run_id = None
        self.attempt = uuid.uuid4().hex[:12]
        self.k = 0
        self.answered = 0  # answers received, so a new broker knows the agent had an earlier conversation
        self.sent = {}  # {rel: sha} of the non-program files already in the box (request or files/)
        self.late = None  # (k, workspace, t0) of the request the run stopped waiting for
        self._wait = time.sleep

    @property
    def folder(self) -> Path:
        return self.box / self.run_id

    def setup(self, run_dir, workspace):
        # the run's subfolder is created at the first request, so a resumed run (whose id is restored from the
        # checkpoint after setup) never leaves an empty run behind for a broker to latch onto
        self.box.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.run_id = self.run_id or uuid.uuid4().hex[:12]

    def _ensure_folder(self):
        if not (self.folder / "RUN").exists():
            self.folder.mkdir(mode=0o700, exist_ok=True)
            _atomic_write(self.folder / "RUN", json.dumps(dict(run_id=self.run_id, started=time.time())).encode())
        (self.folder / "done").unlink(missing_ok=True)
        (self.folder / "interrupted").unlink(missing_ok=True)

    def broker_alive(self) -> bool:
        """A broker serving this run, or one that serves any run of the box (waiting, or --keep-serving), refreshes its
        heartbeat every 30 s, also while the agent thinks."""
        for f in (self.folder / "broker-alive", self.box / "broker-alive"):
            try:
                if f.exists() and time.time() - f.stat().st_mtime < HEARTBEAT_STALE:
                    if f.parent == self.folder or json.loads(f.read_text() or "{}").get("run") is None:
                        return True
            except (OSError, ValueError):
                continue
        return False

    def describe(self) -> str:
        return (
            f"mailbox {self.box}, run id {self.run_id}; answer it with: "
            f"opal broker {self.box} --run {self.run_id} [--ssh user@this-host] --backend claude"
        )

    def _pack(self, workspace):
        """(files, manifest): the programs and every new or changed other file, and {rel: sha} of all other files.
        New or changed other files are also written to <run>/files/ for a broker that lacks them."""
        out, manifest = {}, {}
        for rel, p in _files(Path(workspace), _safe_rel).items():
            if _agent_rel(rel):  # the agent's own files (or what its programs wrote): small text files only
                try:
                    if p.stat().st_size > AGENT_FILE_MAX:
                        continue
                    text = p.read_bytes().decode("utf-8")
                except (OSError, ValueError):
                    continue
            else:
                text = p.read_text(errors="replace")
            if len(PurePosixPath(rel).parts) == 1 and _PROGRAM.fullmatch(rel):
                out[rel] = text
                continue
            sha = _sha(text)
            manifest[rel] = sha
            if self.sent.get(rel) != sha:
                out[rel] = text
                mirror = self.folder / "files" / rel
                mirror.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
                _atomic_write(mirror, text.encode("utf-8", "replace"))
                self.sent[rel] = sha
        return out, manifest

    def decide(self, workspace, message, system, record_dir):
        t0 = time.time()
        if getattr(self, "cancelled", False) or getattr(self, "finished", False):
            return dict(text="", cost_usd=None, returncode=-1, error="the run has ended", seconds=0.0)
        self._ensure_folder()
        self.late = None  # a newer request: an answer to an older one is no longer applied
        self.k += 1
        k = self.k
        files, manifest = self._pack(workspace)
        req = dict(
            run_id=self.run_id,
            attempt=self.attempt,
            k=k,
            system=system,
            message=message,
            files=files,
            manifest=manifest,
            limit=self.timeout,  # what the agent was told it may take: the broker stops it in time for this run
            answered=self.answered,
            utc=time.time(),
            **({"budget_left": self.budget_left} if getattr(self, "budget_left", None) is not None else {}),
        )
        _atomic_write(self.folder / f"request-{k:04d}.json", json.dumps(req).encode())
        latest = dict(k=k, attempt=self.attempt, utc=req["utc"])
        _atomic_write(self.folder / "latest.json", json.dumps(latest).encode())
        # While no broker answers (none seen for broker_wait seconds) or the broker's agent is unavailable, the request
        # is kept and waited for: that time does not count against `timeout`, and the broker is told how long the run
        # had no broker (absent-NNNN.json), so that it does not shorten the agent's time for it either.
        waiting, waited, absent, last = None, 0.0, 0.0, time.time()
        while time.time() - t0 - waited < self.timeout:
            if getattr(self, "cancelled", False) or getattr(self, "finished", False):
                return dict(text="", cost_usd=None, returncode=-1, error="mailbox request cancelled: the run ended",
                            seconds=time.time() - t0 - waited)
            now = time.time()
            down = None
            if now - t0 > self.broker_wait and not self.broker_alive():
                down = ("broker", f"no broker is answering ({self.describe()})")
                absent += now - last
                _atomic_write(self.folder / f"absent-{k:04d}.json", json.dumps(dict(seconds=absent)).encode())
            else:
                down = self._broker_agent_down(k)
            if down is not None:
                if waiting is None:
                    what = "the mailbox's broker" if down[0] == "broker" else "Claude Code (on the broker)"
                    waiting = unavailable.Wait(
                        self._say, what, self.training_waits, self.unavailable_timeout, every=self.poll
                    )
                waited += now - last
                if waiting.note(*down) is None:  # unavailable_timeout is over
                    return self._give_up(k, workspace, t0, down[1], waited)
            elif waiting is not None:
                waiting.end()
                waiting = None
            last = now
            out = self._answer(k, workspace, t0)
            if out is not None:
                if waiting is not None:
                    waiting.end()
                return dict(out, seconds=out["seconds"] - waited, **({"unavailable_seconds": round(waited, 1)} if waited else {}))
            self._wait(self.poll)
        return self._give_up(k, workspace, t0, "mailbox timeout", waited)

    def _say(self, text):
        log = getattr(self, "log", None)
        if log is not None:
            log(text)
        else:
            print(f"[opal] {text}", flush=True)

    def _broker_agent_down(self, k):
        """("limit" | "login" | ..., the CLI's words) while the broker serving request k waits for its agent to be
        available (it writes unavailable-NNNN.json for as long as that lasts), else None."""
        try:
            f = self.folder / f"unavailable-{k:04d}.json"
            if not f.exists():
                return None
            body = json.loads(f.read_text() or "{}")
        except (OSError, ValueError):
            return None
        if body.get("attempt") != self.attempt or not self.broker_alive():
            return None
        return body.get("kind") or "down", str(body.get("reason") or "the broker's agent is unavailable")

    def _answer(self, k, workspace, t0):
        """Response k if it has come and is ours: its programs and the agent's own files are written into the
        workspace. Else None."""
        resp_f = self.folder / f"response-{k:04d}.json"
        if not resp_f.exists():
            return None
        try:
            resp = json.loads(resp_f.read_text())
        except (OSError, ValueError):
            resp = {}
        if resp.get("run_id") != self.run_id or resp.get("attempt") != self.attempt:
            resp_f.unlink(missing_ok=True)  # an answer to an earlier attempt of this request: not ours
            return None
        ws = Path(workspace)
        files = {n: t for n, t in (resp.get("files") or {}).items() if _PROGRAM.fullmatch(n)}
        for p in ws.glob("*"):
            if _PROGRAM.fullmatch(p.name) and p.name not in files and not p.is_symlink():
                p.unlink()
        for name, text in files.items():
            _write_in(ws, name, text)
        for rel, text in (resp.get("agent_files") or {}).items():
            if isinstance(text, str) and _agent_rel(rel) and _write_in(ws, rel, text):
                mirror = self.folder / "files" / rel  # kept for the next broker, and not sent again
                mirror.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
                _atomic_write(mirror, text.encode("utf-8", "replace"))
                self.sent[rel] = _sha(text)
        self.answered += 1
        return dict(
            text=resp.get("text", ""),
            cost_usd=resp.get("cost_usd"),
            returncode=resp.get("returncode", 0),
            error=resp.get("error"),
            seconds=time.time() - t0,
            **{x: resp[x] for x in ("usage", "session", "rollout") if x in resp},
        )

    def _give_up(self, k, workspace, t0, error, waited=0.0):
        """Stop waiting for request k (unavailable_timeout, or the mailbox timeout). It stays the newest request, so a
        broker that comes back still answers it; late_answer() hands that answer over until the next request replaces
        it."""
        self.late = (k, workspace, t0)
        out = dict(text="", cost_usd=None, returncode=-1, error=error, seconds=time.time() - t0 - waited, gave_up=True)
        return dict(out, unavailable_seconds=round(waited, 1)) if waited else out

    def late_answer(self):
        """The answer to the request this backend stopped waiting for, if it has come since and no newer request was
        sent (its programs are then in the workspace), else None."""
        if self.late is None:
            return None
        out = self._answer(*self.late)
        if out is not None:
            self.late = None
        return out

    def finish(self, interrupted=False):
        """`done` releases the broker; an interrupted run (stopped before its last update without close(), e.g. a
        crash) leaves the broker waiting for the resumed attempt."""
        self.finished = True
        if self.run_id:
            self._ensure_folder()
            name = "interrupted" if interrupted else "done"
            _atomic_write(self.folder / name, json.dumps(dict(attempt=self.attempt)).encode())

    def cancel(self):
        """The run ended with this decision in flight: tell the broker to stop the agent."""
        self.cancelled = True
        if self.run_id and self.k and (self.folder / "RUN").exists():
            _atomic_write(self.folder / f"cancel-{self.k:04d}.json", json.dumps(dict(attempt=self.attempt)).encode())

    def state(self):
        return dict(run_id=self.run_id, k=self.k, answered=self.answered)

    def restore(self, state):
        self.run_id = state.get("run_id") or self.run_id
        self.k = state.get("k", 0)
        self.answered = state.get("answered", self.k)
        self.sent = {}


# ----- agent side ---------------------------------------------------------------------------------------------


def _lease_file(path, action, body, expected, stale):
    """Serialize lease transitions on a persistent inode; the OS releases the lock after a crash.

    Kept self-contained so the identical operation runs on the mailbox host over SSH.
    A generation fences a resumed old broker's renewals, releases and response writes.
    """
    import fcntl
    import json
    import os
    import time
    import uuid
    from pathlib import Path

    path = Path(path).expanduser()
    fd = os.open(str(path) + ".lock", os.O_WRONLY | os.O_CREAT, 0o600)
    with os.fdopen(fd, "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            old = json.loads(path.read_bytes())
        except (FileNotFoundError, ValueError):
            old = {}

        def write(target, value):
            tmp = target.with_name(target.name + "." + uuid.uuid4().hex + ".tmp")
            try:
                out = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                with os.fdopen(out, "wb") as stream:
                    stream.write(json.dumps(value).encode())
                os.replace(tmp, target)
            finally:
                try:
                    tmp.unlink()
                except FileNotFoundError:
                    pass

        if action == "claim":
            if (old and old.get("attempt") == body["attempt"] and old.get("broker") != body["broker"]
                    and time.time() - float(old.get("t", 0)) < stale):
                return False
            write(path, body)
            return dict(previous=old)
        if not old or old.get("generation") != expected.get("generation") or old.get("attempt") != expected.get("attempt"):
            return False
        if action == "renew":
            write(path, dict(old, t=time.time()))
        elif action == "release":
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        elif action == "answer":
            write(path.with_name(path.name.replace("claim-", "response-", 1)), body)
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        else:
            raise ValueError("unknown lease action")
        return True


class _Folder:
    """A folder on this machine, or on another one over ssh (one shared connection)."""

    def __init__(self, path, ssh=None):
        self.path, self.ssh = str(path).rstrip("/"), ssh
        if ssh:
            ctl = Path("~/.ssh").expanduser() / "opal-%r@%h-%p"
            self.base = [
                "ssh",
                "-o",
                "BatchMode=yes",
                "-o",
                "ControlMaster=auto",
                "-o",
                f"ControlPath={ctl}",
                "-o",
                "ControlPersist=600",
                ssh,
            ]

    def _q(self, name=""):
        full = f"{self.path}/{name}" if name else self.path
        if full.startswith("~/"):
            return '"$HOME"/' + shlex.quote(full[2:])
        return shlex.quote(full)

    def _local(self, name=""):
        return Path(self.path).expanduser() / name if name else Path(self.path).expanduser()

    def _sh(self, cmd, data=None):
        # without data, give ssh an empty stdin: it would otherwise read (and forward) the broker's own stdin
        io = dict(input=data) if data is not None else dict(stdin=subprocess.DEVNULL)
        p = subprocess.run(self.base + [cmd], capture_output=True, timeout=120, **io)
        if p.returncode == 255 and b"Permission denied" in p.stderr:  # BatchMode: ssh cannot ask for a password
            p.stderr = p.stderr.strip() + (
                " (the broker's ssh cannot ask for a password: use an ssh key, or open a connection it can share "
                f"first: ssh -fNM -o ControlPath=~/.ssh/opal-%r@%h-%p {self.ssh})"
            ).encode()
        return p

    def read(self, name):
        if not self.ssh:
            try:
                return self._local(name).read_bytes()
            except (FileNotFoundError, NotADirectoryError, IsADirectoryError):
                return None  # missing, or removed between a listing and this read
        q = self._q(name)
        p = self._sh(f"if test -f {q}; then cat {q}; else exit 3; fi")  # 3: missing, whatever ssh prints on stderr
        if p.returncode == 0:
            return p.stdout
        if p.returncode == 3:
            return None
        raise RuntimeError(f"ssh read failed ({p.returncode}): {p.stderr.decode()[-300:]}")

    def write(self, name, data: bytes):
        if not self.ssh:
            _atomic_write(self._local(name), data)
            return
        q, t = self._q(name), self._q(f"{name}.{uuid.uuid4().hex[:12]}.tmp")  # a temp name of this write only
        p = self._sh(f"umask 077 && cat > {t} && mv {t} {q}", data)
        if p.returncode != 0:
            raise RuntimeError(f"ssh write failed ({p.returncode}): {p.stderr.decode()[-300:]}")

    def create(self, name, data: bytes) -> bool:
        """Publish complete contents only if `name` does not exist (an exclusive hard link). False if it does."""
        if not self.ssh:
            path = self._local(name)
            tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
            try:
                fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                with os.fdopen(fd, "wb") as f:
                    f.write(data)
                try:
                    os.link(tmp, path)
                except FileExistsError:
                    return False
                return True
            finally:
                tmp.unlink(missing_ok=True)
        temp_name = str(PurePosixPath(name).with_name(f".{PurePosixPath(name).name}.{uuid.uuid4().hex}.tmp"))
        q, tmp = self._q(name), self._q(temp_name)
        cleanup = shlex.quote(f"rm -f {tmp}")
        p = self._sh(f"trap {cleanup} EXIT; umask 077; cat > {tmp} && test ! -e {q} && test ! -L {q} && ln {tmp} {q}", data)
        if p.returncode == 0:
            return True
        probe = self._sh(f"test -e {self._q(name)}")
        if probe.returncode == 0:
            return False
        raise RuntimeError(f"ssh create failed ({p.returncode}): {p.stderr.decode()[-300:]}")

    def lease(self, name, action, body=None, expected=None):
        if not self.ssh:
            return _lease_file(self._local(name), action, body, expected, CLAIM_STALE)
        # Python is already required on the training host. No remote OPAL install is needed.
        script = inspect.getsource(_lease_file) + "\nimport json, sys\nprint(json.dumps(_lease_file(*json.load(sys.stdin))))"
        args = json.dumps([f"{self.path}/{name}", action, body, expected, CLAIM_STALE])
        p = self._sh("python3 -c " + shlex.quote(script), args.encode())
        if p.returncode != 0:
            raise RuntimeError(f"ssh lease failed ({p.returncode}): {p.stderr.decode()[-300:]}")
        return json.loads(p.stdout)

    def remove(self, name):
        if not self.ssh:
            self._local(name).unlink(missing_ok=True)
            return
        self._sh(f"rm -f {self._q(name)}")

    def make(self):
        """Create the box (0700) if it does not exist yet: a broker may start before its job."""
        if not self.ssh:
            self._local().mkdir(mode=0o700, parents=True, exist_ok=True)
        else:
            self._sh(f"mkdir -p -m 700 {self._q()}")

    def runs(self):
        """Run subfolders, newest first."""
        found = []
        if not self.ssh:
            root = self._local()
            try:
                entries = list(root.iterdir()) if root.is_dir() else []
            except OSError:
                entries = []
            for d in entries:
                try:
                    found.append(((d / "RUN").stat().st_mtime, d.name))
                except OSError:
                    continue  # not a run, or removed meanwhile
        else:
            script = (
                f'cd {self._q()} 2>/dev/null || exit 3; for d in */; do test -f "$d/RUN" && '
                f'echo "$(stat -c %Y "$d/RUN" 2>/dev/null || stat -f %m "$d/RUN") ${{d%/}}"; done; exit 0'
            )
            # in sh, not the login shell: zsh fails on a glob that matches nothing ("no matches found: */")
            p = self._sh("sh -c " + shlex.quote(script))
            if p.returncode != 0:
                raise RuntimeError(f"ssh listing failed ({p.returncode}): {p.stderr.decode()[-300:]}")
            for line in p.stdout.decode().splitlines():
                parts = line.split()
                if len(parts) == 2:
                    found.append((float(parts[0]), parts[1]))
        return [name for _, name in sorted(found, reverse=True)]


RETRY_SECONDS = 30  # between failed reads of the mailbox


def _log(msg):
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)  # flushed: a broker's log is usually a file


def broker_id(workdir) -> str:
    """This broker's name in claims: the host and its work folder, so a restarted broker knows its own claims."""
    real = os.path.realpath(os.path.expanduser(str(workdir)))
    return f"{socket.gethostname()}:{hashlib.sha1(real.encode()).hexdigest()[:8]}"


def run_broker(box, backend, workdir, ssh=None, poll=2.0, run=None, log=_log, keep_serving=False, give_up_after=None):
    """Serve the runs of a mailbox. Without keep_serving: one run (the newest with a pending request, else the newest
    unfinished one, or the one `run` names) until it ends with `done`; an interrupted run is waited for, but a broker
    not pinned with `run` moves to another run that is waiting while its own one is idle. With keep_serving: forever,
    taking the runs with pending requests in turn (the longest waiting first). Safe to restart at any point. If reading
    the mailbox keeps failing for `give_up_after` seconds, exit with an error instead of retrying forever."""
    backend.preflight()
    note = backend.reproduction_note() if hasattr(backend, "reproduction_note") else None
    if note:  # another model or effort than the blog's (backends/claude.py)
        log(note)
    last = None
    while True:
        out = _serve_one(box, backend, workdir, ssh, poll, run, log, keep_serving, give_up_after, avoid=last)
        run = None
        if isinstance(out, tuple):  # ("switch", the run it left): serve a run that waits
            last = out[1]
            continue
        if not keep_serving:
            return
        last = None


def _json(raw):
    try:
        out = json.loads(raw) if raw else {}
    except ValueError:
        return {}
    return out if isinstance(out, dict) else {}


def _pending(folder):
    """(k, attempt, utc) of a run's newest request if it has no matching answer yet, else None."""
    latest = folder.read("latest.json")
    if latest is None:
        return None
    cur = json.loads(latest)
    old = folder.read(f"response-{cur['k']:04d}.json")
    if old is not None and json.loads(old).get("attempt") == cur["attempt"]:
        return None
    return cur["k"], cur["attempt"], cur.get("utc", 0.0)


def _pick_run(root, ssh, avoid=None):
    """The run to serve: of the unfinished runs with a pending request the one waiting longest (another than `avoid`
    if there is one), else the newest unfinished run."""
    unfinished = [r for r in root.runs() if _Folder(f"{root.path}/{r}", ssh).read("done") is None]
    waiting = []
    for r in unfinished:
        try:
            p = _pending(_Folder(f"{root.path}/{r}", ssh))
        except (ValueError, KeyError):
            p = None
        if p is not None:
            waiting.append((r == avoid, p[2], r))
    if waiting:
        return min(waiting)[2]
    return unfinished[0] if unfinished else None


def _serve_one(box, backend, workdir, ssh, poll, run, log, keep_serving=False, give_up_after=None, avoid=None):
    root = _Folder(box, ssh)
    root.make()
    state = dict(run=run, stop=False, claim=None, me=broker_id(workdir))

    def beat():
        """Heartbeat every 30 s from its own thread, so that it keeps coming while the agent thinks. A broker that
        serves any run (waiting for one, or --keep-serving) also keeps the box-level one fresh, which every run of the
        box accepts; a claimed request's claim is refreshed too."""
        while not state["stop"]:
            try:
                cur = state["run"]
                if cur is None or keep_serving:
                    body = dict(t=time.time(), run=None, serving=cur, keep_serving=keep_serving, broker=state["me"])
                    root.write("broker-alive", json.dumps(body).encode())
                if cur is not None:
                    body = dict(t=time.time(), run=cur, broker=state["me"])
                    root.write(f"{cur}/broker-alive", json.dumps(body).encode())
                claim = state["claim"]
                if claim is not None:
                    if not root.lease(claim[0], "renew", expected=claim[1]) and state["claim"] is claim:
                        log("broker: lost request lease to another generation; stopping the agent")
                        backend.cancel()
            except (RuntimeError, OSError, subprocess.TimeoutExpired) as exc:
                log(f"broker: heartbeat failed: {exc}")
            until = time.time() + BEAT_SECONDS
            while time.time() < until:
                if state["stop"]:
                    return
                time.sleep(min(1.0, BEAT_SECONDS / 10))

    threading.Thread(target=beat, name="opal-broker-heartbeat", daemon=True).start()
    try:
        return _serve(root, backend, workdir, ssh, poll, run, log, state, keep_serving, give_up_after, avoid)
    finally:
        state["stop"] = True


def _claim(folder, k, attempt, me, log):
    """Atomically acquire a fresh generation, including stale and previous-attempt takeovers."""
    name = f"claim-{k:04d}.json"
    body = dict(broker=me, attempt=attempt, t=time.time(), generation=uuid.uuid4().hex)
    won = folder.lease(name, "claim", body=body)
    if won:
        folder._held = {name: body}
        old = won["previous"]
        if old:
            log(f"broker: taking over request {k} from broker {old.get('broker')}, which stopped answering")
    elif (k, attempt) != getattr(folder, "_backed_off", None):
        folder._backed_off = (k, attempt)
        log(f"broker: request {k} lease belongs to another broker; backing off")
    return bool(won)


def _timeout_for(backend, req, log):
    """Run the agent for at most what it was told (the request's limit), less the time the request waited and a
    margin to write the answer. A broker --timeout shorter than that is kept, and said once."""
    limit = req.get("limit")
    if not limit or not hasattr(backend, "timeout"):
        return None
    if not hasattr(backend, "_own_timeout"):
        backend._own_timeout = backend.timeout if getattr(backend, "timeout_given", True) else None
    # the time the request waited, less the time the run had no broker (which it does not count either)
    absent = float(req.get("absent") or 0.0)
    waited = min(max(0.0, time.time() - float(req.get("utc") or time.time()) - absent), float(limit))
    room = max(60.0, float(limit) - waited - ANSWER_MARGIN)
    own = backend._own_timeout
    if own is None:
        return room
    if own < room and not getattr(backend, "_said_short", False):
        backend._said_short = True
        log(
            f"broker: the agent was told a decision may take up to {int(limit)} s; this broker stops it after "
            f"{int(own)} s (its --timeout), and each message tells the agent so"
        )
    return min(own, room)


def _limit_note(req, timeout) -> str:
    """The first line of a message whose agent this broker stops well before the limit the run told it (a shorter
    --timeout, or a request that waited long for a broker): the limit that holds for this decision, so the agent is
    never stopped earlier than it was told. Empty when the run's limit holds (less the minute kept to write the
    answer)."""
    limit = req.get("limit")
    if not limit or timeout is None or timeout >= float(limit) - 2 * ANSWER_MARGIN:
        return ""
    s = int(timeout)
    s -= s % 60 if s >= 120 else 0  # whole minutes
    stated = "Each decision may take up to" in (req.get("system") or "")  # the neutral prompts say it
    return f"(This decision may take up to {duration(s)}{', less than your instructions state' if stated else ''}.)\n\n"


def _sync_workspace(folder, ws, req, log):
    """Bring the broker's workspace to the run's: the request's files, every manifest file the workspace lacks (from
    <run>/files/), and no inbox/ or reference/ file the run does not have."""
    k = req.get("k")
    for p in ws.glob("*"):
        if _PROGRAM.fullmatch(p.name) and not p.is_symlink():
            p.unlink()
    for rel, text in (req.get("files") or {}).items():
        if not _safe_rel(rel) or not _write_in(ws, rel, text):
            log(f"broker: ignored unsafe path in request {k}: {rel!r}")
    manifest = req.get("manifest")
    if manifest is None:  # a training side older than the manifest
        return
    have = {}
    for rel, p in _files(ws, _safe_rel).items():
        try:
            have[rel] = _sha(p.read_text(errors="replace"))
        except OSError:
            continue
    fetched = 0
    for rel, sha in manifest.items():
        if have.get(rel) == sha or not _safe_rel(rel):
            continue
        raw = folder.read(f"files/{rel}")
        text = raw.decode("utf-8", "replace") if raw is not None else None
        if text is not None and _sha(text) == sha and _write_in(ws, rel, text):
            fetched += 1
        else:
            log(f"broker: request {k}: could not fetch {rel} from the run's files/")
    stale = [rel for rel in have if PurePosixPath(rel).parts[0] in ("inbox", "reference") and rel not in manifest]
    for rel in stale:
        (ws / rel).unlink(missing_ok=True)
    if fetched or stale:
        log(f"broker: request {k}: fetched {fetched} files the workspace lacked, removed {len(stale)} the run has not")


def _serve(root, backend, workdir, ssh, poll, run, log, beat_state, keep_serving, give_up_after=None, avoid=None):
    box = root.path
    pinned = run is not None
    waiting_logged = False
    failing_since = None
    while run is None:
        try:
            run = _pick_run(root, ssh, avoid)
        except (RuntimeError, OSError, ValueError, subprocess.TimeoutExpired) as exc:
            log(f"broker: could not list the runs of the mailbox: {exc}")
            failing_since = time.time() if failing_since is None else failing_since
            if give_up_after and time.time() - failing_since > give_up_after:
                raise SystemExit(f"broker: the mailbox has been unreadable; giving up ({exc})") from None
            time.sleep(RETRY_SECONDS)
            continue
        failing_since = None
        if run is None:
            if not waiting_logged:
                log(f"broker: waiting for a run in {ssh + ':' if ssh else ''}{box}")
                waiting_logged = True
            time.sleep(poll)
    beat_state["run"] = run
    folder = _Folder(f"{root.path}/{run}", ssh)
    work = Path(workdir).expanduser() / run
    ws = work / "workspace"
    ws.mkdir(parents=True, exist_ok=True)
    state_f = work / "broker-state.json"
    saved = json.loads(state_f.read_text()) if state_f.exists() else dict(backend={})
    fresh = not saved.get("backend")
    backend.restore(saved.get("backend", {}))
    backend.setup(work, ws)
    log(f"broker: serving run {run} in {ssh + ':' if ssh else ''}{box}")
    idle_since = time.time()
    failures, failing_since = 0, None
    unsent = None  # (k, attempt, response bytes) of an answer whose write to the mailbox failed
    while True:
        if unsent is not None:
            try:
                if not folder.lease(f"claim-{unsent[0]:04d}.json", "answer", body=_json(unsent[2]), expected=unsent[3]):
                    log(f"broker: request {unsent[0]} lease was lost; discarded the late answer")
                else:
                    log(f"broker: wrote the answer to request {unsent[0]} after a failed write")
                unsent = None
            except (RuntimeError, OSError, ValueError, KeyError, subprocess.TimeoutExpired) as exc:
                log(f"broker: could not retry the answer to request {unsent[0]} ({exc}); still checking new requests")
        try:
            pending = _pending(folder)
            done = folder.read("done")
        except (RuntimeError, OSError, ValueError, KeyError, subprocess.TimeoutExpired) as exc:
            failures += 1
            failing_since = failing_since or time.time()
            minutes = (time.time() - failing_since) / 60
            if failures == 1 or failures % 10 == 0:
                log(f"broker: {failures} failed reads of the mailbox in a row ({minutes:.0f} min); last: {exc}")
            if give_up_after and time.time() - failing_since > give_up_after:
                raise SystemExit(
                    f"broker: the mailbox has been unreadable for {minutes:.0f} min; giving up ({exc})"
                ) from None
            time.sleep(RETRY_SECONDS)
            continue
        if failures:
            log(f"broker: the mailbox is readable again after {failures} failed reads")
            failures, failing_since = 0, None
        if pending is None:
            if done is not None:
                log("broker: run finished")
                return None
            if (keep_serving or not pinned) and time.time() - idle_since > IDLE_SWITCH:
                if _other_waiting(root, ssh, run, log):
                    log(f"broker: run {run} is idle; switching to a run with a pending request")
                    return ("switch", run)
                idle_since = time.time()
            time.sleep(poll)
            continue
        k, attempt, _ = pending
        if done is not None and _json(done).get("attempt") == attempt:
            log("broker: run finished")
            return None
        if unsent is not None and (k, attempt) == unsent[:2]:
            time.sleep(poll)
            continue
        try:
            answered = _answer_request(folder, backend, work, ws, run, k, attempt, fresh, beat_state, log)
        except (RuntimeError, OSError, subprocess.TimeoutExpired) as exc:
            claim = beat_state.get("claim")
            beat_state["claim"] = None
            if claim is not None:
                try:
                    folder.lease(PurePosixPath(claim[0]).name, "release", expected=claim[1])
                except (RuntimeError, OSError, subprocess.TimeoutExpired):
                    pass
            log(f"broker: request {k}: the mailbox failed ({exc}); trying again")
            time.sleep(RETRY_SECONDS)
            continue
        if answered is None:  # cancelled, claimed by another broker, or being rewritten: look again shortly
            time.sleep(poll)
            if keep_serving and unsent is None and _other_waiting(root, ssh, run, log):
                return ("switch", run)
            continue
        fresh = False
        resp, payload = answered
        held = beat_state["claim"][1]
        beat_state["claim"] = None
        state_f.write_text(json.dumps(dict(backend=backend.state()), indent=1))
        try:
            if not folder.lease(f"claim-{k:04d}.json", "answer", body=resp, expected=held):
                log(f"broker: request {k} lease was lost; discarded the answer")
                continue
        except (RuntimeError, OSError, subprocess.TimeoutExpired) as exc:
            log(f"broker: could not write the answer to request {k} ({exc}); trying again")
            unsent = (k, attempt, payload, held)
        idle_since = time.time()
        failed = resp.get("error") or resp.get("returncode") not in (0, None)
        why = f": {error_tail(resp.get('error') or 'no error text')}" if failed else ""
        secs, code = resp.get("seconds") or 0, resp.get("returncode")
        log(f"broker: answered request {k} in {secs:.0f} s, return code {code}{why}")
        if keep_serving and unsent is None and _other_waiting(root, ssh, run, log):
            return ("switch", run)  # take the runs that wait in turn


def _other_waiting(root, ssh, run, log) -> bool:
    try:
        other = _pick_run(root, ssh, avoid=run)
        return other is not None and other != run and _pending(_Folder(f"{root.path}/{other}", ssh)) is not None
    except (RuntimeError, OSError, ValueError, KeyError, subprocess.TimeoutExpired) as exc:
        log(f"broker: could not look at the other runs: {exc}")
        return False


def _answer_request(folder, backend, work, ws, run, k, attempt, fresh, beat_state, log):
    """Serve request k: claim it, sync the workspace, run the agent, collect its answer. (response, bytes), or None
    if it is not ours to answer now. Errors of the agent become an error answer; mailbox errors propagate."""
    if _json(folder.read(f"cancel-{k:04d}.json")).get("attempt") == attempt:
        return None  # the run ended with this request in flight; it will not read an answer
    req = _json(folder.read(f"request-{k:04d}.json"))
    if req.get("attempt") != attempt:
        return None  # latest.json and the request are being rewritten; read again
    if not _claim(folder, k, attempt, beat_state["me"], log):
        if not beat_state.get(f"said-{k}"):
            beat_state[f"said-{k}"] = True
            log(f"broker: request {k} is being answered by another broker")
        return None
    held = folder._held[f"claim-{k:04d}.json"]
    beat_state["claim"] = (f"{run}/claim-{k:04d}.json", held)
    if _json(folder.read(f"response-{k:04d}.json")).get("attempt") == attempt:
        folder.lease(f"claim-{k:04d}.json", "release", expected=held)  # another broker answered it and let go of its claim before ours
        beat_state["claim"] = None
        return None
    folder.remove(f"unavailable-{k:04d}.json")  # left by a broker that stopped while its agent was unavailable
    req["absent"] = _json(folder.read(f"absent-{k:04d}.json")).get("seconds", 0.0)  # the run had no broker this long
    _sync_workspace(folder, ws, req, log)
    known = dict(req.get("manifest") or {})
    rec = work / "decisions" / f"{k:03d}-{attempt}"
    rec.mkdir(parents=True, exist_ok=True)
    message = req.get("message") or ""
    note = getattr(backend, "lost_note", None)
    if fresh and req.get("answered") and note is not None:
        message = note(ws) + message  # this broker has no conversation of the run's agent: say so
    timeout = _timeout_for(backend, req, log)
    if timeout is not None:
        backend.timeout = timeout
    message = _limit_note(req, timeout) + message  # stopped earlier than the run said: the agent is told
    if not folder.lease(f"claim-{k:04d}.json", "renew", expected=held):
        beat_state["claim"] = None
        log(f"broker: request {k} lease was lost before starting the agent; backing off")
        return None
    (rec / "message.txt").write_text(message)
    watching = dict(on=True)
    backend.log = lambda text, k=k: log(f"broker: request {k}: {redact_secrets(text)}")

    def watch(k=k, attempt=attempt, watching=watching):
        """Stop the agent if the run ends or cancels this request while the agent works."""
        while watching["on"]:
            try:
                if (_json(folder.read(f"claim-{k:04d}.json")).get("generation") != held["generation"]
                        and watching["on"]):
                    log(f"broker: request {k} lease was lost; stopping the agent")
                    backend.cancel()
                    return
                c = folder.read(f"cancel-{k:04d}.json")
                d = folder.read("done")
                if (c and _json(c).get("attempt") == attempt) or d is not None:
                    log(f"broker: request {k} was cancelled by the run; stopping the agent")
                    backend.cancel()
                    return
            except (RuntimeError, OSError, ValueError, subprocess.TimeoutExpired):
                pass
            for _ in range(5):
                if not watching["on"]:
                    return
                time.sleep(1)

    threading.Thread(target=watch, name="opal-broker-cancel-watch", daemon=True).start()

    def unavailable_at_run(reason, k=k, attempt=attempt):
        """While the agent is unavailable, the run sees why (unavailable-NNNN.json), so it waits too, and says so."""
        try:
            if reason is None:
                folder.remove(f"unavailable-{k:04d}.json")
            else:
                kind = (unavailable.classify([reason]) or ("down", reason))[0]
                body = dict(attempt=attempt, kind=kind, reason=redact_secrets(reason), t=time.time())
                folder.write(f"unavailable-{k:04d}.json", json.dumps(body).encode())
        except (RuntimeError, OSError, subprocess.TimeoutExpired) as exc:
            log(f"broker: request {k}: could not tell the run that the agent is unavailable ({exc})")

    if hasattr(backend, "on_unavailable"):
        backend.on_unavailable = unavailable_at_run
    backend.budget_left = req.get("budget_left")
    try:
        out = backend.decide(ws, message, req.get("system"), rec)
    except Exception:  # an agent backend's failure is this request's answer, not the broker's end
        out = dict(text="", cost_usd=None, returncode=-1, error=traceback.format_exc(limit=4)[-2000:], seconds=0.0)
    finally:
        watching["on"] = False
    out = dict(out)
    for field in ("error", "text"):
        if out.get(field) is not None:
            out[field] = redact_secrets(out[field])
    backend.cancelled = False  # the backend serves later requests
    files = {}
    for p in sorted(ws.glob("*")):
        if _PROGRAM.fullmatch(p.name) and p.is_file() and not p.is_symlink():
            files[p.name] = p.read_text(errors="replace")
    resp = dict(
        run_id=req["run_id"],
        attempt=attempt,
        k=k,
        files=files,
        agent_files=_agent_files(ws, known, log),
        broker=beat_state["me"],
        agent_timeout=getattr(backend, "timeout", None),
        **{x: out.get(x) for x in ("text", "cost_usd", "returncode", "error", "seconds")},
        **{x: out[x] for x in ("usage", "session", "rollout") if x in out},
        **({"unavailable_seconds": out["unavailable_seconds"]} if out.get("unavailable_seconds") else {}),
    )
    (rec / "response.json").write_text(json.dumps(resp, indent=1))
    return resp, json.dumps(resp).encode()
