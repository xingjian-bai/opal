"""opal broker | say | report | show | watch | compare | doctor | estimate"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _backend(args):
    from .backends import make_backend

    opts = {}
    if args.model:
        opts["model"] = args.model
    if args.effort:
        opts["effort"] = args.effort
    if getattr(args, "timeout", None):
        opts["timeout"] = args.timeout
    if getattr(args, "retries", None) is not None:
        opts["retries"] = args.retries
    if getattr(args, "unavailable_timeout", None) is not None:
        opts["unavailable_timeout"] = args.unavailable_timeout
    if getattr(args, "max_tokens", None) is not None:
        opts["max_tokens"] = args.max_tokens
    return make_backend(args.backend, **opts)


def _run_folder(path):
    run = Path(path)
    if (run / "events.jsonl").exists():
        return run
    inside = sorted(d.name for d in run.iterdir() if (d / "events.jsonl").exists()) if run.is_dir() else []
    hint = f"; runs inside: {', '.join(inside[-5:])}" if inside else ""
    sys.exit(f"opal: not a run folder: {run} (no events.jsonl){hint}")


def _show(run):
    from .report import _jsonl, _response

    events = _jsonl(run / "events.jsonl")  # a line being written is skipped
    start = next((e for e in events if e["kind"] == "start"), {})
    print(f"run {run}")
    if start.get("goal"):
        print(f"goal: {start['goal']}")
    from .report import decision_counts, describe_counts, end_cost, outcome_label, rank_repair_summary

    mode = start.get("blocking")
    mode = "auto-blocking" if mode == "auto" else "blocking" if mode else "non-blocking"
    print(f"{start.get('total_updates', '?')} updates; backend {start.get('backend')}; {mode}")
    repaired = rank_repair_summary(events)
    if repaired:
        print(f"WARNING: {repaired}; the run's trajectory was changed.")
    print(describe_counts(decision_counts(events)))
    print(f"{'k':>4}  {'asked at':>9}  {'outcome':<11} {'at':>9}  {'secs':>6}  {'cost':>6}  rationale")
    req = {e["k"]: e for e in events if e["kind"] == "requested"}
    out = {
        e["k"]: e
        for e in events
        if e["kind"] in ("applied", "kept", "rejected", "load_failed", "cancelled") and e.get("k")
    }
    for k in sorted(req):
        o = out.get(k, {})
        resp = _response(run, k)
        said = " ".join((resp.get("text") or "").split())[:70]
        kind = outcome_label(events, req[k], o)
        cost = resp.get("cost_usd")
        print(
            f"{k:>4}  {req[k]['update']:>9,}  {kind:<11} {o.get('applied_at', ''):>9}  "
            f"{resp.get('seconds', ''):>6}  {('$%.2f' % cost) if cost else '':>6}  {said}"
        )
    for e in events:
        if e["kind"] in ("reverted", "rank_divergence", "ranks_resynced", "resumed", "interrupted", "crashed"):
            print(f"{e['kind']} at update {e.get('update')}")
    end = next((e for e in reversed(events) if e["kind"] == "end"), None)
    if end:
        print(f"finished at update {end['update']:,}; {end.get('decisions')} decisions; {end_cost(start, end)}")


def doctor_mailbox(spec):
    """A mailbox setup: no Claude Code here; the folder must be usable, and a broker should be serving it."""
    import json
    import shutil
    import subprocess
    import time
    import uuid

    from .backends import mailbox

    box = Path(spec.split(":", 1)[1]).expanduser()
    print(f"mailbox {box}: the agent runs on the broker's machine; no Claude Code is needed here")
    python = shutil.which("python3")
    if python is None:
        print("the mailbox host needs python3 (Python 3.6+) on its non-interactive SSH PATH; install it or fix PATH")
        return 1
    try:
        probe = subprocess.run([python, "-c", "import sys; sys.exit(sys.version_info < (3, 6))"],
                               capture_output=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired) as exc:
        print(f"the mailbox host's python3 could not run: {exc}")
        return 1
    if probe.returncode:
        print("the mailbox host needs python3 version 3.6 or newer on its non-interactive SSH PATH")
        return 1
    print("the mailbox host's python3: Python 3.6+ available (also required on the broker's SSH PATH)")
    try:
        box.mkdir(mode=0o700, parents=True, exist_ok=True)
        probe = box / f".doctor-{uuid.uuid4().hex[:8]}"
        probe.write_text("ok")
        probe.unlink()
    except OSError as exc:
        print(f"the mailbox folder cannot be written: {exc}")
        return 1
    print("the mailbox folder: writable")
    beats = [box / "broker-alive"] + [d / "broker-alive" for d in box.iterdir() if d.is_dir()]
    alive = []
    for f in beats:
        try:
            if f.exists() and time.time() - f.stat().st_mtime < mailbox.HEARTBEAT_STALE:
                alive.append(json.loads(f.read_text() or "{}").get("broker") or "a broker")
        except (OSError, ValueError):
            continue
    command = (
        f"opal broker {box} --ssh <you>@<this host> --backend claude --model claude-opus-5-5 --effort high "
        "--keep-serving"
    )
    if alive:
        print(f"a broker is serving this mailbox ({', '.join(sorted(set(alive)))})")
        return 0
    print(
        "no broker is serving this mailbox yet. On a machine with Claude Code logged in, start one, and keep that "
        f"machine awake for the whole run (on a Mac: caffeinate -i {command})"
    )
    return 1


def doctor(a):
    import os
    import platform
    import shutil
    import subprocess
    import tempfile

    import torch

    ok = True
    if a.backend.startswith("mailbox:"):
        return doctor_mailbox(a.backend)
    print(f"python {sys.version.split()[0]} ({sys.executable})")
    print(
        f"torch {torch.__version__}; CUDA available: {torch.cuda.is_available()}"
        + (f" ({torch.cuda.device_count()} x {torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else "")
    )
    if not torch.cuda.is_available() and (shutil.which("nvidia-smi") or os.path.exists("/dev/nvidiactl")):
        built = f"built for CUDA {torch.version.cuda}" if torch.version.cuda else "a CPU-only build"
        print(
            f"this machine has an NVIDIA GPU, but torch {torch.__version__} ({built}) cannot use it: for GPU training, "
            "install the torch build for your driver first (https://pytorch.org/get-started/locally/), or check "
            "CUDA_VISIBLE_DEVICES"
        )
    for name in ("codex" if a.backend == "codex" else "claude",):
        exe = shutil.which(name)
        ver = ""
        if exe:
            try:
                ver = subprocess.run([exe, "--version"], capture_output=True, text=True, timeout=30).stdout.strip()
            except (OSError, subprocess.TimeoutExpired):
                ver = "(version unknown)"
        print(f"{name}: {exe + ' ' + ver if exe else 'not found'}")
    if platform.system() == "Linux":
        for tool, what in (("bwrap", "bubblewrap"), ("socat", "socat")):
            print(f"{what} (for the sandbox of the agent's shell): {'found' if shutil.which(tool) else 'not found'}")
    from .backends import make_backend

    # one call: a failed one (not logged in, a limit) is reported at once, not waited for
    be = make_backend(a.backend, retries=0, timeout=300, unavailable_timeout=0)
    try:
        be.preflight()
        if a.backend == "codex":
            print(f"isolated CODEX_HOME: {be.codex_home} (default home and linked auth rejected)")
            logged_in = be.login_status()
            print(f"isolated Codex login: {'ready' if logged_in else 'not logged in'}")
            if not logged_in:
                print("run CODEX_HOME=~/.codex-opal codex login --device-auth")
                return 1
        print(
            f"backend {a.backend}: found"
            + ("" if a.live or a.backend == "codex" else "; login not checked (run `opal doctor --live`, a few cents)")
        )
    except RuntimeError as exc:
        print(str(exc))
        return 1
    problem = getattr(be, "sandbox_problem", None)
    if problem:
        print(f"the sandbox for the agent's shell cannot start ({problem}): the agent gets file tools only, no shell")
    elif getattr(be, "sandbox", None) is not None:
        print(f"the agent's shell: {'in the OS sandbox' if be.sandbox else 'none (file tools only)'}")
    if a.live:
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp) / "workspace"
            ws.mkdir()
            be.setup(Path(tmp), ws)
            out = be.decide(ws, "Reply with the single word READY and do nothing else.", "You are a test.", Path(tmp))
            reply = (out.get("text") or "").strip()
            ok = out.get("returncode") == 0 and "READY" in reply
            print(
                f"live call: {'ok' if ok else 'failed'} ({out.get('seconds', 0):.0f} s"
                + (f", ${out['cost_usd']:.2f}" if isinstance(out.get("cost_usd"), (int, float)) else "")
                + f"; reply {reply[:80]!r}; error {out.get('error')})"
            )
            if a.backend == "codex":
                print(f"Codex thread: {out.get('session')}; provider rollout: {out.get('rollout')}")
                print(f"tokens (USD unavailable): {json.dumps(out.get('usage'))}")
    return 0 if ok else 1


def estimate_cmd(a):
    from . import estimate
    from .schedule import DEFAULT_DECISIONS, decision_updates

    explicit = a.decisions is not None
    n = len(decision_updates(a.total_steps, a.decisions if explicit else DEFAULT_DECISIONS, a.every))
    capped_from = None
    if not explicit and a.blocking:
        k = estimate.capped_decisions(n, a.total_steps * a.sec_per_step, a.blocking, a.sec_per_step)
        if k < n:
            capped_from, n = n, len(decision_updates(a.total_steps, k, a.every))
    print(f"OPAL estimate, from our runs of {estimate.MODEL} at effort {estimate.EFFORT} (yours may differ):")
    for line in estimate.lines(
        a.total_steps, a.sec_per_step, n, a.blocking, model=a.model, effort=a.effort, mailbox=a.mailbox,
        capped_from=capped_from, explicit=explicit,
    ):  # fmt: skip
        print("  " + line)
    print("  These are estimates from our runs (opal/estimate.py says which); the run prints its own at its first check.")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="opal")
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("broker", help="answer a training run's mailbox with a local agent (laptop or login node)")
    b.add_argument("box", help="mailbox folder of the run (on the GPU machine when --ssh is given)")
    b.add_argument("--ssh", help="user@host of the GPU machine; omit when the folder is on a shared filesystem")
    b.add_argument(
        "--backend",
        default="claude",
        choices=["claude", "codex", "keep"],
        help="keep: answer every decision with 'keep the programs' (to test the mailbox path at no cost)",
    )
    b.add_argument("--model")
    b.add_argument("--effort")
    b.add_argument("--max-tokens", type=int, help="Codex only: soft run cap on input + output tokens; an in-flight turn may exceed it")
    b.add_argument(
        "--workdir", help="local folder for the agent's workspace and records (default ~/.opal/broker/<name>)"
    )
    b.add_argument("--poll", type=float, default=2.0, help="seconds between checks of the mailbox")
    b.add_argument("--run", help="run id (subfolder of the box) to serve; default: the newest unfinished run")
    b.add_argument(
        "--timeout",
        type=int,
        help="seconds per decision; by default the limit the run's agent was told (the mailbox timeout of the run); "
        "a shorter value here is kept, and the agent is told it in each message",
    )
    b.add_argument(
        "--retries",
        type=int,
        help="how often a failed agent turn (for example a tool crash) is made again, after 15 s, 30 s, "
        "... (at most 10 min apart) while the decision's time allows; default: as often as that time allows; 0: never",
    )
    b.add_argument(
        "--unavailable-timeout",
        type=float,
        help="seconds an unavailable agent (not logged in, a usage or spend limit, the API overloaded, no network) is "
        "waited for before the request is answered as failed; default: no deadline",
    )
    b.add_argument("--keep-serving", action="store_true", help="after a run ends, wait for the next run in the box")
    b.add_argument(
        "--give-up-after-min",
        type=float,
        help="exit with an error if the mailbox stays unreadable this long (default: retry forever)",
    )

    s = sub.add_parser("say", help="leave a note for the agent's next decision")
    s.add_argument("run")
    s.add_argument("text")

    r = sub.add_parser("report", help="write <run>/report.html: loss, decisions, rationales and code diffs")
    r.add_argument("run")
    r.add_argument(
        "--summarize",
        action="store_true",
        help="first describe and tag the code changes that have no summary yet (Claude Code CLI, one small call each)",
    )
    r.add_argument("--model", help="model for --summarize (default claude-opus-5-5 at low effort)")

    w = sub.add_parser("show", help="print a run's decisions and where its final program is")
    w.add_argument("run")

    h = sub.add_parser("watch", help="print a run's events as they are written")
    h.add_argument("run")
    h.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="seconds between checks of events.jsonl (default 1)",
    )
    h.add_argument(
        "--timeout",
        type=float,
        help="seconds to wait for a new event, then exit with 1; 0 prints what is there and returns (1 if the run has "
        "not finished); omitted: wait for the run's end event. A run killed without close() (SIGTERM, SIGKILL, out of "
        "memory) never writes one: use --timeout",
    )

    c = sub.add_parser(
        "compare",
        help="set two run folders, or two result.json files, beside each other",
    )
    c.add_argument("a", help="a run folder (events.jsonl) or a result.json")
    c.add_argument("b", help="the other run folder or result.json")
    c.add_argument(
        "--json",
        action="store_true",
        help='print the two summaries and the differing decisions as JSON; NaN and infinite numbers are written as the '
        'strings "nan", "inf" and "-inf"',
    )

    d = sub.add_parser("doctor", help="check that torch and an agent CLI are ready on this machine")
    d.add_argument("--live", action="store_true", help="also make one tiny agent call (costs a few cents)")
    d.add_argument(
        "--backend",
        default="claude",
        help='claude (default), codex, or mailbox:<folder> on a GPU machine whose agent runs elsewhere (checks the folder and '
        "that a broker serves it)",
    )

    e = sub.add_parser(
        "estimate", help="before a run: how long training will wait for the agent, and what the agent will cost"
    )
    e.add_argument("--total-steps", type=int, required=True, help="updates in the run (wrap's total_steps)")
    e.add_argument(
        "--sec-per-step", type=float, required=True, help="seconds per update without the agent (time the smoke test)"
    )
    e.add_argument(
        "--decisions",
        type=int,
        help="as in wrap(decisions=...); not given: OPAL's default, 50, lowered so that the expected waiting stays "
        "within 4x the run's own time",
    )
    e.add_argument("--every", type=int, default=50, help="as in wrap(every=...): decisions fall on checks")
    e.add_argument(
        "--blocking",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="training waits for each decision (the default); --no-blocking: it goes on while the agent thinks",
    )
    e.add_argument("--model", default="claude-opus-5-5")
    e.add_argument("--effort", default="high")
    e.add_argument("--mailbox", action="store_true", help="the agent answers through a mailbox broker")

    a = ap.parse_args(argv)
    if a.cmd == "doctor" and a.backend not in ("claude", "codex") and not a.backend.startswith("mailbox:"):
        ap.error("doctor --backend must be claude, codex or mailbox:<folder>")
    if a.cmd == "watch" and not 0 < a.interval < float("inf"):  # also rejects nan
        ap.error("watch --interval must be greater than 0")
    if a.cmd == "watch" and a.timeout is not None and not 0 <= a.timeout < float("inf"):
        ap.error("watch --timeout must be zero or positive")
    if a.cmd == "estimate":
        return estimate_cmd(a)
    if a.cmd == "broker":
        from .backends.mailbox import run_broker

        name = (a.ssh or "local").replace("@", "_") + "_" + a.box.strip("/").replace("/", "_")
        workdir = Path(a.workdir) if a.workdir else Path("~/.opal/broker").expanduser() / name
        run_broker(
            a.box,
            _backend(a),
            workdir,
            ssh=a.ssh,
            poll=a.poll,
            run=a.run,
            keep_serving=a.keep_serving,
            give_up_after=60 * a.give_up_after_min if a.give_up_after_min else None,
        )
    elif a.cmd == "say":
        run = _run_folder(a.run)
        with open(run / "notes.txt", "a") as f:
            f.write(a.text.strip() + "\n")
        from .report import _jsonl

        kinds = [e.get("kind") for e in _jsonl(run / "events.jsonl")]
        last = lambda kind: max((i for i, k in enumerate(kinds) if k == kind), default=-1)  # noqa: E731
        ended = last("end") > last("resumed")
        print(
            "noted, but this run has finished: no agent will read it"
            if ended
            else f"noted; the agent sees it at its next decision ({run / 'notes.txt'})"
        )
    elif a.cmd == "report":
        from .report import summarize_missing, write_report

        run = _run_folder(a.run)
        if a.summarize:
            from .summaries import Summarizer

            failed = summarize_missing(run, Summarizer(model=a.model), log=print)
            if failed:
                print(
                    f"{failed} summaries failed (see decisions/NNN/summary-error.json); their rows show the rationale"
                )
        print(write_report(run))
    elif a.cmd == "doctor":
        return doctor(a)
    elif a.cmd == "show":
        from .wrapper import final_program

        run = _run_folder(a.run)
        _show(run)
        print(f"final program: {final_program(run)}")
        print("rerun it with no agent: opal.replay(<that folder>, optimizer, model, total_steps=...)")
    elif a.cmd == "watch":
        from . import watch as watchmod

        try:
            return watchmod.follow(_run_folder(a.run), interval=a.interval, timeout=a.timeout)
        except KeyboardInterrupt:
            return 130  # Ctrl-C ends the watch, not the run
        except BrokenPipeError:  # the reader went away (| head): stop quietly, as opal show does
            import os

            os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())  # so the exit flush cannot fail again
            return 0
    elif a.cmd == "compare":
        from . import compare

        try:
            cmp = compare.compare(a.a, a.b)
        except (FileNotFoundError, ValueError) as exc:
            sys.exit(str(exc))
        try:
            if a.json:
                print(json.dumps(compare.jsonable(cmp), default=str, allow_nan=False, indent=2))
            else:
                print("\n".join(compare.lines(cmp)))
        except BrokenPipeError:  # the reader went away (| head): stop quietly, as opal watch does
            import os

            os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())  # so the exit flush cannot fail again
    return 0


if __name__ == "__main__":
    sys.exit(main())
