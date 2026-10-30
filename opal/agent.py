"""The agent side of a run on rank 0: run folder, one decision at a time in a background thread, the message text."""

from __future__ import annotations

import datetime
import json
import math
import os
import secrets
import sys
import threading
import time
import traceback
from pathlib import Path

from . import programs
from .misc import aggressive, audit
from .backends._privacy import redact_secrets


class Run:
    """Everything a run leaves behind: workspace (what the agent sees), programs/vNNN (every applied version),
    decisions/NNN (message, backend transcript, response), events.jsonl, reports.jsonl, losses.jsonl."""

    def __init__(self, path):
        self.dir = Path(path).resolve()
        self.workspace = self.dir / "workspace"
        self.programs = self.dir / "programs"
        self.decisions = self.dir / "decisions"
        for d in (self.workspace / "inbox", self.workspace / "reference", self.programs, self.decisions):
            d.mkdir(parents=True, exist_ok=True)
        self._notes_seen = 0

    def event(self, kind, **kw):
        rec = dict(kind=kind, utc=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"), **kw)
        self._write("events.jsonl", rec)

    def append(self, name, rec):
        self._write(name, rec)

    def _write(self, name, rec):
        """Records never stop training: on a full disk or an exhausted quota, warn once and go on."""
        try:
            with open(self.dir / name, "a") as f:
                f.write(json.dumps(rec, default=str) + "\n")
        except OSError as exc:
            if not getattr(self, "_write_failed", False):
                self._write_failed = True
                print(
                    f"[opal] could not write the run records ({exc}); training continues",
                    file=sys.stderr,
                    flush=True,
                )

    def new_notes(self):
        f = self.dir / "notes.txt"
        if not f.exists():
            return ""
        text = f.read_text()
        new, self._notes_seen = text[self._notes_seen :], len(text)
        return new.strip()


class AgentWorker:
    def __init__(self, run: Run, backend, system_text: str, log=print):
        self.run, self.backend, self.system, self.log = run, backend, system_text, log
        self.thread = None
        self.ready = None
        self.k = 0
        self.cost = 0.0
        self.unknown_cost_decisions = set()  # a request stays unknown until it reports a numeric charge
        self.usage = None  # present only for token-reporting backends (Codex, including through a mailbox)
        self._last = None  # (k, req) of the newest request
        self.after_decision = None  # fn(k, req, resp), called once a response is recorded (a late one: in training)
        backend.setup(run.dir, run.workspace)
        (run.dir / "system.md").write_text(system_text)

    @property
    def recorded_cost(self):
        if self.unknown_cost_decisions or self.usage is not None or self.backend.name == "codex":
            return None
        return round(self.cost, 4)

    @property
    def busy(self) -> bool:
        return self.ready is not None or (self.thread is not None and self.thread.is_alive())

    def submit(self, req: dict) -> int:
        assert not self.busy
        self.k += 1
        self.unknown_cost_decisions.add(self.k)
        self._last = (self.k, req)
        self.thread = threading.Thread(
            target=self._decide, args=(self.k, req), name=f"opal-decision-{self.k}", daemon=True
        )
        self.thread.start()
        return self.k

    def wait(self):
        if self.thread is not None:
            self.thread.join()

    def take(self):
        resp, self.ready = self.ready, None
        return resp

    def collect_late(self):
        """An answer that reached the backend after it stopped waiting (a mailbox whose broker came back) while no
        newer request is out: checked and recorded like any answer, so it is applied at the next update boundary."""
        late = getattr(self.backend, "late_answer", None)
        if late is None or self.busy or self._last is None:
            return
        try:
            out = late()
        except Exception:
            return  # an unreadable mailbox never stops training; the next request says the answer was discarded
        if out is not None:
            self._finish(*self._last, dict(out, late=True))

    def _decide(self, k, req):
        rec = self.run.decisions / f"{k:03d}"
        ws = self.run.workspace
        try:
            rec.mkdir(parents=True, exist_ok=True)
            (ws / "scratch").mkdir(exist_ok=True)
            programs.write_dir(ws, req["files"])  # the workspace starts from the running programs at every request
            (ws / "inbox" / f"decision-{k:03d}.json").write_text(json.dumps(req["inbox"], indent=1, default=str))
            message = format_message(k, req)
            (rec / "message.txt").write_text(message)
            self.backend.log = lambda text, k=k: self.log(f"decision {k}: {redact_secrets(text)}")
            self.backend.training_waits = bool(req.get("blocking"))  # for the line when the agent is unavailable
            out = self.backend.decide(ws, message, self.system, rec)
        except Exception:
            out = dict(text="", cost_usd=None, returncode=-1, error=traceback.format_exc(limit=6)[-3000:], seconds=0.0)
        self._finish(k, req, out)

    def _finish(self, k, req, out):
        """Check the programs the agent left in the workspace and record its answer as ready."""
        rec, ws = self.run.decisions / f"{k:03d}", self.run.workspace
        out = dict(out)
        for field in ("error", "text"):
            if out.get(field) is not None:
                out[field] = redact_secrets(out[field])
        files, changed, errors = None, False, []
        try:
            files = programs.program(programs.read_dir(ws))
            changed = files != req["files"]
            failed = out.get("error") or out.get("returncode") not in (0, None)
            if changed and failed:
                errors = [
                    "the agent call did not finish ("
                    + str(out.get("error") or f"return code {out.get('returncode')}")[:300]
                    + "); its edits were not applied"
                ]
            elif changed:
                errors = programs.check_syntax(files)
        except Exception:
            errors = ["could not read the edited files: " + traceback.format_exc(limit=3)[-800:]]
        charge = out.get("cost_usd")
        if isinstance(charge, (int, float)) and not isinstance(charge, bool) and math.isfinite(charge):
            self.cost += float(charge)
            self.unknown_cost_decisions.discard(k)
        if isinstance(out.get("usage"), dict):
            if self.usage is None:
                self.usage = {}
            for key, value in out["usage"].items():
                self.usage[key] = self.usage.get(key, 0) + value
        error = out.get("error")
        if not error and out.get("returncode") not in (0, None):  # a failed call always carries an error
            error = f"the agent call ended with exit code {out.get('returncode')}"
        resp = dict(
            k=k,
            requested_at=req["update"],
            base=req["files"],
            files=files if changed and not errors else None,
            errors=errors,
            text=(out.get("text") or "")[-8000:],
            cost_usd=out.get("cost_usd"),
            returncode=out.get("returncode"),
            backend_error=error or None,
            seconds=round(out.get("seconds") or 0.0, 1),
            **({"unavailable_seconds": out["unavailable_seconds"]} if out.get("unavailable_seconds") else {}),
            **{x: out[x] for x in ("usage", "session", "rollout") if x in out},
            **{x: True for x in ("gave_up", "late") if out.get(x)},
        )
        try:  # written whole, so that `opal show` never reads half a file
            tmp = rec / "response.json.tmp"
            tmp.write_text(json.dumps({x: v for x, v in resp.items() if x != "base"}, indent=1))
            os.replace(tmp, rec / "response.json")
            if errors and changed and files:  # the workspace goes back to the running programs next time; keep them
                programs.write_dir(rec / "rejected", files)
        except Exception:
            pass
        if self.after_decision is not None:
            try:
                self.after_decision(k, req, resp)
            except Exception:
                pass
        self.ready = resp


def _ms(v):
    return "not measured yet" if v is None else f"{v:.3g} ms"


def _chunks(values, n=100):
    if len(values) <= n:
        return values, 1
    size = -(-len(values) // n)
    return [sum(values[i : i + size]) / len(values[i : i + size]) for i in range(0, len(values), size)], size


def previous_answer(prev, last=None):
    """One sentence on what became of the agent's previous answer: when its edit took effect, or why it did not."""
    if not prev:
        return None
    k, r, u, kind = prev["k"], prev.get("requested_at") or 0, prev.get("applied_at") or 0, prev.get("outcome")
    if kind == "applied":
        s = (
            f"Your edit from decision {k} (requested at update {r:,}) took effect at update {u:,}, {u - r:,} updates "
            "after the state you saw"
        )
        if prev.get("late"):
            s += "; it arrived after the run had stopped waiting for it, and was applied when it came"
        if last and last.get("k") == k and last.get("reverted_at"):
            s += f"; it failed and was reverted at update {last['reverted_at']:,} (the errors below say why)"
        return s + "."
    if prev.get("gave_up"):
        why = str(prev.get("backend_error") or "no answer").split(" (")[0]
        return (
            f"Decision {k} (requested at update {r:,}) was not answered in time: the run stopped waiting for it "
            f"after {prev.get('seconds') or 0:.0f} s ({why}). If you answered it, your answer arrived after the run "
            "had stopped waiting and was discarded, and this folder was reset to the running programs."
        )
    if prev.get("backend_error") is not None:
        why = " ".join(str(prev["backend_error"]).split())[:200] or "no error text"
        return (
            f"Decision {k} was not answered: the agent call failed ({why}). The programs were not changed (requested "
            f"at update {r:,})."
        )
    if kind == "rejected":
        return (
            f"Decision {k}: your edit was not applied and was discarded at update {u:,} (the errors below say why); "
            "this folder holds the running programs."
        )
    if kind == "load_failed":
        return (
            f"Decision {k}: your new programs failed to load at update {u:,} and were discarded (see the errors "
            "below); this folder holds the running programs."
        )
    return f"Decision {k} (requested at update {r:,}, answered by update {u:,}): you kept the programs."


VALIDATION_SHOWN = 50  # validation measurements written into a message; all of them are in the inbox


def validation_line(k, dev) -> str:
    """The validation-set history, oldest first: 'update: loss, accuracy' for the most recent VALIDATION_SHOWN."""
    head = "Validation set (held out from training; not the test set in the goal)"
    if not dev:
        return head + ": no measurement yet."
    keys = [x for x in ("loss", "accuracy") if any(x in d for d in dev)]
    shown = dev[-VALIDATION_SHOWN:]
    which = (
        f"all {len(dev)} measurements so far"
        if len(shown) == len(dev)
        else f"the most recent {len(shown)} of {len(dev)} measurements (all in inbox/decision-{k:03d}.json)"
    )

    def entry(d):
        return f"{d['update']:,}: " + ", ".join(f"{d[x]:.4g}" if x in d else "-" for x in keys)

    return f"{head}, {' and '.join(keys)} at each evaluation, {which} (update: {', '.join(keys)}): " + "; ".join(
        entry(d) for d in shown
    )


def format_message(k, req) -> str:
    u, total = req["update"], req["total_updates"]
    ep = f", epoch {u / total * req['epochs']:.2f} of {req['epochs']:g}" if req.get("epochs") else ""  # epochs= given
    lines = [
        f"Decision {k}. Requested at update {u:,} of {total:,} ({100 * u / total:.1f}% of training{ep}), after that "
        "update's backward pass and before its parameter update."
    ]
    if req["blocking"]:
        lines.append(f"Training is paused until you finish; your edits take effect from update {u:,}.")
    else:
        lines.append(
            "Training continues while you work. Your edits take effect at the first update boundary after you finish; "
            "the next message tells you which update that was."
        )
    lines.append("Why now: " + "; ".join(req["reasons"]) + ".")
    if req.get("last"):  # max_decisions reached with this one
        lines.append("This is your last decision in this run.")
    elif "next_decision" in req:  # show_next_decision=True
        nxt = req["next_decision"]
        lines.append(f"The next scheduled decision is at update {nxt:,}." if nxt else "No scheduled decisions remain.")
    previous = previous_answer(req.get("previous"), req.get("last_applied"))
    if previous:
        lines.append(previous)
    loss, per = _chunks(req["inbox"]["losses"])
    if loss:
        lines.append(
            f"Training loss since your previous decision (mean per {per} update{'s' if per > 1 else ''}, {len(loss)} "
            f"values, from update {req['inbox']['losses_from']:,}): " + ", ".join(f"{v:.4g}" for v in loss)
        )
    else:
        lines.append("No training loss was passed to step(loss=...) since your previous decision.")
    if "validation" in req:  # wrap(validation=N): OPAL 0.1.1-validation
        lines.append(validation_line(k, req["validation"]))
    reports = req["inbox"]["reports"]
    made = max(req["inbox"].get("reports_made", len(reports)), len(reports))
    lines.append(
        f"Reports from your analysis since your previous decision: {made} (all in inbox/decision-{k:03d}.json)."
        if made == len(reports)
        else f"Reports from your analysis since your previous decision: {made}; the most recent {len(reports)} are "
        f"in inbox/decision-{k:03d}.json."
    )
    if reports:
        lines.append("Most recent report: " + json.dumps(reports[-1], default=str)[:6000])
    if req["inbox"].get("metrics"):
        lines.append(
            "Values the training script chose to show you (most recent 20): "
            + json.dumps(req["inbox"]["metrics"][-20:], default=str)[:3000]
        )
    skipped = req["inbox"].get("skipped") or []
    if skipped:
        shown = ", ".join(f"{u:,}" for u in skipped[:20]) + (", ..." if len(skipped) > 20 else "")
        lines.append(f"Updates your program skipped since your previous decision: {len(skipped)} ({shown}).")
    errs = req["inbox"]["errors"]
    if errs:
        lines.append(
            "Errors since your previous decision:\n"
            + "\n".join(f"- update {e['update']} in {e['where']}:\n{e['error']}" for e in errs[-10:])
        )
    else:
        lines.append("No errors since your previous decision.")
    t = req["timing"]
    if t.get("wall_ms_per_update") is not None:
        lines.append(
            f"Timing: {t['wall_ms_per_update']:.1f} ms of wall time per training update recently; your updater "
            f"{_ms(t.get('program_ms'))} per update (the starting program: {_ms(t.get('program_ms_start'))})."
        )
        lines.append(
            f"Profile of one update: your updater, {audit.describe(t.get('audit'))}; the starting program, "
            f"{audit.describe(t.get('audit_start'))}."
            + (
                f" Your analysis took {_ms(t['analysis_ms'])} at its last call."
                if t.get("analysis_ms") is not None
                else ""
            )
        )
    if req.get("notes"):
        lines.append("Note from the person running this training: " + req["notes"])
    so_far = f"Decisions so far: {k - 1}; scheduled decisions left: {req['scheduled_left']}"
    neutral = req.get("neutral")  # the neutral prompts: no cost
    lines.append(so_far + ("." if neutral else f"; agent cost so far: ${req['cost_so_far']:.2f}."))
    lines.append(
        "Your programs are in this folder ("
        + ", ".join(sorted(req["files"]))
        + "); inbox/ holds every input so far"
        + (
            (
                " and reference/ read-only copies of the training script and user code"
                if req.get("script_shared", True)
                else " and reference/ read-only copies of user code"
            )
            if req.get("shared_code", True)
            else ""
        )
        + "; put scratch scripts in scratch/ (only "
        "updater.py, analysis.py and the modules they import are the program). "
        + (aggressive.EDIT_SENTENCE if req.get("aggressive") else "Edit the programs in place, or keep them (a valid decision). ")
        + (
            aggressive.RATIONALE_SENTENCE
            if req.get("aggressive")  # OPAL 0.1.2-aggressive
            else "End with your rationale in the four parts of DECISIONS: observed, changed or kept, prediction, rules."
            if req.get("neutral")  # the neutral prompt
            else "End with a brief rationale: what you observed, what you changed or kept, the effect you predict, and "
            "whether it is a fixed schedule or a feedback rule."
        )
    )
    return "\n".join(lines)


def now_stamp():
    return time.strftime("%Y%m%d-%H%M%S")


def new_run_dir(root: Path) -> Path:
    """A fresh run folder: time, process id and a random suffix, so runs started in the same second never share one."""
    path = Path(root) / f"{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}-{secrets.token_hex(2)}"
    path.mkdir(parents=True, exist_ok=False)
    return path
