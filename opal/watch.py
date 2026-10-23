"""Print a run's events as they are written.

`opal watch <run>` prints one line per event, starting from the beginning of events.jsonl, then waits for more.
It returns 0 when the run has finished (`end` or `interrupted`, and no later `resumed`). A run killed without close()
(SIGTERM, SIGKILL, out of memory) writes neither: a positive `--timeout` gives up after that many seconds without a new
event, and `--timeout 0` prints what is already there; both then return 1.
"""

from __future__ import annotations

import time
from pathlib import Path

from .report import OUTCOMES, _finished, _index, _jsonl, _number, _response, _text, end_cost, outcome_label


def _mode(start) -> str:
    mode = start.get("blocking")
    if mode == "auto":
        return "auto-blocking"
    if mode:
        return "blocking"
    return "non-blocking"


def _cost(value):
    return f"${value:.2f}" if _number(value) else ""


def _secs(value):
    if not _number(value):
        return ""
    return f"{value:.0f} s" if float(value) >= 1 else f"{value:g} s"


def _say(line):
    print(line, flush=True)  # through a pipe (| grep, | tee) each line arrives when it is written


def event_line(run, event, events=None) -> str:
    """One line for one events.jsonl record, in the same register as `opal show`."""
    kind = event.get("kind") or "?"
    update = event.get("update", event.get("applied_at"))
    at = f"{update:,}" if _number(update) else ""
    if kind == "start":
        total = event.get("total_updates", "?")
        total = f"{total:,}" if _number(total) else total
        backend = event.get("backend") or "none"
        return f"start: {total} updates; backend {backend}; {_mode(event)}"
    if kind == "requested":
        return f"requested  k={event.get('k')}  at update {at}"
    if kind in (*OUTCOMES, "cancelled"):
        events = events or []
        k = event.get("k")
        req = next((e for e in events if e.get("kind") == "requested" and e.get("k") == k), {})
        label = outcome_label(events, req, event)
        resp = _response(run, k) if _index(k) else {}
        said = " ".join(_text(resp.get("text")).split())[:70]
        cost, secs = _cost(resp.get("cost_usd")), _secs(resp.get("seconds"))
        extra = "  ".join(x for x in (secs, cost, said) if x)
        applied = f"{event.get('applied_at'):,}" if _number(event.get("applied_at")) else at
        where = f"  at update {applied}" if applied else ""  # a cancelled decision has none
        return f"{label:<10} k={k}{where}" + (f"  {extra}" if extra else "")
    if kind == "estimate":
        n = event.get("decisions", "?")
        extra = f"; capped from {event['capped_from']}" if _number(event.get("capped_from")) else ""
        return f"estimate at update {at}: {n} scheduled{extra}"
    if kind == "end":
        n = event.get("decisions")
        n = f"{n} decisions; " if _number(n) else ""
        start = next((e for e in events or [] if e.get("kind") == "start"), {})
        return f"finished at update {at}; {n}{end_cost(start, event)}"
    if kind == "interrupted":
        return f"interrupted at update {at}"
    if at:
        return f"{kind} at update {at}"
    return kind


def follow(run, *, interval=1.0, timeout=None, sleep=time.sleep, now=time.monotonic, write=_say):
    """Print new events until the run finishes (returns 0), or until `timeout` seconds pass without one (returns 1; 0:
    do not wait)."""
    run = Path(run)
    write(f"watching {run}")
    shown = []  # the records printed so far
    last_new = now()
    while True:
        events = _jsonl(run / "events.jsonl")
        if events[: len(shown)] != shown:
            write(
                "events.jsonl was rewritten (a resume moves the records after its checkpoint to attempts/); printing "
                "it again from the start"
            )
            shown = []
        fresh = events[len(shown) :]
        shown = events
        if fresh:
            last_new = now()
        for e in fresh:
            if not isinstance(e, dict) or not isinstance(e.get("kind"), str):
                continue
            line = event_line(run, e, events)
            write(line)
            if e.get("kind") == "start" and e.get("goal"):
                write(f"goal: {e['goal']}")
        if _finished(events):
            return 0
        if timeout is not None and (timeout <= 0 or now() - last_new >= timeout):
            last = next((e.get("update") for e in reversed(events) if _number(e.get("update"))), None)
            at = f" (last event at update {last:,})" if last is not None else ""
            write(f"no end event yet{at}" if timeout <= 0 else f"no new event for {timeout:g} s{at}")
            return 1
        sleep(interval)
