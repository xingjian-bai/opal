"""Set two OPAL run folders, or two result.json files, beside each other.

`opal compare a b` prints the metric and command when those files exist, the decision counts, cost and waiting,
and the decisions that differ, matched by the update they were asked at. A run folder is one with `events.jsonl`. A
result.json written by an example script (`--out`) is enough for the metric and command alone; when the run folder it
names (`opal_run`) is on this machine, its decisions are compared too.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from .report import (
    OUTCOMES,
    _finished,
    _index,
    _jsonl,
    _list,
    _number,
    _object,
    _response,
    _text,
    decision_counts,
    describe_counts,
    outcome_label,
)


def load_side(path):
    """A run folder (events.jsonl) and/or a result.json. Raises FileNotFoundError or ValueError with an opal: prefix."""
    p = Path(path).expanduser()
    if p.is_file():
        return _from_result_file(p)
    if not p.is_dir():
        raise FileNotFoundError(f"opal: not a run folder or result.json: {p}")
    if (p / "events.jsonl").exists():
        result = _object(p / "result.json") if (p / "result.json").exists() else {}
        return snapshot(p, result=result if isinstance(result, dict) else {})
    if (p / "result.json").exists():
        return _from_result_file(p / "result.json")
    inside = sorted(d.name for d in p.iterdir() if (d / "events.jsonl").exists())
    hint = f"; runs inside: {', '.join(inside[-5:])}" if inside else ""
    raise FileNotFoundError(f"opal: not a run folder: {p} (no events.jsonl){hint}")


def _from_result_file(path: Path):
    try:
        obj = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise ValueError(f"opal: not a result.json: {path} ({exc})") from exc
    if not isinstance(obj, dict) or "metric" not in obj:
        raise ValueError(f"opal: not a result.json: {path} (missing metric)")
    run = None
    listed = obj.get("opal_run")
    if isinstance(listed, str) and listed and (Path(listed) / "events.jsonl").exists():
        run = Path(listed)
    elif (path.parent / "events.jsonl").exists():
        run = path.parent
    if run is not None:
        return dict(snapshot(run, result=obj), path=str(path))
    return _result_only(path, obj)


def snapshot(run, result=None) -> dict:
    """JSON-serialisable summary of one run folder. `result` is the example script's result.json when present."""
    run = Path(run)
    events = _jsonl(run / "events.jsonl")
    start = next((e for e in events if e.get("kind") == "start"), {})
    end = next((e for e in reversed(events) if e.get("kind") in ("end", "interrupted")), {})
    result = result if isinstance(result, dict) else {}
    counts = decision_counts(events) if events else {}
    wait = result.get("agent_wait_seconds")
    if not _number(wait):
        wait = end.get("agent_wait_seconds")
        if not _number(wait):
            wait = next(
                (e.get("agent_wait_seconds") for e in reversed(events) if _number(e.get("agent_wait_seconds"))),
                None,
            )
    cost = result.get("agent_cost_usd")
    if not _number(cost):
        cost = end.get("cost_usd") if _number(end.get("cost_usd")) else None
    return dict(
        path=str(run),
        run=str(run),
        kind="run",
        goal=_text(start.get("goal")) or None,
        backend=start.get("backend"),
        blocking=start.get("blocking"),
        total_updates=start.get("total_updates"),
        update=end.get("update"),
        ended=end.get("kind"),
        finished=_finished(events),
        counts=counts,
        cost_usd=cost,
        agent_wait_seconds=wait if _number(wait) else None,
        last_loss=_last_loss(run),
        metric=result.get("metric") if isinstance(result.get("metric"), dict) else None,
        command=result.get("command"),
        hardware=result.get("hardware"),
        opal_commit=result.get("opal_commit"),
        decisions=_decisions(run, events),
    )


def _result_only(path: Path, obj: dict) -> dict:
    metric = obj.get("metric") if isinstance(obj.get("metric"), dict) else None
    dec = obj.get("decisions") if isinstance(obj.get("decisions"), dict) else None
    counts = {}
    if dec:
        counts = dict(
            scheduled=None,
            asked=dec.get("asked"),
            answered=None,
            applied=dec.get("applied"),
            failed=dec.get("failed"),
        )
    return dict(
        path=str(path),
        run=None,
        kind="result",
        goal=None,
        backend=None,
        blocking=None,
        total_updates=None,
        update=None,
        finished=True,
        counts=counts,
        cost_usd=obj.get("agent_cost_usd"),
        agent_wait_seconds=obj.get("agent_wait_seconds"),
        last_loss=None,
        metric=metric,
        command=obj.get("command"),
        hardware=obj.get("hardware"),
        opal_commit=obj.get("opal_commit"),
        decisions=[],
    )


def _last_loss(run: Path):
    """The last loss recorded, NaN or infinite included: a run that diverged shows it."""
    last = None
    for r in _jsonl(run / "losses.jsonl"):
        for v in _list(r.get("losses")):
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                last = v
    return last


def _decisions(run: Path, events) -> list:
    req = {e["k"]: e for e in events if e.get("kind") == "requested" and _index(e.get("k"))}
    out = {
        e["k"]: e
        for e in events
        if e.get("kind") in (*OUTCOMES, "cancelled") and _index(e.get("k"))
    }
    rows = []
    for k in sorted(set(req) | set(out)):
        o = out.get(k, {})
        r = req.get(k, {})
        resp = _response(run, k)
        rows.append(
            dict(
                k=k,
                asked_at=r.get("update", o.get("requested_at")),
                outcome=outcome_label(events, r, o),
                applied_at=o.get("applied_at"),
                seconds=resp.get("seconds"),
                cost_usd=resp.get("cost_usd"),
                changed=_list(o.get("changed")),
            )
        )
    return rows


def _by_update(decisions) -> dict:
    """Decisions keyed by the update they were asked at (and their order among those asked at that update): k counts
    a run's own requests, so it drifts between runs whose schedules differ."""
    out, seen = {}, {}
    for d in decisions:
        u = d["asked_at"] if _number(d.get("asked_at")) else None
        seen[u] = seen.get(u, -1) + 1
        out[(u, seen[u])] = d
    return out


def compare(a, b) -> dict:
    """Two snapshots and the decisions that differ, matched by the update they were asked at: asked in one run only,
    or with different outcomes (empty when either side has no run folder)."""
    left, right = load_side(a), load_side(b)
    differ, same = [], 0
    if left["kind"] == "run" and right["kind"] == "run":
        by_a, by_b = _by_update(left["decisions"]), _by_update(right["decisions"])
        for key in sorted(set(by_a) | set(by_b), key=lambda key: (key[0] is None, key[0] or 0, key[1])):
            da, db = by_a.get(key), by_b.get(key)
            if da and db and da["outcome"] == db["outcome"]:
                same += 1
                continue
            differ.append(
                dict(
                    asked_at=key[0],
                    a=da["outcome"] if da else None,
                    a_k=da["k"] if da else None,
                    b=db["outcome"] if db else None,
                    b_k=db["k"] if db else None,
                )
            )
    return dict(a=left, b=right, matched_by="asked_at", same_decisions=same, differing_decisions=differ)


def jsonable(obj):
    """obj with NaN and infinite numbers as the strings "nan", "inf" and "-inf", for which JSON has no numbers."""
    if isinstance(obj, float) and not math.isfinite(obj):
        return str(obj)
    if isinstance(obj, dict):
        return {k: jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    return obj


def _cell(value, kind="plain"):
    if value is None or value == "":
        return "—"
    if kind == "cost":
        return f"${value:.2f}" if _number(value) else "—"
    if kind == "wait":
        return f"{value:g} s" if _number(value) else "—"
    if kind == "loss":
        if isinstance(value, float) and not math.isfinite(value):
            return str(value)  # nan or inf: the run diverged
        return f"{value:.5g}" if _number(value) else "—"
    if kind == "updates":
        return f"{value:,}" if _number(value) else "—"
    if kind == "mode":
        if value == "auto":
            return "auto-blocking"
        if value is True or value == "blocking":
            return "blocking"
        if value is False:
            return "non-blocking"
        return str(value)
    if kind == "metric":
        if not isinstance(value, dict) or not value.get("name"):
            return "—"
        direction = value.get("direction") or ""
        shown = f"{value['name']} {value.get('value')}"
        if direction in ("lower", "higher"):
            shown += f" ({direction} is better)"
        return shown
    return str(value)


def _counts_cell(side):
    """Asked, applied and failed agent calls, the same for a run folder and a result.json (which has no schedule)."""
    c = side.get("counts") or {}
    return ", ".join(f"{c[x]} {x}" for x in ("asked", "applied", "failed") if _number(c.get(x))) or "—"


def _ended(side):
    if side["kind"] != "run":
        return "—"
    if not side.get("finished"):
        return "no end event yet"  # still training, or stopped without close()
    at = _cell(side.get("update"), "updates")
    return f"interrupted at {at}" if side.get("ended") == "interrupted" else f"at update {at}"


def _where(side):
    """The path given, and the run folder a result.json points at."""
    run = side.get("run")
    return side["path"] + (f" (run folder {run})" if run and run != side["path"] else "")


def _decision(row, side):
    return f"k={row[side + '_k']} {row[side]}" if row[side] else "not asked"


def lines(cmp) -> list:
    """Human lines for `opal compare`, in the same register as `opal show`."""
    a, b = cmp["a"], cmp["b"]
    out = [f"a: {_where(a)}", f"b: {_where(b)}"]
    if a.get("goal") or b.get("goal"):
        mark = "*" if a.get("goal") != b.get("goal") else " "
        out.append(f"{mark} goal a: {a.get('goal') or '—'}")
        out.append(f"{mark} goal b: {b.get('goal') or '—'}")
    scheduled = [_cell((side.get("counts") or {}).get("scheduled"), "updates") for side in (a, b)]
    rows = [
        ("updates", _cell(a.get("total_updates"), "updates"), _cell(b.get("total_updates"), "updates")),
        ("ended", _ended(a), _ended(b)),
        ("backend", _cell(a.get("backend")), _cell(b.get("backend"))),
        ("mode", _cell(a.get("blocking"), "mode"), _cell(b.get("blocking"), "mode")),
        ("scheduled", *scheduled),
        ("decisions", _counts_cell(a), _counts_cell(b)),
        ("cost", _cell(a.get("cost_usd"), "cost"), _cell(b.get("cost_usd"), "cost")),
        ("wait", _cell(a.get("agent_wait_seconds"), "wait"), _cell(b.get("agent_wait_seconds"), "wait")),
        ("last loss", _cell(a.get("last_loss"), "loss"), _cell(b.get("last_loss"), "loss")),
        ("metric", _cell(a.get("metric"), "metric"), _cell(b.get("metric"), "metric")),
    ]
    width = max(len(name) for name, _, _ in rows)
    col = max(24, max(len(x) for row in rows for x in row[1:]))
    out.append(f"  {'':{width}}  {'a':<{col}}  b")
    for name, left, right in rows:
        mark = "*" if left != right else " "
        out.append(f"{mark} {name:{width}}  {left:<{col}}  {right}")
    if a.get("command") or b.get("command"):  # below the table: commands are long
        mark = "*" if a.get("command") != b.get("command") else " "
        out.append(f"{mark} command a: {_cell(a.get('command'))}")
        out.append(f"{mark} command b: {_cell(b.get('command'))}")
    for side, label in ((a, "a"), (b, "b")):
        c = side.get("counts") or {}
        if side["kind"] == "run" and c:
            out.append(f"{label}: {describe_counts(c)}")
    if a["kind"] != "run" or b["kind"] != "run":
        who = "a and b have" if a["kind"] == b["kind"] else "a has" if a["kind"] != "run" else "b has"
        out.append(f"{who} only a result.json, without its run folder (events.jsonl) here: decisions are not compared.")
        return out
    differ, same = cmp["differing_decisions"], cmp["same_decisions"]
    if not differ and not same:
        out.append("no decisions were asked in either run.")
        return out
    s = "s" if same != 1 else ""
    if not differ:
        out.append(f"all {same} decision{s} asked at the same update{s} in both runs, with the same outcome{s}.")
        return out
    out.append("decisions that differ, matched by the update they were asked at (k: each run's own decision number):")
    out.append(f"{'asked':>9}  {'a':<16}  b")
    for row in differ:
        asked = f"{row['asked_at']:,}" if _number(row.get("asked_at")) else "?"
        out.append(f"{asked:>9}  {_decision(row, 'a'):<16}  {_decision(row, 'b')}")
    if same:
        out.append(f"{same} other decision{s} asked at the same update{s} in both runs: same outcome{s}.")
    return out
