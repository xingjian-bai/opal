"""One self-contained HTML page per run: the loss, when each decision was asked and applied, a table of the code
changes, what the agent said, and the code it changed. No network access needed to view it. During a run, rank 0
rewrites it after each decision (LiveReport); until the run ends the page reloads itself every minute."""

from __future__ import annotations

import ast
import datetime
import difflib
import html
import json
import math
import os
import queue
import re
import threading
from pathlib import Path

from . import summaries

_CONST = re.compile(r"[A-Z][A-Z0-9_]*\Z")
_NOT_LITERAL = object()


_READ = {}  # path -> (bytes parsed so far, the last bytes of them, records): the live report reads only what is new
_READ_LOCK = threading.Lock()


def _jsonl(path):
    try:
        return _read_jsonl(path)
    except OSError:
        return []  # absent, unreadable, or replaced between exists() and open()


def _read_jsonl(path):
    """Records of a .jsonl file; a line being written right now (or broken) is skipped. Appended files are parsed from
    where the previous call stopped; a file that was rewritten (shorter, or other bytes before that point) from the
    start."""
    path = Path(path)
    if not path.exists():
        return []
    with _READ_LOCK, open(path, "rb") as f:
        key = str(path.resolve())
        done, tail, records = _READ.get(key, (0, b"", []))
        size = os.fstat(f.fileno()).st_size
        if done:
            f.seek(max(0, done - len(tail)))
            if size < done or f.read(len(tail)) != tail:
                done, tail, records = 0, b"", []
        f.seek(done)
        data = f.read()
        end = data.rfind(b"\n") + 1  # complete lines only; the rest is read again next time
        records = list(records)
        for line in data[:end].splitlines():
            try:
                record = json.loads(line)
                if isinstance(record, dict):
                    records.append(record)
            except ValueError:
                pass
        done += end
        _READ[key] = (done, (tail + data[:end])[-64:], records)
        out = list(records)
    if data[end:].strip():  # a last line without its newline: complete in a file that is no longer written
        try:
            record = json.loads(data[end:])
            if isinstance(record, dict):
                out.append(record)
        except ValueError:
            pass
    return out


def _diff(old: dict, new: dict) -> str:
    out = []
    for line in summaries.unified_diff(old, new).splitlines():
        cls = (
            "a"
            if line.startswith("+") and not line.startswith("+++")
            else "r"
            if line.startswith("-") and not line.startswith("---")
            else ""
        )
        out.append(f'<span class="{cls}">{html.escape(line)}</span>' if cls else html.escape(line))
    return "\n".join(out)


def _js(obj) -> str:
    """JSON for inside a <script> element: agent text must not be able to close the element."""
    return json.dumps(obj, allow_nan=False).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")


def constants(text: str) -> dict:
    """Module-level UPPER_CASE assignments with a literal value; a name ever bound to anything else is left out."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return {}
    out = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets, value = [node.target], node.value
        else:
            continue
        for t in targets:
            if isinstance(t, ast.Name) and _CONST.match(t.id) and out.get(t.id) is not _NOT_LITERAL:
                try:
                    out[t.id] = ast.literal_eval(value)
                except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
                    out[t.id] = _NOT_LITERAL
    return {k: v for k, v in out.items() if v is not _NOT_LITERAL}


def _short(v, n=48):
    s = repr(v)
    return s if len(s) <= n else s[: n - 1] + "…"


def file_changes(old: dict, new: dict) -> list:
    """One entry per changed file: name, lines added and removed, and whether it is new or deleted."""
    out = []
    for name in sorted(set(old) | set(new)):
        a, b = old.get(name, "").splitlines(), new.get(name, "").splitlines()
        if a == b:
            continue
        lines = list(difflib.unified_diff(a, b, lineterm="", n=0))
        add = sum(1 for x in lines if x.startswith("+") and not x.startswith("+++"))
        rem = sum(1 for x in lines if x.startswith("-") and not x.startswith("---"))
        status = "new" if name not in old else "deleted" if name not in new else ""
        out.append(dict(name=name, added=add, removed=rem, status=status))
    return out


def constant_changes(old: dict, new: dict) -> list:
    """(file, name, old value, new value) for each module-level constant that changed; None where it is absent."""
    out = []
    for name in sorted(set(old) | set(new)):
        if old.get(name) == new.get(name):
            continue
        a, b = constants(old.get(name, "")), constants(new.get(name, ""))
        for c in sorted(set(a) | set(b)):
            ra, rb = (repr(a[c]) if c in a else None), (repr(b[c]) if c in b else None)
            if ra != rb:
                out.append((name, c, _short(a[c]) if c in a else None, _short(b[c]) if c in b else None))
    return out


def _number(value):
    try:
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
    except OverflowError:
        return False


def _index(value):
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _text(value):
    return value if isinstance(value, str) else ""


def _list(value):
    return value if isinstance(value, list) else []


def _events(run):
    return [
        e
        for e in _jsonl(Path(run) / "events.jsonl")
        if isinstance(e.get("kind"), str) and ("k" not in e or _index(e["k"]))
    ]


def _read_text(path):
    try:
        return Path(path).read_text(errors="replace")
    except OSError:
        return ""


def _object(path):
    try:
        obj = json.loads(_read_text(path))
        return obj if isinstance(obj, dict) else {}
    except ValueError:
        return {}


def _cached_summary(record):
    obj = _object(record / "summary.json")
    if not _text(obj.get("summary")).strip():
        return None
    return dict(obj, tags=[t for t in _list(obj.get("tags")) if t in summaries.TAGS])


def _versions(run: Path) -> dict:
    # Static text only: never import agent programs to make a report. Include old knobs-mode snapshots.
    out = {}
    for d in sorted((run / "programs").glob("v[0-9]*")):
        if not re.fullmatch(r"v[0-9]+", d.name) or not d.is_dir() or d.is_symlink():
            continue
        out[d.name] = {
            p.name: _read_text(p)
            for p in [*sorted(d.glob("*.py")), d / "knobs.json"]
            if p.is_file() and not p.is_symlink()
        }
    return out


def _response(run: Path, k: int) -> dict:
    return _object(run / "decisions" / f"{k:03d}" / "response.json")


def _agent_metadata(record):
    # Backend transcripts can be large. Only retain explicit init metadata, not tool messages or token usage.
    try:
        with (record / "backend.jsonl").open(errors="replace") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                except ValueError:
                    continue
                if isinstance(obj, dict) and obj.get("type") == "system" and obj.get("subtype") == "init":
                    return {k: obj[k] for k in ("model", "effort", "cwd", "hostname", "username") if k in obj}
    except OSError:
        pass
    return {}


def decision_rows(run, events=None, versions=None) -> list:
    """One row per decision that changed the programs, in order: when it was asked and applied, the files and
    constants it changed, its rationale, its cached summary (or None), and the two program versions."""
    run = Path(run)
    events = _events(run) if events is None else events
    versions = _versions(run) if versions is None else versions
    requested = {e["k"]: e for e in events if e.get("kind") == "requested" and _index(e.get("k"))}
    start = next((e for e in events if e.get("kind") == "start"), {})
    rows = []
    for e in events:
        if e.get("kind") != "applied" or not _index(e.get("k")):
            continue
        k, v = e["k"], e.get("version")
        old_key, new_key = (f"v{v - 1:03d}", f"v{v:03d}") if _index(v) else (None, None)
        available = old_key in versions and new_key in versions
        old, new = (versions[old_key], versions[new_key]) if available else ({}, {})
        rows.append(
            dict(
                k=k,
                requested=e.get("requested_at", requested.get(k, {}).get("update")),
                applied=e.get("applied_at"),
                version=v,
                snapshots_available=available,
                changed=_list(e.get("changed")),
                analysis_only=e.get("rebuilt_updater") is False or e.get("changed") == ["analysis.py"],
                files=file_changes(old, new),
                constants=constant_changes(old, new),
                rationale=_text(_response(run, k).get("text")).strip(),
                summary=_cached_summary(run / "decisions" / f"{k:03d}"),
                total=start.get("total_updates"),
                old=old,
                new=new,
            )
        )
    return rows


def report_series(run, min_reports=3, limit=12):
    """Numbers the agent's analysis reported (reports.jsonl), as series keyed by path up to depth 2 (`grad_norm.embed`):
    those present in at least `min_reports` reports, at most `limit` of them in order of first appearance. Returns the
    kept series and how many qualified."""
    series = {}
    for r in _jsonl(Path(run) / "reports.jsonl"):
        rep, u = r.get("report"), r.get("update")
        if not isinstance(rep, dict) or not _number(u):
            continue
        for key, v in rep.items():
            if key == "update":  # the x axis itself
                continue
            leaves = v.items() if isinstance(v, dict) else [(None, v)]
            for sub, x in leaves:
                if _number(x):
                    series.setdefault(str(key) if sub is None else f"{key}.{sub}", []).append([u, x])
    keep = [(k, pts) for k, pts in series.items() if len(pts) >= min_reports and len({x for _, x in pts}) > 1]
    return keep[:limit], len(keep)


def _finished(events) -> bool:
    last = lambda kinds: max((i for i, e in enumerate(events) if e.get("kind") in kinds), default=-1)  # noqa: E731
    return last(("end", "interrupted")) > last(("resumed",))


OUTCOMES = ("applied", "kept", "rejected", "load_failed")


def outcome_label(events, requested, outcome) -> str:
    """What `opal show` calls a decision, from its requested event and its outcome event ({} while there is none):
    changed, kept, rejected, load_failed, cancelled, failed (the agent call failed), pending, or lost (no answer, and
    the run ended or resumed after it was asked)."""
    kind = outcome.get("kind", "pending")
    if kind == "pending" and any(
        e.get("kind") in ("resumed", "end") and e.get("utc", "") > requested.get("utc", "") for e in events
    ):
        kind = "lost"
    if kind == "applied":
        kind = "changed"
    elif outcome.get("failed") and not outcome.get("gave_up"):
        kind = "failed"
    return kind


def end_cost(start, end) -> str:
    """The agent cost of a finished run as `opal show` prints it; USD unavailable for a token-reporting backend."""
    tokens = "usage" in end or start.get("backend") == "codex"
    return "USD unavailable" if tokens and end.get("cost_usd") is None else f"${end.get('cost_usd') or 0:.2f}"


def decision_counts(events) -> dict:
    """Scheduled, asked, answered and applied decisions of a run, so a run whose agent was mostly not asked shows at a
    glance. Answered: the agent's answer came (applied, kept or rejected), not a failed or abandoned backend call.
    Applied: it changed the programs. With blocking="auto", how many decisions waited, and the median lag in updates
    from request to effect of the answered ones."""
    start = next((e for e in events if e.get("kind") == "start"), {})
    asked = {e["k"] for e in events if e.get("kind") == "requested" and _index(e.get("k"))}
    schedule = next((e["schedule"] for e in reversed(events)
                     if e.get("kind") in ("start", "estimate", "resumed") and "schedule" in e), [])
    outcomes = [e for e in events if e.get("kind") in OUTCOMES and _index(e.get("k"))]
    answered = [e for e in outcomes if not (e.get("gave_up") or e.get("failed"))]
    lags = sorted(e["lag_updates"] for e in answered if _number(e.get("lag_updates")))
    waits = [e["wait"] for e in events if e.get("kind") == "requested" and e.get("wait")]
    return dict(
        scheduled=len(_list(schedule)),
        asked=len(asked),
        answered=len({e["k"] for e in answered}),
        applied=len({e["k"] for e in outcomes if e.get("kind") == "applied"}),
        failed=len({e["k"] for e in outcomes if e.get("failed") and not e.get("gave_up")}),
        unavailable_seconds=sum(e["unavailable_seconds"] for e in outcomes if _number(e.get("unavailable_seconds"))),
        waited=waits.count("waited"),
        did_not_wait=waits.count("did not wait"),
        median_lag_updates=lags[len(lags) // 2] if lags else None,
        mode=start.get("blocking"),
        max_lag_updates=start.get("max_lag_updates"),
    )


def describe_counts(c) -> str:
    s = (
        f"Decisions: {c['scheduled']} scheduled, {c['asked']} asked, {c['answered']} answered, {c['applied']} applied "
        "(changed the programs)"
    )
    if c.get("failed"):
        s += f"; {c['failed']} agent call{'s' if c['failed'] > 1 else ''} failed"
    if c.get("unavailable_seconds"):
        s += f"; {c['unavailable_seconds'] / 60:.1f} min waiting for the agent to be available"

    if c["mode"] == "auto":
        s += (
            f"; training waited at {c['waited']} and did not wait at {c['did_not_wait']} (blocking=auto, limit "
            f"{c['max_lag_updates']} updates)"
        )
    if c["median_lag_updates"] is not None:
        s += f"; median lag from request to effect {c['median_lag_updates']:,} updates"
    return s + "."


def defaults_flag(events) -> str:
    """The banner of a run whose OPAL install had its default prompt or starting programs edited (opal/defaults.py),
    or that used a custom prompt or starting program; empty otherwise."""
    start = next((e for e in events if e.get("kind") == "start"), {})
    out = ""
    if start.get("defaults_modified"):
        out += (
            '<p class="flag"><b>OPAL\'s own defaults were modified</b> in the install that ran this: '
            + html.escape(", ".join(str(f) for f in _list(start.get("modified_files"))))
            + " differ from the release, so the agent did not get the default prompt or starting programs.</p>\n"
        )
    for key, what in (("custom_prompt", "a custom prompt"), ("custom_program", "a custom starting program")):
        if start.get(key):
            out += f"<p>This run used {what} ({html.escape(str(start[key]))}); the blog's results used OPAL's defaults.</p>\n"
    return out


def rank_repair_summary(events) -> str:
    """A persistent warning, including repairs from earlier attempts in a resumed run folder."""
    updates = {e["update"] for e in events if e.get("kind") == "ranks_resynced" and _index(e.get("update"))}
    for e in events:
        if e.get("kind") == "resumed":
            updates.update(u for u in _list(e.get("ranks_repaired_at")) if _index(u))
    return "ranks repaired at updates " + ", ".join(str(u) for u in sorted(updates)) if updates else ""


def rank_flag(events) -> str:
    summary = rank_repair_summary(events)
    return (
        f'<p class="flag"><b>{html.escape(summary)}.</b> Parameters were broadcast from rank 0; '
        "optimizer state was not repaired. This run's trajectory was changed.</p>\n"
        if summary else ""
    )


def integrity_flag(events) -> str:
    """The banner of a run whose program moved the parameters for validation evaluations and moved them back
    (eval_integrity events, misc/integrity.py); empty otherwise."""
    found = [e for e in events if e.get("kind") == "eval_integrity" and e.get("flagged")]
    if not found:
        return ""
    items = []
    for e in found[:6]:
        tensors_record = [
            r
            for r in _list(e.get("tensors"))
            if isinstance(r, dict)
            and isinstance(r.get("name"), str)
            and _number(r.get("norm_factor", 1))
            and _number(r.get("changed_at"))
            and _number(r.get("restored_at"))
        ]
        tensors = ", ".join(
            f"{r['name']} (x{r.get('norm_factor', 1):.3g} in norm, changed at update {r['changed_at']:,}, back at "
            f"{r['restored_at']:,})"
            for r in tensors_record[:3]
        )
        more = f" and {e['n_tensors'] - 3} more" if _number(e.get("n_tensors")) and e["n_tensors"] > 3 else ""
        items.append(f"update {_n(e.get('update'))}: {tensors or 'tensor details not recorded'}{more}")
    rest = f"; {len(found) - 6} more in events.jsonl" if len(found) > 6 else ""
    return (
        f'<p class="flag"><b>Flagged: evaluation integrity.</b> At {len(found)} validation evaluation'
        f"{'s' if len(found) > 1 else ''} the program had moved parameters to another point and moved them back "
        f"after the evaluation, so those measurements do not describe the trained model: "
        + html.escape("; ".join(items) + rest)
        + ".</p>\n"
    )


def _bi(en, zh):
    return f'<span data-lang="en">{html.escape(en)}</span><span data-lang="zh">{html.escape(zh)}</span>'


def _n(value):
    return f"{value:,}" if _number(value) else "?"


def _sentences(text):
    text = re.sub(r"(?m)^\s*(?:#+\s*|\d+[.)]\s*|[-*]\s+)", "", _text(text))
    text = " ".join(text.replace("**", "").replace("`", "").split())
    return re.split(r'(?<=[.!?。])\s+(?=[A-Z0-9“"\'])', text) if text else []


def _excerpt(text, limit=180, reason=False):
    sentences = _sentences(text)
    if reason:
        # Extract the agent's words; never turn a prediction into a measured explanation.
        introductory = r"\b(because|reason|observ\w*|showed|plateau\w*|stalled|stopped|slow\w*|taking|so I|so we)\b"
        early = [s for s in sentences[:3] if len(s) >= 35 and re.search(introductory, s, re.I)]
        for pattern in (r"\b(because|reason|so that)\b", r"\b(observ\w*|showed|plateau\w*|stalled|indicat\w*)\b"):
            matches = early or [
                s
                for s in sentences
                if len(s) >= 35 and re.search(pattern, s, re.I) and not re.match(r"(?:That|This|It|They)\b", s)
            ]
            if matches:
                sentences = matches
                break
    text = sentences[0] if sentences else ""
    if len(text) > limit:
        text = text[: limit - 1].rsplit(" ", 1)[0].rstrip(" ,;:") + "…"
    return text


def _change(row):
    if row["summary"]:
        text = _text(row["summary"]["summary"])
        first = _excerpt(text, limit=500)
        if len(first) > 140:
            first = " ".join(re.sub(r"\([^()]*\)", "", first).split())
            for separator in (": ", "; ", ", while "):
                clause = first.split(separator, 1)[0]
                if 45 <= len(clause) <= 140 and len(clause) < len(first):
                    return clause.rstrip(".,") + "…"
        return _excerpt(first, limit=140)
    parts = []
    for _, name, old, new in row["constants"][:2]:
        parts.append(
            f"{name} = {new}" if old is None else f"Remove {name}" if new is None else f"{name}: {old} → {new}"
        )
    if parts:
        return "; ".join(parts)
    names = [f["name"] for f in row["files"]] or [str(n) for n in row["changed"]]
    return "Changed " + (", ".join(names) or "programs") + " (summary unavailable)"


def _duration(value, zh=False):
    if not _number(value):
        return "未记录" if zh else "not recorded"
    if value < 60:
        return f"{value:.1f} 秒" if zh else f"{value:.1f} s"
    return f"{value / 60:.1f} 分钟" if zh else f"{value / 60:.1f} min"


def _wall_span(events):
    stamps = []
    for e in events:
        try:
            t = datetime.datetime.fromisoformat(_text(e.get("utc")).replace("Z", "+00:00"))
            if t.tzinfo is not None:
                stamps.append(t)
        except ValueError:
            pass
    return (max(stamps) - min(stamps)).total_seconds() if len(stamps) > 1 else None


def _table(rows):
    if not rows:
        return (
            "<h2>"
            + _bi("Changes", "改动")
            + "</h2><p>"
            + _bi("No applied changes recorded.", "未记录已应用的改动。")
            + "</p>"
        )
    body = []
    for r in rows:
        reason = _excerpt(r["rationale"], reason=True)
        why = f"“{html.escape(reason)}”" if reason else _bi("No rationale recorded.", "未记录理由。")
        tags = ", ".join(r["summary"].get("tags", [])) if r["summary"] else ""
        note = _bi("Analysis only", "仅改分析") if r["analysis_only"] else ""
        if r.get("reverted_at") is not None:
            note += " " + _bi(f"Reverted at update {_n(r['reverted_at'])}", f"在更新步 {_n(r['reverted_at'])} 回退")
        body.append(
            f'<tr><td class="number"><a href="#d{r["k"]}">{r["k"]}</a></td>'
            f'<td class="when" data-label="Update">{_n(r["requested"])} → {_n(r["applied"])}</td>'
            f'<td class="what">{html.escape(r["change"])}<div class="muted">{note}</div></td>'
            f'<td class="why" data-label="Why">{why}</td><td class="tags" data-label="Tags">{html.escape(tags) or "—"}</td></tr>'
        )
    heads = [
        ("#", "#"),
        ("Update requested → applied", "更新步：请求 → 应用"),
        ("What changed", "改动"),
        ("Why, in the agent’s words", "理由，摘自 agent 原话"),
        ("Tags", "标签"),
    ]
    return (
        "<h2>"
        + _bi("Changes", "改动")
        + "</h2><table><thead><tr>"
        + "".join("<th>" + _bi(en, zh) + "</th>" for en, zh in heads)
        + "</tr></thead><tbody>"
        + "".join(body)
        + '</tbody></table><p class="muted">'
        + _bi(
            "Descriptions use saved summaries when available, otherwise literal file or constant changes. Reasons are shortened quotations; tags are saved model labels, not quality judgments.",
            "改动描述优先使用已保存的摘要，否则列出文件或常量改动。理由是原话的缩短摘录；标签来自已保存的模型标注，不代表质量判断。",
        )
        + "</p>"
    )


def _overview(rows, counts, start, raw, metrics, cost, summary_cost, span, waited):
    scheduled = f" of {counts['scheduled']} scheduled" if isinstance(start.get("schedule"), list) else ""
    n_an = sum(r["analysis_only"] for r in rows)
    lag = (
        f"; median request-to-effect lag {_n(counts['median_lag_updates'])} updates"
        if counts["median_lag_updates"] is not None
        else ""
    )
    change_word = "change" if counts["applied"] == 1 else "changes"
    en = f"The agent was asked {counts['asked']}{scheduled} decisions, answered {counts['answered']} and applied {counts['applied']} {change_word} ({len(rows) - n_an} to the updater, {n_an} only to the analysis){lag}."
    schedule_zh = f"计划 {counts['scheduled']} 次决策，" if isinstance(start.get("schedule"), list) else ""
    lag_zh = (
        f"；请求到生效的更新步延迟中位数为 {_n(counts['median_lag_updates'])}"
        if counts["median_lag_updates"] is not None
        else ""
    )
    zh = f"{schedule_zh}agent 收到 {counts['asked']} 次请求，回答 {counts['answered']} 次，应用 {counts['applied']} 次改动（更新器 {len(rows) - n_an} 次，仅分析 {n_an} 次）{lag_zh}。"
    sentences = [_bi(en, zh)]
    updater = [r for r in rows if not r["analysis_only"]]
    highlights = updater or rows
    if highlights:
        chosen = [highlights[0]] + ([highlights[-1]] if len(highlights) > 1 else [])
        for r in chosen:
            reason = _excerpt(r["rationale"], 150, reason=True)
            why = f"; the agent wrote “{reason.rstrip('.')}”" if reason else "; no rationale was recorded"
            why_zh = f"；agent 写道：“{reason.rstrip('.')}”" if reason else "；未记录理由"
            when = f"At update {_n(r['applied'])}" if _number(r["applied"]) else "At an unrecorded update"
            suffix = f" (reverted at update {_n(r['reverted_at'])})" if r.get("reverted_at") is not None else ""
            sentences.append(
                _bi(
                    f"{when}, decision {r['k']}: {r['change'].rstrip('.')}{suffix}{why}.",
                    f"更新步 {_n(r['applied'])} 的决策 {r['k']}：{r['change'].rstrip('.')}{suffix}{why_zh}。",
                )
            )
    finite = [(u, v) for u, v in raw if _number(v)]
    if finite:
        (u0, v0), (u1, v1) = finite[0], finite[-1]
        metric = f"Recorded training loss went from {v0:.5g} at update {_n(u0)} to {v1:.5g} at update {_n(u1)}."
        metric_zh = f"记录的训练损失从更新步 {_n(u0)} 的 {v0:.5g} 到更新步 {_n(u1)} 的 {v1:.5g}。"
    else:
        metric, metric_zh = "No training loss was recorded.", "未记录训练损失。"
    if metrics:
        # Prefer an explicitly named accuracy; otherwise use the first opt.log metric. Do not infer units.
        key = next((k for k in metrics if re.search(r"(?:accuracy|acc)(?:$|[_/.])", k, re.I)), next(iter(metrics)))
        points = metrics[key]
        (a, x), (b, y) = points[0], points[-1]
        logged = f"; {key}: {x:.5g} → {y:.5g} (updates {_n(a)} → {_n(b)})"
        metric = metric.rstrip(".") + logged + "."
        metric_zh = metric_zh.rstrip("。") + f"；{key}：{x:.5g} → {y:.5g}（更新步 {_n(a)} → {_n(b)}）。"
    sentences.append(_bi(metric, metric_zh))
    cost_text = f"${cost:.4f}" if _number(cost) else "not recorded"
    summary_text = f"; saved summaries cost ${summary_cost:.4f} separately" if _number(summary_cost) else ""
    summary_zh = f"；已保存的摘要另计 ${summary_cost:.4f}" if _number(summary_cost) else ""
    cost_zh = f"${cost:.4f}" if _number(cost) else "未记录"
    sentences.append(
        _bi(f"Recorded agent cost: {cost_text}{summary_text}.", f"记录的 agent 费用：{cost_zh}{summary_zh}。")
    )
    availability = counts["unavailable_seconds"]
    available_en = f"; waiting for agent availability: {_duration(availability)}" if availability else ""
    available_zh = f"；等待 agent 恢复可用：{_duration(availability, zh=True)}" if availability else ""
    sentences.append(
        _bi(
            f"Wall span: {_duration(span)} (first to last timestamp, including any gaps); training waiting time: {_duration(waited)}{available_en}.",
            f"运行跨度：{_duration(span, zh=True)}（首末时间戳之差，包含中间停顿）；训练等待时间：{_duration(waited, zh=True)}{available_zh}。",
        )
    )
    # At most six sentences, grouped into three short paragraphs.
    return "".join("<p>" + " ".join(sentences[i : i + 2]) + "</p>" for i in range(0, len(sentences), 2))


def _fact(en, zh, value):
    shown = _bi("Not recorded", "未记录") if value is None else html.escape(str(value))
    return "<dt>" + _bi(en, zh) + "</dt><dd>" + shown + "</dd>"


def boundary_flag(events) -> str:
    """The banner of a run whose program sources crossed OPAL's static boundary check (boundary events,
    misc/boundary.py); empty otherwise. Flags are diagnostics, not a sandbox."""
    found = [e for e in events if e.get("kind") == "boundary"]
    if not found:
        return ""
    locations = "; ".join(
        f"v{e['version']:03d} {e.get('file')}:{e.get('line')} ({e.get('category')})" if isinstance(e.get("version"), int)
        else f"{e.get('file')}:{e.get('line')} ({e.get('category')})"
        for e in found[:6]
    )
    return (f'<p class="flag"><b>Boundary flags:</b> {len(found)} static findings: {html.escape(locations)}. '
            'Review events.jsonl and programs/ for details; flags do not block execution or provide a sandbox.</p>\n')


def write_report(run) -> Path:
    """Read existing records and atomically replace one offline HTML file; no model calls or training imports."""
    from .report_page import render
    from .report_privacy import Privacy

    run = Path(run)
    events = _events(run)
    if not events:
        raise FileNotFoundError("not a run folder (no readable events.jsonl)")
    start = next((e for e in events if e.get("kind") == "start"), {})
    versions = _versions(run)
    rows = decision_rows(run, events, versions)
    requested = {e["k"]: e for e in events if e.get("kind") == "requested" and _index(e.get("k"))}
    outcomes = {e["k"]: e for e in events if e.get("kind") in (*OUTCOMES, "cancelled") and _index(e.get("k"))}
    # An old/partial folder may have an outcome but have lost its request line. Do not lose the answer too.
    decisions = {}
    for k in sorted(requested.keys() | outcomes.keys()):
        record = run / "decisions" / f"{k:03d}"
        decisions[k] = dict(
            request=requested.get(k, {}),
            outcome=outcomes.get(k, {}),
            response=_response(run, k),
            message=_read_text(record / "message.txt"),
            metadata=_agent_metadata(record),
        )
    raw, lr = {}, {}
    for r in _jsonl(run / "losses.jsonl"):
        u = r.get("from_update")
        if not _number(u):
            continue
        for i, v in enumerate(_list(r.get("losses"))):
            raw[u + i] = v if _number(v) else None
        if _number(r.get("to_update")):
            for g, v in enumerate(_list(r.get("lr"))):
                if _number(v):
                    lr.setdefault(g, []).append([r["to_update"], v])
    raw = sorted(raw.items())
    small = []
    if lr:
        distinct = {json.dumps(pts): (g, pts) for g, pts in reversed(list(lr.items()))}
        small.append(
            dict(
                label="Learning rate (exposed parameter groups)",
                series=[dict(name=f"group {g}", points=pts) for g, pts in list(distinct.values())[:6]],
            )
        )
    metrics = {}
    for r in _jsonl(run / "metrics.jsonl"):
        if _number(r.get("update")):
            for key, v in r.items():
                if key != "update" and _number(v):
                    metrics.setdefault(key, []).append([r["update"], v])
    small.extend(
        dict(label=f"{key} (logged with opt.log)", series=[dict(name=key, points=pts)]) for key, pts in metrics.items()
    )
    extra, n_extra = report_series(run)
    for i, (key, pts) in enumerate(extra):
        more = f"; {n_extra - len(extra)} more series not shown" if i == len(extra) - 1 and n_extra > len(extra) else ""
        small.append(dict(label=f"{key} (reported by the agent's analysis{more})", series=[dict(name=key, points=pts)]))
    result = _object(run / "result.json")
    data = dict(events=events, start=start, rows=rows, decisions=decisions, small=small, metrics=metrics, result=result)
    data = Privacy(run, data).clean(data)
    events, start, rows, decisions, small, result = (
        data[k] for k in ("events", "start", "rows", "decisions", "small", "result")
    )
    metrics = data["metrics"]
    counts = decision_counts(events)
    for row in rows:
        row["change"] = _change(row)
        row["reverted_at"] = next(
            (e.get("update") for e in events if e.get("kind") == "reverted" and e.get("version") == row["version"]),
            None,
        )
    by_k = {r["k"]: r for r in rows}
    costs = [
        next((obj["cost_usd"] for obj in (d["response"], d["outcome"]) if _number(obj.get("cost_usd"))), None)
        for d in decisions.values()
    ]
    costs = [v for v in costs if _number(v)]
    settled = next(
        (
            e
            for e in reversed(events)
            if _finished(events) and e.get("kind") in ("end", "interrupted", "cost_settled")
        ),
        {},
    )
    cost = settled.get("cost_usd")
    if costs:
        cost = max(cost, sum(costs)) if _number(cost) else sum(costs)
    pending = settled.get("pending_decision")
    unknown = settled.get("unknown_cost_decisions", [])
    if unknown or (pending is not None and not _number(decisions.get(pending, {}).get("response", {}).get("cost_usd"))):
        cost = None  # pending/unreported is not free, even after the response arrived
    elif "unknown_cost_decisions" not in settled and any(
        e.get("kind") == "cancelled" and not any(
            _number(decisions.get(e.get("k"), {}).get(field, {}).get("cost_usd"))
            for field in ("response", "outcome")
        ) for e in events
    ):
        cost = None  # legacy cancelled records do not carry the persistent unknown set
    summary_costs = [
        r["summary"].get("cost_usd") for r in rows if r["summary"] and _number(r["summary"].get("cost_usd"))
    ]
    summary_cost = sum(summary_costs) if summary_costs else None
    span = _wall_span(events)
    waited = next(
        (obj["agent_wait_seconds"] for obj in [result, *reversed(events)] if _number(obj.get("agent_wait_seconds"))),
        None,
    )
    ticks, blocks, diffs = [], [], []
    for k, d in decisions.items():
        rq, oc, resp = d["request"], d["outcome"], d["response"]
        r = by_k.get(k)
        requested_at = rq.get("update", oc.get("requested_at"))
        applied = oc.get("applied_at")
        status = "failed" if oc.get("failed") else oc.get("kind", "pending" if not _finished(events) else "unanswered")
        reason = _excerpt(_text(resp.get("text")), reason=True)
        change = r["change"] if r else status
        label = f"Update {_n(requested_at)} → {_n(applied)}: {change}" + (f" · “{reason}”" if reason else "")
        ticks.append(
            dict(
                k=k,
                requested=requested_at if _number(requested_at) else None,
                applied=applied if _number(applied) else None,
                changed=r is not None and _number(applied),
                status=status,
                label=label,
            )
        )
        metadata = {
            key: resp[key]
            for key in ("seconds", "cost_usd", "returncode", "backend_error", "errors", "unavailable_seconds")
            if key in resp
        }
        metadata.update(request=rq, outcome=oc)
        if rq.get("wait"):
            metadata.update(wait=rq["wait"], wait_reason=rq.get("wait_reason"))
        blocks.append(
            f'<details id="d{k}"><summary>'
            + _bi(f"Decision {k}", f"决策 {k}")
            + f" · {_n(requested_at)} → {_n(applied)} · {html.escape(str(status))}</summary><h3>"
            + _bi("Request", "请求")
            + "</h3><pre>"
            + html.escape(d["message"] or "(not recorded)")
            + "</pre><h3>"
            + _bi("Answer", "回答")
            + "</h3><pre>"
            + html.escape(_text(resp.get("text")) or "(not recorded)")
            + "</pre><pre>"
            + html.escape(json.dumps(metadata, indent=2, ensure_ascii=False))
            + "</pre></details>"
        )
        if r:
            file_text = "\n".join(f"{f['name']}: +{f['added']} −{f['removed']} {f['status']}" for f in r["files"])
            const_text = "\n".join(f"{name}: {c}: {a} → {b}" for name, c, a, b in r["constants"])
            diff = (
                _diff(r["old"], r["new"])
                if r["snapshots_available"]
                else html.escape("Program snapshots missing; diff unavailable.")
            )
            cached = _text((r["summary"] or {}).get("summary")) or "summary unavailable"
            diffs.append(
                "<details><summary>"
                + _bi(f"Decision {k}", f"决策 {k}")
                + "</summary><p>"
                + html.escape(cached)
                + "</p><pre>"
                + html.escape(file_text + "\n" + const_text)
                + '</pre><pre class="diff">'
                + diff
                + "</pre></details>"
            )
    bins = max(1, math.ceil(len(raw) / 300))
    losses = []
    for i in range(0, len(raw), bins):
        batch = raw[i : i + bins]
        values = [v for _, v in batch if v is not None]
        losses.append(
            [batch[len(batch) // 2][0], sum(v / len(values) for v in values) if len(values) == len(batch) else None]
        )
    models = sorted(
        {_text(obj.get("model")) for d in decisions.values() for obj in (d["metadata"], d["response"])} - {""}
    )
    effort = sorted({_text(obj.get("effort")) for obj in [start, *[d["metadata"] for d in decisions.values()]]} - {""})
    model = start.get("model") or ", ".join(models) or None
    schedule = next((e["schedule"] for e in reversed(events)
                     if e.get("kind") in ("start", "estimate", "resumed") and "schedule" in e), None)
    schedule = ", ".join(_n(v) for v in schedule) if isinstance(schedule, list) else None
    facts = "<p>" + html.escape(describe_counts(counts)) + "</p><dl>"
    for en, zh, value in [
        ("OPAL commit", "OPAL 提交", start.get("opal_commit") or result.get("opal_commit")),
        ("Agent model", "Agent 模型", model),
        ("Agent effort", "Agent 推理强度", start.get("effort") or ", ".join(effort) or None),
        ("Schedule (updates)", "计划决策的更新步", schedule),
        ("Blocking", "是否阻塞训练", start.get("blocking")),
        ("Maximum lag (updates)", "最大延迟（更新步）", start.get("max_lag_updates")),
        ("Training waiting time", "训练等待时间", _duration(waited) if waited is not None else None),
        (
            "Waiting for agent availability",
            "等待 agent 恢复可用",
            _duration(counts["unavailable_seconds"]) if any("unavailable_seconds" in e for e in events) else None,
        ),
        (
            "First / last recorded timestamp",
            "首末记录时间戳",
            " → ".join([_text(events[0].get("utc")), _text(events[-1].get("utc"))])
            if events[0].get("utc") and events[-1].get("utc")
            else None,
        ),
    ]:
        facts += _fact(en, zh, value)
    facts += "</dl>" + defaults_flag(events) + integrity_flag(events) + boundary_flag(events)
    notable = [
        e for e in events if e.get("kind") in ("resumed", "interrupted", "reverted", "crashed", "rank_divergence", "ranks_resynced", "state_dropped")
    ]
    if notable:
        facts += "<pre>" + html.escape(json.dumps(notable, indent=2, ensure_ascii=False)) + "</pre>"
    at = max([u for u, _ in raw] + [e["update"] for e in events if _number(e.get("update"))] + [0])
    live = not _finished(events)
    if live:
        facts = (
            "<p>"
            + _bi(
                f"Live: the run is at update {_n(at)}; this page reloads each minute.",
                f"运行中：当前更新步 {_n(at)}；此页每分钟刷新。",
            )
            + "</p>"
            + facts
        )
    caption = _bi(
        f"Numbers mark applied decisions; faint ticks mark kept decisions. Loss points average up to {bins} update(s); gaps mark non-finite or invalid losses. The vertical axis is not clipped.",
        f"编号标记改动生效的更新步；淡刻度标记保留原程序的决策。每个损失点平均至多 {bins} 个更新步；非有限或无效损失处断开，纵轴不截断。",
    )
    page = render(
        dict(
            refresh='<meta http-equiv="refresh" content="60">\n' if live else "",
            goal=html.escape(_text(start.get("goal"))),
            overview=rank_flag(events) + _overview(rows, counts, start, raw, metrics, cost, summary_cost, span, waited),
            caption=caption,
            table=_table(rows),
            decisions="".join(blocks) or _bi("No decisions recorded.", "未记录决策。"),
            diffs="".join(diffs) or _bi("No applied code diffs recorded.", "未记录已应用的代码改动。"),
            facts=facts,
            losses=_js(losses),
            ticks=_js(ticks),
            small=_js(small),
            total=_js(start.get("total_updates") if _number(start.get("total_updates")) else None),
        )
    )
    out = run / "report.html"
    tmp = run / f".report.html.{os.getpid()}.{threading.get_ident()}.tmp"
    try:
        tmp.write_text(page)
        os.replace(tmp, out)
    finally:
        tmp.unlink(missing_ok=True)
    return out


def summarize_missing(run, summarizer, log=print) -> int:
    """Make the summaries a run's code changes lack (one call each); returns how many failed."""
    run, failed = Path(run), 0
    rows = [r for r in decision_rows(run) if r["summary"] is None]
    for i, r in enumerate(rows):
        rec = run / "decisions" / f"{r['k']:03d}"
        if not r.get("snapshots_available"):
            failed += 1
            log(f"decision {r['k']}: summary unavailable (program snapshots missing)")
            continue
        out = summaries.ensure(rec, summarizer, r["k"], r["rationale"], r["old"], r["new"], r["total"])
        failed += out is None
        log(f"decision {r['k']}: {'summary failed' if out is None else 'summarised'} ({i + 1} of {len(rows)})")
    return failed


class LiveReport:
    """Rank 0's report writer, in a thread of its own so that neither the report nor a summary call runs in the
    training step or holds up the next decision: it makes the summaries of new code changes (when it has a
    summarizer) and rewrites report.html after each decision and at the end. Failures never reach training."""

    def __init__(self, run_dir, summarizer=None, log=print, idle=30.0):
        self.dir, self.summarizer, self.log, self.idle = Path(run_dir), summarizer, log, idle
        self.q, self.lock, self.thread, self._warned = queue.Queue(), threading.Lock(), None, set()
        self.write_lock = threading.Lock()

    def summarize(self, k, rationale, old, new, total=None):
        if self.summarizer is not None:
            self._guard(
                "summary", self._put, ("summary", dict(k=k, rationale=rationale, old=old, new=new, total=total))
            )

    def refresh(self):
        self._guard("report", self._put, ("report", None))

    def finish(self, timeout=120.0):
        """At exit: wait for queued summaries and the last report; past `timeout`, write the report without them."""
        self._guard("report", self._finish, timeout)

    def _finish(self, timeout):
        done = threading.Event()
        self._put(("report", None))
        self._put(("done", done))
        if not done.wait(timeout):
            self._write()

    def _put(self, job):
        with self.lock:
            self.q.put(job)
            if self.thread is None:
                self.thread = threading.Thread(target=self._loop, name="opal-report", daemon=True)
                try:
                    self.thread.start()
                except (Exception, SystemExit):
                    self.thread = None
                    raise

    def _loop(self):
        try:
            self._guard("report", self._work)
        finally:
            # An unexpected queue/thread failure must release finish() too. Future refreshes can retry.
            with self.lock:
                if self.thread is threading.current_thread():
                    self.thread = None
                    while not self.q.empty():
                        kind, job = self.q.get_nowait()
                        if kind == "done":
                            job.set()

    def _work(self):
        while True:
            try:
                jobs = [self.q.get(timeout=self.idle)]
            except queue.Empty:
                with self.lock:
                    if self.q.empty():
                        self.thread = None
                        return
                continue
            while not self.q.empty():
                jobs.append(self.q.get_nowait())
            for kind, job in jobs:
                if kind == "summary":
                    self._guard("summary", self._summary, job)
            if any(kind != "done" for kind, _ in jobs):
                self._write()
            for kind, job in jobs:
                if kind == "done":
                    job.set()

    def _summary(self, job):
        summaries.ensure(self.dir / "decisions" / f"{job['k']:03d}", self.summarizer, **job)

    def _write(self):
        self._guard("report", self._write_if_started)

    def _write_if_started(self):
        if (self.dir / "events.jsonl").exists():  # not before the run has started
            with self.write_lock:
                self._guard("report", write_report, self.dir)

    def _guard(self, what, fn, *args):
        try:
            return fn(*args)
        except (Exception, SystemExit) as exc:  # noqa: BLE001 - the report is never worth a failed run
            if what not in self._warned:
                self._warned.add(what)
                try:
                    message = " ".join(str(exc).split())
                    self.log(f"could not write the {what} ({type(exc).__name__}: {message}); training is not affected")
                except (Exception, SystemExit):
                    pass  # even a broken log callback must not affect training
