"""opal compare: two run folders, or two result.json files, side by side."""

import json

import pytest

from opal import cli, compare


def write_jsonl(path, records):
    path.write_text("".join(json.dumps(r) + "\n" for r in records))


def write_run(
    folder,
    *,
    goal="A small run",
    backend="claude",
    blocking=True,
    outcome="applied",
    k=1,
    update=5,
    cost=0.25,
    wait=12.0,
    loss=1.0,
    result=None,
    extra_events=(),
    extra_requests=(),
    failed=False,
    finished=True,
):
    folder.mkdir(parents=True, exist_ok=True)
    events = [
        dict(
            kind="start",
            utc="2026-10-21T00:00:00Z",
            goal=goal,
            total_updates=30,
            schedule=[update],
            blocking=blocking,
            backend=backend,
        ),
        dict(kind="requested", k=k, update=update, utc="2026-10-21T00:00:01Z"),
        dict(
            kind=outcome,
            k=k,
            requested_at=update,
            applied_at=update + 5,
            version=1 if outcome == "applied" else 0,
            **(dict(changed=["updater.py"]) if outcome == "applied" else {}),
            **(dict(failed=True) if failed else {}),
        ),
        *extra_requests,
        *extra_events,
    ]
    if finished:
        events.append(dict(kind="end", utc="2026-10-21T00:02:00Z", update=30, cost_usd=cost, agent_wait_seconds=wait))
    write_jsonl(folder / "events.jsonl", events)
    write_jsonl(folder / "losses.jsonl", [dict(from_update=1, to_update=3, losses=[3.0, 2.0, loss])])
    rec = folder / "decisions" / f"{k:03d}"
    rec.mkdir(parents=True)
    (rec / "response.json").write_text(json.dumps(dict(text="changed the rate", cost_usd=cost, seconds=15)))
    if result is not None:
        (folder / "result.json").write_text(json.dumps(result))
    return folder


def result_record(**kw):
    rec = dict(
        metric=dict(name="val_loss", value=2.5, direction="lower"),
        train_seconds=1000.0,
        agent_wait_seconds=100.0,
        agent_cost_usd=1.5,
        decisions=dict(asked=10, applied=2, failed=0),
        hardware="1 x GPU",
        opal_commit="abc123",
        command="python train.py --opal",
        opal_run=None,
    )
    rec.update(kw)
    return rec


def test_compare_marks_a_decision_whose_outcome_differs(tmp_path, capsys):
    a = write_run(tmp_path / "a", outcome="applied", loss=0.25)
    b = write_run(tmp_path / "b", outcome="kept", loss=0.30, cost=0.0, wait=0.0, backend="keep")
    assert cli.main(["compare", str(a), str(b)]) == 0
    out = capsys.readouterr().out
    assert "a: " in out and "b: " in out
    assert "* backend" in out and "claude" in out and "keep" in out
    assert "decisions that differ, matched by the update they were asked at" in out
    assert "        5  k=1 changed       k=1 kept" in out
    assert "last loss" in out and "0.25" in out


def test_compare_says_when_every_decision_matches(tmp_path, capsys):
    a = write_run(tmp_path / "a")
    b = write_run(tmp_path / "b", goal="A small run")
    assert cli.main(["compare", str(a), str(b)]) == 0
    out = capsys.readouterr().out
    assert "all 1 decision asked at the same update in both runs, with the same outcome." in out
    assert "decisions that differ" not in out


def test_compare_json_lists_differing_decisions(tmp_path, capsys):
    a = write_run(tmp_path / "a", outcome="applied")
    b = write_run(tmp_path / "b", outcome="kept")
    assert cli.main(["compare", "--json", str(a), str(b)]) == 0
    obj = json.loads(capsys.readouterr().out)
    assert obj["a"]["kind"] == "run" and obj["b"]["kind"] == "run"
    assert obj["matched_by"] == "asked_at" and obj["same_decisions"] == 0
    assert obj["differing_decisions"] == [dict(asked_at=5, a="changed", a_k=1, b="kept", b_k=1)]
    assert obj["a"]["last_loss"] == 1.0 and obj["a"]["cost_usd"] == 0.25


def test_compare_result_json_files_print_the_metric_and_skip_decisions(tmp_path, capsys):
    fa = tmp_path / "official.json"
    fb = tmp_path / "opal.json"
    official = result_record(metric=dict(name="val_loss", value=2.6, direction="lower"))
    official.update(agent_cost_usd=None, decisions=None, command="python train.py --plain-command-that-is-long " * 4)
    fa.write_text(json.dumps(official))
    fb.write_text(json.dumps(result_record()))
    assert cli.main(["compare", str(fa), str(fb)]) == 0
    out = capsys.readouterr().out
    assert "val_loss 2.6 (lower is better)" in out
    assert "val_loss 2.5 (lower is better)" in out
    assert "* metric" in out
    assert "a and b have only a result.json, without its run folder (events.jsonl) here" in out
    assert "* command b: python train.py --opal\n" in out
    table = [line for line in out.splitlines() if line[2:].startswith(("updates", "metric", "decisions"))]
    assert len(table) == 3 and all(len(line) < 100 for line in table)  # the long commands are below the table


def test_compare_uses_result_json_inside_a_run_folder(tmp_path, capsys):
    rec = result_record(metric=dict(name="test accuracy", value=0.81, direction="higher"))
    other = result_record(metric=dict(name="test accuracy", value=0.8, direction="higher"))
    a = write_run(tmp_path / "a", result=rec, loss=0.5)
    b = write_run(tmp_path / "b", result=other, loss=0.6)
    assert cli.main(["compare", str(a), str(b)]) == 0
    out = capsys.readouterr().out
    assert "test accuracy 0.81 (higher is better)" in out
    assert "test accuracy 0.8 (higher is better)" in out


def test_compare_result_json_points_at_its_run_folder(tmp_path, capsys):
    a = write_run(tmp_path / "a", outcome="applied")
    b = write_run(tmp_path / "b", outcome="kept")
    fa = tmp_path / "a.json"
    fa.write_text(json.dumps(result_record(opal_run=str(a), metric=dict(name="loss", value=0.25, direction="lower"))))
    assert cli.main(["compare", str(fa), str(b)]) == 0
    out = capsys.readouterr().out
    assert f"a: {fa} (run folder {a})" in out
    assert "        5  k=1 changed       k=1 kept" in out


def test_compare_a_lost_pending_decision_is_not_the_same_as_kept(tmp_path, capsys):
    a = write_run(tmp_path / "a", outcome="kept")
    b = tmp_path / "b"
    b.mkdir()
    write_jsonl(
        b / "events.jsonl",
        [
            dict(kind="start", utc="2026-10-21T00:00:00Z", goal="x", total_updates=30, schedule=[5], blocking=True),
            dict(kind="requested", k=1, update=5, utc="2026-10-21T00:00:01Z"),
            dict(kind="end", utc="2026-10-21T00:02:00Z", update=30, cost_usd=0),
        ],
    )
    assert cli.main(["compare", str(a), str(b)]) == 0
    out = capsys.readouterr().out
    assert "k=1 kept          k=1 lost" in out


def test_compare_missing_path_exits(tmp_path):
    with pytest.raises(SystemExit) as exited:
        cli.main(["compare", str(tmp_path / "missing"), str(tmp_path / "also")])
    assert exited.value.code
    assert "not a run folder" in str(exited.value)


def test_compare_rejects_a_json_file_without_metric(tmp_path):
    bad = tmp_path / "notes.json"
    bad.write_text(json.dumps(dict(hello=1)))
    good = tmp_path / "ok.json"
    good.write_text(json.dumps(result_record()))
    with pytest.raises(SystemExit) as exited:
        cli.main(["compare", str(bad), str(good)])
    assert "not a result.json" in str(exited.value)


def test_compare_a_parent_folder_names_the_runs_inside(tmp_path):
    write_run(tmp_path / "parent" / "r1")
    write_run(tmp_path / "parent" / "r2")
    with pytest.raises(SystemExit) as exited:
        cli.main(["compare", str(tmp_path / "parent"), str(tmp_path / "parent" / "r1")])
    assert "runs inside: r1, r2" in str(exited.value)


def test_load_side_reads_a_partial_events_line(tmp_path):
    folder = write_run(tmp_path / "a")
    with open(folder / "events.jsonl", "a") as f:
        f.write('{"kind": "requested", "k": 2, "upd')
    snap = compare.load_side(folder)
    assert [d["k"] for d in snap["decisions"]] == [1]


def test_compare_last_loss_shows_a_diverged_run(tmp_path, capsys):
    a = write_run(tmp_path / "a", loss=0.5)
    b = write_run(tmp_path / "b", loss=float("nan"))
    assert cli.main(["compare", str(a), str(b)]) == 0
    row = next(line for line in capsys.readouterr().out.splitlines() if "last loss" in line)
    assert row.startswith("* last loss") and row.split()[-1] == "nan"


def test_compare_matches_decisions_by_the_update_they_were_asked_at(tmp_path, capsys):
    a = write_run(tmp_path / "a", outcome="kept", update=5)
    extra = [dict(kind="requested", k=2, update=10, utc="2026-10-21T00:00:05Z"), dict(kind="kept", k=2, applied_at=10)]
    b = write_run(tmp_path / "b", outcome="kept", update=10, extra_events=extra)  # k=1 at 10: another schedule
    assert cli.main(["compare", str(a), str(b)]) == 0
    out = capsys.readouterr().out
    assert "        5  k=1 kept          not asked" in out
    assert "       10  not asked         k=1 kept" in out
    assert "       10  not asked         k=2 kept" in out
    assert "same outcome" not in out


def test_compare_json_writes_non_finite_numbers_as_strings(tmp_path, capsys):
    diverged = result_record(metric=dict(name="val_loss", value=float("nan"), direction="lower"))
    a = write_run(tmp_path / "a", result=diverged, loss=float("nan"))
    b = write_run(tmp_path / "b", loss=float("inf"))
    assert cli.main(["compare", "--json", str(a), str(b)]) == 0
    obj = json.loads(capsys.readouterr().out)
    assert obj["a"]["metric"]["value"] == "nan" and obj["a"]["last_loss"] == "nan"
    assert obj["b"]["last_loss"] == "inf"


def test_compare_marks_an_unfinished_run_and_failed_calls_and_goals(tmp_path, capsys):
    a = write_run(tmp_path / "a", outcome="kept")
    b = write_run(tmp_path / "b", outcome="kept", failed=True, finished=False, goal="Another run")
    assert cli.main(["compare", str(a), str(b)]) == 0
    out = capsys.readouterr().out.splitlines()
    row = {line[2:].split("  ")[0]: line for line in out}
    assert row["ended"].startswith("* ") and row["ended"].endswith("no end event yet")
    assert row["decisions"].startswith("* ") and row["decisions"].endswith("1 asked, 0 applied, 1 failed")
    assert "* goal a: A small run" in out and "* goal b: Another run" in out


def test_compare_finds_a_relative_opal_run_beside_its_result_json(tmp_path, monkeypatch, capsys):
    # examples/overfit writes opal_run=<out>/opal relative to where training ran; compare from elsewhere
    for name, outcome in (("a", "applied"), ("b", "kept")):
        out = tmp_path / "runs" / name
        write_run(out / "opal", outcome=outcome)
        (out / "result.json").write_text(json.dumps(result_record(opal_run=f"runs/{name}/opal")))
    monkeypatch.chdir(tmp_path / "runs")
    assert cli.main(["compare", "a", "b"]) == 0
    out = capsys.readouterr().out
    assert "a: a/result.json (run folder a/opal)" in out
    assert "        5  k=1 changed       k=1 kept" in out


def test_compare_stops_quietly_when_the_reader_closes_the_pipe(tmp_path):
    import subprocess
    import sys

    many = [dict(kind="requested", k=k, update=k * 10, utc="2026-10-21T00:00:01Z") for k in range(2, 3000)]
    a = write_run(tmp_path / "a", extra_requests=many)  # far more output than a pipe buffer holds
    b = write_run(tmp_path / "b")
    cmd = [sys.executable, "-m", "opal.cli", "compare", "--json", str(a), str(b)]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    assert proc.stdout.readline().startswith("{")
    proc.stdout.close()  # what `| head -1` does
    assert proc.wait(timeout=60) == 0
    assert "Traceback" not in proc.stderr.read()
