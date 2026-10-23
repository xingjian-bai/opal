"""opal watch: print a run's events as they are written."""

import json
import os
import queue
import subprocess
import sys
import threading

import pytest

from opal import cli, watch


def write_jsonl(path, records):
    path.write_text("".join(json.dumps(r) + "\n" for r in records))


def write_run(folder, *, finished=True, extra=(), partial=False):
    folder.mkdir(parents=True, exist_ok=True)
    events = [
        dict(
            kind="start",
            utc="2026-10-21T00:00:00Z",
            goal="A small run",
            total_updates=30,
            schedule=[5, 20],
            blocking=True,
            backend="claude",
        ),
        dict(kind="requested", k=1, update=5, utc="2026-10-21T00:00:01Z"),
        dict(
            kind="applied",
            k=1,
            requested_at=5,
            applied_at=10,
            version=1,
            changed=["updater.py"],
            utc="2026-10-21T00:00:20Z",
        ),
        dict(kind="requested", k=2, update=20, utc="2026-10-21T00:01:00Z"),
        dict(kind="kept", k=2, applied_at=20, utc="2026-10-21T00:01:10Z"),
        *extra,
    ]
    if finished:
        events.append(dict(kind="end", utc="2026-10-21T00:02:00Z", update=30, cost_usd=0.25, decisions=2))
    write_jsonl(folder / "events.jsonl", events)
    if partial:
        with open(folder / "events.jsonl", "a") as f:
            f.write('{"kind": "requested", "k": 3, "upd')
    rec = folder / "decisions" / "001"
    rec.mkdir(parents=True)
    (rec / "response.json").write_text(
        json.dumps(dict(text="I lowered the rate because the loss plateaued.", cost_usd=0.25, seconds=15))
    )
    return folder


def test_watch_prints_a_finished_run_and_returns(tmp_path, capsys):
    run = write_run(tmp_path / "r")
    assert cli.main(["watch", str(run), "--timeout", "0"]) == 0
    out = capsys.readouterr().out
    assert f"watching {run.resolve()}" in out
    assert "start: 30 updates; backend claude; blocking" in out
    assert "goal: A small run" in out
    assert "requested  k=1  at update 5" in out
    assert "changed    k=1  at update 10" in out
    assert "$0.25" in out and "lowered the rate" in out
    assert "kept       k=2  at update 20" in out
    assert "finished at update 30; 2 decisions; $0.25" in out
    assert "still running" not in out


def test_watch_timeout_zero_on_an_unfinished_run_says_so_and_exits_1(tmp_path, capsys):
    run = write_run(tmp_path / "r", finished=False)
    assert cli.main(["watch", str(run), "--timeout", "0"]) == 1
    out = capsys.readouterr().out
    assert "requested  k=2  at update 20" in out
    assert "finished at update" not in out
    assert "no end event yet (last event at update 20)" in out
    assert "still running" not in out


def test_watch_gives_up_after_timeout_seconds_without_a_new_event(tmp_path, capsys):
    run = write_run(tmp_path / "r", finished=False)  # a run killed without close() never writes its end event
    clock = [0.0]

    def sleep(seconds):
        clock[0] += seconds

    assert watch.follow(run, interval=5, timeout=30, sleep=sleep, now=lambda: clock[0]) == 1
    out = capsys.readouterr().out
    assert out.strip().splitlines()[-1] == "no new event for 30 s (last event at update 20)"
    assert clock[0] == 30


def test_watch_prints_each_line_at_once_through_a_pipe(tmp_path):
    run = write_run(tmp_path / "r", finished=False)
    env = {k: v for k, v in os.environ.items() if k != "PYTHONUNBUFFERED"}
    cmd = [sys.executable, "-m", "opal.cli", "watch", str(run), "--interval", "0.05"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True, env=env)
    lines = queue.Queue()
    threading.Thread(target=lambda: [lines.put(line) for line in proc.stdout], daemon=True).start()
    try:
        got = [lines.get(timeout=60) for _ in range(3)]  # the run has not finished, so the watch is still waiting
        assert got[0].startswith("watching") and got[1].startswith("start: 30 updates")
        assert got[2] == "goal: A small run\n"
    finally:
        end = dict(kind="end", utc="2026-10-21T00:02:00Z", update=30, cost_usd=0.1, decisions=2)
        with open(run / "events.jsonl", "a") as f:
            f.write(json.dumps(end) + "\n")
        assert proc.wait(timeout=60) == 0


def test_watch_stops_quietly_when_the_reader_closes_the_pipe(tmp_path):
    many = [dict(kind="requested", k=k, update=k, utc="2026-10-21T00:00:01Z") for k in range(3, 3000)]
    run = write_run(tmp_path / "r", extra=many)  # far more output than a pipe buffer holds
    cmd = [sys.executable, "-m", "opal.cli", "watch", str(run), "--timeout", "0"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    assert proc.stdout.readline().startswith("watching")
    proc.stdout.close()  # what `| head -1` does
    assert proc.wait(timeout=60) == 0
    assert "Traceback" not in proc.stderr.read()


def test_watch_ctrl_c_exits_130_without_a_traceback(tmp_path, monkeypatch):
    run = write_run(tmp_path / "r", finished=False)

    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(watch, "follow", interrupted)
    assert cli.main(["watch", str(run)]) == 130


def test_watch_prints_an_event_written_while_it_waits(tmp_path, capsys):
    run = write_run(tmp_path / "r", finished=False)

    def sleep(_):
        rec = dict(kind="end", utc="2026-10-21T00:02:00Z", update=30, cost_usd=0.1, decisions=2)
        with open(run / "events.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")

    assert watch.follow(run, interval=0.01, sleep=sleep) == 0
    out = capsys.readouterr().out
    assert "finished at update 30; 2 decisions; $0.10" in out
    assert "still running" not in out


def test_watch_skips_a_partial_last_line(tmp_path, capsys):
    run = write_run(tmp_path / "r", partial=True)
    assert cli.main(["watch", str(run), "--timeout", "0"]) == 0
    out = capsys.readouterr().out
    assert "k=3" not in out
    assert "finished at update 30" in out


def test_watch_estimate_and_revert_lines(tmp_path, capsys):
    extra = [
        dict(kind="estimate", update=10, decisions=1, capped_from=38, utc="2026-10-21T00:00:15Z"),
        dict(kind="reverted", update=12, version=1, utc="2026-10-21T00:00:16Z"),
    ]
    run = write_run(tmp_path / "r", extra=extra)
    assert cli.main(["watch", str(run), "--timeout", "0"]) == 0
    out = capsys.readouterr().out
    assert "estimate at update 10: 1 scheduled; capped from 38" in out
    assert "reverted at update 12" in out


def test_watch_cancelled_decision_has_no_update(tmp_path, capsys):
    extra = [
        dict(kind="requested", k=3, update=25, utc="2026-10-21T00:01:20Z"),
        dict(kind="cancelled", k=3, utc="2026-10-21T00:01:30Z"),
    ]
    run = write_run(tmp_path / "r", extra=extra)
    assert cli.main(["watch", str(run), "--timeout", "0"]) == 0
    assert "cancelled  k=3\n" in capsys.readouterr().out


def test_watch_end_cost_reads_as_in_opal_show(tmp_path):
    start, end = dict(kind="start", backend="codex"), dict(kind="end", update=30, decisions=2, cost_usd=None)
    assert watch.event_line(tmp_path, end, [start, end]) == "finished at update 30; 2 decisions; USD unavailable"
    start["backend"] = "claude"
    assert watch.event_line(tmp_path, end, [start, end]) == "finished at update 30; 2 decisions; $0.00"


def test_watch_says_when_a_resume_rewrote_events_jsonl(tmp_path, capsys):
    run = write_run(tmp_path / "r", finished=False)
    kept = (run / "events.jsonl").read_text().splitlines(keepends=True)[:3]
    resumed = dict(kind="resumed", update=10, utc="2026-10-21T00:03:00Z")
    end = dict(kind="end", update=30, utc="2026-10-21T00:04:00Z", cost_usd=0.25, decisions=2)

    def sleep(_):  # the resume keeps the records up to its checkpoint, then the run goes on
        (run / "events.jsonl").write_text("".join(kept) + json.dumps(resumed) + "\n" + json.dumps(end) + "\n")

    assert watch.follow(run, interval=0.01, sleep=sleep) == 0
    out = capsys.readouterr().out.splitlines()
    notice = next(i for i, line in enumerate(out) if line.startswith("events.jsonl was rewritten"))
    assert out[notice + 1] == "start: 30 updates; backend claude; blocking"
    assert out[-2:] == ["resumed at update 10", "finished at update 30; 2 decisions; $0.25"]


def test_watch_missing_folder_exits(tmp_path):
    with pytest.raises(SystemExit) as exited:
        cli.main(["watch", str(tmp_path / "missing"), "--timeout", "0"])
    assert "not a run folder" in str(exited.value)


@pytest.mark.parametrize("args", [["--interval", "nan"], ["--interval", "inf"], ["--timeout", "nan"]])
def test_watch_rejects_a_non_finite_interval_or_timeout(args, capsys):
    with pytest.raises(SystemExit) as exited:
        cli.main(["watch", "r", *args])
    assert exited.value.code == 2


def test_watch_rejects_a_non_positive_interval(capsys):
    with pytest.raises(SystemExit) as exited:
        cli.main(["watch", "r", "--interval", "0", "--timeout", "0"])
    assert exited.value.code == 2
    assert "interval" in capsys.readouterr().err
