"""The mailbox backend and its broker on a local folder (CPU only, no model call): a --keep-serving broker for several runs, a new broker gets the whole workspace; notes come back, the
agent's time limit, the broker survives a failing request, one broker per request and a crashed run
does not hold the broker."""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

import pytest
import torch

from opal.backends import mailbox
from opal.backends.base import Backend, CallableBackend
from opal.backends.claude import ClaudeBackend
from opal.backends.mailbox import MailboxBackend, _Folder, run_broker
from test_claude_backend import fake_cli, quick_waits  # noqa: F401 (a fixture)

UPDATER = "def build(named_params, previous=None, spec=None):\n    pass\n"


@pytest.fixture
def fast(monkeypatch):
    """Heartbeats every 0.3 s that count for 1.5 s, and quick switching, so the timing cases run in seconds."""
    monkeypatch.setattr(mailbox, "BEAT_SECONDS", 0.3)
    monkeypatch.setattr(mailbox, "HEARTBEAT_STALE", 1.5)
    monkeypatch.setattr(mailbox, "IDLE_SWITCH", 0.5)
    monkeypatch.setattr(mailbox, "RETRY_SECONDS", 0.2)


def workspace(tmp_path, name="ws"):
    ws = tmp_path / name
    (ws / "inbox").mkdir(parents=True)
    (ws / "reference").mkdir()
    (ws / "updater.py").write_text(UPDATER)
    (ws / "analysis.py").write_text("EVERY = 10\n")
    return ws


def mailbox_for(tmp_path, box, name="run", **kw):
    kw = {**dict(poll=0.1, broker_wait=30, timeout=120), **kw}
    mb = MailboxBackend(f"mailbox:{box}", **kw)
    mb.setup(tmp_path / name, tmp_path / name / "workspace")
    return mb


class _Stop(BaseException):
    """Raised in a broker thread of serve() when its test ends: run_broker has no stop of its own (its process ends)."""


_started = []  # the broker threads serve() started in the current test


def _stoppable(real):
    def call(*args, **kwargs):
        if getattr(threading.current_thread(), "stopping", False):
            raise _Stop
        return real(*args, **kwargs)

    return call


@pytest.fixture(autouse=True)
def no_broker_outlives_its_test(monkeypatch):
    """Stop every broker of serve() when its test ends, and wait for it and its heartbeat and cancel-watch threads. A
    broker left running keeps polling the module, so it called the fake _pending, _answer_request and lease that tests
    run later in the same process monkeypatch (test_review_035c.py drives _serve with them), and failed them."""
    for name in ("_pick_run", "_pending"):  # each turn of a broker's loop calls one of them
        monkeypatch.setattr(mailbox, name, _stoppable(getattr(mailbox, name)))
    _started.clear()
    yield
    for t in _started:
        t.stopping = True
    end = time.time() + 30
    for t in _started:
        t.join(max(0.0, end - time.time()))

    def left():
        return [t.name for t in threading.enumerate() if t in _started or t.name.startswith("opal-broker-")]

    while left() and time.time() < end:
        time.sleep(0.05)
    assert not left(), f"broker threads outlived their test: {left()}"


def serve(box, backend, work, **kw):
    """run_broker in a daemon thread; the thread's .error holds what it raised. It stops when its test ends."""
    kw = {**dict(poll=0.1, log=lambda m: None), **kw}
    t = threading.Thread(target=lambda: _catch(t, run_broker, str(box), backend, work, **kw), daemon=True)
    t.error = None
    _started.append(t)
    t.start()
    return t


def _catch(t, fn, *a, **kw):
    try:
        fn(*a, **kw)
    except BaseException as exc:  # noqa: BLE001
        t.error = exc


def listing(ws):
    return sorted(p.relative_to(ws).as_posix() for p in Path(ws).rglob("*") if p.is_file() and "fakebin" not in p.parts)


def end_run(mb, box):
    """Write `done` as finish() does, but keep the backend usable (the next decide() clears it)."""
    (box / mb.run_id / "done").write_text(json.dumps(dict(attempt=mb.attempt)))


# ----- B7 ---------------------------------------------------------------------------------------------------------


def test_keep_serving_broker_keeps_every_run_waiting_and_takes_them_in_turn(tmp_path, fast):
    box = tmp_path / "box"
    served = []

    def agent(ws, message, system):
        served.append(message)
        time.sleep(3 if message == "A1" else 0.2)
        return "kept"

    a = mailbox_for(tmp_path, box, "A")
    b = mailbox_for(tmp_path, box, "B", broker_wait=1.0)  # gives up after 1 s without a heartbeat it accepts
    wa, wb = workspace(tmp_path, "wsA"), workspace(tmp_path, "wsB")
    res = {}
    ta = threading.Thread(target=lambda: res.__setitem__("A1", a.decide(wa, "A1", "s", tmp_path)))
    ta.start()
    time.sleep(0.3)
    br = serve(box, CallableBackend(agent), tmp_path / "work", keep_serving=True)
    time.sleep(0.5)
    tb = threading.Thread(target=lambda: res.__setitem__("B1", b.decide(wb, "B1", "s", tmp_path)))
    tb.start()
    ta.join()
    for k in range(2, 6):  # A asks again at once after each answer
        a.decide(wa, f"A{k}", "s", tmp_path)
    tb.join(30)
    assert not res["B1"].get("gave_up") and res["B1"]["returncode"] == 0, res["B1"]
    assert "B1" in served and served.index("B1") <= 2, served  # served right after A1 (and at most one more of A)
    assert br.is_alive() and br.error is None


# ----- B8 ---------------------------------------------------------------------------------------------------------


def test_a_new_broker_gets_the_whole_workspace_and_the_agents_notes_come_back(tmp_path, fast):
    box = tmp_path / "box"
    ws = workspace(tmp_path)
    (ws / "reference" / "training_script_train.py").write_text("# the user's script\n")
    (ws / "inbox" / "decision-001.json").write_text('{"update": 10}')
    mb = mailbox_for(tmp_path, box)

    def agent(w, message, system):
        with open(w / "NOTES.md", "a") as f:
            f.write(f"learned in {message}\n")
        (w / "scratch").mkdir(exist_ok=True)
        (w / "scratch" / "check.py").write_text("print(1)\n")
        return "kept"

    a = serve(box, CallableBackend(agent), tmp_path / "brokerA", run=mb.run_id)
    assert mb.decide(ws, "decision 1", "s", tmp_path)["returncode"] == 0
    end_run(mb, box)
    a.join(10)
    assert (ws / "NOTES.md").read_text() == "learned in decision 1\n"  # the notes are kept on the training side
    assert (ws / "scratch" / "check.py").exists()

    (ws / "inbox" / "decision-002.json").write_text('{"update": 20}')
    ws_b = tmp_path / "brokerB" / mb.run_id / "workspace"
    ws_b.mkdir(parents=True)
    (ws_b / "inbox").mkdir()
    (ws_b / "inbox" / "decision-009.json").write_text("{}")  # a file of a lost attempt, which the run no longer has
    b = serve(box, CallableBackend(agent), tmp_path / "brokerB", run=mb.run_id)
    assert mb.decide(ws, "decision 2", "s", tmp_path)["returncode"] == 0
    end_run(mb, box)
    b.join(10)
    got = listing(ws_b)
    for f in ("reference/training_script_train.py", "inbox/decision-001.json", "inbox/decision-002.json"):
        assert f in got, got
    assert "inbox/decision-009.json" not in got
    assert (ws_b / "NOTES.md").read_text() == "learned in decision 1\nlearned in decision 2\n"
    assert (ws / "NOTES.md").read_text() == "learned in decision 1\nlearned in decision 2\n"


def test_a_new_claude_broker_tells_the_agent_its_conversation_is_new(tmp_path, fast):
    box = tmp_path / "box"
    ws = workspace(tmp_path)
    mb = mailbox_for(tmp_path, box)
    exe = fake_cli(tmp_path)
    t = serve(box, CallableBackend(lambda w, m, s: "kept"), tmp_path / "brokerA", run=mb.run_id)
    (ws / "inbox" / "decision-001.json").write_text("{}")
    mb.decide(ws, "Decision 1.", "s", tmp_path)
    end_run(mb, box)
    t.join(10)
    (ws / "inbox" / "decision-002.json").write_text("{}")
    t = serve(box, ClaudeBackend(binary=str(exe), sandbox=False), tmp_path / "brokerB", run=mb.run_id)
    assert mb.decide(ws, "Decision 2.", "s", tmp_path)["returncode"] == 0
    end_run(mb, box)
    t.join(10)
    sent = next((tmp_path / "brokerB" / mb.run_id / "decisions").glob("002-*")) / "message.txt"
    assert sent.read_text().startswith(
        "(Your earlier conversation could not be resumed, so this is a new conversation: inbox/ holds the inputs "
        "of decision 1, and your current programs are in this folder.)\n\nDecision 2."
    ), sent.read_text()


def test_the_broker_makes_a_failed_call_again_and_logs_the_error_text(tmp_path, fast):
    box = tmp_path / "box"
    ws = workspace(tmp_path)
    mb = mailbox_for(tmp_path, box)
    exe = fake_cli(tmp_path, mode="flaky", fails=1, say="Error: the agent's tool crashed")
    said = []
    t = serve(box, ClaudeBackend(binary=str(exe), sandbox=False, backoff=0.2), tmp_path / "work", run=mb.run_id,
              log=said.append)  # fmt: skip
    out = mb.decide(ws, "Decision 1.", "s", tmp_path)
    assert out["returncode"] == 0 and out["error"] is None, out
    assert any("request 1: agent call 1 failed (exit code 1): Error: the agent's tool crashed" in s
               for s in said), said  # fmt: skip

    class Fails(Backend):
        def decide(self, workspace, message, system, record_dir):
            return dict(text="", cost_usd=None, returncode=1, error="API Error: 529 overloaded", seconds=0.0)

    end_run(mb, box)
    t.join(10)
    said.clear()
    (box / mb.run_id / "done").unlink()  # as the next decide() does, before the broker could read it and leave
    t = serve(box, Fails(), tmp_path / "work2", run=mb.run_id, log=said.append)
    assert mb.decide(ws, "Decision 2.", "s", tmp_path)["returncode"] == 1
    end_run(mb, box)
    t.join(10)
    assert any("return code 1: API Error: 529 overloaded" in s for s in said), said


def test_links_in_the_workspace_are_neither_followed_nor_written_through(tmp_path, fast):
    box = tmp_path / "box"
    ws = workspace(tmp_path)
    secret = tmp_path / "secret.txt"
    secret.write_text("PRIVATE")
    (ws / "inbox" / "leak.json").symlink_to(secret)  # made by a program of the agent, which runs in training
    mb = mailbox_for(tmp_path, box)
    files, manifest = mb._pack(ws)
    assert "inbox/leak.json" not in files and "inbox/leak.json" not in manifest
    outside = tmp_path / "outside"
    outside.mkdir()
    (ws / "scratch").symlink_to(outside)
    assert not mailbox._write_in(ws, "scratch/x.py", "x")
    assert not (outside / "x.py").exists()


# ----- B12 --------------------------------------------------------------------------------------------------------


class Timed(Backend):
    def __init__(self, timeout=None):
        self.timeout_given = timeout is not None
        self.timeout = 3600 if timeout is None else timeout
        self.used, self.messages = [], []

    def decide(self, workspace, message, system, record_dir):
        self.used.append(self.timeout)
        self.messages.append(message)
        return dict(text="kept", cost_usd=0.0, returncode=0, error=None, seconds=0.0)


@pytest.mark.parametrize("given, expect", [(None, (6 * 3600 - 70, 6 * 3600 - 59)), (600, (600, 600))])
def test_the_broker_stops_the_agent_within_what_it_was_told(tmp_path, fast, given, expect):
    box = tmp_path / "box"
    ws = workspace(tmp_path)
    mb = mailbox_for(tmp_path, box, timeout=6 * 3600)
    be = Timed(given)
    logs = []
    t = serve(box, be, tmp_path / "work", run=mb.run_id, log=logs.append)
    system = "... Each decision may take up to 6 hours. ..."  # what the neutral prompt says with this mailbox
    out = mb.decide(ws, "m", system, tmp_path)
    end_run(mb, box)
    t.join(10)
    assert out["returncode"] == 0
    assert expect[0] <= be.used[0] <= expect[1], be.used
    resp = json.loads((box / mb.run_id / "response-0001.json").read_text())
    assert resp["agent_timeout"] == be.used[0]
    assert any("this broker stops it after 600 s" in m for m in logs) == (given == 600)
    # 0.2.1: a broker that stops the agent sooner than the run told it says so in the message (it said nothing)
    if given == 600:
        assert be.messages == ["(This decision may take up to 10 minutes, less than your instructions state.)\n\nm"]
    else:
        assert be.messages == ["m"]  # the run's limit holds, less the minute kept to write the answer


def test_the_limit_note_gives_the_limit_that_holds():
    note = mailbox._limit_note
    assert note(dict(limit=21600, system="x"), 900) == "(This decision may take up to 15 minutes.)\n\n"
    assert note(dict(limit=21600), 21600 - 62) == ""  # the margin to write the answer and a short wait
    assert note(dict(limit=21600), 21600 - 3 * 3600 - 75) == "(This decision may take up to 178 minutes.)\n\n"  # waited
    assert note(dict(limit=None), 900) == note(dict(limit=3600), None) == ""  # an older run, or a backend with no limit
    assert note(dict(limit=3600), 60) == "(This decision may take up to a minute.)\n\n"


# ----- B14 --------------------------------------------------------------------------------------------------------


def test_a_failing_request_does_not_stop_a_keep_serving_broker(tmp_path, fast):
    box = tmp_path / "box"
    ws = workspace(tmp_path)
    mb = mailbox_for(tmp_path, box)
    calls = []

    def agent(w, message, system):
        calls.append(message)
        if message == "first":
            raise RuntimeError("the agent backend broke")
        return "fine"

    t = serve(box, CallableBackend(agent), tmp_path / "work", keep_serving=True)
    first = mb.decide(ws, "first", "s", tmp_path)
    assert first["returncode"] == -1 and "the agent backend broke" in first["error"]
    second = mb.decide(ws, "second", "s", tmp_path)
    assert second["returncode"] == 0 and second["text"] == "fine"
    assert t.is_alive() and t.error is None


class SlowStart(ClaudeBackend):
    """Waits for the run's cancel before starting the CLI: the race window between reading a request and starting."""

    def decide(self, workspace, message, system, record_dir):
        t0 = time.time()
        while not getattr(self, "cancelled", False) and time.time() - t0 < 20:
            time.sleep(0.05)
        return super().decide(workspace, message, system, record_dir)


def test_a_cancel_before_the_agent_starts_does_not_end_the_broker(tmp_path, fast):
    box = tmp_path / "box"
    ws = workspace(tmp_path)
    mb = mailbox_for(tmp_path, box)
    exe = fake_cli(tmp_path)
    threading.Thread(target=mb.decide, args=(ws, "m", "s", tmp_path), daemon=True).start()
    t = serve(box, SlowStart(binary=str(exe), sandbox=False), tmp_path / "work", keep_serving=True)
    for _ in range(100):
        if (tmp_path / "work" / mb.run_id / "decisions").exists():
            break
        time.sleep(0.1)
    mb.cancel()
    time.sleep(3)
    assert t.is_alive() and t.error is None


def test_a_file_removed_while_it_is_read_reads_as_missing(tmp_path):
    f = _Folder(tmp_path)
    assert f.read("nothing.json") is None
    assert f.create("c.json", b"1") and not f.create("c.json", b"2") and f.read("c.json") == b"1"


# ----- B15 --------------------------------------------------------------------------------------------------------


def test_two_brokers_answer_a_request_once(tmp_path, fast):
    box = tmp_path / "box"
    ws = workspace(tmp_path)
    mb = mailbox_for(tmp_path, box)
    served = []

    def agent_for(name):
        def agent(w, message, system):
            served.append((name, message))
            time.sleep(1.0)
            return f"answer from {name}"

        return agent

    for name in ("laptop", "login-node"):
        serve(box, CallableBackend(agent_for(name)), tmp_path / f"work-{name}", keep_serving=True)
    out = [mb.decide(ws, f"decision {k}", "s", tmp_path) for k in (1, 2, 3)]
    time.sleep(1.5)
    assert [m for _, m in served] == ["decision 1", "decision 2", "decision 3"], served
    assert all(o["returncode"] == 0 for o in out)


def test_a_claim_being_written_cannot_be_acquired_by_two_brokers(tmp_path, monkeypatch):
    folder = _Folder(tmp_path / "box")
    folder.make()
    entered, release = threading.Event(), threading.Event()
    results = {}
    real_fdopen = os.fdopen

    def pause_before_writing(fd, *args, **kwargs):
        if threading.current_thread() is creator:
            entered.set()
            assert release.wait(10)
        return real_fdopen(fd, *args, **kwargs)

    creator = threading.Thread(
        target=lambda: results.__setitem__("first", folder.create("claim-0001.json", b'{"broker":"first"}'))
    )
    monkeypatch.setattr(mailbox.os, "fdopen", pause_before_writing)
    try:
        creator.start()
        assert entered.wait(10)
        results["second"] = folder.create("claim-0001.json", b'{"broker":"second"}')
    finally:
        release.set()
        creator.join(10)
    assert not creator.is_alive()
    assert sum(results.values()) == 1, results
    assert json.loads(folder.read("claim-0001.json"))["broker"] == next(k for k, won in results.items() if won)
    assert not list(folder._local().glob("*.tmp"))


@pytest.mark.parametrize("remote", [False, True], ids=["local", "ssh-command"])
def test_exclusive_creation_publishes_complete_private_contents(tmp_path, monkeypatch, remote):
    import subprocess

    folder = _Folder(tmp_path / "box with ' spaces", ssh="fake-host" if remote else None)
    if remote:
        monkeypatch.setattr(
            folder, "_sh",
            lambda cmd, data=None: subprocess.run(["sh", "-c", cmd], input=data, capture_output=True, timeout=10),
        )
    folder.make()
    payload = b'{"broker": "winner", "attempt": "current"}'
    assert folder.create("claim-0001.json", payload)
    assert not folder.create("claim-0001.json", b"another broker")
    assert folder.read("claim-0001.json") == payload
    assert folder._local("claim-0001.json").stat().st_mode & 0o777 == 0o600
    assert not list(folder._local().glob("*.tmp"))


def test_a_broker_that_claims_after_the_answer_does_not_answer_again(tmp_path, fast, monkeypatch):
    """The race behind an occasional failure of the test above on a busy machine: one broker sees request 1 pending,
    the other answers it and removes its claim, then the first one claims it."""
    box = tmp_path / "box"
    ws = workspace(tmp_path)
    mb = mailbox_for(tmp_path, box)
    served = []
    slow, claim = mailbox.broker_id(tmp_path / "work-login-node"), mailbox._claim

    def late_claim(folder, k, attempt, me, log):
        end = time.time() + 10
        while (
            me == slow
            and time.time() < end
            and (folder.read(f"response-{k:04d}.json") is None or folder.read(f"claim-{k:04d}.json") is not None)
        ):
            time.sleep(0.05)
        return claim(folder, k, attempt, me, log)

    monkeypatch.setattr(mailbox, "_claim", late_claim)

    def agent_for(name):
        def agent(w, message, system):
            served.append((name, message))
            time.sleep(1.0)
            return f"answer from {name}"

        return agent

    for name in ("laptop", "login-node"):
        serve(box, CallableBackend(agent_for(name)), tmp_path / f"work-{name}", keep_serving=True)
    assert mb.decide(ws, "decision 1", "s", tmp_path)["returncode"] == 0
    time.sleep(2.0)
    assert [m for _, m in served] == ["decision 1"], served


def test_a_claim_its_broker_stopped_refreshing_is_taken_over(tmp_path, fast, monkeypatch):
    monkeypatch.setattr(mailbox, "CLAIM_STALE", 1.0)
    box = tmp_path / "box"
    ws = workspace(tmp_path)
    mb = mailbox_for(tmp_path, box)
    mb._ensure_folder()
    old = dict(broker="gone-host:00000000", attempt=mb.attempt, t=time.time())
    (box / mb.run_id / "claim-0001.json").write_text(json.dumps(old))
    logs = []
    serve(box, CallableBackend(lambda w, m, s: "late but answered"), tmp_path / "work", keep_serving=True,
          log=logs.append)  # fmt: skip
    out = mb.decide(ws, "m", "s", tmp_path)
    assert out["text"] == "late but answered"
    assert any("taking over request 1 from broker gone-host" in m for m in logs), logs


# ----- D3 ---------------------------------------------------------------------------------------------------------


def test_a_broker_on_a_crashed_run_moves_to_the_run_that_waits(tmp_path, fast):
    box = tmp_path / "box"
    ws = workspace(tmp_path)
    live = mailbox_for(tmp_path, box, "live", broker_wait=5)
    live._ensure_folder()
    crashed = box / "crashed0000"  # newer than the live run, never finished, nothing pending
    crashed.mkdir()
    time.sleep(0.05)
    (crashed / "RUN").write_text("{}")
    os.utime(crashed / "RUN", (time.time() + 5, time.time() + 5))
    t = serve(box, CallableBackend(lambda w, m, s: "answered"), tmp_path / "work")  # one run, not pinned
    time.sleep(0.5)
    out = live.decide(ws, "m", "s", tmp_path)
    assert not out.get("gave_up") and out["text"] == "answered", out
    end_run(live, box)
    t.join(10)
    assert not t.is_alive()


def test_an_answer_whose_write_fails_is_written_again_not_asked_again(tmp_path, fast, monkeypatch):
    box = tmp_path / "box"
    ws = workspace(tmp_path)
    mb = mailbox_for(tmp_path, box)
    real, failed = _Folder.lease, []

    def flaky(self, name, action, body=None, expected=None):
        if action == "answer" and not failed:
            failed.append(name)
            raise RuntimeError("ssh write failed (255): connection reset")
        return real(self, name, action, body=body, expected=expected)

    monkeypatch.setattr(_Folder, "lease", flaky)
    calls = []
    t = serve(box, CallableBackend(lambda w, m, s: calls.append(m) or "once"), tmp_path / "work", keep_serving=True)
    out = mb.decide(ws, "m", "s", tmp_path)
    assert out["text"] == "once" and failed and calls == ["m"]
    assert t.is_alive() and t.error is None


# ----- a broker that is gone for a while is waited for, and the same request is answered --------------


def _fake_broker(box, stop, answers, gone_at=None, gone_for=0.0):
    """A minimal broker: heartbeats, and answers each request k with answers[k] as updater.py (the request's own
    programs otherwise). At request `gone_at` it disappears (no heartbeat, no answer) for `gone_for` seconds."""
    back, seen = None, set()
    while not stop.is_set():
        for run in [d for d in box.iterdir() if (d / "RUN").exists()] if box.exists() else []:
            try:
                latest = json.loads((run / "latest.json").read_text())
            except (OSError, ValueError):
                latest = None
            k = latest["k"] if latest else None
            if k == gone_at and k not in seen:
                seen.add(k)
                back = time.time() + gone_for
            if back is not None and time.time() < back:
                continue
            (run / "broker-alive").write_text(json.dumps(dict(run=run.name, t=time.time())))
            if k is None or (run / f"response-{k:04d}.json").exists():
                continue
            req = json.loads((run / f"request-{k:04d}.json").read_text())
            files = {n: t for n, t in req["files"].items() if n.endswith(".py") and "/" not in n}
            if answers.get(k):
                files["updater.py"] = answers[k]
            resp = dict(run_id=req["run_id"], attempt=req["attempt"], k=k, files=files, text=f"answer {k}", returncode=0)
            (run / f"response-{k:04d}.json").write_text(json.dumps(resp))
        time.sleep(0.05)


def _mailbox_training(tmp_path, name, gone_for, capsys=None):
    import opal
    from test_core import MOMENTUM_UPDATER, SGD_UPDATER, adamw, events, make, train

    box, stop = tmp_path / name / "box", threading.Event()
    answers = {1: SGD_UPDATER, 2: MOMENTUM_UPDATER}
    th = threading.Thread(target=_fake_broker, args=(box, stop, answers, 2, gone_for), daemon=True)
    th.start()
    m, data = make()
    o, s = adamw(m, 120)
    w = opal.wrap(o, m, scheduler=s, total_steps=120, goal="t", decisions=3, every=10, run_dir=tmp_path / name / "r",
                  backend=f"mailbox:{box}", backend_options=dict(poll=0.05, broker_wait=0.5), verbose=True)  # fmt: skip
    t0 = time.time()
    train(m, w, data, 120, sched=s)
    w.close()
    stop.set()
    th.join(5)
    params = [p.detach().clone() for p in m.parameters()]
    return params, events(tmp_path / name / "r"), time.time() - t0


def test_a_broker_gone_longer_than_broker_wait_is_waited_for_and_the_run_is_the_same(tmp_path, fast, capsys):
    calm, calm_events, _ = _mailbox_training(tmp_path, "calm", 0.0)
    capsys.readouterr()
    params, evs, took = _mailbox_training(tmp_path, "gone", 4.0)  # > broker_wait (0.5 s) and the stale heartbeat (1.5 s)
    assert took >= 4.0 and all(torch.equal(a, b) for a, b in zip(params, calm))
    out = [(e["k"], e["kind"]) for e in evs if e["kind"] in ("applied", "kept", "rejected")]
    assert out == [(e["k"], e["kind"]) for e in calm_events if e["kind"] in ("applied", "kept", "rejected")]
    assert out[:2] == [(1, "applied"), (2, "applied")]
    second = next(e for e in evs if e.get("k") == 2 and e["kind"] == "applied")
    assert not second.get("failed") and not second.get("gave_up") and second["unavailable_seconds"] > 1
    err = capsys.readouterr().err
    assert err.count("[opal] decision 2: the mailbox's broker is unavailable (no broker is answering (mailbox ") == 1
    assert "training waits for the agent and checks again every 0.05 s" in err
    assert err.count("[opal] decision 2: the mailbox's broker is available again after") == 1


def test_a_broker_whose_agent_is_unavailable_waits_and_the_run_waits_with_it(tmp_path, fast, quick_waits):
    """The broker's Claude Code is not logged in for two calls: the broker never writes a failed answer; the run sees
    why it waits (unavailable-NNNN.json), says so, and gets the answer."""
    box = tmp_path / "box"
    ws = workspace(tmp_path)
    mb = mailbox_for(tmp_path, box, poll=0.05)
    exe = fake_cli(tmp_path, mode="flaky", fails=2, half=0, say="Not logged in · Please run /login")
    broker_said, run_said = [], []
    t = serve(box, ClaudeBackend(binary=str(exe), sandbox=False), tmp_path / "work", run=mb.run_id,
              log=broker_said.append)  # fmt: skip
    mb.log, mb.training_waits = run_said.append, True
    out = mb.decide(ws, "Decision 1.", "s", tmp_path)
    assert out["returncode"] == 0 and out["error"] is None and out["unavailable_seconds"] > 0, out
    assert any("request 1: Claude Code is unavailable (Not logged in" in s for s in broker_said), broker_said
    assert not any("answered request 1" in s and "return code 1" in s for s in broker_said), broker_said
    assert run_said[0].startswith("Claude Code (on the broker) is unavailable (Not logged in"), run_said
    assert run_said[-1].startswith("Claude Code (on the broker) is available again after"), run_said
    assert not (box / mb.run_id / "unavailable-0001.json").exists()
    end_run(mb, box)
    t.join(10)


def test_doctor_checks_a_mailbox_setup(tmp_path, capsys):
    from opal import cli

    box = tmp_path / "box"
    assert cli.main(["doctor", "--backend", f"mailbox:{box}"]) == 1
    out = capsys.readouterr().out
    assert "no Claude Code is needed here" in out and "writable" in out and "caffeinate -i opal broker" in out
    (box / "broker-alive").write_text(json.dumps(dict(run=None, broker="laptop:1234")))
    assert cli.main(["doctor", "--backend", f"mailbox:{box}"]) == 0
    assert "a broker is serving this mailbox (laptop:1234)" in capsys.readouterr().out


def test_remaining_run_budget_reaches_the_broker_and_is_reset_for_uncapped_requests(tmp_path, fast):
    class BudgetRecorder(Backend):
        def __init__(self):
            self.budgets = []

        def decide(self, workspace, message, system, record_dir):
            self.budgets.append(getattr(self, "budget_left", None))
            return dict(text="kept", returncode=0, error=None, cost_usd=0.05, seconds=0)

    box = tmp_path / "box"
    ws = workspace(tmp_path)
    mb = mailbox_for(tmp_path, box)
    backend = BudgetRecorder()
    broker = serve(box, backend, tmp_path / "broker", run=mb.run_id)
    try:
        for budget in (0.25, 0.1, None):
            mb.budget_left = budget
            assert mb.decide(ws, "Decision.", "s", tmp_path)["returncode"] == 0
        assert backend.budgets == [0.25, 0.1, None]
    finally:
        mb.finish()
        broker.join(10)
    assert not broker.is_alive() and broker.error is None
