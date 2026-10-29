"""CPU regression witnesses from the adversarial review of 0.3.4."""

import copy
import html
import importlib.metadata
import json
from pathlib import Path

import pytest
import torch

import opal
from opal import report
from opal.backends.claude import ClaudeBackend
from test_claude_backend import fake_cli


def wrap(model, raw, path, **kw):
    options = dict(
        total_steps=8,
        goal="release regression",
        backend="keep",
        every=1,
        decisions=0,
        share_code=False,
        summarize=False,
        verbose=False,
        run_dir=path,
    )
    if "schedule" in kw:
        options["decisions"] = None
    return opal.wrap(raw, model, **dict(options, **kw))


SECRETS = [
    "sk-ant-api03-REVIEWONLY-CANARY-1234567890",
    "sk-REVIEWONLY1234567890",
    "ghp_REVIEWONLY1234567890",
    "github_pat_REVIEWONLY1234567890",
    "hf_REVIEWONLY1234567890",
    "AKIA1234567890ABCDEF",
    "Bearer REVIEWONLY.token+/=",
]


def test_backend_secrets_do_not_reach_records_prompts_stderr_or_html(tmp_path, capsys):
    exe = fake_cli(tmp_path, mode="flaky", fails=100, half=0, say="local parser failure; " + " ".join(SECRETS))
    backend = ClaudeBackend(binary=str(exe), sandbox=False, retries=0, unavailable_timeout=0)
    model = torch.nn.Linear(1, 1)
    opt = wrap(
        model,
        torch.optim.SGD(model.parameters(), lr=0.1),
        tmp_path / "run",
        backend=backend,
        total_steps=4,
        schedule=[1, 2],
        verbose=True,
    )
    try:
        for _ in range(4):
            for p in model.parameters():
                p.grad = torch.ones_like(p)
            opt.step(loss=1.0)
    finally:
        opt.close()
    records = [capsys.readouterr().err]
    for name in [
        "decisions/001/attempts.jsonl",
        "decisions/001/backend.jsonl",
        "decisions/001/response.json",
        "decisions/002/message.txt",
        "workspace/inbox/decision-002.json",
        "events.jsonl",
        "report.html",
    ]:
        records.append(html.unescape((opt.run.dir / name).read_text()))
    for secret in SECRETS:
        assert all(secret not in text for text in records), secret
    assert "local parser failure" in records[3]
    assert opt._last_answer is not None


def test_report_redacts_secrets_in_existing_records(tmp_path):
    from opal.report_privacy import Privacy

    text = json.dumps(dict(error="; ".join(SECRETS)))
    cleaned = Privacy(tmp_path, {}).text(text)
    for secret in SECRETS:
        assert secret not in cleaned
    # Redaction must preserve JSON quoting around an Authorization header.
    json.loads(cleaned)


@pytest.mark.parametrize("method", ["step", "zero_grad"])
@pytest.mark.parametrize("mutation", ["add", "replace", "remove"])
def test_raw_parameter_changes_are_refused_before_update_or_zero(tmp_path, method, mutation):
    model = torch.nn.Linear(1, 1)
    raw = torch.optim.SGD([model.weight], lr=0.1)
    opt = wrap(model, raw, tmp_path / "run")
    model.weight.grad = torch.ones_like(model.weight)
    opt.step(loss=1.0)
    if mutation == "add":
        raw.add_param_group(dict(params=[model.bias]))
    elif mutation == "replace":
        raw.param_groups[0]["params"] = [model.bias]
    else:
        raw.param_groups[0]["params"] = []
    before = copy.deepcopy(model.state_dict())
    model.bias.grad = torch.ones_like(model.bias)
    try:
        with pytest.raises(RuntimeError, match="create all parameter groups before opal.wrap"):
            getattr(opt, method)()
        assert opt.update == 1
        assert model.bias.grad is not None
        assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())
    finally:
        opt.close(interrupted=True)


@pytest.mark.parametrize("optimizer", ["AdamW", "SGD"])
def test_unfreeze_existing_group_matches_adding_group_bit_for_bit(tmp_path, optimizer):
    torch.manual_seed(42)
    plain = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Tanh(), torch.nn.Linear(4, 2))
    kept = copy.deepcopy(plain)
    options = dict(lr=0.03, weight_decay=0.1)
    if optimizer == "SGD":
        options.update(momentum=0.9, nesterov=True)
    factory = getattr(torch.optim, optimizer)
    plain[0].requires_grad_(False)
    kept[0].requires_grad_(False)
    raw_plain = factory(plain[2].parameters(), **options)
    raw_kept = factory(
        [{"params": kept[2].parameters()}, {"params": kept[0].parameters(), "lr": 0.02}], **options,
    )
    opt = wrap(kept, raw_kept, tmp_path / "run", total_steps=8)
    try:
        for step in range(8):
            if step == 3:
                plain[0].requires_grad_(True)
                kept[0].requires_grad_(True)
                raw_plain.add_param_group(dict(params=plain[0].parameters(), lr=0.02))
            x, y = torch.randn(5, 3), torch.randn(5, 2)
            for model, optimizer_instance in [(plain, raw_plain), (kept, opt)]:
                optimizer_instance.zero_grad()
                loss = (model(x) - y).square().mean()
                loss.backward()
                if optimizer_instance is opt:
                    opt.step(loss=loss)
                else:
                    optimizer_instance.step()
            for left, right in zip(plain.parameters(), kept.parameters()):
                assert torch.equal(left.detach().view(torch.uint8), right.detach().view(torch.uint8)), (optimizer, step)
                for key, value in raw_plain.state.get(left, {}).items():
                    other = raw_kept.state[right][key]
                    assert torch.equal(value, other) if torch.is_tensor(value) else value == other
            if step < 3:
                assert all(p.grad is None and p not in raw_kept.state for p in kept[0].parameters())
    finally:
        opt.close()


def test_multiple_optimizers_with_gradscaler_are_refused_before_any_update(tmp_path):
    model = torch.nn.Linear(1, 1)
    raw = [torch.optim.SGD([p], lr=0.1) for p in model.parameters()]
    opt = wrap(model, raw, tmp_path / "run")
    scaler = torch.amp.GradScaler("cpu")
    loss = model(torch.ones(1, 1)).sum()
    scaler.scale(loss).backward()
    model.weight.grad.fill_(float("inf"))
    before = copy.deepcopy(model.state_dict())
    try:
        with pytest.raises(RuntimeError, match="multiple optimizers.*GradScaler"):
            scaler.step(opt, loss=loss)
        assert opt.update == 0
        assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())
    finally:
        opt.close(interrupted=True)


UPDATER = """
class Updater:
    def __init__(self, named): self.named, self.k = named, 0
    def step(self, grads, update):
        self.k += 1
        for n, p in self.named:
            if grads[n] is not None: p.add_(grads[n], alpha=-0.25)
    def state_dict(self): return dict(k=self.k)
    def load_state_dict(self, state): self.k = state['k']
def build(named_params, previous=None, spec=None): return Updater(named_params)
"""


def test_accelerate_checkpoint_restores_program_version_updates_and_next_step(tmp_path):
    from accelerate import Accelerator
    from accelerate.state import AcceleratorState, GradientState

    def reset():
        AcceleratorState._reset_state(True)
        GradientState._reset_state()

    def setup(name, backend):
        acc = Accelerator(cpu=True)
        model = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.fill_(1.0)
        model, prepared = acc.prepare(model, torch.optim.SGD(model.parameters(), lr=0.1))
        opt = wrap(model, prepared, tmp_path / name, backend=backend, schedule=[1])
        return acc, model, prepared, opt

    def fake(ws, message, system):
        (ws / "updater.py").write_text(UPDATER)
        return "fixed synthetic update"

    def step(acc, model, opt):
        opt.zero_grad()
        loss = model(torch.ones(1, 1)).sum()
        acc.backward(loss)
        opt.step(loss=loss)

    reset()
    opt = resumed = None
    try:
        acc, model, prepared, opt = setup("continuous", fake)
        for _ in range(3):
            step(acc, model, opt)
        acc.save_state(tmp_path / "checkpoint")
        saved = torch.load(tmp_path / "checkpoint/optimizer.bin", weights_only=True)
        assert saved.get("opal") == 1
        assert prepared.state is opt.user_optimizers[0].state
        assert opt.update == 3 and opt.version == 1
        step(acc, model, opt)
        expected = model.weight.detach().clone()
        reset()
        acc2, model2, prepared2, resumed = setup("resumed", "keep")
        acc2.load_state(tmp_path / "checkpoint")
        assert resumed.update == 3 and resumed.version == 1
        assert resumed.files == opt.files
        assert resumed.updater.k == 3
        step(acc2, model2, resumed)
        assert torch.equal(model2.weight, expected)
        assert resumed.updater.k == opt.updater.k == 4
    finally:
        for obj in (opt, resumed):
            if obj is not None:
                obj.close(interrupted=True)
        reset()


def test_release_version_and_provenance_are_consistent(monkeypatch):
    from opal import results
    import subprocess

    assert opal.__version__ == importlib.metadata.version("opal-optimizer") == "0.3.9"
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: subprocess.CompletedProcess(a, 1))
    assert results.opal_commit() == "0.3.9"


def test_truncated_loss_window_start_tracks_retained_values_between_decisions(tmp_path):
    captured = []

    def fake(ws, message, system):
        k = len(captured) + 1
        captured.append(json.loads((ws / "inbox" / f"decision-{k:03d}.json").read_text()))
        return "keep"

    model = torch.nn.Linear(1, 1, bias=False)
    opt = wrap(
        model,
        torch.optim.SGD(model.parameters(), lr=0),
        tmp_path / "run",
        backend=fake,
        total_steps=300015,
        schedule=[100005, 200010],
        every=100005,
    )
    try:
        for start, end in [(1, 100005), (100006, 200010)]:
            opt.update = end
            opt._rank0_check(end, False, list(range(start, end + 1)), start, False)
            opt._blocking_point(end)
        assert [(r["losses_from"], len(r["losses"]), r["losses"][0], r["losses"][-1]) for r in captured] == [
            (6, 100000, 6, 100005),
            (100011, 100000, 100011, 200010),
        ]
        assert [r["losses_omitted"] for r in captured] == [5, 5]
    finally:
        opt.close(interrupted=True)


def test_cancelled_decision_settles_late_cost_in_end_event_and_report(tmp_path):
    import threading
    from opal.backends.base import Backend

    class ChargedCancellation(Backend):
        name = "fake-cancellable"

        def __init__(self):
            self.started, self.release = threading.Event(), threading.Event()

        def decide(self, workspace, message, system, record_dir):
            self.started.set()
            assert self.release.wait(5)
            return dict(text="cancelled", cost_usd=0.25, returncode=-1, error="cancelled", seconds=0.1)

        def cancel(self):
            self.timer = threading.Timer(0.2, self.release.set)
            self.timer.start()

    backend = ChargedCancellation()
    model = torch.nn.Linear(1, 1)
    opt = wrap(
        model,
        torch.optim.SGD(model.parameters(), lr=0.1),
        tmp_path / "run",
        backend=backend,
        schedule=[1],
        blocking=False,
    )
    opt.step(loss=1.0)
    assert backend.started.wait(2)
    opt.close()
    opt.worker.thread.join(3)
    backend.timer.join(1)
    assert not opt.worker.thread.is_alive()
    events = [json.loads(line) for line in (opt.run.dir / "events.jsonl").read_text().splitlines()]
    end = next(e for e in events if e["kind"] == "end")
    assert end["cost_usd"] == opt.worker.cost == 0.25
    page = report.write_report(opt.run.dir).read_text()
    assert "Recorded agent cost: $0.2500" in page


def test_report_counts_response_cost_recorded_after_end(tmp_path):
    events = [dict(kind="requested", k=1, update=1), dict(kind="end", cost_usd=0.0)]
    (tmp_path / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
    rec = tmp_path / "decisions/001"
    rec.mkdir(parents=True)
    (rec / "response.json").write_text(json.dumps(dict(cost_usd=0.25, text="cancelled")))
    assert "Recorded agent cost: $0.2500" in report.write_report(tmp_path).read_text()


def test_wake_reasons_do_not_accumulate_after_decision_budget_is_spent(tmp_path):
    def fake(ws, message, system):
        (ws / "analysis.py").write_text(
            'EVERY = 1\ndef analyze(obs, memory):\n    return {"wake": "update " + str(obs.update)}\n'
        )
        return "wake each update"

    model = torch.nn.Linear(1, 1, bias=False)
    opt = wrap(
        model,
        torch.optim.SGD(model.parameters(), lr=0),
        tmp_path / "run",
        backend=fake,
        total_steps=3001,
        schedule=[1],
        max_decisions=1,
    )
    try:
        for u in range(2000):
            opt.step(loss=1.0)
            assert len(opt._reasons) <= 10
        assert opt.worker.k == 1
        assert not opt._reasons
    finally:
        opt.close(interrupted=True)


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('pathlib').Path('cache')",
        "range(3)",
        "__import__('collections').deque([1])",
        "(lambda x: (setattr(x, 'path', __import__('pathlib').Path('cache')), x)[1])(__import__('torch').ones(1))",
        "(lambda x: (setattr(x, 'path', __import__('pathlib').Path('cache')), x)[1])(__import__('collections').Counter())",
    ],
)
def test_pickleable_nonplain_agent_state_is_rejected_and_safe_checkpoint_loads(tmp_path, expression):
    source = UPDATER.replace("return dict(k=self.k)", "return dict(cache=" + expression + ")")

    def fake(ws, message, system):
        (ws / "updater.py").write_text(source)
        return "nonplain state"

    model = torch.nn.Linear(1, 1)
    opt = wrap(model, torch.optim.SGD(model.parameters(), lr=0.1), tmp_path / "run", backend=fake, schedule=[1])
    try:
        opt.step(loss=1.0)
        assert opt.version == 0
        path = tmp_path / "checkpoint.pt"
        torch.save(opt.state_dict(), path)
        assert torch.load(path, weights_only=True)["version"] == 0
        assert any("state_dict" in str(error) for error in opt.errors)
    finally:
        opt.close(interrupted=True)


def test_agent_state_cannot_be_cyclic_but_shared_plain_values_are_valid():
    import io
    from collections import Counter, OrderedDict

    from opal import programs

    state = []
    state.append(state)
    assert programs.unpicklable(state)
    shared = [torch.ones(2), 1, None, True, "value"]
    assert not programs.unpicklable(dict(first=shared, second=(shared,)))
    safe = dict(milestones=Counter({30: 1}), ordered=OrderedDict(first=shared))
    assert not programs.unpicklable(safe)
    buffer = io.BytesIO()
    torch.save(safe, buffer)
    buffer.seek(0)
    loaded = torch.load(buffer, weights_only=True)
    assert type(loaded["milestones"]) is Counter and type(loaded["ordered"]) is OrderedDict


def _permutation_ddp_worker(rank, directory):
    from datetime import timedelta
    import torch.distributed as dist

    torch.set_num_threads(1)
    root = Path(directory)
    dist.init_process_group(
        "gloo", init_method=(root / "init").as_uri(), rank=rank, world_size=2, timeout=timedelta(seconds=20)
    )
    opt = None
    try:
        model = torch.nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            model.weight.copy_(torch.tensor([[1.0, 2.0]]))
        ddp = torch.nn.parallel.DistributedDataParallel(model)

        def fake(ws, message, system):
            (ws / "updater.py").write_text("""
class Updater:
    def __init__(self, named, rank): self.named, self.rank = named, rank
    def step(self, grads, update):
        if update == 1 and self.rank == 1:
            for n, p in self.named: p.copy_(p.flip(-1))
    def state_dict(self): return {}
    def load_state_dict(self, state): pass
def build(named_params, previous=None, spec=None): return Updater(named_params, spec['rank'])
""")
            return "rank-local permutation"

        opt = wrap(
            ddp,
            torch.optim.SGD(model.parameters(), lr=0),
            root / "run",
            backend=fake,
            schedule=[1],
            total_steps=4,
            rank_check_every=1,
        )
        for _ in range(4):
            opt.zero_grad()
            loss = ddp(torch.tensor([[1.0, 3.0]])).sum()
            loss.backward()
            opt.step(loss=loss)
        assert torch.equal(model.weight, torch.tensor([[1.0, 2.0]]))
        if rank == 0:
            events = [json.loads(line) for line in (root / "run/events.jsonl").read_text().splitlines()]
            flags = [e for e in events if e["kind"] == "rank_divergence"]
            assert len(flags) == 1 and flags[0]["names"] == ["weight"]
    finally:
        if opt is not None:
            opt.close(interrupted=True)
        dist.destroy_process_group()


def test_ddp_permutation_collision_is_detected_and_trial_rolls_back(tmp_path):
    torch.multiprocessing.spawn(_permutation_ddp_worker, args=(str(tmp_path),), nprocs=2, join=True)


def test_custom_backend_exception_and_log_are_redacted(tmp_path, capsys):
    from opal.backends.base import Backend

    class FailingBackend(Backend):
        name = "fake-failure"

        def decide(self, workspace, message, system, record_dir):
            self.log(" ".join(SECRETS))
            raise RuntimeError(" ".join(SECRETS))

    model = torch.nn.Linear(1, 1)
    opt = wrap(
        model,
        torch.optim.SGD(model.parameters(), lr=0.1),
        tmp_path / "run",
        backend=FailingBackend(),
        total_steps=4,
        schedule=[1, 2],
        verbose=True,
    )
    try:
        for _ in range(4):
            opt.step(loss=1.0)
    finally:
        opt.close()
    recorded = capsys.readouterr().err + "".join(p.read_text() for p in opt.run.dir.glob("decisions/*/response.json"))
    recorded += (opt.run.dir / "decisions/002/message.txt").read_text()
    for secret in SECRETS:
        assert secret not in recorded


def test_codex_errors_and_client_version_use_release_contract(tmp_path):
    from test_codex_backend import backend, decide

    be, ws = backend(tmp_path, turn_fails=1, kind="rateLimitExceeded", error=" ".join(SECRETS))
    out = decide(be, ws)
    recorded = json.dumps(out) + (ws.parent / "001/backend.jsonl").read_text()
    for secret in SECRETS:
        assert secret not in recorded
    calls = [json.loads(line) for line in (tmp_path / "provider/calls.jsonl").read_text().splitlines()]
    initialize = next(call for call in calls if call["method"] == "initialize")
    assert initialize["params"]["clientInfo"]["version"] == opal.__version__ == "0.3.9"


def test_later_nonplain_agent_state_is_omitted_before_weights_only_checkpoint(tmp_path):
    source = UPDATER.replace("return dict(k=self.k)", "return dict(k=self.k, cache=getattr(self, 'cache', None))")

    def fake(ws, message, system):
        (ws / "updater.py").write_text(source)
        return "initially plain state"

    model = torch.nn.Linear(1, 1)
    opt = wrap(model, torch.optim.SGD(model.parameters(), lr=0.1), tmp_path / "run", backend=fake, schedule=[1])
    try:
        opt.step(loss=1.0)
        assert opt.version == 1
        opt.updater.cache = Path("relative-cache")
        state = opt.state_dict()
        assert state["updater"] is None
        path = tmp_path / "checkpoint.pt"
        torch.save(state, path)
        assert torch.load(path, weights_only=True)["updater"] is None
    finally:
        opt.close(interrupted=True)


def test_cancel_cost_stays_unknown_until_worker_settles_after_close_timeout(tmp_path):
    import threading
    from opal.backends.base import Backend

    class SlowSettlement(Backend):
        name = "fake-slow-settlement"

        def __init__(self):
            self.started, self.release = threading.Event(), threading.Event()

        def decide(self, workspace, message, system, record_dir):
            self.started.set()
            assert self.release.wait(10)
            return dict(text="cancelled", cost_usd=0.25, returncode=-1, error="cancelled", seconds=2)

        def cancel(self):
            pass

    backend = SlowSettlement()
    model = torch.nn.Linear(1, 1)
    opt = wrap(
        model,
        torch.optim.SGD(model.parameters(), lr=0.1),
        tmp_path / "run",
        backend=backend,
        schedule=[1],
        blocking=False,
    )
    try:
        opt.step(loss=1.0)
        assert backend.started.wait(2)
        opt.close()
        before = report.write_report(opt.run.dir).read_text()
        assert "Recorded agent cost: $0.0000" not in before
        backend.release.set()
        opt.worker.thread.join(3)
        assert not opt.worker.thread.is_alive()
        opt._live.finish(timeout=3)
        assert "Recorded agent cost: $0.2500" in (opt.run.dir / "report.html").read_text()
        events = [json.loads(line) for line in (opt.run.dir / "events.jsonl").read_text().splitlines()]
        assert events[-1]["kind"] == "cost_settled" and events[-1]["cost_usd"] == 0.25
    finally:
        backend.release.set()
        if opt.worker.thread:
            opt.worker.thread.join(3)
        opt.close()


def test_summary_backend_errors_are_redacted_before_cache(tmp_path, monkeypatch):
    from opal import summaries

    reply = json.dumps(dict(type="result", is_error=True, result=" ".join(SECRETS)))
    monkeypatch.setattr(summaries, "_run_cli", lambda *a, **kw: (1, reply, " ".join(SECRETS)))
    summarizer = summaries.Summarizer()
    summarizer._flags = set()
    assert summaries.ensure(tmp_path, summarizer, 1, "change", {"updater.py": "a=1"}, {"updater.py": "a=2"}) is None
    recorded = (tmp_path / "summary-error.json").read_text()
    for secret in SECRETS:
        assert secret not in recorded


def test_mailbox_broker_errors_and_unavailable_reason_are_redacted_before_records(tmp_path):
    from opal.backends.base import Backend
    from opal.backends.mailbox import _Folder, _answer_request

    class FailingBackend(Backend):
        name = "fake-broker-failure"
        on_unavailable = None

        def decide(self, workspace, message, system, record_dir):
            self.log(" ".join(SECRETS))
            self.on_unavailable(" ".join(SECRETS))
            raise RuntimeError(" ".join(SECRETS))

    folder = _Folder(tmp_path / "mailbox")
    folder.make()
    folder.write("request-0001.json", json.dumps(dict(attempt="attempt", run_id="run", message="test")).encode())
    ws, work = tmp_path / "workspace", tmp_path / "broker"
    ws.mkdir()
    logs = []
    result = _answer_request(
        folder, FailingBackend(), work, ws, "run", 1, "attempt", True, dict(me="broker", claim=None), logs.append
    )
    recorded = result[1].decode() + "\n".join(logs)
    recorded += (work / "decisions/001-attempt/response.json").read_text()
    recorded += folder.read("unavailable-0001.json").decode()
    for secret in SECRETS:
        assert secret not in recorded


def test_codex_provider_stderr_is_redacted_before_persistent_diagnostics(tmp_path):
    from test_codex_backend import backend, decide

    be, ws = backend(tmp_path)
    exe = Path(be.binary)
    source = exe.read_text()
    source = source.replace(
        "cfg = json.loads", "sys.stderr.write(" + repr(" ".join(SECRETS) + "\n") + ")\ncfg = json.loads"
    )
    exe.write_text(source)
    assert decide(be, ws)["error"] is None
    recorded = (be.control / "codex-stderr.txt").read_text()
    assert "<hidden>" in recorded
    for secret in SECRETS:
        assert secret not in recorded


def test_codex_preflight_error_text_is_redacted(tmp_path, monkeypatch):
    import subprocess
    import sys
    from opal.backends.codex import CodexBackend

    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: subprocess.CompletedProcess(a, 1, "", " ".join(SECRETS)))
    be = CodexBackend(binary=sys.executable, codex_home=tmp_path / "isolated")
    with pytest.raises(RuntimeError) as caught:
        be.preflight()
    for secret in SECRETS:
        assert secret not in str(caught.value)


@pytest.mark.parametrize("encoding", ["newline_bearer", "unicode_key", "plain_bearer", "stderr_bearer"])
def test_json_encoded_credentials_are_redacted_before_backend_transcript(tmp_path, encoding):
    if encoding in ("newline_bearer", "plain_bearer", "stderr_bearer"):
        token = "REVIEW-ESCAPED-BEARER-1234567890"
        error = "Bearer\n" + token
    else:
        token = "sk-REVIEWONLY1234567890"
        error = token
    said = json.dumps(dict(type="result", is_error=True, result=error))
    if encoding == "plain_bearer":
        said = error
    if encoding == "unicode_key":
        said = said.replace("sk-", r"\u0073\u006b\u002d")
    exe = fake_cli(tmp_path, mode="flaky", fails=100, half=0, say=said)
    if encoding == "stderr_bearer":
        exe.write_text(exe.read_text().replace("argv = sys.argv[1:]", "sys.stdout = sys.stderr\nargv = sys.argv[1:]"))
    backend = ClaudeBackend(binary=str(exe), sandbox=False, retries=0, unavailable_timeout=0)
    model = torch.nn.Linear(1, 1)
    opt = wrap(model, torch.optim.SGD(model.parameters(), lr=0.1), tmp_path / "run", backend=backend, schedule=[1])
    try:
        opt.step(loss=1.0)
        transcript = (opt.run.dir / "decisions/001/backend.jsonl").read_text()
        decoded = []
        for line in transcript.splitlines():
            try:
                decoded.append(json.loads(line))
            except ValueError:
                decoded.append(line)
        assert all(token not in str(event) for event in decoded)
        assert token not in opt.run.dir.joinpath("decisions/001/response.json").read_text()
    finally:
        opt.close(interrupted=True)
