"""examples/overfit/train.py on CPU with synthetic data: the agent-facing texts (decision messages, inbox, system prompt,
obs.metrics) carry the validation measurements and never the test metrics, which only the harness's log holds."""

import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SEEN = """
EVERY = 2
SEEN = []
def analyze(obs, memory):
    SEEN.extend(obs.metrics)
"""
SMOKE = ["--data", "synthetic", "--device", "cpu", "--width", "4", "--epochs", "3", "--batch-size", "32"]


def records():
    sys.path.insert(0, str(ROOT / "examples" / "overfit"))
    import records as mod

    return mod


def harness():
    spec = importlib.util.spec_from_file_location("overfit_train", ROOT / "examples" / "overfit" / "train.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("setting", ["label10", "small20"])
def test_the_agent_never_sees_test_metrics(tmp_path, setting):
    messages = []

    def agent(ws, msg, system):
        if not messages:
            (ws / "analysis.py").write_text(SEEN)
        messages.append(msg)
        return "kept"

    out = tmp_path / "run"
    argv = SMOKE + ["--setting", setting, "--opal", "--validation", "--evals-per-epoch", "3", "--every", "2"]
    result, opt = harness().main(argv + ["--decisions", "4", "--out", str(out)], backend=agent)
    log = list(map(json.loads, open(out / "log.jsonl")))
    evals, per_epoch = [r for r in log if r["kind"] == "eval"], log[0]["steps_per_epoch"]
    assert [r["update"] for r in evals] == records().eval_updates(per_epoch, 3, 0, 3) == log[0]["eval_updates"]
    assert all("test_acc" in r and "test_loss" in r for r in evals)  # the test set at the same moments, for us
    seen = sys.modules[opt.analysis.__name__].SEEN
    run = out / "opal"
    inbox = [p.read_text() for p in sorted((run / "workspace" / "inbox").glob("*.json"))]
    shown = messages + inbox + [(run / "system.md").read_text(), json.dumps(seen), (run / "metrics.jsonl").read_text()]
    assert len(messages) >= 3 and seen and inbox
    for text in shown:
        assert "test_acc" not in text and "test_loss" not in text
        for r in evals:
            assert repr(r["test_loss"]) not in text and repr(r["test_acc"]) not in text
    # positive control: the validation measurements do reach the agent
    assert all(set(m) <= {"update", "val_loss", "val_accuracy"} for m in seen)
    assert any(m["val_loss"] == pytest.approx(evals[-2]["dev_loss"]) for m in seen)
    assert json.loads(inbox[-1])["validation"][0]["loss"] == pytest.approx(evals[0]["dev_loss"])
    assert f"{evals[0]['update']}: {evals[0]['dev_loss']:.4g}, {evals[0]['dev_acc']:.4g}" in messages[-1]
    system = (run / "system.md").read_text()
    n = records().eval_interval(per_epoch, 3)
    assert f"about every {n} updates, at times that are not announced" in system and f"Every {n} updates" not in system
    assert (out / "best.pt").exists() and result["early_stopping_on_dev_acc"]["test_acc"] is not None


def test_static_run_and_no_dev_shown_without_validation(tmp_path):
    result, _ = harness().main(SMOKE + ["--setting", "small20", "--out", str(tmp_path / "static")])
    assert result["method"] == "adamw-lr0.001" and result["opal_run"] is None
    messages = []
    harness().main(
        SMOKE + ["--setting", "small20", "--opal", "--decisions", "3", "--every", "2", "--out", str(tmp_path / "nodev")],
        backend=lambda ws, msg, system: messages.append(msg) or "kept",
    )
    assert messages and not any("Validation set" in m or "val_" in m or "dev_" in m for m in messages)
    assert "validation set" not in (tmp_path / "nodev" / "opal" / "system.md").read_text()


def test_the_files_the_agent_reads_say_nothing_about_the_experiment():
    """OPAL shows the agent the training script and the model's source; the experiment's question, the wrong labels,
    early stopping and the setting names live in README.md, splits.py and records.py only."""
    for name in ("train.py", "resnet.py"):
        text = (ROOT / "examples" / "overfit" / name).read_text().lower()
        for word in ("overfit", "wrong", "noise", "noisy", "corrupt", "early", "memoris", "memoriz", "label10", "small20"):
            assert word not in text, (name, word)
        # 0.2.2: nor when the validation set is evaluated (records.py draws the times)
        cadence = ("dev_every", "dev-every", "every epoch", "evaluate every", "evaluates every", "eval time")
        for word in cadence + ("evaluation time",):
            assert word not in text, (name, word)
        assert not re.search(r"\bu\s*%", text), name  # no modulo of the update counter


def test_evaluation_times_vary_across_epochs_and_follow_the_seed():
    rec = records()
    times = rec.eval_updates(78, 150, seed=0)
    assert times[-1] == 78 * 150  # the last update is always evaluated, as before
    drawn = times if len(times) == 150 else times[:-1]
    assert len(drawn) == 150 and all((u - 1) // 78 == e for e, u in enumerate(drawn))  # exactly one in each epoch
    offsets = [u - 78 * ((u - 1) // 78) for u in drawn]
    assert len(set(offsets)) > 30  # anywhere in the epoch, not a cadence
    assert len({b - a for a, b in zip(drawn, drawn[1:])}) > 30
    assert rec.eval_updates(78, 150, seed=0) == times and rec.eval_updates(78, 150, seed=1) != times
    assert rec.eval_interval(78) == 78 and rec.eval_interval(32, 16) == 2
    two = rec.eval_updates(32, 3, seed=0, per_epoch=16)  # one in each pair of updates
    two = two if len(two) == 48 else two[:-1]  # without the last update when it was not the last pair's draw
    assert [(u - 1) // 2 for u in two] == list(range(48)) and len({u % 2 for u in two}) == 2
