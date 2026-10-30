"""Opt-in v1 options: decisions evenly spaced (schedule="uniform"), the next scheduled decision's update in
each message (show_next_decision=True), and the draft neutral prompt (prompt="neutral", task_facts=...).
The defaults are unchanged."""

import re
import types

import pytest
import torch

import opal
from opal.schedule import decision_updates
from test_core import adamw, make, train
from test_review1 import scripted


def wrapped(tmp_path, agent, steps=200, **kw):
    m, data = make()
    o, s = adamw(m, steps)
    kw = dict(dict(decisions=4, every=10, blocking=True, verbose=False), **kw)
    w = opal.wrap(o, m, scheduler=s, total_steps=steps, goal="t", run_dir=tmp_path / "r", backend=agent, **kw)
    train(m, w, data, steps, sched=s)
    return w


def test_uniform_spacing_is_v1s_and_the_default_is_unchanged():
    assert decision_updates(64626, 100, schedule="uniform") == [646 * k for k in range(1, 101)]  # 646, 1292, ... 64600
    at = decision_updates(64626, 100, cadence=50, schedule="uniform")
    assert at[:3] == [650, 1300, 1950] and len(at) == 99  # 64600 falls in the last 50 updates: no answer could act
    # the default, front-loaded placement (values from the previous release)
    assert decision_updates(1000, 10, cadence=10) == [30, 50, 160, 270, 370, 480, 580, 690, 790, 900]
    assert len(decision_updates(3250, 100, cadence=50)) == 64
    with pytest.raises(ValueError, match="schedule must be"):
        decision_updates(100, 4, schedule="even")


STEERING = [
    "We encourage you to use this freedom actively",
    "is a good use of a decision",
    "late\n  cooldown",
    "Plan the cooldown early",
    "beta2 0.95 is common",
    "0.85 to 0.95",
    "A-B-A",
    "negative cosine",
    "0.8 * total",
    "betas=(0.9, 0.95)",
    "lr=3e-4",
]


def test_neutral_prompt_formats_and_leaves_out_the_steering(tmp_path):
    w = wrapped(tmp_path / "a", scripted(), prompt="neutral", task_facts="FACTS: 211.8M tokens, 10 epochs, batch 64.")
    system = w.worker.system
    assert "TASK FACTS" in system and "FACTS: 211.8M tokens, 10 epochs, batch 64." in system
    assert "PUBLIC DIRECTIONS, NOT REQUIRED SOLUTIONS" in system and "precision contract" in system
    assert not re.search(r"\{[a-z_]+\}", system)  # every field filled
    assert not [s for s in STEERING if s in system]
    bare = wrapped(tmp_path / "c", scripted(), prompt="neutral").worker.system
    assert "TASK FACTS" not in bare and "Not given" not in bare  # the section only for free-text task_facts


def test_neutral_example_is_a_reference_file_and_carries_state_over(tmp_path):
    w = wrapped(tmp_path, scripted(), prompt="neutral")
    assert "reference/example_updater.py shows how to write your own updater" in w.worker.system
    assert "def build(" not in w.worker.system  # the code is not in the prompt
    ns = {}
    src = (tmp_path / "r" / "workspace" / "reference" / "example_updater.py").read_text()
    exec(compile(src, "example", "exec"), ns)
    m, _ = make()
    named = list(m.named_parameters())
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3, weight_decay=0.1)
    spec = dict(total_updates=100, user_optimizers=[opt], user_schedulers=[])
    u1 = ns["build"](named, None, spec)
    g = u1.param_groups[0]
    assert g["lr"] == 3e-3 and g["weight_decay"] == 0.1  # the training script's settings
    m(torch.randn(2, 8)).sum().backward()
    u1.step(None, 1, types.SimpleNamespace(loss=None, update=1))
    u2 = ns["build"](named, u1, spec)
    a, b = u1.state_dict()["adamw"]["0.weight"]["exp_avg"], u2.state_dict()["adamw"]["0.weight"]["exp_avg"]
    assert torch.equal(a, b) and a.data_ptr() != b.data_ptr()  # carried over as a copy


@pytest.mark.parametrize("validation", [None, 25])
def test_the_default_prompt_is_the_neutral_one(tmp_path, validation):
    """Since 0.2.0 a run with no prompt argument gets the neutral prompt (and its starting analysis), with the
    validation-set option off and on. The names of earlier releases ("legacy", "default", "v1") raise."""
    kw = dict(validation=validation) if validation else {}
    implicit = wrapped(tmp_path / "a", scripted(), **kw)
    explicit = wrapped(tmp_path / "b", scripted(), prompt="neutral", **kw)
    assert implicit.worker.system == explicit.worker.system
    assert implicit.files["analysis.py"] == explicit.files["analysis.py"]
    assert not [s for s in STEERING if s in implicit.worker.system]
    assert ("about every 25 updates, at times that are not announced" in implicit.worker.system) == bool(validation)
    if not validation:
        for gone in ("legacy", "default", "v1"):
            with pytest.raises(ValueError, match="was removed"):
                wrapped(tmp_path / gone, scripted(), prompt=gone)
        with pytest.raises(ValueError, match="neither a name nor a file"):
            wrapped(tmp_path / "x", scripted(), prompt="nutral")


def test_code_is_the_only_mode(tmp_path):
    with pytest.raises(ValueError, match='mode="knobs" is not supported in this release'):
        wrapped(tmp_path / "k", scripted(), mode="knobs")
    with pytest.raises(ValueError, match='mode must be "code"'):
        wrapped(tmp_path / "x", scripted(), mode="menu")
    assert wrapped(tmp_path / "c", scripted(), mode="code").mode == "code"
