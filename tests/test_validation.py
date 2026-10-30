"""OPAL 0.1.1-validation (wrap(validation=N), opt.report_validation): the agent sees the validation set's loss and
accuracy in obs.metrics and in each decision message; the neutral prompt says so only when the option is on, and is
byte-identical to v0.1.1's with it off."""

import json
import sys
from pathlib import Path

import pytest
import torch

import opal
from opal.agent import validation_line
from opal.misc import setup as setup_mod
from test_core import adamw, make
from test_neutral_task import system

SEEN = """
EVERY = 10
SEEN = []
def analyze(obs, memory):
    SEEN.extend(obs.metrics)
"""


def test_prompt_with_the_option_on_states_the_development_set_and_nothing_more(tmp_path):
    off = system(tmp_path / "off", epochs=2)[0].worker.system
    on = system(tmp_path / "on", epochs=2, validation=25, validation_size="5,000 images")[0].worker.system
    task = (
        " The training script evaluates the model on a validation set of 5,000 images held out from training about "
        "every 25 updates, at times that are not announced, and shows you its loss, and its accuracy when the script "
        "reports one. The performance in the goal is measured on a separate test set that you never see.\n"
    )
    assert "final" not in on.lower() and "final" not in off.lower()
    assert "each epoch" not in task and ("about 2 epochs." + task) in on
    assert "Evaluation data are not shown to you, nor are\nthe test results. The validation-set loss" in on
    assert "Evaluation data and results are not shown" not in on
    # the only differences are those two places
    assert (
        on.replace(task, "\n").replace(opal.misc.taskfacts.EVALUATION_ON, opal.misc.taskfacts.EVALUATION_OFF)
        == off
    )
    added = (task + opal.misc.taskfacts.EVALUATION_ON).lower()  # the text the option adds
    for word in ("overfit", "early stop", "regulari", "generaliz", "generalis", "checkpoint", "after the run", "should"):
        assert word not in added, word
    no_size = system(tmp_path / "ns", validation=40)[0].worker.system
    assert (
        " The training script evaluates the model on a validation set held out from training about every 40 updates, "
        "at times that are not announced," in no_size.replace("\n", " ")
    )


def test_the_prompt_states_no_exact_evaluation_cadence(tmp_path):
    """0.2.2: two overfit runs read "Every N updates, the training script evaluates ..." and timed changes of the
    parameters to the evaluations. The prompt now gives only an approximate interval and says the times are not
    announced; nothing else in it states when the script evaluates."""
    for kw in (dict(validation=78), dict(validation=351, validation_size="5,000 images", epochs=150)):
        text = system(tmp_path / str(kw["validation"]), **kw)[0].worker.system
        flat = " ".join(text.split())
        n = f"{kw['validation']:,}"
        assert f"about every {n} updates, at times that are not announced" in flat
        assert f"Every {n} updates" not in flat and f"every {n} updates" not in flat.replace(f"about every {n}", "")
        assert flat.count(n) == 1  # the interval appears once, in that sentence
        for word in ("gaming", "probe", "integrity", "check whether", "detect"):
            assert word not in flat.lower(), word


def test_validation_must_be_a_positive_int(tmp_path):
    for bad in (True, 0, 2.5, "10"):
        with pytest.raises(ValueError, match="validation"):
            system(tmp_path / str(bad), validation=bad)


def run_dev(tmp_path, validation=10, steps=200):
    from test_review1 import scripted, wrapped

    agent = scripted({"analysis.py": SEEN})
    kw = dict(prompt="neutral", **({"validation": validation} if validation else {}))
    m, data, s, w = wrapped(tmp_path, agent, steps=steps, **kw)
    dev = []
    for i in range(steps):
        x, y = data[i % len(data)]
        loss = torch.nn.functional.mse_loss(m(x), y)
        w.zero_grad()
        loss.backward()
        w.step(loss=loss)
        s.step()
        if (i + 1) % 10 == 0:
            d = dict(loss=1.0 + (i + 1) / 1000, accuracy=0.5 + (i + 1) / 10000)
            dev.append(dict(update=i + 1, **d))
            w.report_validation(loss=torch.tensor(d["loss"]), accuracy=d["accuracy"])
            w.log(test_acc=0.987654, test_loss=3.21098)  # record-only: never shown
    run = tmp_path / "r"
    messages = [p.read_text() for p in sorted((run / "decisions").glob("*/message.txt"))]
    inbox = [json.loads(p.read_text()) for p in sorted((run / "workspace" / "inbox").glob("*.json"))]
    seen = sys.modules[w.analysis.__name__].SEEN if w.analysis is not None and hasattr(w.analysis, "SEEN") else []
    records = [json.loads(x) for x in open(run / "metrics.jsonl")]
    return w, dev, messages, inbox, seen, records


def test_dev_measurements_reach_obs_metrics_and_the_message_with_their_history(tmp_path):
    w, dev, messages, inbox, seen, records = run_dev(tmp_path)
    assert len(messages) >= 3
    lines = [ln for msg in messages for ln in msg.splitlines() if ln.startswith("Validation set (held out")]
    assert len(lines) == len(messages)  # one line per message
    last = lines[-1]
    assert last.startswith(
        "Validation set (held out from training; not the test set in the goal), loss and accuracy at each evaluation, all "
    )
    assert "(update: loss, accuracy): 10: 1.01, 0.501; 20: 1.02, 0.502;" in last  # the history, not only the latest
    k = len(messages)
    n_last = int(last.split("all ")[1].split(" measurements")[0])
    assert n_last == len(inbox[-1]["validation"]) and inbox[-1]["validation"][0] == pytest.approx(dev[0])
    assert any(r.get("val_loss") == pytest.approx(1.01) and r["update"] == 10 for r in seen)
    assert all(set(r) <= {"update", "val_loss", "val_accuracy"} for r in seen)
    # programs written before 0.3.10 read dev_loss / dev_accuracy, and still find them; the agent sees only val_
    assert any(r.get("dev_loss") == pytest.approx(1.01) and r["dev_accuracy"] == pytest.approx(0.501) for r in seen)
    assert all("dev_loss" in r for r in seen if "val_loss" in r)
    assert "dev_" not in json.dumps(seen) and not any("dev_" in m or "Development" in m for m in messages)
    assert sum("val_loss" in r for r in records) == len(dev)
    for text in messages + [json.dumps(x) for x in inbox] + [json.dumps(seen)]:
        assert "test_acc" not in text and "test_loss" not in text and "0.987654" not in text and "3.21098" not in text
    assert "Values the training script chose to show you" not in messages[-1]  # dev has its own line only
    assert k == w.worker.k


def test_report_validation_without_the_option_is_recorded_only(tmp_path):
    w, dev, messages, inbox, seen, records = run_dev(tmp_path, validation=None)
    assert messages and not any("Validation set" in m or "val_loss" in m for m in messages)
    assert not seen and all("validation" not in x for x in inbox)
    assert sum("val_loss" in r for r in records) == len(dev)


def test_validation_line_shows_the_most_recent_fifty():
    dev = [dict(update=10 * i, loss=1.0, accuracy=0.5) for i in range(1, 61)]
    line = validation_line(7, dev)
    assert "the most recent 50 of 60 measurements (all in inbox/decision-007.json)" in line
    assert line.endswith("600: 1, 0.5") and "(update: loss, accuracy): 110: 1, 0.5; 120: 1, 0.5;" in line and " 100: " not in line
    assert validation_line(1, []).endswith(": no measurement yet.")
    assert validation_line(1, [dict(update=5, loss=2.0)]).endswith(
        "loss at each evaluation, all 1 measurements so far (update: loss): 5: 2"
    )


def test_dev_history_survives_a_checkpoint(tmp_path):
    w, dev, *_ = run_dev(tmp_path, steps=105)
    state = w.state_dict()
    assert state["dev"] == [dict(d) for d in w._dev] and len(state["dev"]) == 10
    m, _ = make()
    o, s = adamw(m, 200)
    w2 = opal.wrap(
        o,
        m,
        scheduler=s,
        total_steps=200,
        goal="t",
        run_dir=tmp_path / "r2",
        backend="keep",
        validation=10,
        verbose=False,
    )
    w2.load_state_dict(state)
    assert w2._dev == state["dev"]


def test_regular_evaluation_times_warn_the_person_running_training(tmp_path, capfd):
    """0.2.2: the agent is told the times are not announced and sees each measurement's update, so a script that
    evaluates at a fixed interval lets it predict the next one; the person running the training is told (stderr), and
    the agent is not."""
    for name, gaps in (("regular", [10, 10, 10, 10]), ("irregular", [7, 13, 9, 11])):
        m, data = make()
        o, s = adamw(m, 60)
        w = opal.wrap(
            o, m, scheduler=s, total_steps=60, goal="t", run_dir=tmp_path / name, backend="keep", validation=10
        )
        at = 0
        evals = {at := at + g for g in gaps}
        for i in range(1, 61):
            x, y = data[i % len(data)]
            loss = torch.nn.functional.mse_loss(m(x), y)
            w.zero_grad()
            loss.backward()
            w.step(loss=loss)
            s.step()
            if i in evals:
                w.report_dev(loss=1.0)
        err = capfd.readouterr().err
        assert ("the last three evaluations were 10 updates apart" in err) == (name == "regular"), err
