"""CPU-only regression checks for delayed test loading; no model/API backend is called."""

import hashlib
import importlib.util
import json
import os
import pickle
import random
import struct
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(os.environ.get("OFIT_PROBE_ROOT", Path(__file__).resolve().parents[1]))


def harness(root=ROOT):
    folder = root / "examples" / "overfit"
    sys.path.insert(0, str(folder))
    for name in ("records", "resnet", "splits"):
        sys.modules.pop(name, None)
    spec = importlib.util.spec_from_file_location("overfit_train", folder / "train.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def tensor_hash(items):
    h = hashlib.sha256()
    for name, value in items:
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def rng_state():
    return (torch.get_rng_state().numpy().tobytes(), pickle.dumps(np.random.get_state()), pickle.dumps(random.getstate()))


def run_probe(mod, out, opal, deferred, augment, monkeypatch, data_args=None):
    losses, models, accesses, member_opens = [], [], [], []
    state = dict(updates=0, closed=False, evals=0)
    argv = ["--data", "synthetic", "--device", "cpu", "--width", "2", "--epochs", "3",
            "--batch-size", "64", "--evals-per-epoch", "2", "--lr", "3e-3", "--deterministic",
            "--out", str(out)]
    if data_args:
        argv += data_args
    if augment:
        argv += ["--augment"]
    if deferred:
        argv += ["--test-at-end"]
    if opal:
        argv += ["--opal", "--validation", "--backend", "keep", "--decisions", "4", "--every", "2"]
    original_model = mod.resnet.ResNet18
    original_backward = torch.Tensor.backward
    original_eval = mod.records.Recorder.on_eval
    original_score = mod.records.Recorder.test_scores if hasattr(mod.records.Recorder, "test_scores") else None
    original_getitem = np.lib.npyio.NpzFile.__getitem__
    original_open = zipfile.ZipFile.open

    def constructor(*a, **kw):
        model = original_model(*a, **kw)
        models.append(model)
        return model

    def backward(loss, *a, **kw):
        losses.append(struct.pack("!d", loss.detach().item()))
        return original_backward(loss, *a, **kw)

    def on_eval(recorder, rec, predict, *a):
        state["updates"] = rec["update"]
        state["evals"] += 1
        before = rng_state(), tensor_hash(models[0].state_dict().items())
        result = original_eval(recorder, rec, predict, *a)
        assert before == (rng_state(), tensor_hash(models[0].state_dict().items()))
        if deferred:
            assert not hasattr(recorder, "test_x") and not hasattr(recorder, "test_y")
            assert "test_x" not in recorder.S and "test_y" not in recorder.S
            assert not any(k.startswith("test_") for k in rec)
        return result

    def test_scores(recorder, rec, predict):
        before = rng_state(), tensor_hash(models[0].state_dict().items())
        modes = [m.training for m in models[0].modules()]
        result = original_score(recorder, rec, predict)
        assert before == (rng_state(), tensor_hash(models[0].state_dict().items()))
        assert modes == [m.training for m in models[0].modules()]
        return result

    def check_end():
        if deferred:
            assert state["updates"] == len(losses) > 0
            if opal:
                assert state["closed"]

    def getitem(z, key):
        if key in ("x_test", "y_test"):
            check_end()
        accesses.append((key, len(losses), state["closed"]))
        return original_getitem(z, key)

    def member_open(z, name, *a, **kw):
        if isinstance(name, str) and name in ("x_test.npy", "y_test.npy"):
            check_end()
            member_opens.append((name, len(losses), state["closed"]))
        return original_open(z, name, *a, **kw)

    with monkeypatch.context() as m:
        m.setattr(mod.resnet, "ResNet18", constructor)
        m.setattr(torch.Tensor, "backward", backward)
        m.setattr(mod.records.Recorder, "on_eval", on_eval)
        if original_score:
            m.setattr(mod.records.Recorder, "test_scores", test_scores)
        m.setattr(np.lib.npyio.NpzFile, "__getitem__", getitem)
        m.setattr(zipfile.ZipFile, "open", member_open)
        if opal:
            import opal.wrapper
            # Intercept the concrete wrapper returned by wrap, without changing its backend.
            import opal as package
            original_wrap = package.wrap

            def wrap(*a, **kw):
                opt = original_wrap(*a, **kw)
                close = opt.close

                def closed(*a, **kw):
                    result = close(*a, **kw)
                    state["closed"] = True
                    return result

                opt.close = closed
                return opt

            m.setattr(package, "wrap", wrap)
        result, opt = mod.main(argv)
    model = models[0]
    logs = [json.loads(line) for line in (out / "log.jsonl").read_text().splitlines()]
    evals = [{k: v for k, v in rec.items() if k not in ("test_loss", "test_acc", "seconds")}
             for rec in logs if rec["kind"] == "eval"]
    observations = []
    if opal:
        for p in sorted((out / "opal/workspace/inbox").glob("*.json")):
            rec = json.loads(p.read_text())
            # Timing is also agent-visible, and cannot be identical between runs.
            # Compare all numerical training/validation observations, excluding only clocks.
            for report in rec.get("reports", []):
                report["report"].pop("program_ms_per_update", None)
            observations.append({k: rec.get(k) for k in
                                 ("update", "losses", "losses_from", "reports", "metrics", "development")})
        assert observations
    return dict(loss_bytes=b"".join(losses).hex(), loss_sha256=hashlib.sha256(b"".join(losses)).hexdigest(),
                updates=len(losses), parameter_sha256=tensor_hash(model.named_parameters()),
                state_sha256=tensor_hash(model.state_dict().items()), evals=evals,
                observations=observations, result=result, accesses=accesses, member_opens=member_opens,
                rng=[hashlib.sha256(s).hexdigest() for s in rng_state()])


@pytest.mark.parametrize("opal", [False, True], ids=["plain", "keep"])
@pytest.mark.parametrize("augment", [False, True], ids=["noaug", "aug"])
def test_training_is_bitwise_unchanged(tmp_path, monkeypatch, opal, augment):
    off = run_probe(harness(), tmp_path / "off", opal, False, augment, monkeypatch)
    on = run_probe(harness(), tmp_path / "on", opal, True, augment, monkeypatch)
    for key in ("loss_bytes", "updates", "parameter_sha256", "state_sha256", "evals", "observations"):
        assert off[key] == on[key], key
    # Python and NumPy global RNG are not reset by main; compare torch, which is.
    assert off["rng"][0] == on["rng"][0]
    for key in ("final", "early_stopping_on_dev_acc", "early_stopping_on_dev_loss"):
        assert off["result"][key] == on["result"][key]
    assert on["result"]["peak_test_acc_oracle"] is None
    assert on["result"]["overfit_gap_test_acc"] is None
    evidence = os.environ.get("OFIT_EVIDENCE_DIR")
    if evidence:
        p = Path(evidence)
        p.mkdir(parents=True, exist_ok=True)
        (p / f"bitwise-{opal}-{augment}.json").write_text(json.dumps(dict(off=off, on=on), indent=2))


def test_npz_test_members_open_only_after_last_update_and_close(tmp_path, monkeypatch):
    mod = harness()
    data = tmp_path / "data"
    data.mkdir()
    raw = mod.splits.synthetic(n_train=256, n_test=24)
    np.savez(data / "cifar10_uint8.npz", **dict(zip(("x_train", "y_train", "x_test", "y_test"), raw)))
    result = run_probe(mod, tmp_path / "run", True, True, False, monkeypatch,
                       ["--data", "cifar10", "--data-dir", str(data)])
    assert [r[0] for r in result["accesses"]] == ["x_train", "y_train", "x_test", "y_test"]
    assert result["member_opens"]
    assert all(r[1:] == (result["updates"], True) for r in result["member_opens"])


def test_missing_cache_never_downloads_or_decodes(tmp_path, monkeypatch):
    mod = harness()

    def forbidden(*a):
        pytest.fail("download/decoder called")

    monkeypatch.setattr(mod.splits, "_from_hf", forbidden)
    monkeypatch.setattr(mod.splits, "_from_toronto", forbidden)
    with pytest.raises(FileNotFoundError, match="prepare-data separately"):
        mod.records.deferred_setting("label10", "cifar10", tmp_path, 0)
    with pytest.raises(SystemExit):
        mod.parse(["--test-at-end", "--prepare-data"])


@pytest.mark.parametrize("save_best", [False, True])
def test_both_dev_selected_models_scored_and_final_restored(tmp_path, save_best):
    mod = harness()
    S = dict(train_wrong=np.zeros(2, bool), train_clean_y=np.zeros(2, np.int64))
    rec = mod.records.Recorder(tmp_path, S, lambda x: x, "cpu", save_best=save_best, test_at_end=True)
    model = torch.nn.Linear(1, 2, bias=False)
    x, y = torch.ones(2, 1), torch.zeros(2, dtype=torch.long)

    def predict(x):
        return model(x).detach()

    expected = {}
    for u, acc, loss in ((1, .8, 2.), (2, .7, 1.), (3, .6, 3.)):
        with torch.no_grad():
            model.weight.copy_(torch.tensor([[u], [-u]]))
        expected[u] = torch.nn.functional.cross_entropy(predict(x), y).item()
        rec.on_eval(dict(update=u, epoch=u, dev_acc=acc, dev_loss=loss), predict, x, y, model)
    before = tensor_hash(model.state_dict().items()), rng_state()
    rec.score_at_end(lambda: (x, y.numpy()), predict, model)
    assert before == (tensor_hash(model.state_dict().items()), rng_state())
    result = rec.finish({}, "label10", "test", None)
    for name, u in (("final", 3), ("early_stopping_on_dev_acc", 1), ("early_stopping_on_dev_loss", 2)):
        assert result[name]["update"] == u
        assert result[name]["test_loss"] == expected[u]
    assert (tmp_path / "best.pt").exists() == save_best
