"""Bookkeeping for examples/overfit/train.py that the agent does not see.

OPAL gives the agent read-only copies of the training script and of the model's source, not of this file or of
splits.py, so the script it reads describes only the training. This module holds what is for us: the test-set scores,
the training images' wrong labels (label10) and the model's predictions on them, the early-stopping references and the
run's result. See README.md in this folder for the experiment.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


EVAL_SEED = 31337  # with --eval-seed: the evaluation times


def deferred_setting(setting, data, data_dir, seed):
    """Read only training arrays; return a test loader that must run after optimizer.close().

    A prepared cache is required: the download/fallback routines decode both splits.
    NPZ members are lazy, so opening the combined cache does not read its test arrays.
    """
    import splits

    if data == "synthetic":
        g = np.random.default_rng(seed)
        x = g.integers(0, 256, (1024, 32, 32, 3), dtype=np.uint8)
        y = g.integers(0, 10, 1024)
        n_test = 256

        def load_test():
            return g.integers(0, 256, (n_test, 32, 32, 3), dtype=np.uint8), g.integers(0, 10, n_test)

    else:
        cache = Path(data_dir) / "cifar10_uint8.npz"
        if not cache.is_file():
            raise FileNotFoundError(f"{cache}: run --prepare-data separately before --test-at-end")
        with np.load(cache) as z:
            x, y = z["x_train"], z["y_train"]
        n_test = 10000  # official CIFAR-10 size; no test member/header is read here

        def load_test():
            with np.load(cache) as z:
                return z["x_test"], z["y_test"]

    # Reuse exactly the existing split/noise code with an empty test split, then
    # remove it. Only the public split size is available during training.
    S = splits.make_setting(setting, x, y, None, ())
    del S["test_x"], S["test_y"]
    S["info"]["test"] = n_test
    return S, load_test


def eval_updates(steps_per_epoch, epochs, seed=0, per_epoch=1):
    """The updates after which train.py evaluates the validation set (and the test set, for us): in each epoch, one
    update drawn uniformly at random in each of `per_epoch` equal parts of the epoch (a seeded generator, --eval-seed),
    and the run's last update. Kept here, out of the files the agent reads, so that the training script shows no fixed
    cadence: the agent is told only that the script evaluates about every eval_interval(...) updates, at times that are
    not announced (OPAL 0.2.2, after two runs timed changes of the parameters to the evaluations). It sees each past
    measurement with its update."""
    rng = np.random.default_rng([EVAL_SEED, seed])
    per = max(1, min(per_epoch, steps_per_epoch))
    part = steps_per_epoch // per
    out = set()
    for e in range(epochs):
        for k in range(per):
            lo = e * steps_per_epoch + k * part + 1
            hi = e * steps_per_epoch + ((k + 1) * part if k < per - 1 else steps_per_epoch)
            out.add(int(rng.integers(lo, hi + 1)))
    out.add(epochs * steps_per_epoch)
    return sorted(out)


def eval_interval(steps_per_epoch, per_epoch=1):
    """The mean number of updates between evaluations, as wrap(validation=N) states it."""
    return max(1, steps_per_epoch // max(1, min(per_epoch, steps_per_epoch)))


class Recorder:
    def __init__(self, out, S, to_x, device, save_best=True, test_at_end=False):
        self.out, self.S, self.save_best = out, S, save_best
        self.test_at_end, self.to_x, self.device = test_at_end, to_x, device
        self.log_path = out / "log.jsonl"
        if not test_at_end:
            self.test_x = to_x(S["test_x"])
            self.test_y = torch.from_numpy(S["test_y"]).to(device)
        self.wrong = torch.from_numpy(np.flatnonzero(S["train_wrong"])).to(device)
        self.clean_y = torch.from_numpy(S["train_clean_y"]).to(device)
        self.evals, self.best = [], dict(dev_acc=None, dev_loss=None)
        self.best_weights = {}
        self.t0 = time.time()

    def test_scores(self, rec, predict):
        logits = predict(self.test_x)
        rec["test_loss"] = F.cross_entropy(logits, self.test_y).item()
        rec["test_acc"] = (logits.argmax(1) == self.test_y).float().mean().item()

    def score_at_end(self, load_test, predict, model):
        """Called only after the final update and opt.close(); restore final weights afterwards."""
        x, y = load_test()
        self.test_x, self.test_y = self.to_x(x), torch.from_numpy(y).to(self.device)
        final_weights = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        was_training = model.training
        try:
            if self.evals:
                self.test_scores(self.evals[-1], predict)
            for key, weights in self.best_weights.items():
                model.load_state_dict(weights)
                self.test_scores(self.best[key], predict)
        finally:
            model.load_state_dict(final_weights)
            model.train(was_training)
        self.best_weights.clear()

    def log(self, rec):
        with open(self.log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")

    def on_eval(self, rec, predict, train_x, train_y, model):
        """rec holds the script's own measurements (update, losses, validation scores); adds the test scores and, for
        label10, how the model labels the wrongly labelled training images; keeps the best validation checkpoint."""
        if not self.test_at_end:
            self.test_scores(rec, predict)
        if len(self.wrong):
            pred = predict(train_x[self.wrong]).argmax(1)
            rec["wrong_labelled_predicted_wrong_label"] = (pred == train_y[self.wrong]).float().mean().item()
            rec["wrong_labelled_predicted_true_label"] = (pred == self.clean_y[self.wrong]).float().mean().item()
        self.log(rec)
        self.evals.append(rec)
        b = self.best
        if b["dev_acc"] is None or rec["dev_acc"] > b["dev_acc"]["dev_acc"]:  # ties keep the earlier update
            b["dev_acc"] = rec
            if self.test_at_end:
                self.best_weights["dev_acc"] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if self.save_best:
                torch.save(dict(update=rec["update"], model=model.state_dict()), self.out / "best.pt.tmp")
                os.replace(self.out / "best.pt.tmp", self.out / "best.pt")
        if b["dev_loss"] is None or rec["dev_loss"] < b["dev_loss"]["dev_loss"]:
            b["dev_loss"] = rec
            if self.test_at_end:
                self.best_weights["dev_loss"] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        return rec

    def finish(self, config, setting, method, opal_run, uniform=None):
        """uniform: the fields every example's result.json has (opal/results.py), written first."""
        def pick(rec):
            keys = ("update", "epoch", "dev_loss", "dev_acc", "test_loss", "test_acc")
            return {k: rec[k] for k in keys} if rec else None

        ev = self.evals
        final = ev[-1] if ev else None
        peak = max(ev, key=lambda r: r["test_acc"]) if ev and not self.test_at_end else None
        result = dict(
            kind="result",
            setting=setting,
            method=method,
            final=pick(final),
            early_stopping_on_dev_acc=pick(self.best["dev_acc"]),  # the early-stopping reference
            early_stopping_on_dev_loss=pick(self.best["dev_loss"]),
            peak_test_acc_oracle=pick(peak),  # chosen on the test set: for reading the curves only, not a baseline
            overfit_gap_test_acc=(peak["test_acc"] - final["test_acc"]) if peak else None,
            seconds=round(time.time() - self.t0, 1),
            opal_run=opal_run,
        )
        self.log(result)
        (self.out / "result.json").write_text(json.dumps({**(uniform or {}), "config": config, **result}, indent=1))
        print("RESULT " + json.dumps({k: result[k] for k in ("final", "early_stopping_on_dev_acc")}), flush=True)
        return result
