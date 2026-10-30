#!/usr/bin/env python3
"""ResNet-18 on CIFAR-10 with AdamW and a warmup-cosine schedule; optionally wrapped by OPAL.

    python train.py --prepare-data         # download CIFAR-10 once
    python train.py                        # plain AdamW
    python train.py --opal --validation # with the agent, which is shown the validation set's scores
"""

from __future__ import annotations

import argparse
import json
import math
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import records  # noqa: E402
import resnet  # noqa: E402
import splits  # noqa: E402

MEAN, STD = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--setting", choices=splits.SETTINGS, default=splits.SETTINGS[0])
    p.add_argument("--optimizer", choices=["adamw"], default="adamw", help="the training script's optimizer")
    p.add_argument("--lr", type=float, default=1e-3, help="AdamW peak learning rate")
    p.add_argument("--wd", type=float, default=0.01, help="AdamW weight decay (decoupled), on every tensor")
    p.add_argument("--schedule", choices=["cosine", "constant"], default="cosine")
    p.add_argument("--warmup-epochs", type=float, default=1.0)
    p.add_argument("--epochs", type=int, default=None, help="default: set per data setting")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--width", type=int, default=64, help="channels of the first stage (64 = ResNet-18)")
    p.add_argument("--augment", action="store_true", help="random flips and 4-pixel reflect-pad crops (off by default)")
    p.add_argument("--evals-per-epoch", type=int, default=1, help="validation evaluations per epoch (see records.py)")
    p.add_argument("--eval-seed", type=int, default=0, help="the updates at which the validation set is evaluated")
    p.add_argument("--test-at-end", action="store_true", help="load and score the test set only after training and OPAL close")
    p.add_argument("--seed", type=int, default=0, help="model initialisation")
    p.add_argument("--data-seed", type=int, default=0, help="data order (and augmentation)")
    p.add_argument("--data", choices=["cifar10", "synthetic"], default="cifar10")
    p.add_argument("--data-dir", default="./data/cifar10")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--deterministic", action="store_true", help="deterministic kernels (slower)")
    p.add_argument("--out", default=None, help="output folder (default runs/cifar/run-<time>)")
    p.add_argument("--no-save-best", dest="save_best", action="store_false", help="do not save best.pt")
    p.add_argument("--prepare-data", action="store_true", help="download CIFAR-10, then exit")
    # OPAL
    p.add_argument("--opal", action="store_true", help="wrap the AdamW and its schedule with opal.wrap")
    p.add_argument("--validation", action="store_true", help="OPAL 0.1.1-validation: show the agent the validation set")
    p.add_argument("--backend", default="claude", help="claude, keep, or mailbox:<dir>")
    p.add_argument("--model", default=None, help="agent model (default: claude-opus-5-5, at effort high)")
    p.add_argument("--effort", default=None, help="agent effort (default: high)")
    p.add_argument("--save", default=None, help="write the final weights (model.pt) into this folder")
    p.add_argument("--decisions", type=int, default=30)
    p.add_argument("--decision-schedule", choices=["uniform", "front"], default="uniform")  # when the agent is asked
    p.add_argument("--every", type=int, default=50, help="OPAL's check interval, in updates")
    p.add_argument("--blocking", choices=["true", "false"], default="true")
    p.add_argument("--prompt", default="neutral")
    p.add_argument("--no-share-code", dest="share_code", action="store_false")
    p.add_argument("--max-cost-usd", type=float, default=None, help="stop asking the agent after this much spend")
    p.add_argument("--goal", default=None, help="replace the goal sentence")
    a = p.parse_args(argv)
    a.epochs = a.epochs or splits.EPOCHS[a.setting]
    if a.validation and not a.opal:
        p.error("--validation needs --opal")
    if a.test_at_end and a.prepare_data:
        p.error("prepare the data separately, before using --test-at-end")
    return a


def git_commit():
    try:
        out = subprocess.run(["git", "-C", str(HERE), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10)
        dirty = subprocess.run(
            ["git", "-C", str(HERE), "status", "--porcelain"], capture_output=True, text=True, timeout=10
        )
        return out.stdout.strip() + ("+local changes" if dirty.stdout.strip() else "")
    except Exception:  # noqa: BLE001
        return None


def lr_factor_fn(total, warmup, schedule):
    def lr_factor(step):  # step = updates applied so far
        if step < warmup:
            return (step + 1) / warmup
        if schedule == "constant":
            return 1.0
        return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))

    return lr_factor


def main(argv=None, backend=None):
    """Returns (result dict, the optimizer). backend overrides --backend (a callable, for tests)."""
    a = parse(argv)
    if a.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = not a.deterministic
    device = torch.device(a.device)
    cuda = device.type == "cuda"

    if a.test_at_end:
        S, load_test = records.deferred_setting(a.setting, a.data, a.data_dir, a.data_seed)
    else:
        raw = splits.synthetic(seed=a.data_seed) if a.data == "synthetic" else splits.load_cifar10(a.data_dir)
        if a.prepare_data:
            print(f"data ready: {len(raw[0])} training and {len(raw[2])} test images", flush=True)
            return None, None
        S = splits.make_setting(a.setting, *raw)
    method = ("opal-dev" if a.validation else "opal") if a.opal else f"adamw-lr{a.lr:g}"
    out = Path(a.out or f"runs/cifar/run-{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}")
    out.mkdir(parents=True, exist_ok=True)
    if (out / "result.json").exists():
        raise SystemExit(f"{out} already holds a finished run")
    mean = torch.tensor(MEAN, device=device)[:, None, None]
    std = torch.tensor(STD, device=device)[:, None, None]
    fmt = torch.channels_last if cuda else torch.contiguous_format

    def to_x(x):
        x = torch.from_numpy(x).to(device).permute(0, 3, 1, 2).float().div_(255)
        return ((x - mean) / std).contiguous(memory_format=fmt)

    train_x, dev_x = to_x(S["train_x"]), to_x(S["dev_x"])
    train_y, dev_y = (torch.from_numpy(S[k]).to(device) for k in ("train_y", "dev_y"))
    rec_ = records.Recorder(out, S, to_x, device, save_best=a.save_best, test_at_end=a.test_at_end)
    n_train, B = len(train_y), a.batch_size
    steps_per_epoch = n_train // B  # the last partial batch of each epoch is dropped
    total = a.epochs * steps_per_epoch
    eval_at = records.eval_updates(steps_per_epoch, a.epochs, a.eval_seed, a.evals_per_epoch)
    order_gen = torch.Generator().manual_seed(a.data_seed)
    aug_gen = torch.Generator().manual_seed(10_000 + a.data_seed)

    torch.manual_seed(a.seed)
    model = resnet.ResNet18(a.width).to(device).to(memory_format=fmt)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.999), weight_decay=a.wd)
    warmup = max(1, round(a.warmup_epochs * steps_per_epoch))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor_fn(total, warmup, a.schedule))
    goal = a.goal or (
        f"ResNet-18 ({n_params / 1e6:.1f}M parameters) trained from scratch on {n_train:,} CIFAR-10 training images for "
        f"{a.epochs} epochs; judged by accuracy on the {S['info']['test']:,} CIFAR-10 test images after the last update."
    )
    if a.opal:
        import opal

        opt = opal.wrap(
            opt,
            model,
            scheduler=sched,
            total_steps=total,
            goal=goal,
            backend=backend if backend is not None else a.backend,
            backend_options={k: v for k, v in dict(model=a.model, effort=a.effort).items() if v} or None,
            decisions=a.decisions,
            schedule=a.decision_schedule,
            every=a.every,
            blocking={"true": True, "false": False}.get(a.blocking, a.blocking),
            prompt=a.prompt,
            share_code=a.share_code,
            max_cost_usd=a.max_cost_usd,
            epochs=a.epochs,
            train_size=f"{n_train:,} images",
            batch_size=f"{B} images",
            same_order_each_epoch=False,
            validation=records.eval_interval(steps_per_epoch, a.evals_per_epoch) if a.validation else None,
            validation_size=f"{len(dev_y):,} images",
            run_dir=out / "opal",
        )

    def autocast():
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=cuda)

    @torch.no_grad()
    def predict(x):
        model.eval()
        out_ = []
        for i in range(0, len(x), 1000):
            with autocast():
                out_.append(model(x[i : i + 1000]).float())
        model.train()
        return torch.cat(out_) if out_ else torch.zeros(0, 10, device=device)

    def scores(x, y):
        logits = predict(x)
        return F.cross_entropy(logits, y).item(), (logits.argmax(1) == y).float().mean().item()

    def augment(x):
        n = len(x)
        flip = (torch.rand(n, generator=aug_gen) < 0.5).to(device)
        oy, ox = (torch.randint(0, 9, (n,), generator=aug_gen).to(device) for _ in range(2))
        x = torch.where(flip[:, None, None, None], x.flip(-1), x)
        x = F.pad(x, (4, 4, 4, 4), mode="reflect")
        ar = torch.arange(32, device=device)
        rows, cols = oy[:, None] + ar, ox[:, None] + ar
        x = x[torch.arange(n, device=device)[:, None, None], :, rows[:, :, None], cols[:, None, :]]
        return x.permute(0, 3, 1, 2).contiguous(memory_format=fmt)

    config = dict(
        kind="config",
        args=vars(a),
        argv=sys.argv if argv is None else list(argv),
        split=S["info"],
        n_params=n_params,
        total_updates=total,
        steps_per_epoch=steps_per_epoch,
        eval_updates=eval_at,
        warmup_updates=warmup,
        goal=goal if a.opal else None,
        opal_run=str(out / "opal") if a.opal else None,
        git=git_commit(),
        torch=torch.__version__,
        device=torch.cuda.get_device_name(device) if cuda else str(device),
        host=socket.gethostname(),
    )
    rec_.log(config)
    print(
        f"{n_train:,} train / {len(dev_y):,} dev images, {n_params / 1e6:.2f}M "
        f"params, {a.epochs} epochs x {steps_per_epoch} updates of batch {B}, out {out}",
        flush=True,
    )

    loss_sum = torch.zeros((), device=device)
    correct_sum = torch.zeros((), device=device)
    seen, u, t0, eval_at = 0, 0, time.time(), set(eval_at)
    for epoch in range(a.epochs):
        perm = torch.randperm(n_train, generator=order_gen).to(device)
        for i in range(steps_per_epoch):
            idx = perm[i * B : (i + 1) * B]
            x, y = train_x[idx], train_y[idx]
            if a.augment:
                x = augment(x)
            with autocast():
                logits = model(x)
            loss = F.cross_entropy(logits.float(), y)
            loss.backward()
            if a.opal:
                opt.step(loss=loss)
            else:
                opt.step()
            opt.zero_grad(set_to_none=True)
            sched.step()
            u += 1
            loss_sum += loss.detach() * len(y)
            correct_sum += (logits.detach().argmax(1) == y).sum()
            seen += len(y)
            if u not in eval_at:
                continue
            dev_loss, dev_acc = scores(dev_x, dev_y)
            if a.validation:
                opt.report_dev(loss=dev_loss, accuracy=dev_acc, update=u)  # after the last update: recorded only
            rec = dict(
                kind="eval",
                update=u,
                epoch=round(u / steps_per_epoch, 4),
                lr=float(opt.param_groups[0]["lr"]),
                train_loss=loss_sum.item() / seen,
                train_acc=correct_sum.item() / seen,
                dev_loss=dev_loss,
                dev_acc=dev_acc,
                seconds=round(time.time() - t0, 1),
            )
            loss_sum.zero_()
            correct_sum.zero_()
            seen = 0
            rec = rec_.on_eval(rec, predict, train_x, train_y, model)  # adds the test scores, for us only
            test_scores = (
                "" if a.test_at_end else f"  test {rec['test_loss']:.4f}/{rec['test_acc']:.4f}"
            )
            print(
                f"update {u:6d} epoch {rec['epoch']:7.2f}  train {rec['train_loss']:.4f}/{rec['train_acc']:.4f}  "
                f"dev {dev_loss:.4f}/{dev_acc:.4f}{test_scores}  {rec['seconds']:.0f}s",
                flush=True,
            )
    if a.opal:
        opt.close()
    if a.test_at_end:
        rec_.score_at_end(load_test, predict, model)

    from opal.results import save_weights, summary

    if a.save:
        save_weights(a.save, model=model.state_dict())
    final = rec_.evals[-1]["test_acc"] if rec_.evals else None
    seconds = time.time() - rec_.t0
    uniform = summary("test_accuracy", final, "higher", seconds, opt if a.opal else None, device)
    result = rec_.finish(config, a.setting, method, str(out / "opal") if a.opal else None, uniform)
    return result, opt


if __name__ == "__main__":
    main()
