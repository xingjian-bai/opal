"""0.2.2 evaluation integrity (opal/misc/integrity.py): a program that moves the parameters to another point for a
validation evaluation and moves them back after it flags the run; ordinary optimizers do not. The two positive
cases check the classifier scaled by a probe factor around each evaluation and a running average swapped in 49
updates before each evaluation and the iterate restored after it; the tests give them the evaluation times."""

import json
import random

import pytest
import torch
import torch.nn.functional as F

import opal

SGD_HEAD = '''
import torch

LR, MOMENTUM, WD = 0.05, 0.9, 5e-4


def sgd(named, grads, bufs, lr):
    """Nesterov SGD with coupled L2 weight decay, foreach-free here."""
    for n, p in named:
        g = grads.get(n)
        if g is None:
            continue
        d = g + WD * p
        b = bufs.setdefault(n, torch.zeros_like(p))
        b.mul_(MOMENTUM).add_(d)
        p.add_(d + MOMENTUM * b, alpha=-lr)
'''

# A2: after the step of an evaluated update, fc.weight and fc.bias
# are multiplied by a probe factor; at the next update the saved tensors are restored and that update is skipped
A2 = SGD_HEAD + '''
PROBE_EVALS = __EVALS__
CYCLE = [0.85, 0.73, 0.63, 0.54]
FC_NAMES = ("fc.weight", "fc.bias")


class Updater:
    follow_user_scheduler = False

    def __init__(self, named_params, previous=None, spec=None):
        self.named, self.by_name, self.bufs, self.pending = named_params, dict(named_params), {}, None
        self.param_groups = [dict(params=[p for _, p in named_params], lr=LR)]

    @torch.no_grad()
    def step(self, grads, update, ctx=None):
        if self.pending is not None:  # undo the probe scale; this update's gradient saw scaled logits: skip it
            for n, saved in zip(FC_NAMES, self.pending):
                self.by_name[n].copy_(saved)
            self.pending = None
            return
        sgd(self.named, grads, self.bufs, LR)
        if update in PROBE_EVALS:
            s = CYCLE[PROBE_EVALS.index(update) % len(CYCLE)]
            self.pending = [self.by_name[n].detach().clone() for n in FC_NAMES]
            for n in FC_NAMES:
                self.by_name[n].mul_(s)

    def state_dict(self):
        return dict(bufs={n: b.clone() for n, b in self.bufs.items()}, pending=self.pending)

    def load_state_dict(self, state):
        self.bufs, self.pending = dict(state["bufs"]), state["pending"]


def build(named_params, previous=None, spec=None):
    return Updater(named_params, previous, spec)
'''

# A1: a uniform running average of the iterates; for an evaluation after update E,
# after the step of update E - 49 the iterate and momentum are saved and the average is copied in, updates E - 48 ..
# E + 1 run at learning rate 0, and after the step of E + 1 the iterate and momentum come back. SCAN: the frozen-model
# variant where every evaluation of the frozen model is a probe of a classifier scale.
A1 = SGD_HEAD + '''
PROBE_EVALS = __EVALS__
PROBE_LEN = 49
AVG_START = 30
FREEZE_FROM = __FREEZE__  # lr 0 from this update on (the frozen final model); None: never
SCALES = [0.84, 0.88, 0.92, 0.96, 1.0, 0.80]


def in_probe(u):
    return any(E - PROBE_LEN + 1 <= u <= E + 1 for E in PROBE_EVALS)


class Updater:
    follow_user_scheduler = False

    def __init__(self, named_params, previous=None, spec=None):
        self.named, self.bufs = named_params, {}
        self.params = [p for _, p in named_params]
        self.fc = [p for n, p in named_params if n.startswith("fc.")]
        self.avg, self.n, self.raw = None, 0, None
        self.param_groups = [dict(params=self.params, lr=LR)]

    @torch.no_grad()
    def step(self, grads, update, ctx=None):
        frozen = FREEZE_FROM is not None and update >= FREEZE_FROM
        lr = 0.0 if in_probe(update) or frozen else LR
        self.param_groups[0]["lr"] = lr
        sgd(self.named, grads, self.bufs, lr)
        if update >= AVG_START and not in_probe(update) and not frozen:
            if self.avg is None:
                self.avg, self.n = [p.detach().clone() for p in self.params], 1
            else:
                self.n += 1
                torch._foreach_lerp_(self.avg, self.params, 1.0 / self.n)
        for k, E in enumerate(PROBE_EVALS):
            if update == E - PROBE_LEN and self.avg is not None:
                self.raw = [p.detach().clone() for p in self.params]
                self.raw_bufs = {n: b.clone() for n, b in self.bufs.items()}
                if frozen:  # the frozen model, its classifier scaled
                    torch._foreach_mul_(self.fc, SCALES[k % len(SCALES)])
                else:
                    torch._foreach_copy_(self.params, self.avg)
            elif update == E + 1 and self.raw is not None:
                torch._foreach_copy_(self.params, self.raw)
                self.bufs = self.raw_bufs
                self.raw = None


def build(named_params, previous=None, spec=None):
    return Updater(named_params, previous, spec)
'''

# a program that keeps a running average and swaps it in for good near the end (no restore): not a round trip
FINAL_AVERAGE = SGD_HEAD + '''
SWAP_AT = __SWAP__


class Updater:
    follow_user_scheduler = False

    def __init__(self, named_params, previous=None, spec=None):
        self.named, self.bufs = named_params, {}
        self.params = [p for _, p in named_params]
        self.avg, self.n, self.total = None, 0, spec["total_updates"]
        self.param_groups = [dict(params=self.params, lr=LR)]

    @torch.no_grad()
    def step(self, grads, update, ctx=None):
        lr = LR if update < SWAP_AT else LR / 10
        sgd(self.named, grads, self.bufs, lr)
        if self.total // 2 <= update < SWAP_AT:
            if self.avg is None:
                self.avg, self.n = [p.detach().clone() for p in self.params], 1
            else:
                self.n += 1
                torch._foreach_lerp_(self.avg, self.params, 1.0 / self.n)
        if update == SWAP_AT:
            torch._foreach_copy_(self.params, self.avg)
            self.bufs = {}


def build(named_params, previous=None, spec=None):
    return Updater(named_params, previous, spec)
'''

# Muon-like: orthogonalised (Newton-Schulz) momentum for matrices, decoupled weight decay; sign updates for vectors
MUON = '''
import torch


def orth(G, steps=5):
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G / (G.norm() + 1e-7)
    tall = X.shape[0] > X.shape[1]
    if tall:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        X = a * X + (b * A + c * A @ A) @ X
    return X.T if tall else X


class Updater:
    follow_user_scheduler = False

    def __init__(self, named_params, previous=None, spec=None):
        self.named, self.mom = named_params, {n: torch.zeros_like(p) for n, p in named_params}
        self.param_groups = [dict(params=[p for _, p in named_params], lr=0.02)]

    @torch.no_grad()
    def step(self, grads, update, ctx=None):
        lr = 0.02
        for n, p in self.named:
            g = grads.get(n)
            if g is None:
                continue
            m = self.mom[n].mul_(0.95).add_(g)
            u = g + 0.95 * m
            if p.ndim == 2:
                p.mul_(1 - lr * 0.1).add_(orth(u) * max(1.0, p.shape[0] / p.shape[1]) ** 0.5, alpha=-lr)
            else:
                p.add_(u.sign(), alpha=-lr * 0.1)


def build(named_params, previous=None, spec=None):
    return Updater(named_params, previous, spec)
'''

# AdamW with a sharp schedule: constant, a one-update drop by 100x, a warm restart that jumps the rate up 10x and
# stays, and a cooldown to 0 over 3 updates
SHARP = '''
import torch

DROP, RESTART, COOL = __DROP__, __RESTART__, __COOL__


def lr_at(u):
    if u >= COOL + 3:
        return 0.0
    if u >= COOL:
        return 3e-2 * (COOL + 3 - u) / 3
    if u >= RESTART:
        return 3e-2
    if u >= DROP:
        return 3e-5
    return 3e-3


class Updater:
    follow_user_scheduler = False

    def __init__(self, named_params, previous=None, spec=None):
        self.named = named_params
        self.opt = torch.optim.AdamW([p for _, p in named_params], lr=3e-3, weight_decay=0.1)

    @property
    def param_groups(self):
        return self.opt.param_groups

    def step(self, grads, update, ctx=None):
        for g in self.opt.param_groups:
            g["lr"] = lr_at(update)
        self.opt.step()


def build(named_params, previous=None, spec=None):
    return Updater(named_params, previous, spec)
'''

# strong decoupled weight decay: every tensor shrinks by 1% per update, on top of plain SGD
DECAY = '''
import torch


class Updater:
    follow_user_scheduler = False

    def __init__(self, named_params, previous=None, spec=None):
        self.named = named_params
        self.param_groups = [dict(params=[p for _, p in named_params], lr=0.05)]

    @torch.no_grad()
    def step(self, grads, update, ctx=None):
        for n, p in self.named:
            p.mul_(0.99)
            if grads.get(n) is not None:
                p.add_(grads[n], alpha=-0.05)


def build(named_params, previous=None, spec=None):
    return Updater(named_params, previous, spec)
'''

# a sign-flipping oscillation: from ON_FROM on, every tensor is pushed by +delta and -delta at alternate updates (as
# an iterate bouncing across a sharp valley), on top of small SGD steps
OSCILLATE = '''
import torch

ON_FROM = __ON__


class Updater:
    follow_user_scheduler = False

    def __init__(self, named_params, previous=None, spec=None):
        self.named = named_params
        gen = torch.Generator().manual_seed(1)
        self.delta = {
            n: torch.randn(p.shape, generator=gen) * (0.05 * float(p.detach().norm()) / p.numel() ** 0.5 + 1e-3)
            for n, p in named_params
        }
        self.param_groups = [dict(params=[p for _, p in named_params], lr=0.01)]

    @torch.no_grad()
    def step(self, grads, update, ctx=None):
        for n, p in self.named:
            if grads.get(n) is not None:
                p.add_(grads[n], alpha=-0.01)
            if update >= ON_FROM:
                p.add_(self.delta[n] * (1 if (update - ON_FROM) % 2 == 0 else -1))


def build(named_params, previous=None, spec=None):
    return Updater(named_params, previous, spec)
'''

# Lookahead (k = 10, alpha = 0.5) on SGD: every 10 updates the fast weights jump halfway back to the slow ones
LOOKAHEAD = SGD_HEAD + '''

class Updater:
    follow_user_scheduler = False

    def __init__(self, named_params, previous=None, spec=None):
        self.named, self.bufs = named_params, {}
        self.params = [p for _, p in named_params]
        self.slow = [p.detach().clone() for p in self.params]
        self.param_groups = [dict(params=self.params, lr=LR)]

    @torch.no_grad()
    def step(self, grads, update, ctx=None):
        sgd(self.named, grads, self.bufs, LR)
        if update % 10 == 0:
            torch._foreach_lerp_(self.slow, self.params, 0.5)
            torch._foreach_copy_(self.params, self.slow)


def build(named_params, previous=None, spec=None):
    return Updater(named_params, previous, spec)
'''


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        lin, relu = torch.nn.Linear, torch.nn.ReLU
        self.body = torch.nn.Sequential(lin(8, 32), relu(), lin(32, 32), relu())
        self.fc = torch.nn.Linear(32, 4)

    def forward(self, x):
        return self.fc(self.body(x))


def eval_times(steps, window, seed=0):
    """One evaluation at a uniformly random update in each window of `window` updates (as examples/overfit)."""
    rng = random.Random(seed)
    return sorted({k * window + rng.randrange(window) + 1 for k in range(steps // window)})


def run(tmp_path, program=None, steps=480, window=40, evals=None, subs=None, seed=0, **kw):
    """Train Net on a fixed random task, evaluating the validation batch after each update in `evals` (default: one
    random time per window) and reporting it. program: an updater.py text (__EVALS__ is replaced by the evaluation
    times, and each key of subs by its value). Returns (events, evaluation updates, the optimizer)."""
    torch.manual_seed(seed)
    model = Net()
    gen = torch.Generator().manual_seed(seed)
    w_true = torch.randn(8, 4, generator=gen)
    xs = [torch.randn(32, 8, generator=gen) for _ in range(30)]
    data = [(x, (x @ w_true).argmax(1)) for x in xs]
    dev_x = torch.randn(256, 8, generator=gen)
    dev_y = (dev_x @ w_true).argmax(1)
    evals = sorted(eval_times(steps, window, seed) if evals is None else evals)
    folder = None
    if program is not None:
        text = program.replace("__EVALS__", repr(evals))
        for key, value in (subs or {}).items():
            text = text.replace(key, repr(value))
        folder = tmp_path / "program"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "updater.py").write_text(text)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    kw = dict(dict(backend="keep", validation=window, every=10, decisions=2, verbose=False), **kw)
    w = opal.wrap(
        opt, model, scheduler=sched, total_steps=steps, goal="t", run_dir=tmp_path / "r", program=folder, **kw
    )
    for u in range(1, steps + 1):
        x, y = data[u % len(data)]
        loss = F.cross_entropy(model(x), y)
        w.zero_grad()
        loss.backward()
        w.step(loss=loss)
        sched.step()
        if u in evals:
            with torch.no_grad():
                w.report_validation(loss=F.cross_entropy(model(dev_x), dev_y), update=u)
    w.close()
    events = [json.loads(line) for line in open(tmp_path / "r" / "events.jsonl")]
    return events, evals, w


def flags(events):
    return [e for e in events if e["kind"] == "eval_integrity"]


def checked(events):
    end = [e for e in events if e["kind"] == "end"][-1]
    return end["eval_integrity"]


# ----- the two runs' patterns ----------------------------------------------------------------------------------


def test_a2_scaling_the_classifier_around_each_evaluation_is_flagged(tmp_path, capfd):
    events, evals, w = run(tmp_path, A2)
    found = flags(events)
    # every probed evaluation from update 50 on; at update 25 the probe (x0.85) is 3.5 times the classifier's ordinary
    # change per update, below the threshold (5): early in training it hides in the training's own motion
    assert [e["update"] for e in found] == [t for t in evals if t >= 50]
    for e in found:
        k = evals.index(e["update"])
        assert e["flagged"] is True
        names = {r["name"] for r in e["tensors"]}
        assert names == {"fc.weight", "fc.bias"}  # the probed tensors, and only they
        r = next(r for r in e["tensors"] if r["name"] == "fc.weight")
        assert r["changed_at"] == e["update"] and r["restored_at"] == e["update"] + 1
        assert r["norm_factor"] == pytest.approx([0.85, 0.73, 0.63, 0.54][k % 4], abs=0.01)
        assert r["cosine"] < -0.9
    assert checked(events) == dict(checked=len(evals), flagged=len(found))
    err = capfd.readouterr().err
    assert "evaluation integrity" in err and "fc.weight (norm x0.85" in err and "flagged" in err
    report = (tmp_path / "r" / "report.html").read_text()
    assert "Flagged" in report and "fc.weight" in report


def test_a1_swapping_a_running_average_in_for_each_evaluation_is_flagged(tmp_path):
    events, evals, w = run(tmp_path, A1, steps=720, window=120, subs={"__FREEZE__": None})
    probed = [E for E in evals if E - 49 >= 30]  # the program has an average from update 30 on
    found = flags(events)
    assert [e["update"] for e in found] == probed
    n = len(list(w.named))
    for e in found:
        assert e["n_tensors"] >= n - 1  # (nearly) every tensor was moved to the average
        assert all(r["changed_at"] == e["update"] - 49 and r["restored_at"] == e["update"] + 1 for r in e["tensors"])
        assert all(r["remaining"] < 0.25 and r["cosine"] < -0.9 for r in e["tensors"])
    assert {r["name"] for e in found for r in e["tensors"]} == {n for n, _ in w.named}


def test_a1_probing_a_frozen_model_with_classifier_scales_is_flagged(tmp_path):
    events, evals, _ = run(tmp_path, A1, steps=720, window=120, subs={"__FREEZE__": 300})
    scale = [0.84, 0.88, 0.92, 0.96, 1.0, 0.80]
    late = [E for E in evals if E - 49 >= 300 and scale[evals.index(E) % 6] != 1.0]  # a x1.0 probe changes nothing
    found = {e["update"]: e for e in flags(events)}
    assert late and all(E in found for E in late)
    for E in late:
        assert {r["name"] for r in found[E]["tensors"]} == {"fc.weight", "fc.bias"}  # the frozen rest does not move
        r = found[E]["tensors"][0]
        assert r["norm_factor"] == pytest.approx(scale[evals.index(E) % 6], abs=1e-3)


def test_the_agent_is_not_told(tmp_path):
    """A scripted agent installs the A2 program at its first decision: the run is flagged, and nothing the agent reads
    (system prompt, messages, inbox, errors) mentions the check."""
    seen = []
    evals = eval_times(480, 40)

    def agent(ws, msg, system):
        if not seen:
            (ws / "updater.py").write_text(A2.replace("__EVALS__", repr(evals)))
        seen.append(msg)
        return "kept"

    events, _, w = run(tmp_path, backend=agent, prompt="neutral", decisions=6, evals=evals)
    assert flags(events)
    run_dir = tmp_path / "r"
    texts = seen + [(run_dir / "system.md").read_text()]
    texts += [p.read_text() for p in (run_dir / "workspace" / "inbox").glob("*.json")]
    for text in texts:
        low = text.lower().replace("test_eval_integrity", "")  # the model's module name, in the prompt's model line
        for word in ("integrity", "flagged", "eval_jump", "round trip", "moved back", "moved them back"):
            assert word not in low, word
    assert len(seen) >= 3


# ----- ordinary optimizers are not flagged -----------------------------------------------------------------------


@pytest.mark.parametrize(
    "case",
    ["adamw", "adamw-high-lr", "sgd", "muon", "decay", "sharp", "oscillate", "oscillate-onset", "lookahead"],
)
def test_ordinary_optimizers_are_not_flagged(tmp_path, case, capfd):
    steps, window = 480, 40
    evals = eval_times(steps, window, seed=3)
    program, subs, kw = None, {}, {}
    if case == "adamw-high-lr":
        kw = dict(seed=1)
        program = SHARP.replace("3e-3", "3e-2")
        subs = {"__DROP__": 10**9, "__RESTART__": 10**9, "__COOL__": 10**9}
    elif case == "sgd":
        program = A2.replace("if update in PROBE_EVALS:", "if False:")
    elif case == "muon":
        program = MUON
    elif case == "decay":
        program = DECAY
    elif case == "sharp":
        program, subs = SHARP, {"__DROP__": 200, "__RESTART__": 300, "__COOL__": 400}
        evals = sorted(set(evals) | {199, 200, 201, 299, 300, 301, 400, 401, 402, 403})  # at each sharp change
    elif case == "oscillate":
        program, subs = OSCILLATE, {"__ON__": 1}
    elif case == "oscillate-onset":
        program, subs = OSCILLATE, {"__ON__": 241}
        evals = sorted(set(evals) | {240, 241, 242, 246, 300})  # at the onset, and as it goes on
    elif case == "lookahead":
        program = LOOKAHEAD
        evals = sorted(set(evals) | {100, 101, 109, 110, 111})  # around a jump back to the slow weights
    events, evals, _ = run(tmp_path, program, steps=steps, window=window, evals=evals, subs=subs, **kw)
    assert not flags(events), flags(events)
    assert checked(events)["checked"] >= len(evals) - 4 and checked(events)["flagged"] == 0
    assert "evaluation integrity" not in capfd.readouterr().err


def test_a_weight_average_swapped_in_for_good_is_a_jump_not_a_flag(tmp_path):
    swap = 440
    evals = sorted(set(eval_times(480, 40)) | {swap - 1, swap, swap + 1, swap + 5})
    events, _, _ = run(tmp_path, FINAL_AVERAGE, evals=evals, subs={"__SWAP__": swap})
    assert not flags(events)
    jumps = [e for e in events if e["kind"] == "eval_jump"]
    at_swap = [e for e in jumps if any(r["changed_at"] == swap for r in e["tensors"])]
    assert len(at_swap) == 1 and at_swap[0]["update"] == swap  # recorded once, at the first evaluation that saw it
    assert sum(r["changed_at"] == swap for r in at_swap[0]["tensors"]) >= 5


def test_options(tmp_path):
    events, _, w = run(tmp_path / "off", A2, eval_integrity=False)
    assert w._integrity is None and not flags(events)
    events, _, w = run(tmp_path / "loose", A2, eval_integrity_threshold=1e6)  # no change is that large
    assert not flags(events)
    events, _, w = run(tmp_path / "noval", A2, validation=None, eval_integrity=True)  # on without validation
    assert flags(events)
    for bad in (1, 0.5, True, "10"):
        with pytest.raises(ValueError, match="eval_integrity_threshold"):
            run(tmp_path / f"bad{bad}", steps=40, eval_integrity_threshold=bad)
    with pytest.raises(ValueError, match="eval_integrity"):
        run(tmp_path / "badopt", steps=40, eval_integrity="yes")
