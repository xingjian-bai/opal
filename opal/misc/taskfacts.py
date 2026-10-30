"""The TASK paragraph of the draft neutral prompt (prompts/neutral.md): the run's facts and when decisions happen, in
plain sentences. Facts come from wrap(epochs=, train_size=, batch_size=, same_order_each_epoch=); a clause whose value
was not given is left out. The schedule is described from the actual decision updates."""

from __future__ import annotations


def _v(x):
    return f"{x:,}" if isinstance(x, int) and not isinstance(x, bool) else str(x)


def run_facts(total, epochs=None, train_size=None, batch_size=None, same_order=None) -> str:
    s = f"The run is planned for {total:,} training updates"
    if epochs is not None and train_size is not None:
        s += f", about {_v(epochs)} epochs over a training set of {_v(train_size)}"
    elif epochs is not None:
        s += f", about {_v(epochs)} epochs"
    elif train_size is not None:
        s += f", over a training set of {_v(train_size)}"
    if batch_size is not None:
        s += f", with a batch of {_v(batch_size)} per update"
    if same_order is True:
        s += "; each epoch passes over the same data again"
    elif same_order is False:
        s += "; each epoch passes over the data in a new order"
    return s + "."


def _shown(updates, head=3):
    if len(updates) <= head + 1:
        return ", ".join(f"{u:,}" for u in updates)
    return ", ".join(f"{u:,}" for u in updates[:head]) + f", ..., {updates[-1]:,}"


def _every(updates):
    gaps = [b - a for a, b in zip(updates, updates[1:])]
    if not gaps:
        return None, True
    return round(sum(gaps) / len(gaps)), len(set(gaps)) == 1


def _spacing(prev, updates):
    """Mean gap from `prev` through `updates`, and whether every gap is the same."""
    gaps = [y - x for x, y in zip([prev] + updates, updates)]
    return round(sum(gaps) / len(gaps)), len(set(gaps)) == 1


def _per_epoch(k, total, epochs):
    if not isinstance(epochs, (int, float)) or isinstance(epochs, bool) or epochs <= 0:
        return ""
    return f" (about once every {k * epochs / total:.2g} epochs)"


def _every_phrase(k, exact):
    return f"{'every' if exact else 'about every'} {k:,} updates"


def schedule_sentence(updates, kind, total, epochs=None, segments=None) -> str:
    """When the agent is called, computed from the actual decision updates. kind is wrap(schedule=...)'s form
    (opal/schedule.py): one phase for "uniform", an early phase (the first 5% of training) and a late one for the
    front-loaded default, one phase per segment for "segments" (segments: the updates of each), the updates themselves
    for "updates"."""
    n = len(updates)
    if n == 0:
        return "No decisions are scheduled."
    if n == 1:
        return f"You are called once, at update {updates[0]:,}."
    if kind == "updates":
        return f"You are called at updates {_shown(updates)}; {n} scheduled decisions in all."
    if kind == "segments":
        return _segments_sentence([s for s in segments or [updates] if s], total, epochs) + (
            f"; {n} scheduled decisions in all."
        )
    if kind == "uniform":
        k, exact = _spacing(0, updates)
        return (
            f"You are called {_every_phrase(k, exact)}{_per_epoch(k, total, epochs)}: at updates {_shown(updates)}, "
            f"{n} scheduled decisions in all."
        )
    early = [u for u in updates if u <= max(1, int(total * 0.05))] or updates[:1]
    late = updates[len(early) :]
    if len(early) == 1:
        s = f"You are called at update {early[0]:,}"
    else:
        k1, ex1 = _spacing(0, early)
        s = f"You are called {_every_phrase(k1, ex1)}{_per_epoch(k1, total, epochs)} for the first {early[-1]:,} "
        s += f"updates ({len(early)} decisions)"
    if len(late) == 1:
        s += f", then at update {late[0]:,}"
    elif late:
        k2, ex2 = _every(late)  # the late phase's own spacing, not the jump from the early phase
        s += (
            f", then {_every_phrase(k2, ex2)}{_per_epoch(k2, total, epochs)} from update {late[0]:,} to update "
            f"{late[-1]:,} ({len(late)} decisions)"
        )
    return s + f"; {n} scheduled decisions in all."


def _segments_sentence(phases, total, epochs):
    """One clause per segment, in the front-loaded default's words: "every K updates for the first U updates (n
    decisions)", then "every K updates from update A to update B (n decisions)"; a segment of one decision is "at
    update A"."""
    parts = []
    for i, seg in enumerate(phases):
        if len(seg) == 1:
            parts.append(f"at update {seg[0]:,}")
            continue
        k, exact = _spacing(0, seg) if i == 0 else _every(seg)
        where = f"for the first {seg[-1]:,} updates" if i == 0 else f"from update {seg[0]:,} to update {seg[-1]:,}"
        parts.append(f"{_every_phrase(k, exact)}{_per_epoch(k, total, epochs)} {where} ({len(seg)} decisions)")
    return "You are called " + ", then ".join(parts)


def mode_sentence(blocking, max_lag=None) -> str:
    if max_lag is not None:
        return (
            f"Training waits for your answer when it would otherwise take effect more than {max_lag:g} updates late "
            "(judged from the time per update and your recent answer times, and always at the first decision), and "
            "keeps running otherwise."
        )
    if blocking:
        return "Training pauses until you finish, and your edits take effect from that update."
    return (
        "Training keeps running while you work; your edits take effect at the first update boundary after you finish."
    )


def decisions(updates, kind, total, blocking, max_lag=None, show_next=False, epochs=None, segments=None) -> str:
    """kind, segments: as in schedule_sentence."""
    s = schedule_sentence(updates, kind, total, epochs, segments)
    s += ' You are also called when your analysis returns {"wake": reason}, and when the training loss becomes non-finite.'
    s += " " + mode_sentence(blocking, max_lag)
    if show_next and updates:
        s += " Each message gives the update of the next scheduled decision."
    return s + (
        " Between decisions, your programs run by themselves at every update, so whatever should change between "
        "decisions must be written into them."
    )


def protocol_note(every, max_decisions, blocking, sync_every, pause_on_nonfinite=True) -> str:
    """The technical details PROTOCOL keeps: the check cadence, the delay before an edit takes effect, one decision in
    flight, merged reasons."""
    s = (
        f"The wrapper checks every {every} updates (and at your analysis steps), whatever EVERY you set. A decision "
        "is asked only at a check: the first check at or after a scheduled update, or after your analysis asks; every "
        f"check also looks for a non-finite loss. At most {max_decisions} decisions are asked in all, wake-ups included."
    )
    if not blocking:
        s += (
            " When training keeps running while you work, an edit takes effect within a few updates of your answer on "
            f"one GPU and within {2 * sync_every} updates in data-parallel runs; each message tells you how late your "
            "previous edit took effect."
        ) + (" A non-finite training loss pauses training until you answer." if pause_on_nonfinite else "")
    return (
        s + " Only one decision is in flight at a time; reasons that arise meanwhile are merged into the next request."
    )


# ----- the starting optimizer in one line (the full dump goes to reference/starting_optimizer.txt) -------------------

SCHED_KEYS = (
    "T_max",
    "eta_min",
    "total_iters",
    "start_factor",
    "end_factor",
    "step_size",
    "gamma",
    "milestones",
    "max_lr",
    "total_steps",
    "pct_start",
    "T_0",
    "T_mult",
    "factor",
    "power",
)


def _num(v):
    if hasattr(v, "item") and getattr(v, "numel", lambda: 0)() == 1:
        v = v.item()
    return f"{v:g}" if isinstance(v, float) else str(v)


def _same(groups, key):
    vals = [repr(g.get(key)) for g in groups if key in g]
    return len(set(vals)) == 1


def _group_value(groups, key, name):
    present = [g for g in groups if key in g]
    if not present:
        return None
    if _same(present, key):
        v = present[0][key]
        return f"{name} {tuple(_num(x) for x in v) if isinstance(v, (tuple, list)) else _num(v)}".replace("'", "")
    return f"{name} per group " + ", ".join(
        _num(g[key]) if not isinstance(g[key], (tuple, list)) else str(tuple(_num(x) for x in g[key])).replace("'", "")
        for g in present
    )


def _scheduler(s, shared=True):
    cls = type(s).__name__
    fns = getattr(s, "lr_lambdas", None)
    if fns:  # a function's shape cannot be read reliably
        if not shared:
            return f"a {cls} schedule (its function is not shared)"
        return f"a {cls} schedule; its source is in reference/starting_optimizer.txt"
    try:
        st = s.state_dict()
    except Exception:
        st = {}
    args = [f"{k} {_num(st[k])}" for k in SCHED_KEYS if k in st and isinstance(st[k], (int, float, list, tuple))]
    return (
        f"{cls} ({', '.join(args)})" if args else f"a {cls} schedule; its source is in reference/starting_optimizer.txt"
    )


def starting_summary(optimizers, schedulers, clip=None, shared=True) -> str:
    """One line from the user's objects: class, lr, betas or momentum, eps, weight decay with tensors per group, and the
    scheduler (its class and settings, or the lambda's name or first line when that is reliable)."""
    parts = []
    for o in optimizers:
        groups = o.param_groups
        bits = [type(o).__name__]
        # a scheduler sets the current lr (0 at the first update of a warmup): name its base, initial_lr
        base = all("initial_lr" in g for g in groups)
        lr = ("initial_lr", "base lr") if base else ("lr", "lr")
        for key, name in (lr, ("betas", "betas"), ("momentum", "momentum"), ("eps", "eps")):
            v = _group_value(groups, key, name)
            if v:
                bits.append(v)
        if any("weight_decay" in g for g in groups):
            wd = ", ".join(f"{_num(g.get('weight_decay', 0))} on {len(g['params'])} tensors" for g in groups)
            bits.append(f"weight decay {wd}")
        parts.append(", ".join(bits))
    if schedulers:
        parts.append("scheduler " + "; ".join(_scheduler(s, shared) for s in schedulers))
    else:
        parts.append("no scheduler")
    if clip is not None:
        parts.append(f"gradients clipped to a global norm of {float(clip):g} before each update")
    return "; ".join(parts)


def duration(seconds) -> str:
    """'an hour', '6 hours', '15 minutes', '90 seconds'."""
    s = int(seconds)
    if s % 3600 == 0:
        return "an hour" if s == 3600 else f"{s // 3600} hours"
    if s % 60 == 0:
        return "a minute" if s == 60 else f"{s // 60} minutes"
    return f"{s} seconds"


def time_limit(seconds) -> str:
    """The decision time limit, from the backend's timeout (seconds)."""
    if not seconds:
        return ""
    return f"Each decision may take up to {duration(seconds)}."


# ----- OPAL 0.1.1-validation (wrap(validation=N)): what the neutral prompt says about the validation set ---------

EVALUATION_OFF = (
    "Evaluation data and results are not shown to\nyou unless the person running the training chose to show a value, "
    "which then appears in obs.metrics and in the\ndecision message; do not look for them elsewhere."
)  # the v0.1.1 sentence, line breaks included, so the prompt is unchanged with the option off
EVALUATION_ON = (
    "Evaluation data are not shown to you, nor are\nthe test results. The validation-set loss and accuracy, and any "
    "other value the person running the training\nchose to show, appear in obs.metrics and in the decision message; "
    "do not look for them elsewhere."
)


def validation_note(every=None, size=None) -> str:
    """The TASK sentences (appended to the run's facts) when the training script reports a validation set:
    about every N updates, at times that are not announced."""
    if not every:
        return ""
    of = f" of {_v(size)}" if size is not None else ""
    return (
        f" The training script evaluates the model on a validation set{of} held out from training about every "
        f"{every:,} updates, at times that are not announced, and shows you its loss, and its accuracy when the "
        "script reports one. The performance in the goal is measured on a separate test set that you never see."
    )


def evaluation_note(every=None) -> str:
    return EVALUATION_ON if every else EVALUATION_OFF
