"""The wrapper: a torch optimizer whose update is a small program that a coding agent rewrites while training runs.

This file is the main loop: build and swap programs, step, check, ask the agent. Checkpoints, recovery from a failing
program, DDP collectives and one-time setup live in opal/misc/.

Cost rules (DESIGN.md section 9): training waits for the agent at each decision by default (blocking=True); with
blocking=False it waits only when the loss turns non-finite, with "auto" only when an answer would land late; an
unchanged answer costs nothing, and an analysis-only change never rebuilds the updater; the wrapper
adds no per-update host sync (losses are copied into a device buffer and read at check steps, every `every` updates)
and, in DDP, no per-update collective outside the short trial window after a swap (swaps are agreed every `sync_every`
updates over a separate gloo group, asynchronously; one small broadcast per check step); no hooks touch the model
unless activations are requested.
"""

from __future__ import annotations

import atexit
import copy
import datetime
import json
import math
import numpy as np
import shutil
import sys
import tempfile
import time
import traceback
import types
import weakref
from collections import deque
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.algorithms.join import Join, Joinable

from . import context, defaults, estimate, programs
from .agent import AgentWorker, Run, new_run_dir, now_stamp
from .backends import make_backend
from .backends._privacy import redact_secrets
from .misc import audit, boundary, integrity, neutral_analysis, waiting
from .misc.checkpoint import CheckpointMixin
from .misc.ddp import DDPMixin
from .misc.precision import PrecisionMixin
from .misc.recovery import RecoveryMixin
from .misc.schedulers import ScheduleMixin
from .misc.schedulers import _rebuild_scheduler  # noqa: F401  (checkpoints of v0.1.0 pickle this path)
from .misc.setup import SetupMixin
from .misc.setup import resolve_prompt as setup_prompt
from .misc.skip import SkipMixin, takes_ctx
from .report import LiveReport
from . import schedule as decision_schedule
from .summaries import Summarizer


class _AccelerateCheckpointAdapter:
    """Keep Accelerate's public optimizer API on the original optimizer; adapt checkpoint I/O only."""

    def __init__(self, optimizer, opal):
        object.__setattr__(self, "_optimizer", optimizer)
        object.__setattr__(self, "_opal", opal)

    def __getattr__(self, name):
        return getattr(self._optimizer, name)

    def __setattr__(self, name, value):
        setattr(self._optimizer, name, value)

    def state_dict(self):
        return self._opal.state_dict()

    def load_state_dict(self, state):
        if state.get("opal") == 1:
            return self._opal.load_state_dict(state)
        if "state" in state and "param_groups" in state:
            key = "accelerate-legacy-state"
            if key not in self._opal._warned:
                self._opal._warned.add(key)
                print(
                    "[opal] WARNING: the legacy Accelerate checkpoint contains only the original optimizer state; "
                    "it holds no OPAL program or progress. Load a separately saved OPAL state_dict to restore them.",
                    file=sys.stderr, flush=True,
                )
            with programs.keep_param_group_identity():
                return self._optimizer.load_state_dict(state)
        return self._opal.load_state_dict(state)


# ----- public entry points ------------------------------------------------------------------------------------


def wrap(optimizer, model, *, total_steps, goal, scheduler=None, decisions=None, schedule="front", **kw):
    """Wrap a torch optimizer (or a list of them) so that a coding agent can rewrite the update during training.

    optimizer        the training script's optimizer(s); the starting program calls them unchanged. With Accelerate,
                     wrap after accelerator.prepare (the torch optimizer inside is wrapped; no GradScaler, no
                     accumulation through Accelerate) and call the wrapped optimizer's step(loss=...). The prepared
                     optimizer delegates checkpoint save/load to OPAL, including accelerator.save_state/load_state
    model            the model being trained, bare or wrapped by DDP / torch.compile (names, shapes, source code)
    total_steps      number of updates in the run (opt.step() calls that apply an update)
    goal             one sentence: what the run trains and what it is judged by
    scheduler        the training script's LR scheduler(s); their step() runs while the program follows them
    backend          "claude" (default), "codex", "mailbox:<dir>", "keep" (no agent, every decision keeps the
                     programs), or a function(workspace, message, system) -> rationale str that may edit the files
    backend_options  claude: model= (default "claude-opus-5-5") and effort= (default "high"), the agent of the blog's
                     runs; another model or effort prints one line at the start saying the blog's results may not be
                     reproduced with it. unavailable_timeout= (claude and mailbox; default None, no deadline): an agent
                     that cannot work at all (not logged in, a usage or spend limit, the API overloaded, no network,
                     no broker answering the mailbox) is waited for, with a line when the wait starts and ends, and the
                     same decision goes on once it is back; with a number of seconds the decision fails after that
                     long. timeout= (seconds per decision), retries= (a failed call is made again
                     after backoff=15 s, then twice as long, ... at most backoff_max=600 s apart, while the decision's
                     timeout allows; None, the default: as often as it allows; 0: never), shell_timeout= (seconds per
                     shell command of the agent); mailbox: poll=, broker_wait= (seconds without a broker before it
                     counts as unavailable), timeout=
    decisions        how many times the agent is asked over the run (default 50). Each time it may change the programs
                     or keep everything as it is: this is the number of chances it gets, not the number of changes it
                     makes. Its analysis may ask for more (wake-ups); max_decisions (default: twice the scheduled
                     ones) caps them all. A short run gets fewer (decisions fall on checks, see every). Not set, a run
                     whose training waits for the agent asks at most as many as keep the expected waiting within 4x
                     its own training time (measured at its first check; at least one, in the same schedule), and says
                     so; a decisions you set is kept, with a line when its waiting is above 4x (opal/estimate.py;
                     `opal estimate` before launch)
    schedule         when those decisions are asked (not the learning-rate schedule, which is `scheduler`):
                     "front" (default): a fifth of them evenly over the first 5% of training, the rest evenly after;
                     "uniform": every total_steps // decisions updates; [(0.05, 20), (1.0, 30)]: segments of
                     (end as a fraction of training, count), here 20 evenly over the first 5% and then 30 evenly over
                     the rest; [500, 1000, 5000]: these updates. A list gives the count itself (leave decisions
                     out, or make it agree). Each is moved up to the next check (opal/schedule.py)
    max_cost_usd     stop asking when the remaining USD budget cannot cover another estimated call (since 0.3.4);
                     training continues with the current programs. Codex reports tokens and cannot enforce a USD cap
    every            how often OPAL looks at the run, in updates (default 50). At each check it reads the training
                     losses, looks for a non-finite one, asks a scheduled decision when one is due (decisions happen
                     only at checks), and runs the agent's analysis program (whose own interval, EVERY, the agent may
                     set). Between checks OPAL only runs the update
    blocking         True (default): training waits at each decision. False: it continues while the agent thinks.
                     "auto": it waits only when an answer would land more than max_lag_updates (50) updates late,
                     judged from the time per update and the agent's recent answer times, and at the first decision
    share_code       False keeps the training script and model source out of what the agent (and its provider) sees
    clip_grad_norm   if your loop clips gradients, hand the clipping to OPAL: delete the loop's
                     clip_grad_norm_(model.parameters(), x) and pass clip_grad_norm=x. The starting program then clips
                     the same way (the run starts identical, bitwise when the optimizer holds all of the model's
                     trainable parameters), and the agent, which sees the unclipped gradients, may change it. Clipping
                     left in the loop happens outside the optimizer: the agent's program gets clipped gradients and
                     cannot change the threshold
    run_dir          where the run's records go (default ./opal_runs/<time>-<pid>-<random>); report.html there is
                     rewritten after each decision and at the end
    summarize        True (default): with backend "claude", a language model (claude-opus-5-5 at low effort; a model
                     name here picks another) describes and tags each code change for the report, in a background thread. False: off
    mode             "code" (default, and the only mode): the agent writes the update rule. "knobs" (a fixed updater
                     with a menu of settings, up to 0.3.3) is not supported in this release and raises
    Advanced: sync_every (20), trial_steps (20), rank_check_every (100; also after the last update), pause_on_nonfinite (True),
    activations (module names to capture, eager models only), prompt ("neutral", the default; "aggressive", the
    neutral prompt with a review of every group's hyperparameters at each decision (misc/aggressive.py); or the path of
    a template file, whose {fields} are filled as in prompts/neutral.md), program (replay folder),
    epochs, train_size, batch_size (ints or strings such as "223M tokens"), same_order_each_epoch: facts the
    neutral prompt states in its TASK paragraph (a fact not given is left out),
    snapshot_limit_gb (2.0: largest CPU copy kept to undo a failing update exactly).
    validation       OPAL 0.1.1-validation: N (an int) means the training script evaluates the model on a
                     validation set held out from training about every N updates, at times it does not announce, and
                     reports it with opt.report_validation(loss=..., accuracy=...). The agent sees every measurement
                     (obs.metrics, and a line with their history in each decision message) and the neutral prompt says
                     so in its TASK paragraph ("about every N updates, at times that are not announced");
                     validation_size (e.g. "5,000 images") names the set's size there. None (default): off, and the
                     prompt is unchanged. N and the irregular times are your promise: OPAL does not run the
                     evaluation (examples/overfit draws one time at random in each epoch), and warns you if the times
                     are regular. Needs a prompt that states it (the neutral one, or a template with
                     {validation_note}). Never report the test set the run is judged by this way.
    eval_integrity   0.2.2: check that the program does not move the parameters to another point for a validation
                     evaluation and move them back after it (misc/integrity.py). None (default): on with validation=N;
                     True/False: on/off. A run that does is flagged (an eval_integrity event, the report, stderr); the
                     agent is not told of the check. eval_integrity_threshold (5): how many times a tensor's typical
                     change per update a jump must be. Cost: one sampled copy per update (at most 32K entries).
    """
    return AgentOptimizer(
        optimizer,
        model,
        total_steps=total_steps,
        goal=goal,
        scheduler=scheduler,
        decisions=decisions,
        schedule=schedule,
        **kw,
    )


def replay(program, optimizer, model, *, total_steps, scheduler=None, **kw):
    """Run a saved program with no agent: a run folder (its final program) or a programs/vNNN folder. Returns an
    AgentOptimizer used like the wrapped one (opt.step(loss=...)); it writes its own run folder."""
    return AgentOptimizer(
        optimizer, model, total_steps=total_steps, goal="", scheduler=scheduler, program=program, backend=None, **kw
    )


def final_program(path) -> Path:
    path = Path(path)
    if (path / "updater.py").exists():
        return path
    if (path / "FINAL").exists():
        return path / "programs" / (path / "FINAL").read_text().strip()
    versions = sorted((p for p in (path / "programs").glob("v*") if p.name[1:].isdigit()),
                      key=lambda p: int(p.name[1:]))
    if not versions:
        raise FileNotFoundError(f"no program found in {path}")
    return versions[-1]


WAKE = "your analysis asked for a decision: "
MAX_REASONS = 10


def _reasons_shown(reasons):
    """At most MAX_REASONS reasons: the scheduled decision, a non-finite loss and skip alarms always, then the most
    recent wake-up requests of the analysis, and how many earlier ones are left out."""
    reasons = list(dict.fromkeys(reasons))
    if len(reasons) <= MAX_REASONS:
        return reasons
    keep = [r for r in reasons if not r.startswith(WAKE)][:MAX_REASONS]
    wakes = [r for r in reasons if r.startswith(WAKE)]
    room = MAX_REASONS - len(keep)
    shown = keep + (wakes[-room:] if room > 0 else [])
    return shown + [f"{len(wakes) - max(room, 0)} earlier requests from your analysis, not listed"]


class _ValidationAliases(dict):
    """An obs.metrics entry. Before 0.3.10 the validation values were named dev_loss and dev_accuracy; programs written
    then still find them under those names. The entry holds, prints and serializes only the val_ names."""

    def _key(self, k):
        if isinstance(k, str) and k.startswith("dev_") and not dict.__contains__(self, k):
            return "val_" + k[4:]
        return k

    def __getitem__(self, k):
        return dict.__getitem__(self, self._key(k))

    def get(self, k, default=None):
        return dict.get(self, self._key(k), default)

    def __contains__(self, k):
        return dict.__contains__(self, self._key(k))


def _plain_json(obj, bad, depth=0):
    """A report as plain JSON values: one-element tensors and numpy scalars become numbers, small tensors lists;
    anything else becomes its text, and its type is noted in `bad`."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if depth > 20:
        bad.append("nesting deeper than 20")
        return str(obj)
    if torch.is_tensor(obj):
        if obj.numel() == 1:
            return obj.detach().reshape(()).item()
        if obj.numel() <= 10_000:
            return obj.detach().cpu().tolist()
        bad.append(f"a tensor of {obj.numel()} elements")
        return f"tensor{tuple(obj.shape)}"
    if isinstance(obj, dict):
        return {str(k): _plain_json(v, bad, depth + 1) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain_json(v, bad, depth + 1) for v in obj]
    item = getattr(obj, "item", None)
    if callable(item) and hasattr(obj, "dtype"):  # numpy scalars
        try:
            return _plain_json(item(), bad, depth + 1)
        except Exception:
            pass
    bad.append(type(obj).__name__)
    return str(obj)


def _rounded(recs):
    return [
        {k: (float(f"{v:.4g}") if isinstance(v, float) else v) for k, v in r.items() if k != "tensor"} for r in recs
    ]


def _weak_log(ref):
    """The optimizer's log for objects that outlive it (the report writer stays registered at exit)."""
    return lambda msg: ref() is not None and ref()._log(msg)


def _is_accelerate(obj):
    return type(obj).__module__.startswith("accelerate.") and type(obj).__name__ in (
        "AcceleratedOptimizer",
        "AcceleratedScheduler",
    )


def _inside_accelerate(opt):
    """The torch optimizer inside Accelerate's AcceleratedOptimizer (wrap after accelerator.prepare): OPAL wraps that
    one, and the loop calls the wrapped optimizer's step(). That is exact when Accelerate neither scales the gradients
    (a GradScaler, for fp16) nor accumulates them over several step() calls; those are refused."""
    if not _is_accelerate(opt):
        return opt
    if getattr(opt, "scaler", None) is not None:
        raise ValueError(
            "opal: this AcceleratedOptimizer scales fp16 gradients with a GradScaler, which OPAL cannot wrap exactly; "
            "use bf16 or fp32 mixed precision with Accelerate"
        )
    steps = getattr(getattr(opt, "gradient_state", None), "num_steps", 1) or 1
    if steps > 1:
        raise ValueError(
            f"opal: Accelerate accumulates gradients over {steps} step() calls here (gradient_accumulation_steps), "
            "and OPAL's step() is an update each time; accumulate in your loop instead and call opt.step(loss=...) "
            "once per update (Accelerator(gradient_accumulation_steps=1))"
        )
    return opt.optimizer


def _close_at_exit(ref):
    opt = ref()
    if opt is not None:
        try:
            opt.close(interrupted=opt.update < opt.total)
        except Exception:
            pass


class AgentOptimizer(
    CheckpointMixin,
    RecoveryMixin,
    ScheduleMixin,
    DDPMixin,
    PrecisionMixin,
    SetupMixin,
    SkipMixin,
    torch.optim.Optimizer,
    Joinable,
):
    def __init__(
        self,
        optimizer,
        model,
        *,
        total_steps,
        goal,
        scheduler=None,
        backend="claude",
        backend_options=None,
        decisions=None,
        max_decisions=None,
        max_cost_usd=None,
        blocking=True,
        max_lag_updates=50,
        pause_on_nonfinite=True,
        every=50,
        sync_every=20,
        trial_steps=20,
        rank_check_every=100,
        activations=None,
        run_dir=None,
        prompt=None,
        program=None,
        share_code=True,
        schedule="front",
        show_next_decision=None,
        task_facts=None,
        epochs=None,
        train_size=None,
        batch_size=None,
        same_order_each_epoch=None,
        snapshot_limit_gb=2.0,
        summarize=True,
        clip_grad_norm=None,
        mode="code",
        validation=None,
        validation_size=None,
        eval_integrity=None,
        eval_integrity_threshold=5.0,
        verbose=True,
    ):
        Joinable.__init__(self)
        self.user_optimizers = list(optimizer) if isinstance(optimizer, (list, tuple)) else [optimizer]
        accelerated = [o for o in self.user_optimizers if _is_accelerate(o)]
        self.user_optimizers = [_inside_accelerate(o) for o in self.user_optimizers]
        self._user_parameter_ids = self._parameter_ids()
        if scheduler is None:
            self.user_schedulers = []
        else:
            self.user_schedulers = list(scheduler) if isinstance(scheduler, (list, tuple)) else [scheduler]
            # Accelerate's AcceleratedScheduler steps the torch scheduler inside it, which OPAL follows
            self.user_schedulers = [getattr(s, "scheduler", s) if _is_accelerate(s) else s for s in self.user_schedulers]
        self.model, self.total, self.goal = model, int(total_steps), goal
        self._share_code = bool(share_code)
        self._clip_grad_norm = clip_grad_norm
        if mode == "knobs":
            raise ValueError('opal: mode="knobs" is not supported in this release; mode="code" (the default) is the only mode')
        if mode != "code":
            raise ValueError(f'opal: mode must be "code", got {mode!r}')
        prompt = setup_prompt(prompt, validation if backend is not None else None)
        if prompt is None:
            prompt = "neutral"  # the default prompt
        self._aggressive = prompt == "aggressive"  # OPAL 0.1.2-aggressive: the neutral prompt, edited
        if show_next_decision is None:  # on by default with the neutral prompts
            show_next_decision = prompt in ("neutral", "aggressive")
        self._show_next, self._task_facts = bool(show_next_decision), task_facts
        n_decisions = decision_schedule.count(schedule, decisions)  # raises on a bad schedule or a mismatch
        self._schedule_kind = decision_schedule.kind(schedule)  # "front", "uniform", "segments", "updates"
        # decisions not set (and no list schedule): the first check may lower the count (opal/estimate.py)
        self._may_cap = decisions is None and self._schedule_kind in ("front", "uniform")
        self._max_decisions_given = max_decisions is not None
        self._neutral = prompt in ("neutral", "aggressive")
        # the neutral analysis compares with one epoch earlier only when the data repeat in the same order
        self._neutral_epochs = epochs if same_order_each_epoch is True and isinstance(epochs, (int, float)) else None
        self._run_facts = dict(  # the neutral prompt's TASK sentence (misc/taskfacts.py)
            epochs=epochs, train_size=train_size, batch_size=batch_size, same_order=same_order_each_epoch
        )
        if self.total < 1:
            raise ValueError("opal: total_steps must be the number of updates in the run (>= 1)")
        if backend is not None and not (isinstance(goal, str) and goal.strip()):
            raise ValueError("opal: goal must say in one sentence what the run trains and what it is judged by")
        if int(every) < 1 or int(sync_every) < 1:
            raise ValueError("opal: every and sync_every must be >= 1")
        self.mode = mode
        if validation is not None and (
            isinstance(validation, bool) or not isinstance(validation, int) or validation < 1
        ):
            raise ValueError(
                "opal: validation must be None (off) or the number of updates between validation evaluations "
                f"(an int >= 1), got {validation!r}"
            )
        self._validation, self._validation_size = validation, validation_size  # OPAL 0.1.1-validation
        self._dev = []  # the validation measurements so far: dict(update=, loss=, accuracy=)
        if eval_integrity not in (None, True, False):
            raise ValueError(
                f"opal: eval_integrity must be None (on with validation), True or False, got {eval_integrity!r}"
            )
        if isinstance(eval_integrity_threshold, bool) or not (
            isinstance(eval_integrity_threshold, (int, float)) and eval_integrity_threshold > 1
        ):
            raise ValueError(
                f"opal: eval_integrity_threshold must be a number > 1, got {eval_integrity_threshold!r}"
            )
        self._integrity_on = validation is not None if eval_integrity is None else eval_integrity
        self._integrity_threshold = float(eval_integrity_threshold)
        self._integrity = None  # misc/integrity.py, rank 0 (set up once the starting program is installed)
        self.named = context.named_params(model, self.user_optimizers)
        if not self.named:
            raise ValueError("opal: the optimizer has no parameters of this model")
        self._record_dtypes()
        super().__init__([{"params": [p for _, p in self.named], "lr": float("nan")}], {})
        self._own_groups = self.param_groups
        self._initialised = True
        self.device = self.named[0][1].device

        self.dist = dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1
        self.rank = dist.get_rank() if self.dist else 0
        self.world = dist.get_world_size() if self.dist else 1
        self.ctrl = dist.new_group(backend="gloo", timeout=datetime.timedelta(hours=12)) if self.dist else None
        self.rank0 = self.rank == 0
        self.verbose = verbose and self.rank0

        if isinstance(blocking, str) and blocking != "auto":
            raise ValueError(f'opal: blocking must be False, True or "auto", got {blocking!r}')
        self.blocking, self.pause_on_nonfinite = (blocking != "auto" and bool(blocking), bool(pause_on_nonfinite))
        self._auto = waiting.AutoWait(max_lag_updates) if blocking == "auto" else None  # misc/waiting.py
        self._block_now = False  # this check's decision waits (blocking="auto")
        self.default_every, self.sync_every = int(every), int(sync_every)
        self.trial_steps, self.rank_check_every = (int(trial_steps), int(rank_check_every))
        self.max_cost = max_cost_usd
        self._cost_said = False  # the line that max_cost_usd is reached was printed
        self.snapshot_limit = snapshot_limit_gb * 2**30
        # the scheduled decision updates, one list per segment (the prompt describes each), and all of them
        self._schedule_segments = (
            decision_schedule.decision_segments(self.total, n_decisions, self.default_every, schedule)
            if backend is not None
            else [[]]
        )
        self.schedule = [u for seg in self._schedule_segments for u in seg]
        self.max_decisions = max_decisions if max_decisions is not None else 2 * len(self.schedule)

        self.update = 0
        self.version = 0
        self.memory = {}
        self.errors = deque(maxlen=50)
        self.reports, self.metrics, self._metrics_new, self._shown_new = deque(maxlen=400), [], [], []
        self._float_losses, self._loss_n = [], 0
        self._loss_buf = torch.zeros(max(1, self.default_every), dtype=torch.float32, device=self.device)
        self._losses_from = 1
        self._pending_losses, self._pending_from = [], 1
        self._pending_omitted = 0
        self._analysis_losses, self._analysis_metrics, self._next_analysis = ([], [], self.default_every)
        self._reasons, self._sched_i = [], 0
        self._nonfinite_episode = False
        self._last_applied, self._last_answer = None, None
        self._prev_files, self._prev_updater, self._trial_until = None, None, 0
        self._snapshot, self._hypers, self._first_pending = None, None, False
        self._rank_snapshot, self._rank_trial_equal = None, False
        self._rank_checked_at = None
        self._ranks_repaired_at = []
        # the tags (programs.load) of the running updater's and analysis' modules and of the fallback updater's; other
        # versions' modules are dropped from sys.modules (_forget_retired)
        self._updater_tag = self._analysis_tag = self._prev_tag = None
        self._program_tags = set()
        self.updater = None
        self._ledger, self._install_rec, self._build_changes = None, None, None  # misc/schedulers.py
        self._resume_pending, self._observed_loss = None, None
        self._in_updater = False
        self._flag_work = None
        self._timing = dict(
            t=None,
            u=None,
            wall_ms_per_update=None,
            program_ms=None,
            program_ms_start=None,
            analysis_ms=None,
            audit=None,
            audit_start=None,
        )
        self._audit_at = 2  # profile the second update of each new updater (rank 0)
        self._step_loss, self._step_ctx, self._skips_in_a_row, self._skipped, self._skip_alarm = (
            None,
            False,
            0,
            [],
            False,
        )
        self._events = None
        self._paused_s = 0.0  # seconds training waited for the agent since the last check (rank 0)
        self._reports_made = 0  # analysis reports since the previous decision (the inbox keeps the last 400)
        self._capture_next = False
        self._estimate_due, self._estimated, self._schedule_changed = False, False, False  # opal/estimate.py
        self._t_mark, self._steps_here = None, 0  # (update, time) from which this run's time per update is measured
        self._loss_hook = None  # fn() -> this update's loss, for a framework that calls step() without it (opal/hf.py)
        self.agent_wait_seconds = 0.0  # seconds training waited for the agent in this process (rank 0; opal/results.py)
        self._closed = False
        self._resume_note = ""
        self._warned = set()
        self.activations = {}

        # run folder and records (rank 0); other ranks keep program files in a private temporary folder
        if self.rank0:
            self._auto_run_dir = run_dir is None
            self.run = Run(Path(run_dir) if run_dir else new_run_dir(Path("opal_runs")))
            self._prog_root = self.run.programs
            self._live = LiveReport(self.run.dir, log=_weak_log(weakref.ref(self)))
            atexit.register(self._live.finish)  # runs after _close_at_exit (registered below): the last report
        else:
            self._live = None
            self._auto_run_dir = False
            self.run = None
            self._prog_root = Path(tempfile.mkdtemp(prefix=f"opal-rank{self.rank}-"))

        names_by_id = {id(p): n for n, p in self.named}
        if program is not None:
            files = programs.program(programs.read_dir(final_program(program)))
        else:
            files = programs.starting_files(
                self.user_optimizers,
                self.user_schedulers,
                names_by_id,
                self.default_every,
                clip=clip_grad_norm,
                sources=share_code,
            )
            if self._neutral:  # its own starting analysis (misc/neutral_analysis.py)
                files["analysis.py"] = neutral_analysis.text(self.default_every, self._neutral_epochs)
        if self.dist:  # every rank runs rank 0's program text (descriptions of user objects may differ per rank)
            box = [files, str(self.run.dir) if self.rank0 else None]
            dist.broadcast_object_list(box, src=0, group=self.ctrl)
            files, self._run_path = box[0], Path(box[1])
        else:
            self._run_path = self.run.dir
        self._starting_files = files
        self._starting_digest = programs.side_digests(files)[0]
        err = self._install(files, previous=None, first=True)
        if err:
            raise RuntimeError(f"opal: the starting program failed to build:\n{err}")
        self._new_install_record()
        self._patch_schedulers()
        self._hook_user_optimizers()
        self._watch(activations)
        if self._integrity_on and self.rank0:
            self._integrity = integrity.Integrity(self.named, self._integrity_threshold)
            self._integrity_record(0)  # the starting parameters

        self.worker = None
        self._backend_label = "none"
        if backend is not None and self.rank0:
            be = make_backend(backend, **(backend_options or {}))
            be.preflight()
            note = be.reproduction_note() if hasattr(be, "reproduction_note") else None
            if note:  # another model or effort than the blog's (backends/claude.py)
                self._log(note)
            if share_code:
                refs = context.reference_files(model, self.user_optimizers, self.user_schedulers)
                for name, text in refs.items():
                    (self.run.workspace / "reference" / name).write_text(text)
                self._script_copied = any(n.startswith("training_script_") for n in refs)  # not in a notebook
            self._agent_shell = getattr(be, "sandbox", None)  # False: the agent has file tools only
            self._decision_timeout = getattr(be, "timeout", None)  # the neutral prompt states the limit
            self._prompt_name, self._names_by_id = prompt, names_by_id  # to write the prompt again (_set_schedule)
            self.worker = AgentWorker(self.run, be, self._system_prompt(prompt, names_by_id), log=self._log)
            # the estimate of the waiting and the cost at the first check, measured from this run (opal/estimate.py)
            self._estimate_due = getattr(be, "name", None) in ("claude", "mailbox")
            self.worker.after_decision = self._after_decision
            self._backend_label = backend if isinstance(backend, str) else getattr(be, "name", "custom")
            if summarize and getattr(be, "name", None) == "claude":
                model = summarize if isinstance(summarize, str) else None
                self._live.summarizer = Summarizer(model=model, binary=getattr(be, "binary", "claude"))
        checked = self._check_defaults(prompt, program) if self.rank0 and backend is not None else {}
        if self.rank0:
            self.run.event(
                "start",
                **checked,
                goal=self.goal,
                total_updates=self.total,
                world=self.world,
                blocking="auto" if self._auto else self.blocking,
                **(dict(max_lag_updates=self._auto.max_lag) if self._auto else {}),
                schedule=self.schedule,
                backend=getattr(self.worker.backend, "name", None) if self.worker else None,
                version=self.version,
                **(dict(validation=self._validation) if self._validation else {}),
                **(dict(eval_integrity=dict(threshold=self._integrity_threshold)) if self._integrity else {}),
            )
        atexit.register(_close_at_exit, weakref.ref(self))
        # Accelerator retains these objects. Public state/param_groups/defaults still delegate to the raw
        # optimizer (including after load_state_dict replaces state); only checkpoint I/O goes through OPAL.
        for prepared in accelerated:
            prepared.optimizer = _AccelerateCheckpointAdapter(prepared.optimizer, self)

    def _check_defaults(self, prompt, program):
        """OPAL's defaults are fixed (opal/defaults.py): say loudly when the files this run is about to use differ from
        the released ones (an edited install), and say when it uses a custom prompt or starting program. Returns what
        the start event records."""
        named = prompt in ("neutral", "aggressive")
        used = defaults.SHARED_FILES + (defaults.PROMPT_FILES if named else ()) + (
            defaults.PROGRAM_FILES if program is None else ()
        )
        out = {}
        changed = defaults.modified(used)
        if changed:
            out.update(defaults_modified=True, modified_files=changed)
            print(
                f"[opal] WARNING: OPAL's own defaults were modified ({', '.join(changed)}): these files differ from the "
                "release, so this run's agent does not get the prompt or starting programs of the blog's and the README's results. "
                "Do not edit files inside the OPAL package: reinstall OPAL, and customise through wrap()'s arguments "
                "(prompt=<a template file>, program=<a program folder>, task_facts=).",
                file=sys.stderr,
                flush=True,
            )
        for what, value in (("prompt", None if named else prompt), ("starting program", program)):
            if value is not None:
                out["custom_" + what.split()[-1]] = str(value)
                print(
                    f"[opal] this run uses a custom {what} ({value}); the blog's results used OPAL's defaults",
                    file=sys.stderr,
                    flush=True,
                )
        return out

    def _announce(self):
        """Printed at the first update, after any load_state_dict, so it names the folder the run really writes to."""
        self._announced = True
        if not self.rank0:
            return
        self._log(
            f"run folder {self.run.dir}; {len(self.schedule)} scheduled decision{'s' * (len(self.schedule) != 1)}; "
            f"{'auto-blocking' if self._auto else 'blocking' if self.blocking else 'non-blocking'}; "
            f"backend {self._backend_label}"
        )
        if self._auto:
            self._notice(
                'blocking="auto" is experimental and untested on real training runs; use True (default) or False'
            )
        describe = getattr(self.worker.backend, "describe", None) if self.worker else None
        if describe:
            self._log(describe())

    def __repr__(self):
        where = f", run {self.run.dir}" if self.run is not None else ""
        return f"AgentOptimizer(program v{self.version:03d}, update {self.update}/{self.total}{where})"

    # ----- program versions -----------------------------------------------------------------------------------

    def _spec(self):
        return dict(
            total_updates=self.total,
            update=self.update,
            world_size=self.world,
            rank=self.rank,
            user_optimizers=self.user_optimizers,
            user_schedulers=self.user_schedulers,
            clip_grad_norm=None if self._clip_grad_norm is None else float(self._clip_grad_norm),
        )

    def _groups_of(self, updater):
        """The groups the wrapper exposes: the updater's param_groups, plus one group (lr NaN) with every parameter
        they leave out, so GradScaler unscales every gradient once and LR logging never meets a group without "lr"."""
        groups = getattr(updater, "param_groups", None)
        if not (isinstance(groups, list) and groups and all(isinstance(g, dict) and "params" in g for g in groups)):
            return self._own_groups
        ids = [id(p) for g in groups for p in g["params"]]
        if len(ids) != len(set(ids)):
            raise ValueError(
                "param_groups lists a parameter in more than one group (a gradient scaler would unscale it twice)"
            )
        covered = set(ids)
        rest = [p for _, p in self.named if id(p) not in covered]
        return list(groups) + ([{"params": rest, "lr": float("nan")}] if rest else [])

    @property
    def param_groups(self):
        """The groups the training script sees (GradScaler, LR logging): the running updater's param_groups, read
        again whenever its group dicts change (a property that builds new dicts each call shows its current rates), plus
        the group of the parameters it leaves out. Cached while the updater returns the same dicts."""
        d = self.__dict__
        up = d.get("updater")
        if up is None:
            return d["_param_groups"]
        groups = getattr(up, "param_groups", None)
        cache = d.get("_groups_cache")
        same = (
            cache is not None
            and cache[0] is up
            and isinstance(groups, list)
            and isinstance(cache[1], list)
            and len(groups) == len(cache[1])
            and all(a is b for a, b in zip(groups, cache[1]))
        )
        if same or (cache is not None and cache[0] is up and groups is None and cache[1] is None):
            return cache[2]
        try:
            exposed = self._groups_of(up)
        except ValueError:  # a program that lists a parameter twice was refused when built; keep what it had
            return cache[2] if cache is not None and cache[0] is up else d["_param_groups"]
        d["_groups_cache"] = (up, list(groups) if isinstance(groups, list) else None, exposed)
        return exposed

    @param_groups.setter
    def param_groups(self, value):  # torch.optim.Optimizer.__init__
        self.__dict__["_param_groups"] = value

    def _every_of(self, an):
        e = getattr(an, "EVERY", self.default_every) if an is not None else self.default_every
        if isinstance(e, bool) or not isinstance(e, int) or e < 1:
            raise TypeError(f"EVERY must be a positive int, got {e!r}")
        return e

    def _build(self, files, version, previous, updater=True, restoring=False):
        """Load `files` and, unless only the analysis side changed, build the updater. Raises on any problem and
        registers nothing on failure."""
        if self.rank0:
            boundary.record(files, version, self.run)
        d = self._prog_root / f"v{version:03d}"
        programs.write_dir(d, files)
        tag = f"r{self.rank}v{version}"
        try:
            an = programs.load(files, "analysis.py", tag, d)
            if an is not None and not callable(getattr(an, "analyze", None)):
                raise TypeError("analysis.py must define analyze(obs, memory); this one does not, so it would never run")
            every, up, groups = self._every_of(an), None, None
            if updater:
                mod = programs.load(files, "updater.py", tag, d)
                with programs.keep_param_group_identity(), self._dtypes_kept():
                    up = mod.build(self.named, previous=previous, spec=self._spec())
                    starting = programs.side_digests(files)[0] == self._starting_digest
                    methods = ("step",) if starting else ("step", "state_dict", "load_state_dict")
                    for method in methods:
                        if not callable(getattr(up, method, None)):
                            raise TypeError(f"build() must return an updater with a callable {method} method")
                    bad = programs.unpicklable(programs.normalize_state(up.state_dict())) if not (starting or restoring) else []
                if bad:
                    raise TypeError(
                        "state_dict() must contain tensors and weights-only-safe values in acyclic containers, "
                        "so that checkpoints load with weights_only=True; found: "
                        + "; ".join(bad)
                    )
                groups = self._groups_of(up)
        except BaseException:
            programs.forget(tag)
            if self.rank0:
                shutil.move(str(d), str(self._prog_root / f"rejected-v{version:03d}-{now_stamp()}"))
            else:
                shutil.rmtree(d, ignore_errors=True)
            raise
        return an, up, every, groups

    def _install(self, files, previous, first=False, updater=True, handover=True, restoring=False):
        """Make `files` the next version. updater=False: only analysis.py (or its helpers) changed, so the running
        updater object and its state stay. On failure on any rank, keep the current programs (with their
        hyperparameters as they were) and return the error text. handover=False: the caller puts the user's schedule
        in place itself (a revert, a resume)."""
        version = 0 if first else self.version + 1
        swap = updater and not first
        hypers = self._hyper_state() if swap else None
        before = self._schedule_state() if swap else None  # the user's schedule as the running program left it
        ledger = copy.deepcopy(self._ledger) if swap else None
        user_dtypes = self._user_state_dtypes() if swap else None  # misc/precision.py
        err = None
        try:
            an, up, every, groups = self._build(files, version, previous, updater, restoring=restoring)
        except programs.AGENT_ERRORS:
            err = traceback.format_exc(limit=6)[-3000:]
        if not self._agree(err is None):
            if err is None:  # built here, failed on another rank
                programs.forget(f"r{self.rank}v{version}")
            self._restore_hypers(hypers)
            if before is not None:
                self._set_schedule_state(before)
            return err or "failed on another rank"
        digests = programs.side_digests(files)
        # an analysis stopped for changing training stays stopped until analysis.py (or a module it imports) changes
        off = not first and getattr(self, "_analysis_off", False) and digests[1] == self.digests[1]
        self.files, self.version, self.digests = files, version, digests
        self.analysis, self.every, self._analysis_off = an, every, off
        tag = f"r{self.rank}v{version}"
        self._analysis_tag = tag
        self._program_tags.add(tag)
        if updater:
            old = self.updater
            self.updater, self._hypers = up, hypers  # param_groups follows the updater (the property below)
            self._updater_tag = tag
            self._step_ctx, self._skips_in_a_row, self._ctx_any = takes_ctx(up), 0, None
        if swap:
            self._build_changes = self._changed_settings(before)  # what build() set on the user's groups
            if handover:
                self._schedule_handover(old, up, before, self._build_changes)
            self._install_rec = dict(before=before, ledger=ledger, calls=[], user_dtypes=user_dtypes)
        if not first:
            self._next_analysis = min(self._next_analysis, self.update + every)
        if self.rank0:
            (self.run.dir / "FINAL").write_text(f"v{version:03d}\n")
        return None

    def _forget_retired(self):
        """Drop from sys.modules the modules of the versions that can no longer run (all but the running updater and
        analysis and the fallback updater of a trial window), so that what their module globals hold is freed."""
        keep = {self._updater_tag, self._analysis_tag, self._prev_tag if self._prev_updater is not None else None}
        for tag in self._program_tags - keep:
            programs.forget(tag)
        self._program_tags &= keep

    def _apply(self, resp, u):
        """Swap in an agent's answer at the start of update u (every rank calls this with the same resp)."""
        if not resp:
            return
        k = resp.get("k")
        rec = dict(k=k, requested_at=resp.get("requested_at"), applied_at=u)
        rec.update({x: True for x in ("gave_up", "late") if resp.get(x)})  # backend stopped waiting / answered after
        if resp.get("backend_error") is not None:
            rec["failed"] = True
        if self._auto is not None and self.rank0:
            self._auto.answered(resp)
        for e in resp.get("errors") or []:
            self.errors.append(dict(update=u, where=f"your answer to decision {k} (previous programs kept)", error=e))
        if resp.get("backend_error") is not None:
            err = str(resp["backend_error"]) or 'the agent call failed'
            self.errors.append(dict(update=u, where=f"agent backend (decision {k})", error=err[-1500:]))
            self._log(f"decision {k}: the agent backend failed: {err[:300]}")
        if resp.get("errors"):
            return self._outcome("rejected", rec, resp)
        agent, base = resp.get("files"), resp.get("base") or {}
        if not agent:
            return self._outcome("kept", rec, resp)
        # three-way merge at file level: apply only what the agent changed relative to the files it was shown, so an
        # answer computed before a revert does not bring the reverted program back
        files = dict(self.files)
        for name in set(agent) | set(base):
            if agent.get(name) != base.get(name):
                if name in agent:
                    files[name] = agent[name]
                else:
                    files.pop(name, None)
        files = programs.program(files)
        up_d, an_d = programs.side_digests(files)
        if (up_d, an_d) == self.digests:
            return self._outcome("kept", rec, resp)
        rank_equal = False
        if self.dist and up_d != self.digests[0]:
            version = self.version
            rank_equal = not self._check_ranks(u - 1)  # before build() can change shared parameters or state
            if self.version != version:  # the outgoing trial failed; merge the answer against the fallback instead
                return self._apply(resp, u)
        changed = sorted(n for n in set(files) | set(self.files) if files.get(n) != self.files.get(n))
        t0 = time.time()
        old_files, old_updater, old_tag = self.files, self.updater, self._updater_tag
        new_updater = up_d != self.digests[0]
        # the old updater is still in its trial window: the fallback stays the program before it, which passed one,
        # with its group settings and the record of the user's schedule since it ran
        held = (self._hypers, self._install_rec) if new_updater and u <= self._trial_until else None
        # taken before build(), so that a failing build or first update is undone exactly (misc/recovery.py)
        snap = self._take_snapshot() if new_updater else None
        err = self._install(files, previous=old_updater, updater=new_updater)
        if err:
            if isinstance(snap, dict):
                self._put_back_placement()
                self._restore_snapshot(snap, old_updater)
            self.errors.append(dict(update=u, where="loading your new programs (previous programs kept)", error=err))
            return self._outcome("load_failed", rec, resp, "new programs failed to load; previous kept", error=err)
        self._last_applied = rec
        if new_updater:  # trial window: a failure in it goes back to the old updater, the first update exactly
            self._timing["program_ms"], self._events = None, None
            if held is None:
                self._prev_files, self._prev_updater, self._prev_tag = old_files, old_updater, old_tag
                self._rank_snapshot, self._rank_trial_equal = (snap, rank_equal) if self.dist else (None, False)
            else:  # the old updater is dropped (it never passed its trial window); the fallback stays
                self._rank_trial_equal = self._rank_trial_equal and rank_equal
                self._hypers, self._install_rec = held
                if self._first_pending and isinstance(self._snapshot, dict):
                    snap = self._snapshot  # the old updater never stepped: the copy taken before it undoes both builds
                elif isinstance(snap, dict):
                    snap["copies"]["old"] = None  # the old updater's state, which is not the fallback's
            self._trial_until, self._snapshot, self._first_pending = u + self.trial_steps, snap, True
            self._timing["audit"], self._audit_at = None, u + 1
        self._forget_retired()
        note = (
            f"applied at update {u} as v{self.version:03d} (requested at {rec['requested_at']}): {', '.join(changed)}"
        )
        self._outcome(
            "applied",
            rec,
            resp,
            note,
            version=self.version,
            changed=changed,
            rebuilt_updater=new_updater,
            swap_seconds=round(time.time() - t0, 3),
        )

    def _outcome(self, kind, rec, resp, note=None, **extra):
        # what became of this answer: the next message says it in one line (agent.previous_answer)
        self._last_answer = dict(rec, outcome=kind, **{x: resp.get(x) for x in ("backend_error", "seconds")})
        if self.rank0:
            lag = None if rec.get("applied_at") is None else rec["applied_at"] - (rec.get("requested_at") or 0)
            waited = resp.get("unavailable_seconds")  # waiting for an unavailable agent, apart from its own time
            self.run.event(
                kind,
                **rec,
                **extra,
                lag_updates=lag,
                seconds=resp.get("seconds"),
                cost_usd=resp.get("cost_usd"),
                **({"unavailable_seconds": waited} if waited else {}),
            )
            cost = resp.get("cost_usd") or 0
            cost_text = "USD unavailable" if "usage" in resp and resp.get("cost_usd") is None else f"${cost:.2f}"
            then = f", after {waited:.0f} s waiting for the agent to be available" if waited else ""
            self._log(f"decision {rec['k']}: {note or 'programs ' + kind} ({resp.get('seconds')} s{then}, {cost_text})")
            if self.worker is not None:
                self._cost_reached()  # said as soon as an answer spends the rest
            self._live.refresh()  # only queued: the report is written in its own thread

    def _after_decision(self, k, req, resp):
        """In the agent's thread, once a decision's response is recorded: queue the summary of a proposed code change
        (when there is a summarizer) and a new report."""
        if resp.get("files"):
            self._live.summarize(k, resp.get("text"), req["files"], resp["files"], self.total)
        if self._closed:
            self.run.event(
                "cost_settled", k=k,
                cost_usd=self.worker.recorded_cost,
                unknown_cost_decisions=sorted(self.worker.unknown_cost_decisions),
            )
        self._live.refresh()

    # ----- the update -------------------------------------------------------------------------------------------

    def _parameter_ids(self):
        return tuple(frozenset(id(p) for g in o.param_groups for p in g["params"]) for o in self.user_optimizers)

    def _check_user_parameters(self):
        if self._parameter_ids() != self._user_parameter_ids:
            raise RuntimeError(
                "opal: the raw optimizer's parameter set changed after wrapping; "
                "create all parameter groups before opal.wrap"
            )

    @property
    def _step_supports_amp_scaling(self):
        # GradScaler queries this before unscaling or deciding whether to skip the wrapper's step.
        if len(self.user_optimizers) > 1:
            raise RuntimeError(
                "opal: multiple optimizers with GradScaler are unsupported: "
                "a combined step cannot preserve each optimizer's independent overflow handling; "
                "use a single optimizer or disable gradient scaling"
            )
        self._check_user_parameters()
        return False

    def observe(self, loss):
        """Record this update's loss when a framework calls step() for you (Accelerate, Lightning, HF Trainer). Called
        before step(), it is also the loss a step(grads, update, ctx) program gets as ctx.loss."""
        self._push_loss(loss)
        self._observed_loss = loss

    def _push_loss(self, loss):
        if torch.is_tensor(loss):
            v = loss.detach()
            if v.numel() != 1:
                self._warn("loss-vector", f"the loss passed to step() has {v.numel()} elements; its mean is used")
                v = v.float().mean()
            if self._loss_n < self._loss_buf.numel():
                self._loss_buf[self._loss_n].copy_(v.reshape(()).float())  # async, CUDA-graph safe
                self._loss_n += 1
        else:
            self._float_losses.append((self._loss_n, float(loss)))

    @torch.no_grad()
    def step(self, closure=None, loss=None):
        Join.notify_join_context(self)
        self._check_join_inputs()
        self._check_user_parameters()
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        if loss is None and self._observed_loss is None and self._loss_hook is not None:
            loss = self._loss_hook()  # opal/hf.py: the update's loss, which the Trainer does not pass to step()
        if not getattr(self, "_announced", False):
            self._announce()
        if getattr(self.step, "_wrapped_by_lr_sched", False):
            self._warn(
                "sched-on-wrapper",
                "a scheduler was built on the wrapper itself; pass your scheduler to wrap() "
                "instead, or it will overwrite the learning rates the agent's program sets",
            )
        if self._resume_pending is not None:
            self._finish_resume()
        self.update += 1
        u = self.update
        if self._estimate_due and self._t_mark is None and self._steps_here == 1:
            # The second update: warm-up is behind; agent waiting is not training time.
            self._t_mark = (u, time.perf_counter() - self.agent_wait_seconds)
        self._steps_here += 1
        if u == self.total + 1:
            self._warn(
                "past-total",
                f"step() called past total_steps={self.total}; programs written as functions of "
                "update / total_updates may misbehave from here (check total_steps)",
            )
        self._maybe_swap(u)
        self._step_loss = loss if loss is not None else self._observed_loss
        self._observed_loss = None
        if loss is not None:
            self._push_loss(loss)
        analyze = u >= self._next_analysis or u == self.total
        if analyze or u % self.default_every == 0:
            self._at_check(u, analyze)
        self._do_update(u)
        if self._integrity is not None:
            self._integrity_record(u)
        if self.dist and self.rank_check_every and self._rank_checked_at != u and (
            u % self.rank_check_every == 0 or u == self.total
        ):
            self._check_ranks(u)
        self._capture_next = self.rank0 and bool(self._hooks) and u + 1 >= self._next_analysis
        if u == self.total:
            self.close()
        return loss

    def _do_update(self, u):
        grads = {n: p.grad for n, p in self.named}
        trial = u <= self._trial_until
        first = trial and self._first_pending
        profile = self.rank0 and u == self._audit_at
        # one update per check, and the run's first updates, so that the starting program is measured before any swap
        measure = ((u - 1) % self.default_every == 0 or u <= 3) and not profile
        if measure:
            if self.device.type == "cuda":
                ev = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                ev[0].record()
            else:
                if self.device.type == "mps":
                    torch.mps.synchronize()  # asynchronous like CUDA: time the work, not the launches
                t0 = time.perf_counter()
        err = None
        self._in_updater = True
        try:
            if profile:
                self._record_audit(audit.audited(lambda: self._call_updater(grads, u), self.device), u)
            else:
                self._call_updater(grads, u)
        except programs.AGENT_ERRORS:
            err = traceback.format_exc(limit=6)[-3000:]
        finally:
            self._in_updater = False
        if err is None:  # a changed dtype fails like an exception (misc/precision.py); the state at some updates only
            err = self._precision_error(grads, first, state=trial or u % self.default_every == 0)
        failed = (not self._agree(err is None)) if trial else err is not None
        if failed:
            self._recover(u, grads, err or "raised on another rank", trial, first)
            return
        if self.dist and trial and (first or u == self._trial_until or u == self.total):
            version = self.version
            self._check_ranks(u)
            if self.version != version:  # exact rollback; do not step on gradients from the rejected trajectory
                return
        if first:
            self._snapshot, self._first_pending = None, False  # the exact undo is only for the first update
        if trial and u == self._trial_until:
            self._prev_updater = self._hypers = None  # trial passed: release the old updater
            self._rank_snapshot, self._rank_trial_equal = None, False
            self._forget_retired()
        if measure:
            starting = self.digests[0] == self._starting_digest
            if self.device.type == "cuda":
                ev[1].record()
                self._events = (ev, self.version, u, starting)
            else:
                if self.device.type == "mps":
                    torch.mps.synchronize()
                self._set_program_ms((time.perf_counter() - t0) * 1000, self.version, u, starting)

    def _set_program_ms(self, ms, version, measured_at, starting):
        """Updater time per update, measured at one update per check and at the run's first updates. The first
        measurement of a run includes warm-up and is not kept; the starting program's figure is its latest measurement
        (also after a revert to it), so while it runs the two agree."""
        if measured_at <= 1:
            return
        if version == self.version:
            self._timing["program_ms"] = round(ms, 4)
        if starting:
            self._timing["program_ms_start"] = round(ms, 4)

    def _record_audit(self, counts, u):
        self._timing["audit"] = counts
        if self.digests[0] == self._starting_digest:
            self._timing["audit_start"] = counts
        if self.rank0 and counts:
            self.run.event("audit", update=u, version=self.version, **counts)

    def _read_losses(self):
        """This rank's losses since the last check (rank 0's are what the agent sees; no collective)."""
        n, self._loss_n = self._loss_n, 0
        floats, self._float_losses = self._float_losses, []
        if not self.rank0:
            return []
        tensors = self._loss_buf[:n].tolist() if n else []
        if not floats:
            return tensors
        losses, start = [], 0
        for at, value in floats:
            losses.extend(tensors[start:at])
            losses.append(value)
            start = at
        losses.extend(tensors[start:])
        return losses

    def _at_check(self, u, analyze):
        """Every `every` updates (and whenever the agent's analysis is due): rank 0 reads its losses, watches for a
        non-finite loss and for scheduled decisions, runs analysis.py when it is due and may request a decision. Ranks
        learn from rank 0 whether to pause, in one small broadcast."""
        losses_from = self._losses_from
        losses = self._read_losses()
        self._losses_from = u + 1
        new_episode = False
        if self.rank0:
            if not losses and u >= self.default_every:
                self._warn(
                    "no-loss", "no loss received: call opt.step(loss=loss) (or opt.observe(loss)) so the agent sees it"
                )
            nonfinite = any(not math.isfinite(x) for x in losses)
            new_episode = nonfinite and not self._nonfinite_episode
            if losses:
                self._nonfinite_episode = nonfinite
            now = time.perf_counter()
            if self._timing["t"] is not None and u > self._timing["u"]:  # the pauses for the agent left out
                busy = max(0.0, now - self._timing["t"] - self._paused_s)
                self._timing["wall_ms_per_update"] = round(1000 * busy / (u - self._timing["u"]), 2)
            self._timing["t"], self._timing["u"], self._paused_s = now, u, 0.0
            if self._auto is not None:
                self._auto.tick(u, now)
        if self._events is not None and analyze:
            (a, b), v, mu, starting = self._events
            b.synchronize()
            self._set_program_ms(a.elapsed_time(b), v, mu, starting)
            self._events = None
        if self.rank0:
            self._rank0_check(u, analyze, losses, losses_from, new_episode)
        else:
            self._next_analysis = u + self.every if analyze else self._next_analysis
        pause = self._from_rank0(self.blocking or self._block_now or (new_episode and self.pause_on_nonfinite))
        if pause:
            self._blocking_point(u)
            # a non-finite loss found while an older decision was in flight: ask about it now, and wait again
            nan_left = self.rank0 and any(r.startswith("non-finite") for r in self._reasons)
            if nan_left and self.worker is not None and not self.worker.busy and self._budget_left():
                self._submit(u, pause=True)
            if self._from_rank0(nan_left):
                self._blocking_point(u)
        self._block_now = False

    def _rank0_check(self, u, analyze, losses, losses_from, new_episode):
        lrs = []
        for g in self.param_groups:
            try:
                lrs.append(float(g.get("lr")))
            except (TypeError, ValueError, AttributeError):
                pass
        self.run.append("losses.jsonl", dict(from_update=losses_from, to_update=u, losses=losses, lr=lrs))
        new_metrics = [self._plain(m) for m in self._metrics_new]
        self._metrics_new = []
        for m in new_metrics:
            self.run.append("metrics.jsonl", m)
        self._analysis_losses.extend(losses)
        self._analysis_metrics.extend(self._plain(m) for m in self._shown_new)  # only values shown to the agent
        self._shown_new = []
        del self._analysis_losses[:-100_000]
        wake = self._run_analysis(u) if analyze else None
        self._pending_losses.extend(losses)
        dropped = max(0, len(self._pending_losses) - 100_000)
        self._pending_from += dropped
        self._pending_omitted += dropped
        del self._pending_losses[:-100_000]
        if new_episode:
            bad = sum(1 for _, p in self.named if not bool(torch.isfinite(p.detach()).all()))
            self._reasons.append(
                f"non-finite training loss before update {u:,} ({bad} of {len(self.named)} parameter tensors are "
                "non-finite now)"
            )
        if self._estimate_due and self._t_mark is not None and u - self._t_mark[0] >= 2:
            self._first_estimate(u)
        while self._sched_i < len(self.schedule) and u >= self.schedule[self._sched_i]:
            if "scheduled decision" not in self._reasons:
                self._reasons.append("scheduled decision")
            self._sched_i += 1
        if wake:
            self._reasons.append(f"{WAKE}{str(wake)[:300]}")
        if self._skip_alarm:
            self._skip_alarm = False
            self._reasons.append("your program kept skipping updates (see errors)")
        if self._reasons and self.total - u < self.default_every:
            self._reasons = []  # an answer could no longer act on this run
        if self.worker is not None:
            self.worker.collect_late()  # an answer that came after the backend stopped waiting, before a newer request
        if self.worker is None or self.worker.k >= self.max_decisions or (
            not self.worker.busy and not self._budget_left()
        ):
            self._reasons = []  # no future call can consume these; keep analysis reports, not unbounded wake strings
        wakes_only = all(r.startswith(WAKE) for r in self._reasons)
        if self._reasons and self.worker is not None and not self.worker.busy and self._budget_left(wakes_only):
            self._submit(u, pause=new_episode and self.pause_on_nonfinite)

    def _first_estimate(self, u):
        """At the first check (rank 0): print the estimate of the waiting and the cost from this run's own time per
        update. With decisions not set, a blocking run keeps the expected waiting within estimate.MAX_WAIT_FACTOR (4)
        times its own training time by scheduling fewer decisions (at least one), in the same shape."""
        self._estimate_due, self._estimated = False, True
        since_u, since_t = self._t_mark
        sps = max(1e-9, (time.perf_counter() - self.agent_wait_seconds - since_t) / (u - since_u))
        blocking = "auto" if self._auto is not None else self.blocking
        max_lag = self._auto.max_lag if self._auto is not None else 50
        n, capped_from = len(self.schedule), None
        if self._may_cap and blocking is not False and n > 1:
            k = estimate.capped_decisions(n, self.total * sps, blocking, sps, max_lag)
            if k < n or self.worker.k > self._sched_i:
                capped_from = n
                if self._sched_i or self.worker.k:
                    # Keep consumed checks; thin only the rest, counting calls already asked (including wake-ups).
                    remaining = self.schedule[self._sched_i :]
                    left = min(len(remaining), max(0, k - self.worker.k))
                    keep = set(self.schedule[: self._sched_i])
                    keep.update(remaining[i * len(remaining) // left] for i in range(left))
                    segments = [[x for x in seg if x in keep] for seg in self._schedule_segments]
                else:
                    segments = decision_schedule.decision_segments(self.total, k, self.default_every, self._schedule_kind)
                # A one-decision uniform grid lands at the excluded final update. Keep a usable original check.
                self._set_schedule(segments if any(segments) else [self.schedule[:1]])
        be = self.worker.backend
        mailbox = getattr(be, "name", None) == "mailbox"
        model = estimate.MODEL if mailbox else getattr(be, "model", None) or "the CLI's default model"
        effort = estimate.EFFORT if mailbox else getattr(be, "effort", None) or "default"
        said = estimate.lines(
            self.total, sps, len(self.schedule), blocking, max_lag, model, effort, mailbox, capped_from,
            explicit=not self._may_cap,
        )  # fmt: skip
        self._log(
            f"estimate from this run's updates {since_u:,} to {u - 1:,} ({sps:.3g} s each) and our runs of "
            f"{estimate.MODEL} at effort {estimate.EFFORT}:"
        )
        for line in said:
            self._log("  " + line)
        self.run.event(
            "estimate",
            update=u,
            sec_per_update=round(sps, 6),
            decisions=len(self.schedule),
            **(dict(capped_from=capped_from, schedule=self.schedule) if capped_from is not None else {}),
        )

    def _set_schedule(self, segments):
        """Replace the scheduled decisions (the first check's cap, or a resume of a capped run) and write the prompt
        again, which states them."""
        self._schedule_segments, self._schedule_changed = segments, True
        self.schedule = [x for seg in segments for x in seg]
        if not self._max_decisions_given:
            self.max_decisions = 2 * len(self.schedule)
        if self.worker is not None:
            self.worker.system = self._system_prompt(self._prompt_name, self._names_by_id)
            (self.run.dir / "system.md").write_text(self.worker.system)

    def _run_analysis(self, u):
        """Call the agent's analysis.analyze(obs, memory); keep its report, return its wake request (if any)."""
        self._next_analysis = u + self.every
        wake = None
        if self.analysis is not None and not self._analysis_off:
            obs = types.SimpleNamespace(
                update=u,
                total_updates=self.total,
                losses=self._analysis_losses,
                params={n: p for n, p in self.named},
                grads={n: p.grad for n, p in self.named},
                updater=self.updater,
                metrics=[_ValidationAliases(m) for m in self._analysis_metrics],
                program_ms_per_update=self._timing["program_ms"],
                wall_ms_per_update=self._timing["wall_ms_per_update"],
                activations=self.activations,
            )
            marks = self._analysis_marks()
            t0 = time.perf_counter()
            try:
                res = self.analysis.analyze(obs, self.memory) or {}
            except programs.AGENT_ERRORS:
                self.errors.append(
                    dict(update=u, where="analysis.analyze", error=traceback.format_exc(limit=6)[-3000:])
                )
                res = {}
            self._timing["analysis_ms"] = round(1000 * (time.perf_counter() - t0), 2)
            touched = self._analysis_changes(marks)
            if touched:  # it changed training; it cannot be undone, so it is stopped and the agent told
                self._put_back_placement()
                self._analysis_off = True
                self.errors.append(
                    dict(
                        update=u,
                        where="analysis.analyze (not run again until analysis.py changes)",
                        error="analysis.analyze changed parameters or gradients (in place, or their dtype or device), "
                        "which it may not do: " + ", ".join(touched[:8]) + (f" ({len(touched)} in all)" if len(touched) > 8 else ""),
                    )
                )
            if isinstance(res, dict):
                if "report" in res:
                    bad = []
                    report = _plain_json(res["report"], bad)  # converted now, not when the message is written
                    if bad:
                        self.errors.append(
                            dict(
                                update=u,
                                where="analysis.analyze report",
                                error="values that are not JSON were stored as text: " + ", ".join(sorted(set(bad))[:8]),
                            )
                        )
                    self.reports.append(dict(update=u, report=report))
                    self._reports_made += 1
                    self.run.append("reports.jsonl", dict(update=u, version=self.version, report=report))
                wake = res.get("wake")
        self._analysis_losses, self._analysis_metrics = [], []
        self.activations = {}
        return wake

    def _budget_left(self, wakes_only=False):
        """Whether one more decision may be asked. Wake-ups of the analysis alone may not use the decisions the
        remaining scheduled ones need."""
        left = self.max_decisions - self.worker.k
        if left <= 0 or (wakes_only and left <= len(self.schedule) - self._sched_i):
            return False
        return not self._cost_reached()

    def _cost_reached(self):
        """True once max_cost_usd cannot cover another estimated call; says so on stdout the first time."""
        if self.max_cost is None:
            return False
        left = self.max_cost - self.worker.cost
        needed = max(0.01, self.worker.cost / max(1, self.worker.k))
        if left >= needed:
            return False
        if not self._cost_said:
            self._cost_said = True
            why = f", ${left:.3g} remaining is below the ${needed:.3g} call estimate" if left > 0 else ""
            self._notice(
                f"max_cost_usd reached (${self.worker.cost:.2f} spent, cap ${self.max_cost:.2f}{why}); training continues "
                "with the current programs, no more agent calls."
            )
        return True

    def _submit(self, u, pause=False):
        """Ask a decision at update u. pause: training waits for it whatever the waiting mode (a non-finite loss)."""
        inbox = dict(
            update=u,
            losses=self._pending_losses,
            losses_omitted=self._pending_omitted,
            losses_from=self._pending_from,
            reports=list(self.reports),
            reports_made=self._reports_made,
            errors=list(self.errors),
            metrics=self._metrics_since(consume=True),
            skipped=list(self._skipped),
            timing=dict(self._timing, t=None),
            **({"validation": list(self._dev)} if self._validation else {}),
        )
        notes = self.run.new_notes()
        if self._resume_note:
            notes, self._resume_note = ((self._resume_note + (" " + notes if notes else "")), "")
        wait = self._auto.decide() if self._auto is not None else {}  # blocking="auto": wait at this decision?
        self._block_now = wait.get("wait") == "waited"
        left = self.max_decisions - (self.worker.k + 1)  # decisions that may still be asked after this one
        upcoming = self.schedule[self._sched_i :][: max(0, left)]
        if self.max_cost is not None:  # what the run may still spend, for a backend that can cap one call
            self.worker.backend.budget_left = max(0.0, self.max_cost - self.worker.cost)
        req = dict(
            update=u,
            total_updates=self.total,
            blocking=self.blocking or self._block_now or pause,
            reasons=_reasons_shown(self._reasons),
            last_applied=self._last_applied,
            previous=self._last_answer,
            files=self.files,
            inbox=inbox,
            timing=self._timing,
            notes=notes,
            scheduled_left=len(upcoming),
            **({"last": True} if left <= 0 else {}),
            **({"next_decision": next(iter(upcoming), None)} if self._show_next else {}),
            **({"neutral": True} if self._neutral else {}),
            **({"aggressive": True} if self._aggressive else {}),
            **({"epochs": self._run_facts["epochs"]} if isinstance(self._run_facts["epochs"], (int, float)) else {}),
            **({"validation": list(self._dev)} if self._validation else {}),
            shared_code=self._share_code,
            script_shared=getattr(self, "_script_copied", False),
            cost_so_far=self.worker.cost,
        )
        k = self.worker.submit(req)
        self.run.event("requested", k=k, update=u, reasons=req["reasons"], **wait)
        if getattr(self.worker.backend, "name", None) != "keep":  # keep answers at once: no line to wait on
            then = "training waits for the answer" if req["blocking"] else "training continues meanwhile"
            self._log(f"decision {k}: sent to the agent at update {u} ({then})")
        self.reports.clear()
        self.errors.clear()
        self._reasons, self._last_applied, self._last_answer = [], None, None
        self._pending_losses, self._pending_from = [], u + 1
        self._pending_omitted = 0
        self._skipped, self._reports_made = [], 0

    def _blocking_point(self, u):
        """Training waits here (all ranks) for the decision in flight, then applies it before update u's step."""
        busy = self._from_rank0(self.rank0 and self.worker is not None and self.worker.busy)
        if not busy:
            return
        resp = None
        if self.rank0:
            t0 = time.perf_counter()
            self.worker.wait()
            resp = self.worker.take()
            self._paused_s += time.perf_counter() - t0
            self.agent_wait_seconds += time.perf_counter() - t0
            if self._auto is not None:
                self._auto.paused(time.perf_counter() - t0)
        if self.dist:
            box = [resp]
            dist.broadcast_object_list(box, src=0, group=self.ctrl)
            resp = box[0]
        self._apply(resp, u)

    def _maybe_swap(self, u):
        """Non-blocking swaps happen at update boundaries. One process: as soon as the answer is there. DDP: rank 0
        announces readiness in an asynchronous broadcast every sync_every updates; every rank reads the announcement
        one interval later, so no rank waits for the host or the network in between."""
        if not self.dist:
            if self.worker is not None and self.worker.ready is not None:
                self._apply(self.worker.take(), u)
            return
        if u % self.sync_every:
            return
        if self._flag_work is not None:
            self._flag_work.wait()
            if int(self._flag_buf.item()):
                box = [self.worker.take() if self.rank0 and self.worker is not None else None]
                dist.broadcast_object_list(box, src=0, group=self.ctrl)
                self._apply(box[0], u)
        ready = self.rank0 and self.worker is not None and self.worker.ready is not None
        self._flag_buf = torch.tensor([1 if ready else 0])
        self._flag_work = dist.broadcast(self._flag_buf, src=0, group=self.ctrl, async_op=True)

    # ----- user-facing extras -------------------------------------------------------------------------------------

    def log(self, values=None, *, show_agent=False, **kw):
        """Record values (validation loss, accuracy, ...) in metrics.jsonl and the report: opt.log(test_acc=a), or a
        dict as in wandb, opt.log({"test_acc": a}). The agent does not see them
        unless show_agent=True: then its next analysis call sees them in obs.metrics and its next decision message lists
        them, and it may tune the run to them (show only a split you accept that for, never the test set). Scalars only
        (numbers or one-element tensors); tensors are read on the host at the next check step, not here. Only rank 0's
        values are kept."""
        if not self.rank0:
            return
        if values is not None and not isinstance(values, dict):
            raise TypeError("opt.log takes a dict of values or keyword values, e.g. opt.log(test_acc=a)")
        values = dict(values or {}, **kw)
        rec = dict(update=self.update)
        for k, v in values.items():
            if torch.is_tensor(v):
                if v.numel() != 1:
                    self._warn(f"log-{k}", f"opt.log: {k} has {v.numel()} elements; only scalars are kept")
                    continue
                rec[k] = v.detach()
            elif isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool):
                rec[k] = int(v) if isinstance(v, (int, np.integer)) else float(v)
            else:
                self._warn(f"log-{k}", f"opt.log: {k} is {type(v).__name__}; only numbers are kept")
        if len(rec) == 1:
            self._warn("log-empty", "opt.log was called with no number to keep; nothing was recorded")
            return
        if self._closed:  # e.g. the final evaluation after the last update: keep it in the records and the report
            self.run.append("metrics.jsonl", self._plain(rec))
            self._live.refresh()
            return
        self._metrics_new.append(rec)
        if show_agent:
            self.metrics.append(rec)
            self._shown_new.append(rec)
            if len(self.metrics) > 1000:
                self.metrics = self.metrics[-1000:]

    def report_validation(self, loss=None, accuracy=None, update=None):
        """OPAL 0.1.1-validation: report one evaluation on the validation set (held out from training; never the
        test set the run is judged by). With wrap(validation=N) the agent sees it in obs.metrics (as val_loss and
        val_accuracy) and, with the history of earlier measurements, in each decision message; it is also kept in
        metrics.jsonl. Without validation= it is only recorded. update defaults to the number of updates applied so
        far. Values are read on the host here (a validation evaluation synchronises anyway). Rank 0 only.

        Call it right after evaluating the parameters as they are after the latest update: with eval_integrity on,
        it opens the check of that evaluation (misc/integrity.py)."""
        if not self.rank0:
            return
        if self._integrity is not None and not self._closed:
            self._integrity.open(self.update, version=self.version)
        rec = dict(update=int(self.update if update is None else update))
        for key, v in (("loss", loss), ("accuracy", accuracy)):
            if v is not None:
                rec[key] = float(v.detach().float().mean()) if torch.is_tensor(v) else float(v)
        shown = dict(update=rec["update"], **{f"val_{k}": v for k, v in rec.items() if k != "update"})
        if not self._validation:
            self._warn("validation-off", "opt.report_validation without wrap(validation=N): recorded only; the agent does not see it")
            self.log(**{k: v for k, v in shown.items() if k != "update"})
            return
        if self._closed:
            self.run.append("metrics.jsonl", shown)
            return
        gaps = [b["update"] - a["update"] for a, b in zip(self._dev[-3:], self._dev[-2:] + [rec])]
        if len(gaps) == 3 and len(set(gaps)) == 1:
            self._warn(
                "validation-regular",
                f"opt.report_validation: the last three evaluations were {gaps[0]} updates apart; the agent is told the "
                f"script evaluates about every {self._validation} updates at times that are not announced, and it sees "
                "the update of each measurement, so regular times let it predict the next one (draw each time at "
                "random, as examples/overfit does)",
            )
        mean_gap = (rec["update"] - self._dev[0]["update"]) / len(self._dev) if self._dev else None
        if len(self._dev) >= 4 and not 0.5 <= mean_gap / self._validation <= 2:
            self._warn(
                "validation-interval",
                f"opt.report_validation: {len(self._dev) + 1} evaluations from update {self._dev[0]['update']} to "
                f"{rec['update']}; the agent is told the script evaluates about every {self._validation} updates "
                "(wrap(validation=N) is your promise of that: OPAL does not run the evaluation)",
            )
        self._dev.append(rec)
        self._metrics_new.append(dict(shown))
        self._shown_new.append(dict(shown))

    def report_dev(self, loss=None, accuracy=None, update=None):
        """The name of report_validation before 0.3.10; kept so that existing scripts keep working."""
        return self.report_validation(loss=loss, accuracy=accuracy, update=update)

    # ----- evaluation integrity (misc/integrity.py; rank 0; the agent is not told) ------------------------------

    def _integrity_record(self, u):
        """Sample the parameters after update u, and check the evaluations whose window is complete. A failure turns the
        check off, with a note to the person running the training, and never stops training."""
        try:
            self._integrity.record(u)
            if self._integrity.due(u):
                self._integrity_report(self._integrity.resolve(u))
        except Exception:
            self._integrity_failed()

    def _integrity_report(self, results):
        for t, meta, trips, jumps in results:
            if jumps:  # informational: a lone change that stayed (a weight average swapped in for good)
                self.run.event("eval_jump", update=t, **meta, n_tensors=len(jumps), tensors=_rounded(jumps[:20]))
            if not trips:
                continue
            self.run.event(
                "eval_integrity", update=t, flagged=True, **meta, n_tensors=len(trips), tensors=_rounded(trips[:20])
            )
            names = ", ".join(
                f"{r['name']} (norm x{r['norm_factor']:.3g}, changed at update {r['changed_at']:,}, back at "
                f"{r['restored_at']:,})"
                for r in trips[:4]
            )
            more = f" and {len(trips) - 4} more tensors" if len(trips) > 4 else ""
            print(
                f"[opal] evaluation integrity: for the validation evaluation after update {t:,}, the program "
                f"(v{meta.get('version', self.version):03d}) moved {names}{more} to another point and moved them back "
                "after the evaluation. The run is flagged (eval_integrity in events.jsonl, report.html); read "
                "programs/ before trusting its validation or test results.",
                file=sys.stderr,
                flush=True,
            )

    def _integrity_failed(self):
        err = traceback.format_exc(limit=3)
        self._integrity = None
        if self.rank0:
            self.run.event("eval_integrity_off", update=self.update, error=err[-1500:])
        print(
            "[opal] the evaluation integrity check failed and is off for the rest of this run: "
            + (err.strip().splitlines() or ["?"])[-1],
            file=sys.stderr,
            flush=True,
        )

    @staticmethod
    def _plain(m):
        return {k: (float(v) if torch.is_tensor(v) else v) for k, v in m.items()}

    def _metrics_since(self, consume=False):
        out = [self._plain(m) for m in self.metrics]
        if consume:
            self.metrics = []
        return out

    def zero_grad(self, set_to_none=True):
        self._check_user_parameters()
        for _, p in self.named:
            if p.grad is not None:
                if set_to_none:
                    p.grad = None
                else:
                    if p.grad.grad_fn is not None:
                        p.grad.detach_()
                    else:
                        p.grad.requires_grad_(False)
                    p.grad.zero_()

    def add_param_group(self, param_group):
        if getattr(self, "_initialised", False):
            raise NotImplementedError("opal: create all parameter groups before opal.wrap")
        super().add_param_group(param_group)

    def close(self, interrupted=False):
        """End of the run: called at the last update, by you after early stopping, and at interpreter exit (where a run
        that stopped before total_steps counts as interrupted: a mailbox broker then keeps waiting for the resumed
        run). Marks the final program and stops a decision in flight."""
        if self._closed:
            return
        self._closed = True
        self._rank_snapshot = None  # early stopping can end a trial before its usual release point
        if not self.rank0:
            shutil.rmtree(self._prog_root, ignore_errors=True)
            return
        for m in self._metrics_new:
            self.run.append("metrics.jsonl", self._plain(m))
        self._metrics_new = []
        be = self.worker.backend if self.worker else None
        if self.worker is not None and self.worker.busy and be is not None:
            be.cancel()  # its answer could no longer be applied
            self.run.event("cancelled", k=self.worker.k)
            if self.worker.thread is not None:
                self.worker.thread.join(timeout=1.0)  # bounded settlement; slower answers record cost_settled
        settling = self.worker is not None and self.worker.thread is not None and self.worker.thread.is_alive()
        checked = None
        if self._integrity is not None:  # the evaluations of the last updates, with what is left of their window
            try:
                self._integrity_report(self._integrity.resolve(self.update, final=True))
                checked = dict(checked=self._integrity.checks, flagged=self._integrity.flags)
            except Exception:
                self._integrity_failed()
        self.run.event(
            "interrupted" if interrupted else "end",
            update=self.update,
            version=self.version,
            decisions=self.worker.k if self.worker else 0,
            cost_usd=self.worker.recorded_cost if self.worker else 0,
            unknown_cost_decisions=sorted(self.worker.unknown_cost_decisions) if self.worker else [],
            agent_wait_seconds=round(self.agent_wait_seconds, 1),  # the report's "training waiting time"
            **(dict(pending_decision=self.worker.k) if settling else {}),
            **(dict(usage=self.worker.usage) if self.worker and self.worker.usage is not None else {}),
            **(dict(eval_integrity=checked) if checked else {}),
        )
        if be is not None:
            be.finish(interrupted=interrupted)
        self._live.finish(timeout=30.0)  # the final report is on disk when close() returns
        self._log(
            f"{'stopped' if interrupted else 'done'} at update {self.update}: program v{self.version:03d} in "
            f"{self.run.dir / 'programs' / f'v{self.version:03d}'}"
        )

    def _log(self, msg):
        if self.verbose:
            print(f"[opal] {redact_secrets(msg)}", file=sys.stderr, flush=True)

    def _notice(self, msg):
        """A line the user should not miss (the agent will not be asked again): on stdout."""
        if self.verbose:
            print(f"[opal] {msg}", flush=True)

    def _warn(self, key, msg):
        if key not in self._warned:
            self._warned.add(key)
            self._log(msg)
