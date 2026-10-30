"""Setup done once per run: the system prompt, the user's schedulers, activation hooks."""

from __future__ import annotations

import re
import weakref
from pathlib import Path

import torch

from .. import context, programs
from . import aggressive, neutral_analysis, taskfacts
from .schedulers import OWNER, follow_class
from .skip import MAX_SKIPS_IN_A_ROW, user_optimizer_pre_step

PROMPT_DIR = Path(__file__).parent.parent / "prompts"
PROMPT = PROMPT_DIR / "neutral.md"  # the default prompt
# the prompt names; any other prompt is the path of a template file. "aggressive" (OPAL 0.1.2-aggressive): the
# neutral prompt with the authors' edits of misc/aggressive.py
CODE_PROMPTS = ("neutral", "aggressive")
# names of earlier releases, no longer shipped
REMOVED = ("legacy", "default", "v1")


def prompt_template(prompt) -> str:
    """The template text of a prompt: "neutral" (or None), "aggressive", or the path of a template file whose {fields}
    are filled as in prompts/neutral.md. Raises ValueError for a removed name or a path that is not a file."""
    if prompt is None or prompt == "neutral":
        return PROMPT.read_text()
    if prompt == "aggressive":
        return aggressive.template(PROMPT.read_text())
    if prompt in REMOVED:
        raise ValueError(
            f'opal: prompt={prompt!r} was removed; prompt is "neutral" (the default), "aggressive" or the path of a '
            "template file"
        )
    path = Path(prompt)
    if not path.is_file():
        raise ValueError(
            f'opal: prompt must be "neutral" (the default), "aggressive" or the path of a template file; {prompt!r} '
            "is neither a name nor a file"
        )
    return path.read_text()


def resolve_prompt(prompt, validation=None):
    """The prompt a run uses: "neutral" when None. Raises for a prompt that is not a name or a file, and for
    validation=N with a template that does not state the validation set (it has no {validation_note})."""
    text = prompt_template(prompt)
    if validation is not None:
        if "{validation_note}" not in text:
            raise ValueError(
                "opal: validation=N shows the agent a validation set, but this prompt does not say so (it has no "
                "{validation_note} field): use the default neutral prompt, or a template with that field"
            )
    return prompt


NEUTRAL_REF = "reference/ holds read-only copies of this training run's code: the training script, the source files of its model\nclass and of any custom optimizer or scheduler (code that is not part of torch), and starting_optimizer.txt with the\nstarting optimizer's full settings. Read them closely before your first edit, and again whenever they would help: the\narchitecture (which layers exist, how they are initialised, which are tied or zero-initialised), the loss, the data\nand batching, and what the training loop does around the optimizer (gradient clipping, accumulation, weight averaging,\nevaluation) tell you what this optimization problem looks like, and where a generic optimizer may not fit it. Let what\nyou find there shape your design. These copies are for reading only; your programs reach the user's optimizer and\nscheduler objects through spec (below)."
NEUTRAL_REF_PRIVATE = "The person running this training chose not to share the training script or the model source: learn the architecture\nfrom the parameter names and shapes below and from the model summary. reference/starting_optimizer.txt holds the\nstarting optimizer's full settings. Your programs reach the user's optimizer and scheduler objects through spec\n(below)."
NO_FACTS = "Not given; learn what you need from reference/ (if it is shared) and the model summary below."
# the neutral prompt's sentences on the agent's shell, and their form when the backend gives it no shell
SHELL = (
    "Your programs do not run in your shell: they run inside\nthe training process, which reports import and runtime "
    "errors back to you. Your shell runs the training's Python with\ntorch, on CPU only, for checking your programs on "
    "small tensors (shapes, numerics, a few steps on random data). It\ncannot see the model, the data, the evaluation "
    "or the GPUs."
)
NO_SHELL = "Your programs run inside the training process, which reports import and runtime errors back to you."
# reference/ when the training script could not be copied (a notebook, or a script above 300 KB)
NEUTRAL_REF_NO_SCRIPT = "reference/ holds read-only copies of this training run's code: the source files of its model class and of any custom\noptimizer or scheduler (code that is not part of torch; the training script itself could not be copied), and\nstarting_optimizer.txt with the starting optimizer's full settings. Read them closely before your first edit, and again\nwhenever they would help: the architecture (which layers exist, how they are initialised, which are tied or\nzero-initialised) tells you what this optimization problem looks like, and where a generic optimizer may not fit it.\nLet what you find there shape your design. These copies are for reading only; your programs reach the user's optimizer\nand scheduler objects through spec (below)."


def fill_template(template: str, fields: dict) -> str:
    """Replace {name} for the known fields only, in one pass (so braces in code examples need no escaping, and a
    field's value is never itself searched for placeholders)."""
    pattern = re.compile(r"\{(" + "|".join(re.escape(k) for k in fields) + r")\}")
    return pattern.sub(lambda m: str(fields[m.group(1)]), template)


class SetupMixin:
    def _patch_schedulers(self):
        for sch in self.user_schedulers:
            if not hasattr(type(sch), "_opal_base"):
                sch.__class__ = follow_class(type(sch))
            OWNER[sch] = weakref.ref(self)
            if isinstance(sch, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self._warn(
                    "plateau-metric",
                    "ReduceLROnPlateau keeps the metric you pass to its step() (best, num_bad_epochs), and the agent's "
                    "programs can read it through spec['user_schedulers']: pass it only a value you would show the agent",
                )

    def _hook_user_optimizers(self):
        """A skipped update also leaves the training script's optimizers as they were (misc/skip.py)."""
        self._skip_capture = None
        ref = weakref.ref(self)
        self._opt_hooks = [o.register_step_pre_hook(user_optimizer_pre_step(ref)) for o in self.user_optimizers]

    def _watch(self, names):
        self._hooks = []
        if not names:
            return
        if hasattr(self.model, "_orig_mod") or hasattr(getattr(self.model, "module", None), "_orig_mod"):
            self._warn(
                "hooks-compiled", "activation capture on a torch.compile'd model adds a graph; prefer an eager model"
            )
        modules = dict(context.bare_model(self.model).named_modules())
        for name in names:
            if name not in modules:
                raise ValueError(f"opal: no module named {name!r} to watch")

            def hook(_m, _inp, out, _n=name):
                if self._capture_next and torch.is_tensor(out):
                    self.activations[_n] = out.detach()

            self._hooks.append(modules[name].register_forward_hook(hook))

    def _system_prompt(self, prompt, names_by_id):
        template = prompt_template(prompt)
        nan_pause = " A non-finite training loss pauses training until you answer." if self.pause_on_nonfinite else ""
        auto = (
            "Training pauses at a decision until you finish when your answer would otherwise take effect more than "
            f"{self._auto.max_lag:g} updates late, judged from the time per update and your recent answer times, and "
            "at the first decision, before they are measured; then your edits take effect from the update at which the "
            "decision was requested. Otherwise training continues while you work and your edits take effect at the "
            "first update boundary after you finish. Each message says which applies." + nan_pause
            if getattr(self, "_auto", None) is not None
            else None
        )
        mode = auto or (
            "Training pauses at each decision until you finish; your edits take effect from the update at which the "
            "decision was requested."
            if self.blocking
            else "Training does not pause while you work. Your edits take effect at the first update boundary after "
            f"you finish (within a few updates on one GPU, within {2 * self.sync_every} updates in data-parallel runs); "
            "the next message tells you the update and how many updates passed since the state you saw." + nan_pause
        )
        early = sum(1 for u in self.schedule if u <= max(1, int(self.total * 0.05)))
        placed = f"({early if early else 'none'} of them in the first 5% of training)"
        kind = getattr(self, "_schedule_kind", "front")
        if kind != "front" and self.schedule:
            at = ", ".join(f"{u:,}" for u in self.schedule[:3]) + (
                f", ... {self.schedule[-1]:,}" if len(self.schedule) > 3 else ""
            )
            placed = f"evenly spaced (at updates {at})" if kind == "uniform" else f"(at updates {at})"
        sched = (
            f"{len(self.schedule)} scheduled decisions {placed}, plus any your "
            f"analysis asks for by returning {{'wake': reason}} (at most {self.max_decisions} decisions in all). "
            f"Decisions, and the check for a non-finite loss, happen every {self.default_every} updates (and at your "
            "analysis steps), whatever EVERY you set."
        ) + (
            " Each message gives the update of the next scheduled decision."
            if getattr(self, "_show_next", False)
            else ""
        )
        full = programs.describe_user_objects(
            self.user_optimizers, self.user_schedulers, names_by_id, sources=self._share_code
        )
        clip = getattr(self, "_clip_grad_norm", None)
        if clip is not None:
            full += (
                f"\ngradient clipping: the starting program clips the gradients of all these parameters to a global norm "
                f"of {float(clip):g} before the optimizer step (CLIP_GRAD_NORM in updater.py)"
            )
        if "{starting_summary}" in template:  # the neutral draft: one line here, the full dump in reference/
            (self.run.workspace / "reference" / "starting_optimizer.txt").write_text(full + "\n")
            example = (PROMPT_DIR / "example_updater.py").read_text()  # structure only, no values of ours
            if getattr(self, "_aggressive", False):  # with a feedback controller of the learning rates, off
                example = aggressive.example(example)
            (self.run.workspace / "reference" / "example_updater.py").write_text(example)
        groups = context.group_index(self.user_optimizers, self.named)
        table = context.param_table(self.named, groups)
        if len(self.named) > 300:
            (self.run.workspace / "reference" / "parameters.txt").write_text(table + "\n")
            table = (
                f"{len(self.named)} tensors; the full table (name, shape, dtype, group) is in reference/parameters.txt. "
                "First 40:\n" + "\n".join(table.splitlines()[:40])
            )
        fields = dict(
            goal=self.goal,
            model_summary=context.model_summary(self.model, self.named, self.world, self.device),
            param_table=table,
            starting_program=full,
            starting_summary=taskfacts.starting_summary(
                self.user_optimizers, self.user_schedulers, clip=clip, shared=self._share_code
            ),
            neutral_reference_note=(
                (NEUTRAL_REF if getattr(self, "_script_copied", True) else NEUTRAL_REF_NO_SCRIPT)
                if self._share_code
                else NEUTRAL_REF_PRIVATE
            ),
            shell_note=NO_SHELL if getattr(self, "_agent_shell", True) is False else SHELL,
            analysis_note=neutral_analysis.note(getattr(self, "_neutral_epochs", None)),
            task_facts_section=(
                "TASK FACTS\nFrom the person running the training:\n" + str(self._task_facts).strip() + "\n\n"
                if getattr(self, "_task_facts", None)
                else ""
            ),
            time_note=taskfacts.time_limit(getattr(self, "_decision_timeout", None)),
            script_note=(
                ", and the training script itself is in reference/"
                if self._share_code and getattr(self, "_script_copied", True)
                else ""
            ),
            total_updates=f"{self.total:,}",
            every=self.default_every,
            schedule_note=sched,
            mode_note=mode,
            trial=self.trial_steps,
            max_skips=MAX_SKIPS_IN_A_ROW,
            task_facts=str(getattr(self, "_task_facts", None) or NO_FACTS).strip(),
            run_facts=taskfacts.run_facts(self.total, **getattr(self, "_run_facts", {})),
            validation_note=taskfacts.validation_note(
                getattr(self, "_validation", None), getattr(self, "_validation_size", None)
            ),
            evaluation_note=taskfacts.evaluation_note(getattr(self, "_validation", None)),
            decisions_note=taskfacts.decisions(
                self.schedule,
                getattr(self, "_schedule_kind", "front"),
                self.total,
                self.blocking,
                self._auto.max_lag if getattr(self, "_auto", None) is not None else None,
                getattr(self, "_show_next", False),
                getattr(self, "_run_facts", {}).get("epochs"),
                segments=getattr(self, "_schedule_segments", None),
            ),
            protocol_note=taskfacts.protocol_note(
                self.default_every,
                self.max_decisions,
                self.blocking,
                self.sync_every,
                self.pause_on_nonfinite,
            ),
            reference_note=(
                (
                    "reference/ holds read-only copies of the training script and of user code that is not part of "
                    "torch (the model class, a custom optimizer or scheduler). Read them to learn the loss, the data, "
                    "what the loop does around the optimizer, and the architecture."
                    if getattr(self, "_script_copied", True)
                    else "reference/ holds read-only copies of user code that is not part of torch (the model class, a "
                    "custom optimizer or scheduler; the training script itself could not be copied). Read them to learn "
                    "the architecture."
                )
                + " They are copies for reading: your programs reach the user's optimizer and scheduler objects "
                "through spec (below)."
                if self._share_code
                else "The person running this training chose not to share the training script or the model source: "
                "learn the architecture from the parameter names and shapes below and from the model summary."
                " Your programs reach the user's optimizer and scheduler objects through spec (below)."
            ),
        )
        return fill_template(template, fields)
