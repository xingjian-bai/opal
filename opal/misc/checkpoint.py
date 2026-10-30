"""Checkpoints: state_dict / load_state_dict, resuming into the same run folder, and the fallback when a program
crashed after its trial window in an earlier attempt."""

from __future__ import annotations

import bisect
import copy
import io
import json
import shutil
import traceback
from pathlib import Path

import torch
import torch.distributed as dist

from .. import programs
from ..agent import Run, now_stamp
from .nested import devices, restore_devices
from .schedulers import follows


_OBSERVATION_LISTS = (
    "_float_losses", "_pending_losses", "_analysis_losses", "_analysis_metrics", "_metrics_new",
    "_shown_new", "metrics", "reports", "errors", "_reasons", "_skipped",
)


def _empty_observations(update):
    return dict(
        **{name: [] for name in _OBSERVATION_LISTS},
        _losses_from=update + 1, _pending_from=update + 1, _pending_omitted=0, _reports_made=0,
        _nonfinite_episode=False, _skips_in_a_row=0, _skip_alarm=False,
        _last_applied=None, _last_answer=None, _observed_loss=None, activations={},
    )


class CheckpointMixin:
    def state_dict(self):
        """Everything needed to resume, including the programs and the agent's session. Never raises because of the
        agent's state: if the updater's state cannot be saved (its state_dict() raises, or holds what a new process
        could not load), the checkpoint leaves it out (the program then restarts with fresh state on resume) and the
        agent is told."""
        up, bad = None, []
        try:
            with programs.keep_param_group_identity():
                up = self.updater.state_dict()
        except programs.AGENT_ERRORS:
            if self.digests[0] == self._starting_digest:
                raise  # the training script's own optimizers, as in the plain run
            tb = traceback.format_exc(limit=6)
            self.errors.append(
                dict(
                    update=self.update,
                    where="checkpointing your updater state",
                    error="state_dict() raised; checkpoints leave it out until you fix it:\n" + tb[-3000:],
                )
            )
            last = tb.strip().splitlines()[-1]
            self._warn("state-raised", f"the updater's state_dict() raised ({last}); checkpoints leave it out")
            self._state_dropped("updater", last)
        else:
            if self.digests[0] != self._starting_digest:
                up = programs.normalize_state(up)
            bad = programs.unpicklable(up) if self.digests[0] != self._starting_digest else []
        if bad:
            msg = "the updater's state_dict() holds objects a new process could not load (" + "; ".join(bad) + ")"
            self.errors.append(
                dict(
                    update=self.update,
                    where="checkpointing your updater state",
                    error=msg + "; checkpoints leave it out until you fix it",
                )
            )
            self._warn("state-nonplain", msg + "; checkpoints leave it out")
            self._state_dropped("updater", msg)
            up = None
        mem = io.BytesIO()
        try:
            torch.save(self.memory, mem)
        except Exception as exc:
            mem = io.BytesIO()
            torch.save({}, mem)
            self.errors.append(
                dict(
                    update=self.update,
                    where="checkpointing analysis memory",
                    error=f"memory could not be saved and is left out of checkpoints: {exc}",
                )
            )
            self._warn("memory-pickle", f"analysis memory could not be pickled ({exc}); checkpoints leave it out")
            self._state_dropped("memory", str(exc))
        observations = self._observation_state()
        try:
            event_offset = (self.run.dir / "events.jsonl").stat().st_size if self.run is not None else None
        except OSError:
            event_offset = None
        return dict(
            opal=1,
            observations=observations,
            update=self.update,
            version=self.version,
            files=self.files,
            prev_files=self._prev_files,
            updater=up,
            devices=devices(up),
            memory=mem.getvalue(),
            decisions=self.worker.k if self.worker else 0,
            cost=self.worker.cost if self.worker else 0.0,
            unknown_cost_decisions=sorted(self.worker.unknown_cost_decisions) if self.worker else [],
            **(dict(usage=dict(self.worker.usage)) if self.worker and self.worker.usage is not None else {}),
            backend=self.worker.backend.state() if self.worker else {},
            notes_seen=self.run._notes_seen if self.run is not None else 0,
            dev=list(getattr(self, "_dev", [])),  # validation=N: the validation measurements so far
            run_dir=str(self.run.dir) if self.run is not None else None,
            event_offset=event_offset,
            **(dict(ranks_repaired_at=list(self._ranks_repaired_at)) if self._ranks_repaired_at else {}),
            # the training script's own optimizers and schedulers, whatever the program's state_dict() holds, and the
            # user's schedule while a program owns it (misc/schedulers.py)
            user_optimizers=[o.state_dict() for o in self.user_optimizers],
            user_schedulers=[s.state_dict() for s in self.user_schedulers],
            ledger=self._ledger,
            follows=follows(self.updater),
            next_analysis=self._next_analysis,
            # the decisions the first check scheduled when decisions was not set (opal/estimate.py), and whether it ran
            decision_schedule=self._schedule_segments if getattr(self, "_schedule_changed", False) else None,
            estimated=getattr(self, "_estimated", False),
        )

    def _observation_state(self):
        values = {name: getattr(self, name) for name in _empty_observations(self.update)}
        values["reports"], values["errors"] = list(self.reports), list(self.errors)
        if torch.is_tensor(self._observed_loss):
            values["_observed_loss"] = self._observed_loss.detach().clone()
        values["loss_buffer"] = self._loss_buf[:self._loss_n].detach().clone()
        defaults = dict(_empty_observations(self.update), loss_buffer=None)
        for name, value in values.items():
            value = programs.normalize_state(value)
            bad = programs.unpicklable(value)
            if bad:
                self._state_dropped("observations." + name, "; ".join(bad))
                value = defaults[name]
            values[name] = value
        return copy.deepcopy(values)

    def _state_dropped(self, field, reason):
        if self.run is not None:
            self.run.event("state_dropped", update=self.update, version=self.version, field=field, reason=reason)
            self._live.refresh()

    def _continue_run_dir(self, old):
        """On resume, keep writing into the checkpoint's run folder when this run's folder was auto-named and that folder
        is the same run interrupted: it lies in this run's opal_runs/ (under the current working directory) and its run
        did not finish. Otherwise (a folder named with run_dir=, a checkpoint copied from another place, or one of a
        run that finished) this run writes to its own folder and never touches the checkpoint's."""
        if not self.rank0 or not old or not Path(old).is_dir() or Path(old).resolve() == self.run.dir:
            return
        why = (
            "as you asked (run_dir=)"
            if not self._auto_run_dir
            else _not_the_same_run(Path(old), self.run.dir.parent)
        )
        if why:
            self._log(f"the checkpoint's run folder was {old}; this run writes to {self.run.dir}, {why}")
            # the agent's inbox holds every input so far: bring the earlier ones (the lost attempt's are archived later)
            for f in sorted((Path(old) / "workspace" / "inbox").glob("decision-[0-9][0-9][0-9]*.json")):
                dest = self.run.workspace / "inbox" / f.name
                if not dest.exists() and not f.is_symlink():
                    shutil.copy2(f, dest)
            return
        new = self.run.dir
        self.run = Run(old)
        self._prog_root = self.run.programs
        self._run_path = self.run.dir  # crash markers go to the folder the run writes to
        self._live.dir = self.run.dir
        if self.worker is not None:
            self.worker.run = self.run
        shutil.rmtree(new, ignore_errors=True)

    def _archive_lost_attempt(self, version, decisions, update, event_offset=None):
        """Records written after the checkpoint by the attempt that was interrupted move to attempts/<time>/, so the
        resumed run neither overwrites them nor mixes them into its curves."""
        dest = self.run.dir / "attempts" / now_stamp()
        moved = 0
        for d in list(self.run.decisions.glob("[0-9][0-9][0-9]*")):
            if d.name.isdigit() and int(d.name) > decisions:
                (dest / "decisions").mkdir(parents=True, exist_ok=True)
                shutil.move(str(d), str(dest / "decisions" / d.name))
                moved += 1
        for d in list(self.run.programs.glob("v[0-9][0-9][0-9]*")):
            if d.name[1:].isdigit() and int(d.name[1:]) > version:
                (dest / "programs").mkdir(parents=True, exist_ok=True)
                shutil.move(str(d), str(dest / "programs" / d.name))
                moved += 1
        for f in list((self.run.workspace / "inbox").glob("decision-[0-9][0-9][0-9]*.json")):
            try:
                k = int(f.stem.split("-")[1])
            except ValueError:
                continue
            if k > decisions:  # the agent reads inbox/ as every input so far: not the lost attempt's
                (dest / "inbox").mkdir(parents=True, exist_ok=True)
                shutil.move(str(f), str(dest / "inbox" / f.name))
                moved += 1
        for name, key in (("losses.jsonl", "to_update"), ("metrics.jsonl", "update"), ("reports.jsonl", "update")):
            f = self.run.dir / name
            if not f.exists():
                continue
            keep, later = [], []
            for line in f.read_text().splitlines():
                try:
                    (later if json.loads(line).get(key, 0) > update else keep).append(line)
                except json.JSONDecodeError:
                    keep.append(line)
            if later:
                dest.mkdir(parents=True, exist_ok=True)
                (dest / name).write_text("\n".join(later) + "\n")
                f.write_text("".join(x + "\n" for x in keep))
                moved += 1
        f = self.run.dir / "events.jsonl"
        if f.exists():
            data = f.read_bytes()
            if event_offset is not None and 0 <= event_offset <= len(data):
                keep, later = data[:event_offset], data[event_offset:]
            else:  # checkpoints from before the exact event offset was recorded
                keep, later = [], []
                for line in data.splitlines(keepends=True):
                    try:
                        e = json.loads(line)
                        lost = (e.get("update", e.get("applied_at", 0)) > update or e.get("k", 0) > decisions
                                or e.get("version", 0) > version or e.get("kind") in ("end", "interrupted"))
                    except (ValueError, TypeError):
                        lost = False
                    (later if lost else keep).append(line)
                keep, later = b"".join(keep), b"".join(later)
            if later:
                dest.mkdir(parents=True, exist_ok=True)
                (dest / "events.jsonl").write_bytes(later)
                f.write_bytes(keep)
                moved += 1
        return dest if moved else None

    def _finish_resume(self):
        """At the first step after load_state_dict, when the training script has also loaded its own scheduler states:
        a crashed program that was replaced hands the user's schedule over as at a swap, and the trial window that a
        resume opens gets its record of the schedule."""
        pending, self._resume_pending = self._resume_pending, None
        if pending.get("crashed"):
            self._schedule_handover(None, self.updater, self._schedule_state(), None)
        self._new_install_record()

    def _crashed_digests(self, *folders):
        out = set()
        for folder in folders:
            if folder and Path(folder).is_dir():
                for f in Path(folder).glob("crashed-*-r*.json"):
                    out.add(f.name.split("-")[1])
        return out

    def load_state_dict(self, state):
        """Resume: call after wrap(...) with the same arguments, on every rank."""
        if state.get("opal") != 1:
            raise ValueError("opal: not an OPAL optimizer state")
        self._continue_run_dir(state.get("run_dir"))
        if self.dist:  # every rank writes its crash marker to the folder rank 0 writes to
            box = [str(self.run.dir) if self.rank0 else None]
            dist.broadcast_object_list(box, src=0, group=self.ctrl)
            self._run_path = Path(box[0])
        files, prev = state["files"], state.get("prev_files")
        # programs that crashed after their trial window (markers left by any rank of any earlier attempt, in this
        # run folder or the checkpoint's) are not resumed: fall back to the predecessor, or to the starting program
        choice = None
        if self.rank0:
            crashed = self._crashed_digests(self.run.dir, state.get("run_dir"))
            digest = lambda fs: programs.side_digests(fs)[0] if fs else None  # noqa: E731
            if digest(files) in crashed:
                choice = prev if prev and digest(prev) not in crashed else self._starting_files
        if self.dist:
            box = [choice]
            dist.broadcast_object_list(box, src=0, group=self.ctrl)
            choice = box[0]
        crashed = choice is not None
        if crashed:
            self.errors.append(
                dict(
                    update=state["update"],
                    where=f"program v{state['version']:03d} (crashed after "
                    "this checkpoint in an earlier attempt; the run resumed with an earlier program)",
                    error="see crashed-*.json in the run folder",
                )
            )
        if self.rank0:
            same_folder = state.get("run_dir") and Path(state["run_dir"]).resolve() == self.run.dir
            lost = self._archive_lost_attempt(state["version"], state.get("decisions", 0), state["update"],
                                              state.get("event_offset") if same_folder else None)
            if lost:
                self._log(f"records of the interrupted attempt after the checkpoint moved to {lost}")
        self.update = state["update"]
        self._ranks_repaired_at = list(state.get("ranks_repaired_at", []))
        if getattr(self, "_integrity", None) is not None:  # misc/integrity.py: the sampled history is not this run's
            self._integrity.reset()
        self.version = state["version"] - (0 if crashed else 1)
        # Existing checkpoints may contain values outside today's new-program allowlist. The checkpoint
        # loader already reconstructed them; do not reject a legacy program's initial bookkeeping state.
        err = self._install(choice if crashed else files, previous=None, handover=False, restoring=True)
        if err:
            raise RuntimeError(f"opal: the checkpointed programs failed to load:\n{err}")
        with programs.keep_param_group_identity():  # before the program's own state, which may hold them too
            for o, s in zip(self.user_optimizers, state.get("user_optimizers") or []):
                o.load_state_dict(s)
        for s, st in zip(self.user_schedulers, state.get("user_schedulers") or []):
            s.load_state_dict(st)
        self._ledger = state.get("ledger")
        if state.get("updater") is not None and not crashed:
            with programs.keep_param_group_identity():
                self.updater.load_state_dict(restore_devices(state["updater"], state.get("devices"), self.device))
        elif not crashed:
            self._warn(
                "state-missing",
                "the checkpoint has no updater state (it could not be saved); the program restarts its state",
            )
        self.memory = torch.load(
            io.BytesIO(state["memory"]), weights_only=False,
            map_location=lambda storage, location: storage if location == "cpu" else storage.to(device=self.device),
        )
        self._dev = [d for d in state.get("dev", []) if d.get("update", 0) <= self.update]
        if state.get("estimated"):  # the first check of the checkpointed run measured it (and may have capped it)
            self._estimate_due, self._estimated = False, True
        if state.get("decision_schedule") is not None and self.rank0:
            self._set_schedule([list(seg) for seg in state["decision_schedule"]])
        self._sched_i = bisect.bisect_right(self.schedule, self.update)
        observations = _empty_observations(self.update)
        observations.update(copy.deepcopy(state.get("observations", {})))
        loss_buffer = observations.pop("loss_buffer", None)
        self._loss_n = 0 if loss_buffer is None else loss_buffer.numel()
        self._loss_buf = torch.zeros(max(self.default_every, self._loss_n), dtype=torch.float32,
                                     device=self.device)
        if self._loss_n:
            self._loss_buf[:self._loss_n].copy_(loss_buffer)
        # Restore only known fields, keeping the report/error bounds and clearing legacy buffers.
        for name in _empty_observations(self.update):
            if name in ("reports", "errors"):
                target = getattr(self, name)
                target.clear()
                target.extend(observations[name])
            else:
                setattr(self, name, observations[name])
        self._next_analysis = state.get("next_analysis") or self.update + self.every  # the analysis keeps its grid
        self._audit_at = self.update + 2
        # A checkpoint has no pre-swap rollback copy or evidence that ranks agreed before its updater was built.
        self._rank_snapshot, self._rank_trial_equal, self._rank_checked_at = None, False, None
        # a new trial window (decided by rank 0): if the restored program raises soon, every rank goes back together
        if self._from_rank0(self.files != self._starting_files):
            fallback = self._starting_files if crashed else (prev or self._starting_files)
            self._prev_files, self._prev_updater = fallback, None
            self._trial_until, self._snapshot, self._first_pending = self.update + self.trial_steps, None, False
        self._forget_retired()
        # finished at the first step, once the training script has loaded its own scheduler states too
        self._resume_pending = dict(crashed=crashed)
        if self.worker:
            self.worker.k, self.worker.cost = state.get("decisions", 0), state.get("cost", 0.0)
            self.worker.unknown_cost_decisions = set(state.get("unknown_cost_decisions", []))
            self.worker.usage = dict(state["usage"]) if state.get("usage") is not None else None
            self.worker.backend.restore(state.get("backend", {}))
            self.worker.backend.setup(self.run.dir, self.run.workspace)
        if self.run is not None:
            self.run._notes_seen = state.get("notes_seen", 0)
            self._resume_note = (
                f"The run was resumed from a checkpoint at update {self.update:,}; anything you decided after that "
                "point in the interrupted attempt was lost, and the programs are the ones running at the checkpoint"
                + (" (except that a program which crashed after it was replaced by an earlier one)" if crashed else "")
                + "."
            )
            self.run.event(
                "resumed", update=self.update, version=self.version, crashed_program_replaced=crashed,
                schedule=self.schedule,
                **(dict(ranks_repaired_at=self._ranks_repaired_at) if self._ranks_repaired_at else {}),
            )


def _not_the_same_run(old: Path, runs: Path):
    """Why a checkpoint's run folder is not this run's to continue (None if it is): it lies outside this run's
    opal_runs/ (a checkpoint copied from another place), or its run finished (its last start, resume or end is an
    end), so a resume from it is a new run."""
    try:
        if old.resolve().parent != Path(runs).resolve():
            return f"since that folder is not in {runs} (a checkpoint copied from another place)"
        last = None
        for line in (old / "events.jsonl").read_text().splitlines():
            try:
                kind = json.loads(line).get("kind")
            except json.JSONDecodeError:
                continue
            if kind in ("start", "resumed", "end", "interrupted"):
                last = kind
        if last == "end":
            return "since the run of that folder finished (a resume from its checkpoint is a new run)"
    except OSError:
        return "since that folder could not be read"
    return None
