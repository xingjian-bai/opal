---
name: train-with-agent
description: Put the OPAL agent into this repository's PyTorch training script, so that a coding agent can rewrite the optimizer while the model trains. Use when the user asks to "train with an agent", "add OPAL", or similar.
---

# Add OPAL to a training script

Goal: the user's training run, unchanged except that its optimizer is wrapped by `opal.wrap(...)`. Work in small
steps and show the user the diff before running anything that costs GPU time.

**Do not edit any file inside the OPAL repository** (its prompt, starting programs or code): configure everything
through `opal.wrap` arguments (`prompt=`, `program=`, `task_facts=`, ...). An edited install changes every run without
a word, and its results no longer match the blog's; OPAL warns when it finds one.

The agent's programs run inside your training process with its permissions and are not sandboxed; eval_integrity detects some evaluation-time parameter moves and boundary flags some file I/O and introspection, but neither prevents a program from reading what your process can read.
Keep test data and labels out of the training process, evaluate the final checkpoint in a separate process, and review boundary flags in report.html and events.jsonl (see docs/GUIDE.md, "What OPAL does not protect against").

1. **Find the loop.** Locate the training script and the place where the optimizer and LR scheduler are built and
   where `optimizer.step()` is called. Note: number of optimizer steps in the run (`total_steps`), data-parallel setup
   (single GPU, DDP, Accelerate, HF Trainer, Lightning, FSDP/DeepSpeed), gradient clipping in the loop, GradScaler,
   gradient accumulation, EMA, checkpoint save/load, and whether parameter groups are added during training (for
   example layers unfrozen later).
2. **Check it is supported.** FSDP, ZeRO/DeepSpeed, tensor or pipeline parallelism and JAX are not supported: stop
   and tell the user. With an HF Trainer, use `opal.hf.OpalTrainer` (docs/GUIDE.md, "Hugging Face Trainer").
   With Accelerate, wrap after `accelerator.prepare` (GUIDE, "Accelerate"). Lightning is untested: say so before
   continuing. All parameter groups must exist before `wrap`: for layers unfrozen later, create their group up front
   with `requires_grad=False` and set it to True when they are unfrozen. With DDP, every rank must run the same number
   of steps; uneven inputs under `Join` stop with an error.
3. **Write the goal sentence with the user**: what the run trains and what it is judged by at the end (for example
   "ResNet-18 on CIFAR-10, judged by test accuracy after 30 epochs"). Ask; do not invent the metric.
4. **Wrap.** Right after the optimizer and scheduler are created:
   ```python
   import opal
   optimizer = opal.wrap(optimizer, model, scheduler=scheduler, total_steps=TOTAL_STEPS, goal=GOAL)
   ```
   The agent's prompt states the run's facts that you pass: `epochs=`, `train_size=`, `batch_size=`,
   `same_order_each_epoch=` and free text in `task_facts=`; ask the user for them rather than guessing. Pass several
   optimizers or schedulers as lists. Replace `optimizer.step()` with `optimizer.step(loss=loss)` (with a
   GradScaler: `scaler.step(optimizer, loss=loss)`). Keep `scheduler.step()` calls as they are. If the script logs a
   validation metric, add `optimizer.log(val_loss=...)` next to it: it goes into the run's report and is not shown
   to the agent. Add `show_agent=True` only if the user asks for it on a split the agent may tune on, never the test
   set. Checkpoints: save and load `optimizer.state_dict()` as before; it now also holds the agent's programs.
5. **Gradient clipping goes to OPAL.** Clipping is part of the optimizer, so if the loop clips gradients, delete the
   loop's `clip_grad_norm_(model.parameters(), x)` and pass `clip_grad_norm=x` to `wrap`: the starting program clips
   the same way (the smoke test below must still match bit for bit) and the agent may change it. Tell the user you
   did this. Keep the clipping in the loop only if the user asks, or if it clips other parameters than the optimizer
   holds (then the bitwise check would fail); with the HF Trainer see the GUIDE.
6. **Backend.** Default `backend="claude"` needs Claude Code logged in on the machine that runs rank 0. If the GPU
   machine has no agent, use `backend="mailbox:<folder on the GPU machine>"`, check it there with
   `opal doctor --backend mailbox:<folder>`, and give the user the broker command:
   `opal broker <folder> --ssh <user@gpu-host> --backend claude --model claude-opus-5-5 --effort high`. With a
   mailbox the model and effort are set on the broker, not in `backend_options`. Over ssh, the GPU host needs
   `python3` (3.6 or newer) on its non-interactive PATH; the doctor checks it. The broker's ssh cannot answer a
   password prompt: if the host asks for one, tell the user to log in with a key or to open a shared connection
   first (`ssh -fNM -o ControlPath=~/.ssh/opal-%r@%h-%p <user@gpu-host>`). The broker's machine must stay awake
   for the whole run (on a Mac: `caffeinate -i opal broker ...`); while no broker answers, the run waits.
7. **Smoke test before the real run**: run about 100 updates twice with a reduced `total_steps`, once unwrapped and
   once wrapped with `backend="keep"` (no agent, no cost), and check that the losses match step for step. Time the
   updates of the unwrapped run (seconds per update, after the first few).
8. **Estimate, and wait for the user's go-ahead.** Run `opal estimate --total-steps N --sec-per-step S` with the real
   run's `total_steps` and the measured seconds per update (add `--decisions K`, `--every E`, `--no-blocking` or
   `--mailbox` as the wrap call will have them). Show the user its lines as they are: the run's time without the
   agent, the expected waiting for the agent (a range, and as a multiple of the run's own time) and the expected
   cost. If `decisions` is not set, OPAL lowers it so that the waiting stays within 4x the run's own time, but keeps
   at least one decision, even when one already waits longer (short runs); say so.
   Ask for `max_cost_usd`, and give the launch command only after the user confirms. Then switch to the real
   backend, with `max_cost_usd`. The agent defaults to claude-opus-5-5 at effort high, the blog's
   setting; set `backend_options=dict(model=..., effort=...)` only if the user asks (the run then warns that the blog's
   results may not be reproduced).
9. **DDP and long jobs:** wrap on every rank with the same arguments; call `optimizer.close()` when the loop ends early
   (early stopping, GradScaler skips); to resume, call `wrap(...)` first and then `optimizer.load_state_dict(...)` on
   every rank. On a cluster without an agent login, use one mailbox folder and run
   `opal broker <folder> --ssh <user@gpu-host> --backend claude --keep-serving` on the laptop or login node.
10. Tell the user: where records go (`opal_runs/<time>-<pid>-<random>/`), `report.html` there (rewritten after each
   decision), `opal show <run>` for one line per decision, `opal watch <run>` to print events as they are written,
   `opal report <run>` to rewrite the page, `opal compare <a> <b>` to set two runs or two result.json files beside
   each other, `opal say <run> "..."` to leave a note for the agent, and that the agent's code runs inside their
   training process.
