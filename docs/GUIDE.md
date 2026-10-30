# OPAL guide

This guide covers how OPAL runs, what the agent edits, and how to fit it into your script.
Start with the [README](../README.md). Cost, privacy, safety and clusters are covered here too.

## What you get

Each run writes `report.html` in `opal_runs/<folder>/`. It shows the training loss with a tick at each decision,
when the decision was asked and when its edit took effect. A table lists the changed files and constants, a short
description and tags. You can read what the agent said and the diff of each change. The page updates after every
decision and at the end of the run.

`opal show <folder>` prints one line per decision: when it was asked, what happened, its time and cost, and the start
of the rationale. `opal watch <folder>` prints each event as `events.jsonl` grows, and returns when the run finishes.
A run killed without `close()` (SIGTERM, SIGKILL, out of memory) never writes its end, so give it `--timeout`.
`--timeout 0` prints what is already there. `opal report <folder>` rebuilds the page at any time. `opal compare a b`
sets two run folders, or two `result.json` files, beside each other: metric, command, decision counts, cost, waiting,
and the decisions that differ, matched by the update they were asked at. Every program version is in `programs/`.
Each decision's message and answer are in `decisions/`. Events are in `events.jsonl`.

## How it works

Only rank 0 talks to the agent. By default training waits for each answer.

```
rank 0, main thread:   step → step → step → … → swap → step …
rank 0, background:       └─ claude process (one conversation for the whole run) ─┘ answer ready
other ranks:           step → step → step → … → swap → step …
```

The picture shows `blocking=False`. With the default `blocking=True`, every rank waits at the decision.

- Every `every` updates, rank 0 reads its losses and runs the agent's analysis when due. The default is 50.
  At scheduled points, or when the analysis asks, a background thread runs the agent CLI.
  Training waits for the answer by default. With `blocking=False`, it continues on the current program.
- A new program takes over at the next update boundary. On one GPU this happens at once. In DDP, rank 0 announces
  it through an asynchronous broadcast every `sync_every` updates on a separate gloo group. The default is 20.
  Every rank swaps at the same update.
- With `blocking=True`, all ranks wait at every decision, as in most of our published runs. With `blocking=False`, they wait
  only once when the loss turns non-finite. Waiting uses the gloo group, so a long agent turn does not trip an NCCL timeout.
- A new updater runs on trial for `trial_steps` updates, by default 20. If it raises on any rank, every rank returns
  to the previous updater. The fallback is the last updater that passed its trial, or the starting one.
  An updater replaced during its trial does not become the fallback.

Before building a new updater, the wrapper keeps a CPU copy of the parameters, gradients, your optimizers' state
and the old updater's state. This must fit in `snapshot_limit_gb` on every rank. The default is 2.
With that copy, a failed build or a failure at the first update is undone exactly, including changes `build()` made
to shared objects. A later failure in the trial window is not undone, since the update may be partly applied.
A failure without a copy is not undone either. The event log, the run's log line and the agent's next message say
which happened.

If an updater fails after its trial window, a single-process run rebuilds the previous program and continues.
A DDP run stops on that exception.
When you restart from a checkpoint, the run replaces the crashed program with its predecessor. In each case, your
learning-rate schedule returns to where the previous program would have it. See learning-rate logging below.

DDP compares parameter fingerprints before a new updater, after its first update and at the end of its trial.
If equal ranks diverge, OPAL restores the pre-trial snapshot and reverts the updater. Without a snapshot, it stops
for checkpoint recovery. The agent is told which tensors differed. Later divergence broadcasts rank 0's parameters
and is recorded in the event log and stderr. The report and `opal show` mark “ranks repaired at updates …”.
Optimizer state is not repaired.

## The two programs the agent edits

The agent works in `opal_runs/<folder>/workspace/`. It contains the programs, `inbox/` for each decision's inputs,
`reference/` for read-only copies of your training script and model source, and `scratch/`.
The program consists of `updater.py`, `analysis.py` and the modules they import. Other files in the folder do not count.

```python
# updater.py: turns gradients into updates, on every rank
def build(named_params, previous=None, spec=None):   # previous = the updater it replaces (carry its state over)
    return Updater(...)                                # spec: total_updates, update, world_size, rank,
                                                       #       user_optimizers, user_schedulers, clip_grad_norm
class Updater:
    follow_user_scheduler = False                      # True: your loop's scheduler.step() keeps stepping your schedulers
    def step(self, grads, update): ...                 # grads[name] is param.grad; update counts from 1
    def state_dict(self): ...                          # plain tensors, numbers, strings, lists, tuples, dicts only
    def load_state_dict(self, state): ...
    param_groups = [...]                               # recommended: torch.optim-style groups with "lr", each param once

# analysis.py: what the agent wants to see, on rank 0
EVERY = 50
def analyze(obs, memory):                              # obs.losses, obs.params, obs.grads, obs.metrics, obs.updater, ...
    return {"report": {...}, "wake": "reason"}         # report: kept for the next decision; wake: ask for a decision now
```

The starting `updater.py` calls your own optimizer and scheduler. The default instructions are in
`opal/prompts/neutral.md`. You can also use `prompt="aggressive"` or the path of your own template.
See `prompt` under cost and limits.

## Skipping an update

The agent's updater can discard an update it does not trust. Set `can_skip = True` on the updater.
Before each update, the wrapper copies the parameters and the updater's `state_dict()`.
It also copies your optimizers' state just before stepping them, unless the updater's `state_dict()` holds it.
If `step` returns `True`, as a bool or a 0-d bool tensor, the update is undone and training continues from that state.

`step(grads, update, ctx)` also receives `ctx.loss`, the update's training loss on the device.
It is the mean over ranks, so every rank makes the same decision. After 10 skips in a row, the next update is applied
and the agent is asked for a decision.

Only programs that enable skipping pay its cost: memory for one copy of the parameters and optimizer state,
one device copy and one host synchronisation per update. On a small test model on an RTX 3090, this added about 10%
per update on both one and two GPUs. Things outside the optimizer are not undone: your loop's EMA, GradScaler,
BatchNorm statistics from the forward pass, and the batch itself. The skipped batch is not learned from.

## Try it without an agent

`backend="keep"` answers every decision with "keep the programs". The run must match the unwrapped one.
`opal doctor` takes no `--backend keep`; a keep run needs no Claude installation or login.

To test a code change, pass a function as the agent. It receives the agent's folder, the decision message and the
system prompt. It can edit the program files there and return its rationale:

```python
def fake_agent(workspace, message, system):
    (workspace / "updater.py").write_text(MY_UPDATER_SOURCE)
    return "switching to SGD with momentum"

opt = opal.wrap(opt, model, total_steps=steps, goal="...", backend=fake_agent)
```

Run the plain and keep scripts in separate processes with the same seed, data order and software versions.
Add this after training in both and compare the hashes. Hash the tensors rather than a checkpoint file, whose
container metadata can differ. Compare the per-update training losses too.

```python
import hashlib, json
import torch

digest = hashlib.sha256()
for name, p in sorted(model.named_parameters()):
    a = p.detach().cpu().contiguous()
    header = json.dumps([name, str(a.dtype), list(a.shape)], separators=(",", ":")).encode()
    digest.update(len(header).to_bytes(8, "little") + header + a.reshape(-1).view(torch.uint8).numpy().tobytes(order="C"))
print("parameter sha256:", digest.hexdigest())
```

For the Hugging Face Trainer, install the `hf` extra and replace `Trainer` with `OpalTrainer`.
For a CPU smoke test, set `use_cpu=True` in your `TrainingArguments` and keep everything else.
Keep the Trainer's `max_grad_norm`. Moving clipping into OPAL can change rounding, so the run is no longer
byte-identical.

```bash
pip install "opal-optimizer[hf]"
```

```python
from opal.hf import OpalTrainer

trainer = OpalTrainer(model=model, args=training_args, train_dataset=train, eval_dataset=dev,
                      opal=dict(backend="keep", summarize=False, goal="My classifier; judged by held-out accuracy"))
trainer.train()
```

### Accelerate on CPU

Install Accelerate in the same environment as OPAL:

```bash
python -m pip install accelerate
```

Run this with Python for a single-process CPU check. It uses no fp16 GradScaler or Accelerate gradient accumulation.

```python
from accelerate import Accelerator
import opal
import torch
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(0)
steps = 100
model = torch.nn.Linear(2, 2)
train_loader = DataLoader(TensorDataset(torch.randn(128, 2), torch.randint(2, (128,))), batch_size=16)
criterion = torch.nn.CrossEntropyLoss()
accelerator = Accelerator(cpu=True, mixed_precision="no", gradient_accumulation_steps=1)
opt = torch.optim.AdamW(model.parameters(), lr=0.01)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
model, opt, train_loader, sched = accelerator.prepare(model, opt, train_loader, sched)
opt = opal.wrap(opt, model, scheduler=sched, total_steps=steps,
                goal="A linear classifier on synthetic data; judged by cross-entropy after the last update",
                backend="keep", summarize=False, clip_grad_norm=1.0)
try:
    update = 0
    while update < steps:
        for x, y in train_loader:
            opt.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            accelerator.backward(loss)
            opt.step(loss=loss)
            sched.step()
            update += 1
            if update == steps:
                break
finally:
    opt.close()
```

Wrap after `accelerator.prepare`, using its returned optimizer and scheduler. Keep `sched.step()` after each update.
The loop needs a nonempty loader and makes one update per batch. If your plain loop calls
`accelerator.clip_grad_norm_(model.parameters(), 1.0)`, remove that call and pass `clip_grad_norm=1.0` to `wrap`.
If the plain loop has no clipping, omit it from `wrap` too. Clipping transfer matched parameter bytes on CPU float32
without a scaler or accumulation. Compare your plain and keep loops with the same seed, loader order and threshold.
Accelerate fp16 GradScaler and `gradient_accumulation_steps` greater than 1 are not supported.

Use `opt.log(validation_accuracy=value)` to record a measurement without showing it to the agent.
Finish a keep check with `opal show` and `opal report` on the printed run folder. Leave out `--summarize` to avoid
model calls. During a longer run, `opal watch` prints events as they are written. The `profiler_start` /
`profiler_stop` lines on stderr come from OPAL's one audited update per new program and are harmless.

## Experimental Codex backend

Claude is the default. To use Codex, pass
`backend="codex", backend_options=dict(model="gpt-6.1-sol", effort="high")`. `xhigh` works too.
For a broker, use `opal broker <box> --backend codex --model gpt-6.1-sol --effort high`.

Log in separately with `CODEX_HOME=~/.codex-opal codex login --device-auth`.
Do not copy or link the default home's `auth.json`. `max_cost_usd` does not cap Codex.
`backend_options=dict(max_tokens=N)` is a soft input + output token cap. A turn in progress can exceed it.
The broker flag is `--max-tokens N`. The run stops with an error if Codex compacts its conversation history,
at about 450k context tokens with Codex CLI 0.160.1 defaults.

Our single runs on one H100, Codex against plain: `gpt_small` validation loss 4.3125 vs 4.4763;
`diffusion_mnist` EMA validation loss 0.1687 vs 0.1696.

## What the agent sees, and what leaves your machine

The agent sees your goal sentence, every parameter's name and shape, training losses and its own analysis reports.
It sees errors, timings and `opt.log` values passed with `show_agent=True`, which is off by default.
`reference/` holds copies of your training script and the user code for your model, optimizer and scheduler.
All of this goes to the backend's model provider, such as Anthropic for Claude Code.
`share_code=False` keeps the script and source files out.

The agent's CLI receives an allowlist of environment variables: PATH, HOME, user and locale, proxies and CA files,
and what Python needs. These include `LD_LIBRARY_PATH`, `PYTHONPATH`, `VIRTUAL_ENV`, `CONDA_PREFIX` and thread counts.
It also receives the backend's login variables, including AWS/Google variables for Claude Code on Bedrock or Vertex.
The login variables are unset again for the agent's shell through Claude Code's `sandbox.credentials`.
Other variables do not reach the agent, such as a database URL, an ssh agent or a token with an unrecognised name.
If your Python needs a variable outside the list, it will be missing from the agent's CPU checks too.

The agent gets none of your CLAUDE.md, memory, skills, plugins or MCP servers. It gets no git status from surrounding
folders: no branch, user name, file names or commits. Claude Code still tells it your account's e-mail address and
whether its folder is inside a git repository.

## Cost and limits

With `max_cost_usd`, calls stop when the remainder is below the larger of $0.01 and the reported spend per decision
asked so far.

| Option | Default | Meaning |
|---|---|---|
| `decisions` | 50 | How many times the agent is asked. Each time it may keep the programs. A fifth fall in the first 5% of training, the rest evenly over the remainder. Short runs get fewer. When unset, blocking Claude and mailbox runs cap expected waiting at 4x their training time, as described below. This cap does not apply to Codex. |
| `max_decisions` | 2 x the scheduled decisions | Cap including the agent's wake-up calls. Wake-ups leave room for the remaining scheduled decisions. The last decision's message says it is the last. |
| `max_cost_usd` | none | Stop asking once Claude Code reports this spend. Checked before each decision. The Claude backend also caps each decision at the remainder with `--max-budget-usd`, which can overshoot by one model call. `backend_options=dict(max_budget_usd=...)` caps each decision. |
| `every` | 50 | Updates between checks. Decisions and the non-finite check happen at checks. |
| `backend_options` | `{}` | `model=` defaults to `claude-opus-5-5`, `effort=` to `high`, as in the blog's runs. Another model or effort prints a warning at the start that the blog's results may differ. `timeout=` is seconds per decision, default 3600. `shell_timeout=` is seconds per agent shell command. `max_budget_usd=` caps each decision through Claude Code's `--max-budget-usd`. It can overshoot by one model call. `retries=` retries a failed turn, such as a tool crash, from the programs it started with. Pauses start at `backoff=15` s and double up to `backoff_max=600` s while `timeout` allows. `None`, the default, retries as often as that allows. `0` never retries. Each failure's error goes to the run's log, `decisions/NNN/attempts.jsonl` and the broker's log. The broker flag is `opal broker --retries N`. An unavailable agent has a separate wait, described below. `unavailable_timeout=` sets its deadline. |
| `blocking` | True | Training waits for each decision, as in most of our published runs. False keeps training. |
| `share_code` | True | False keeps the training script and source files out of the agent's folder. |
| `schedule` | "front" | When decisions are asked. The learning-rate schedule is `scheduler`. "front" puts a fifth evenly over the first 5% of training and the rest evenly over the remainder. "uniform" asks every total_steps // decisions updates. Segments such as `[(0.05, 20), (1.0, 30)]` give an end fraction and count: here 20 in the first 5%, then 30 over the rest. Update numbers such as `[500, 1000, 5000]` give the exact updates. A list sets the count: omit `decisions` or make it agree, or `wrap` raises. Each decision moves to the next check, every `every` updates. Two on one check merge. None falls in the last `every` updates. The first log line gives the remaining count. |
| `show_next_decision` | on with the neutral and aggressive prompts | True gives the next scheduled decision's update in each message. False turns it off. |
| `prompt` | "neutral" | The run's facts, protocol and short directions without recommended values. `task_facts="..."` adds facts about data, repetition, batch and the loss, as do `epochs=`, `train_size=`, `batch_size=` and `same_order_each_epoch=`. "aggressive" (OPAL 0.1.2-aggressive) asks for a review of each parameter group's hyperparameters at every decision. Its rationale has five parts: observed, review, changed, prediction, rules. It drops the statement that keeping the programs is valid. `reference/example_updater.py` includes a learning-rate feedback controller, off by default. `opal/misc/aggressive.py` holds every difference. You can also pass a template path. Its `{fields}` are filled as in `opal/prompts/neutral.md`. |

Before launching, run `opal estimate --total-steps N --sec-per-step S [--decisions K] [--every E]
[--no-blocking] [--mailbox]`. `opal estimate` forecasts a Claude run; a keep run makes no model calls and is free.
It prints training time without the agent, expected waiting as a range and a multiple of training time, and cost.
The per-decision numbers come from our claude-opus-5-5 runs at effort high.
`opal/estimate.py` lists the numbers and sources. The first decision read the code and wrote the first program.
It took 100 to 430 s and cost $0.30 to $1.60. Later decisions averaged 10 to 50 s and $0.05 to $0.38 each.

Claude and mailbox runs print the same estimate at their first check, using their own time per update.
This first-check estimate and the 4x waiting cap do not apply to Codex. If `decisions` is unset and training waits,
OPAL schedules at most as many decisions as keep expected waiting within 4x training time.
It keeps at least one, even if that exceeds 4x, and keeps the same schedule.
The log line says what it chose, and `events.jsonl` records an `estimate` event.
An explicit `decisions` count stays as set. The log says when its expected waiting exceeds 4x.
A 66-second CIFAR-10 run gets one decision this way, a five-hour run gets 50.
A resumed run keeps the choice from its first check.

If the agent is unavailable, OPAL keeps the decision and waits. This covers Claude Code not being logged in, an
invalid key, usage, spend or rate limits, an overloaded API, a network outage or a missing CLI.
With a mailbox, it also covers no broker answering for `broker_wait` seconds, by default 120, or a broker whose
Claude Code is unavailable. A broker never answers such a request as failed.

The run prints a line when waiting starts, with the CLI's words and what to do. For example:
"decision 7: Claude Code is unavailable (Not logged in · Please run /login); training waits for the agent and tries
again in 60 s, then at most every 5 min. To go on: log in on this machine ...".
It prints a reminder every 10 minutes and a line when the agent returns. The broker logs the same.
Retries start after 60 s, then at most every 5 minutes. A reset time in the message can bring the next retry forward.

When the agent returns, the same decision continues with the same message. If a call began editing before it stopped,
the folder is restored and the message says so, as for a failed call. Waiting has no deadline and does not count
against the decision's `timeout`. The report and `events.jsonl` show it separately as `unavailable_seconds`.
`backend_options=dict(unavailable_timeout=s)` or `opal broker --unavailable-timeout s` fails the decision after s
seconds. The run can still ask again. With `blocking=False`, training continues on the current programs while waiting.
No new decision is asked meanwhile. A failed turn, such as a crash, time limit or spent budget, follows `retries`
instead of the unavailable-agent wait.

By default, `blocking=True` waits for the answer at each decision. Its edit takes effect from the update where the
decision was asked, and the agent's thinking adds to wall time. `blocking=False` allows faster
runs. Without waiting, an answer takes effect answer time / time per update updates after the state the agent saw.
No new decision is asked while one is in progress, so fast runs may ask far fewer times.

## The report

Each run keeps `report.html` in its folder. It shows the loss with a tick per decision, your `opt.log` values,
code changes, and every decision's rationale and diff. It also plots up to 12 numbers from the analysis program's
`report` dict. These can be nested one level, such as `grad_norm.embed`, and must appear in at least three reports.
Rank 0 rebuilds the page in a background thread after each decision and at the end. During the run it reloads every
minute. `opal report <run>` rebuilds it at any time.

The first line, and `opal show`, count scheduled, asked, answered and applied decisions.
Answered means the answer arrived. Applied means it changed the programs. The median lag from request to effect
helps you see when the agent was rarely asked.

The page opens with an overview from the records, a loss plot with numbered applied decisions, and a table.
Hover, focus or tap a decision for the change and the agent's reason. Kept decisions are faint ticks.
The table gives the decision number, requested → applied update, a one-line change, a shortened quote from the
rationale, and tags. Analysis plots, full messages and answers, code diffs, changed files and constants, and run facts
start folded.

The page shows recorded costs and the first-to-last timestamp span. Missing timing, commit, model and effort fields
say "not recorded". The single HTML file works offline, on phones, in light or dark mode. Run paths are relative.
Private locations and recorded identities are omitted from messages, code and plot data. Rendering makes no model
calls.

Cached descriptions and tags come from Claude Opus at low effort through the Claude Code CLI, with no tools.
One call per code change costs a few cents ($0.02 to $0.06 in our checks). It is recorded in `decisions/NNN/summary.json` and excluded from
`max_cost_usd`. A change can have any number of these tags:

- `common practice`: standard recipes, such as Muon, warmup, cosine or linear decay, clipping and AdamW betas.
- `niche practice`: published but uncommon, such as SOAP, Shampoo, per-tensor LR controllers and NorMuon.
- `block-specific`: some groups, layers or blocks are treated differently.
- `observation-originated`: the change cites something the agent saw in this run.
- `time-step-specific`: the program treats a particular update or phase differently.
- `fix or revert`: repairs or undoes the agent's earlier change.

Tags describe a change, not its quality. Read the rationale and diff before relying on them.
On 95 decisions from earlier runs, two labellers agreed on the full tag set for a third.
With `backend="claude"`, summaries are made during the run. `summarize=False` turns them off, and
`summarize="<model>"` picks another model. For other backends, or to fill gaps, run `opal report <run> --summarize`
on a machine with Claude Code.

## Comparing with our runs

Every example script with `--opal` can write a result.json with `--out <file>`.
The first fields are shared: metric name, value and direction, training and waiting time, agent cost, decisions
asked, applied and failed, hardware, OPAL commit, full command and run folder. Plain runs have the same fields with
the OPAL fields empty. `--save <folder>` writes `model.pt` and, for scripts that keep an EMA, `ema.pt`.
`opal compare a b` sets two result.json files, or two run folders, beside each other. It prints the metric and
command, decision counts, cost and waiting and, when both run folders are on this machine, the decisions that differ,
matched by the update they were asked at. `--json` prints the two summaries and the differing decisions as JSON.
Read one run with `opal show <run folder>`.

Our run folders are not published. `opal.replay("<run folder>", optimizer, model, total_steps=...)` reruns the final
program of one of your own runs in place of `opal.wrap(...)`. It runs the version named by `FINAL` deterministically.

The example scripts take `--decisions`, `--decision-schedule`, `--blocking`, `--model` and `--effort`.
Most default to 50 decisions, `--decision-schedule front`, blocking, claude-opus-5-5 at effort high.
`examples/overfit/train.py` defaults to 30 uniformly spaced decisions. Its `--schedule` controls the learning-rate
schedule.

## Fitting it into your training script

- Create all parameter groups before you call `opal.wrap`. If the raw optimizer's parameters change afterwards, OPAL raises an error at the next `step` or `zero_grad`.
  To unfreeze layers during training, put them in a parameter group from the start with `requires_grad=False` and set it to True when you unfreeze them. torch.optim optimizers skip parameters without a gradient, so this gives the same updates as adding the group later.
- For gradient accumulation, call `opt.step(loss=...)` once per update with that update's mean loss.
- The agent reads your scheduler's settings and a `LambdaLR` function's source, but not its closure values.
  If warmup and decay lengths come from variables or flags outside the code it reads, state them in `task_facts`.
  For example: `task_facts="linear warmup over 7,935 updates, then linear decay to 0 at update 79,355"`.
- OPAL needs torch 2.5 or newer and Python 3.10. Repositories pinned to older torch or Transformers need a newer torch,
  which may require small fixes. Older Transformers with recent torch need `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`
  to resume from a checkpoint.
- Hand gradient clipping to OPAL. Delete `clip_grad_norm_(model.parameters(), x)` from your loop and pass
  `clip_grad_norm=x` to `wrap`. The starting program clips first, bit for bit like the loop when the optimizer holds
  all trainable model parameters. The agent sees unclipped gradients and can change clipping.
  If you leave clipping in the loop, the agent receives clipped gradients and cannot change the threshold.
  See the HF Trainer and Accelerate instructions for their clipping rules.
- Log `opt.param_groups[i]["lr"]` after `opt.step()` and before `scheduler.step()`.
  A program that sets rates inside `step` has not set the current rate before that call.
  While the agent's program owns the schedule, your `scheduler.step()` does nothing.
  Where possible, `get_last_lr()` reports the rate the program gives each optimizer group's parameters.
  When a program returns control to your scheduler, the schedule continues as if it had never stopped.
  This happens when the agent restores the starting updater or a failing program reverts to one that follows it.
  The optimizer's group settings return to when the agent took over, and later `scheduler.step()` calls are replayed,
  including across a checkpoint. `ReduceLROnPlateau` keeps the metric passed to `step()` as `best`.
  Programs can read it through `spec["user_schedulers"]`, so pass only values you would show the agent.
- For several optimizers or schedulers, pass lists: `wrap([muon, adamw], model, scheduler=[s1, s2], ...)`.
  OPAL cannot combine several optimizers with an enabled GradScaler: `scaler.step` raises an error before any update.
- For fp16 with GradScaler, use `scaler.step(opt, loss=loss)`. Skipped updates are not counted.
  Call `opt.close()` when the loop ends. With a fused optimizer, the scaler takes its generic unscale path.
  The starting program then matches the plain run numerically, with possible differences in parameter bytes.
- For DDP on one node, wrap the bare or DDP-wrapped model on every rank with the same arguments.
  Call `opt.step(loss=...)` and `opt.close()` on every rank. The agent sees rank 0's loss.
  `opt.log` values from rank 0 are kept. Save `opt.state_dict()` from rank 0 alongside the model.
  Load it on every rank after `wrap(...)`. A program crash after its trial window stops the job.
  Restarting from a checkpoint replaces it with its predecessor, or the starting program if that also crashed.
  You can use the same run folder or a new one. The checkpoint remembers the old folder.
  In DDP, the consistency checks compare SHA-256 digests of the parameter bytes, copied to CPU in chunks at each check.
  Uneven inputs across DDP ranks (`torch.distributed.algorithms.Join`) are not supported: OPAL stops with a clear error when a rank runs out of data. Give every rank the same number of steps.
- For checkpoints, save `opt.state_dict()` and your scheduler's `state_dict()`.
  OPAL's state holds the programs, their state, the agent session, and your optimizers' and schedulers' state,
  whatever the program's `state_dict()` holds. It also holds analysis memory.
  Checkpoints made with `opt.state_dict()` keep the observations the agent has not seen yet (loss windows, analysis reports, and metrics logged with `show_agent=True`), so a resumed run shows the agent the same thing as an uninterrupted one. Older checkpoints cannot recover observations that were never saved.
  State that the agent's updater keeps must use plain types: numbers, strings, bytes, None, tensors, torch sizes, dtypes, devices, layouts and memory formats, and lists, tuples, dicts (including OrderedDict and Counter), sets and frozensets of these. Sets, frozensets and memory formats must also be supported by the installed PyTorch weights-only loader. NumPy values are converted to numbers or tensors. Anything else is left out of the checkpoint with a warning and a `state_dropped` event; custom serialization hooks are never run.
  The starting analysis keeps a tensor per parameter role, including GPT-2's whole token embedding.
  To resume, rebuild the optimizer and scheduler, call `wrap(...)` with the same arguments, then `load_state_dict`.
  With the default run folder, an interrupted run continues in the checkpoint's folder and conversation if it is
  the same unfinished run in this run's `opal_runs/`. Records written after the checkpoint move to `attempts/`.
  The agent is told its later decisions were lost. A finished run's checkpoint, or one copied from elsewhere,
  resumes in a new folder and leaves the original as it was.
- For early stopping or an exception, call `opt.close()`. It also runs at exit.
  A run stopped before `total_steps` without `close()` counts as interrupted, so its mailbox broker waits for it
  to resume. For a planned interruption, such as a job's time limit, call `opt.close(interrupted=True)`.
  The broker keeps waiting. When `close()` returns, the final `report.html` is written.
  It waits up to 30 s for pending Claude summaries.
- `blocking=True` is deterministic given the agent's answers. Replaying a final program with `opal.replay` is
  deterministic too. With `blocking=False`, timing decides when an edit takes effect.
  Two runs with the same answers can differ.
- To turn the agent off mid-run, resume with `backend="keep"` or start a fresh run with `opal.replay(...)`.
- For the Hugging Face Trainer, use `opal.hf.OpalTrainer` in place of `Trainer`.
  Install it with `pip install "opal-optimizer[hf]"` (or `pip install "opal-optimizer[hf] @ git+https://github.com/xingjian-bai/opal"`):

  ```python
  from opal.hf import OpalTrainer

  trainer = OpalTrainer(model=model, args=training_args, train_dataset=train, eval_dataset=dev,
                        opal=dict(goal="GPT-2 small on Swahili; judged by FLORES-200 bits", max_cost_usd=20))
  trainer.train()
  ```

  It wraps the optimizer after the Trainer builds it and its schedule. The schedule stays on the Trainer's optimizer.
  It gives OPAL each update's mean loss, since the Trainer calls `step()` without it.
  It records `eval_*` values in the report without showing them to the agent.
  With `backend="keep"`, it matches the plain Trainer byte for byte on CPU with transformers 5.20.
  The Trainer's `max_grad_norm` clipping stays outside the optimizer.
  To let the agent control it, pass `TrainingArguments(max_grad_norm=0)` and `opal=dict(clip_grad_norm=1.0)`.
  The starting run then matches up to rounding, since accelerate computes the norm differently.
  Older Transformers with recent torch need `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` to resume.
  Their RNG state does not load with `weights_only=True`.
- For Accelerate, wrap after `accelerator.prepare(...)` with the prepared optimizer and scheduler:
  `opt = opal.wrap(opt, model, scheduler=sched, ...)`. OPAL wraps the torch optimizer inside Accelerate's.
  Call `opt.step(loss=loss)` once per update. See the complete CPU recipe above for clipping.
  CPU runs match byte for byte without fp16 GradScaler or Accelerate gradient accumulation.
  `wrap` refuses those modes. Wrapping before `prepare` fails because `prepare` saves and reloads the wrapper's state.
  If you wrap after `accelerator.prepare`, `accelerator.save_state` and `accelerator.load_state` also save and restore OPAL's program, its version and the update count.
  The model may have any number of DDP, Accelerate or `torch.compile` wrappers.
- If another framework calls `step()` for you, keep a handle to the wrapped optimizer.
  Call `opt.observe(loss)` once per update and `opt.log(...)` for metrics. Lightning has no test yet.
- Closure-only optimizers such as LBFGS are not supported as the starting program.

## What it costs your training

The wrapper copies loss into a device buffer and reads it every `every` updates, with no host synchronisation per
update. In DDP, it adds a few small gloo broadcasts per check: one, or three in blocking mode.
It adds an asynchronous broadcast every `sync_every` updates and one gloo all-reduce per update during a new updater's
`trial_steps` window. An `EVERY` below `every` adds a check at each analysis step.

Before building a new updater, it copies parameters, gradients and your optimizer's state to CPU.
Above `snapshot_limit_gb`, by default 2, it skips the copy and warns you.
A 125M-parameter model with AdamW needs about 1.9 GiB, and more once the previous updater has its own state.

For an agent's updater, it reads each parameter's and gradient's dtype and device after every update.
This takes tens of microseconds for a few hundred tensors. It walks the updater's state at check steps.
It adds no model hooks and does not recompile a `torch.compile`d model when the program changes.
The latter has been tested. It adds nothing when the agent keeps the programs.
Each decision shows the agent its program's time per update beside the starting program's.

## GPU machines without an agent

Claude Code can run on the GPU machine if it reaches the internet and you have run `claude` there to log in.
On a cluster, login usually carries over to compute nodes through the shared home directory.
Use a mailbox if the machine has no outbound internet or you prefer to keep your agent login elsewhere.
Point the run at a folder and serve it from a laptop or login node over ssh or a shared filesystem:

```python
opt = opal.wrap(opt, model, total_steps=steps, goal="...", backend="mailbox:/scratch/me/opal-box")
```

```bash
opal broker /scratch/me/opal-box --ssh me@gpu-host --backend claude --keep-serving
```

On the GPU machine, `opal doctor --backend mailbox:/scratch/me/opal-box` checks that the folder is writable and
whether a broker serves it. That machine needs no Claude Code. Set the model and effort on the broker with
`opal broker ... --model claude-opus-5-5 --effort high`, the defaults, rather than in the run's `backend_options`.
The broker's machine must stay awake for the whole run. On a Mac, use `caffeinate -i opal broker ...`.
The broker's ssh runs with `BatchMode=yes`, so it cannot ask for a password or a second factor. Log in with a key,
or open a connection it can share before you start it: `ssh -fNM -o ControlPath=~/.ssh/opal-%r@%h-%p me@gpu-host`.
The run waits while no broker answers, as described under "When the agent is unavailable".
Mailbox runs do not make code-change summaries during training.
Run `opal report <run> --summarize` where Claude Code is logged in to add them.

At its first update, the run prints its id and the exact broker command.
Each run gets a subfolder of the box with mode 0700. Every request and answer carries the run id and an attempt id.
One box can serve many runs, and a resumed run does not pick up an old answer.
The broker refreshes its heartbeat every 30 s, including while the agent thinks.
After 2 minutes without a broker, the run keeps the request, waits and prints how to start one.
`unavailable_timeout` limits that wait. A returning broker can still answer.
The answer is applied if it arrives before the next request, and discarded otherwise.
The next message tells the agent what happened to the previous answer: when its edit took effect or why it did not.

Without `--keep-serving`, the broker serves one run until it ends. It chooses the newest pending request,
or the run given by `--run <id>`. It waits for an interrupted run to resume.
While its run is idle, it serves another waiting run in the box, unless pinned with `--run`.
With `--keep-serving`, it serves all runs in turn, longest waiting first, and keeps a box-level heartbeat accepted
by every run. A run that asks often cannot starve another. When training ends with a decision in progress,
the broker stops that agent turn.

Each request carries a workspace manifest. Files other than programs are also kept in `<box>/<run>/files/`.
A broker on another machine or with a new `--workdir` fetches missing earlier inputs in `inbox/` and `reference/`,
and drops inbox files the run no longer has. The agent's own `NOTES.md` and `scratch/` files return with each answer
and stay in the workspace. They must be small text files, at most 256 KB each and 2 MB per answer.
Notes survive a change of broker. A broker without the run's conversation tells the agent at its first message.

Before starting the agent, a broker claims a request by exclusively creating `claim-NNNN.json`.
This stops two brokers from answering and billing the same fresh request.
A claim with no refresh for 2 minutes can be taken over.
Mailbox brokers hold a lease on each request; a lease taken over by another broker cannot be renewed, released or answered by the old one. The mailbox host must provide Python 3.6 or newer as `python3` on the broker's non-interactive SSH PATH; run `opal doctor --backend mailbox:<folder>` in that same environment before starting the broker.

The broker runs the agent for at most the time the agent was told it has: the mailbox `timeout`, by default 6 hours,
less the request's wait and a minute to write the answer. A shorter broker `--timeout` stays in force and is logged.
If the broker stops the agent earlier than the run told it, the message starts with the actual limit.
This can happen with a shorter `--timeout` or a request that waited for a broker.
For example: "This decision may take up to 15 minutes, less than your instructions state."
Give the run `backend_options=dict(timeout=...)` to state the same limit from the start.

`--backend keep` tests the whole path at no cost. Broker log lines have timestamps and flush at once,
including under `nohup` or a file. Failed mailbox reads are reported every ten failures.
`--give-up-after-min M` exits with an error after M minutes of failures.
The mailbox backend takes `backend_options=dict(poll=2, broker_wait=120, timeout=21600)`.
Run `opal show`, `watch`, `report` and `say` on the training side's run folder.

## Safety

Claude Code's file tools are confined to its folder. Its shell runs in the OS sandbox with no network and can read
only its folder and your Python installation. On Linux, the sandbox needs `bubblewrap` and `socat`.
Bubblewrap must be allowed to create namespaces, which some distributions forbid for unprivileged users.
OPAL tries bubblewrap once at the start. If it fails, the agent gets file tools only and OPAL says why.
With `backend_options=dict(sandbox=True)`, the run stops instead.

A watchdog stops the agent if training dies. After each decision, OPAL stops processes the CLI left running,
such as shell background jobs. On macOS, the shell cannot reach the GPU through Metal.
With Claude Code 2.1.302, torch imports but `torch.backends.mps.is_available()` is False in the sandbox.
The code the agent writes runs inside your training process with your permissions. It is not sandboxed.
Every version is kept in `programs/vNNN`. The final program is Python you can read before reusing it.

The shell can run one Python: the interpreter running the Claude backend.
With `backend="claude"`, that is your training process's Python. With a mailbox, it is the `opal broker` process's,
so the broker's environment needs torch for the agent's CPU checks. It comes first on PATH as `python`.
The sandbox can read its venv, site-packages and base installation, with links resolved, and each link's containing
folder on the path to the real binary. It never opens a folder containing your home folder or the run folder.
For `~/bin/python`, a link to conda or uv Python, it opens `~/bin`, not `~`.

The macOS sandbox checks real paths. A venv on Homebrew's Python works:
`/opt/homebrew/opt/python@3.12` links into `Cellar/`. This was checked on macOS with Claude Code 2.1.302 and
`python -m venv` on Homebrew's 3.12. A uv-managed Python venv works too: `uv venv --managed-python`.
If an essential folder must be left out, such as an interpreter installed in your home folder itself,
OPAL says so at the start.

## What OPAL does not protect against

The agent's `updater.py`, `analysis.py` and imported helpers run inside training with its permissions.
They are not sandboxed, even when the agent's shell is. They can read accessible files, inspect the training script's
globals or change objects outside the optimizer interface. Analysis reports go back to the agent, so data read by
a program can reach the agent and its provider. `share_code=False` stops source copies going into `reference/`.
It does not prevent these accesses.

`eval_integrity` detects some evaluation-time parameter moves. The static `boundary` check separately flags common
file/network I/O, dynamic imports, builtins access, introspection, hook registration and model/loss method assignments.
It checks source before loading each candidate on rank 0. Findings go into `events.jsonl` with the version, file,
line, category and snippet. Flagged versions get one stderr line and a **Boundary flags** note in `report.html`.
Candidates that fail to build can have flags too.

The boundary check records findings without blocking a program or changing its execution.
If the check fails, it logs one line and lets the existing program-loading checks proceed.
Its heuristics can give false positives and miss indirect or hidden accesses. No flags is not evidence of isolation.

Keep test data and labels out of training and evaluate the final checkpoint in a separate process.
Separate evaluation does not hide files that training can still read with its permissions.
Review the report's boundary flags, their details in `events.jsonl` and saved programs before trusting a result.
Prompt rules describe intended behavior. They do not enforce a security boundary.

## What it can and cannot change

| Works | Not supported |
|---|---|
| Plain PyTorch loops on one GPU or several GPUs of one node (DDP) | Multi-node training (untested), FSDP, ZeRO, DeepSpeed |
| `torch.compile`d models, bf16 autocast, fp16 GradScaler, gradient accumulation | Tensor/pipeline parallelism, JAX, whole-step CUDA graphs |
| One or several optimizers (e.g. Muon + AdamW), LR schedulers | Closure-only optimizers as the starting program |
| Anything an update can express: algorithm per tensor, per-group and per-block LR, schedules, momentum and betas, preconditioners, clipping inside the optimizer, weight decay | Anything outside the optimizer: model, data, batch size, loss, precision, number of steps |

Precision is fixed. The agent may not change parameters' or gradients' dtype or device.
It keeps new state in the parameters' dtype or float32, and computes updates at the algorithm's standard precision.
For example, Muon's Newton–Schulz uses bfloat16, as in `torch.optim.Muon`.
A program is reverted if it changes a dtype or device, casts your optimizers' state, or keeps parameter-shaped state
in a dtype other than the parameter's or float32. This includes bfloat16 moments for float32 weights.

The check reads tensor attributes without device work around `build()`.
After every agent update, it checks every parameter and `p.grad` dtype and device.
The `grads` dict is scratch space and is not checked. It checks updater state and your optimizers' state after each
trial-window update and at check steps. State matches parameters by name when keys are parameter names, and by shape
otherwise. Your optimizer's own state dtype is accepted. The prompt sets arithmetic precision, which is not checked.

The agent's `analysis.py` may not change parameters or gradients. Their version counters, dtypes and devices are
compared around each call. An analysis that changes them stops running until the agent edits it.
The change cannot be undone, and the agent is told.

`opt.log(...)` accepts keyword values or a dict. It records values in `metrics.jsonl` and the report.
Only `show_agent=True` shows them to the agent in `obs.metrics` and its next decision message.
An agent can tune to any value it sees. Show only a split you accept it tuning on, never the test set used to judge
the run. Programs run inside training and can read its folder, including `metrics.jsonl`, and anything else the
process can read. The prompt forbids this and the boundary check flags some patterns. Neither prevents it.
The sandbox stops the agent's shell from reading the run folder. Program files are never read or written through
symbolic links, so a workspace link cannot bring a file into the agent's folder.
Read `programs/` before trusting a result.

With `wrap(..., validation=N)`, added in OPAL 0.1.1-validation, your script evaluates a held-out validation set
about every N updates at unannounced times. Report it with `opt.report_validation(loss=..., accuracy=...)`
(`opt.report_dev`, its name before 0.3.10, still works).
`validation_size="5,000 images"` can name its size.
The agent sees each value in `obs.metrics` as `val_loss` and `val_accuracy`.
Programs written before 0.3.10 that read `dev_loss` and `dev_accuracy` still find them under those names.
Each decision message has a validation-set line with the history:
"Validation set (held out from training; not the test set in the goal), ...".
Every measurement is in the inbox.

With the neutral prompt, two sentences in TASK state the evaluation and the separate, unseen test set.
The evaluation-data sentence says validation values are shown. Nothing else changes.
With the default `validation=None`, neither sentence appears and `report_validation` only records.
`validation=N` is your promise of cadence and irregular times. OPAL does not run evaluation.
It warns you once, without telling the agent, if the last three evaluations were equally spaced or if their average
frequency differs greatly from every N updates. The agent sees each measurement's update and could predict the next.
Draw each time at random, as `examples/overfit/records.py` does with one uniformly random update per epoch.
The prompt must state this. Use the neutral prompt or a template with `{validation_note}`.
Without that field, `wrap` raises. `examples/overfit/train.py` uses this feature.

Evaluation integrity was added in 0.2.2. Two of our overfit runs read evaluation times from the prompt, which then
stated them, and from the training script. They timed parameter changes to the evaluations.
One scaled the classifier for each evaluation and restored it at the next update.
The other copied a running average into the model 49 updates before each evaluation and restored the iterate after it.
Both planned a final classifier scale chosen from these probes.

Along with keeping times unannounced, `eval_integrity` checks every evaluation you report.
It is on by default with `validation=N`. Set `True` or `False` to choose.
After every update, rank 0 copies a fixed sample of each parameter tensor's entries, at most 32K entries in all.
One `torch.cat` writes into a ring of 138 updates on the device, with no host synchronisation.
For a ResNet-18, this costs about 0.04 ms per update on CPU and 0.16 ms on Apple MPS.
Eight updates after each `report_validation`, it reads that window once.

A tensor is flagged when its last unusually large change at or before evaluation is undone by its first unusually
large change after evaluation, with ordinary updates on either side.
A large change exceeds `eval_integrity_threshold`, by default 5, times the window's median change per update,
and exceeds 0.1% of the tensor's norm. Undoing means returning to within a quarter of the excursion from the old value.
The run gets an `eval_integrity` event in `events.jsonl`, naming tensors, change and return updates, and the norm factor.
It also gets a `report.html` banner and a stderr message.
A lone lasting change, such as a permanent weight average swap, is recorded as `eval_jump` without flagging the run.
The agent is told neither the check nor its results.

The check covers program changes from 128 updates before evaluation through 7 after it.
It misses changes no larger than about five ordinary updates of that tensor.
Early in training, a small probe can hide in training's own motion.
A program undoing its own update on the next one looks the same if evaluation falls between them.
Read `programs/` before trusting a flag or its absence.
Call `report_validation` right after evaluating parameters as the latest update left them.
Your script's own EMA swap around evaluation happens outside the updater and is not attributed to the program.

Accelerate and HF Trainer have CPU tests, as described above. Lightning has no test yet.
The wrapper keeps the `torch.optim.Optimizer` interface and accepts `step(closure)` and `observe(loss)`.
