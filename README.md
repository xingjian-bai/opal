
<h1 align="center">OPAL: a reactive optimizer</h1>

<p align="center"><b>A PyTorch optimizer wrapper that lets a coding agent change the optimizer while your model trains.</b></p>

<p align="center">
  <a href="https://xingjianbai.com/reactive-optimizers/">Blog post</a> ·
  <a href="https://arxiv.org/abs/2610.33826">Paper</a> ·
  <a href="https://www.alphaxiv.org/abs/2610.33826">alphaXiv</a> ·
  <a href="https://github.com/xingjian-bai/opal/blob/main/docs/GUIDE.md">Guide</a> ·
  <a href="#start-with-your-coding-agent">Start with your agent</a> ·
  <a href="https://github.com/xingjian-bai/opal/blob/main/examples/README.md">Examples</a>
</p>

Most networks are trained with an optimizer that is fixed before training starts. OPAL (OPtimizer with an Agent in
the Loop) turns the optimizer into a program that an agent keeps rewriting during the run. You wrap your PyTorch
optimizer in one line. Every so often the agent (Claude Code) reads what the run reports and decides whether to
rewrite two small programs: the optimizer program, which turns gradients into updates at every step, and an analysis
program, which measures whatever the agent wants to look at.

The agent can change anything that can be written as optimizer code: the update rule of each parameter group, the
learning rates and their schedules, momentum and β₂, preconditioning, clipping, guards against loss spikes. It is
asked not to touch your model, data, loss, batch size, number of steps or evaluation. Its programs run inside your
training process, so this is a rule, not a sandbox (see [Safety](https://github.com/xingjian-bai/opal/blob/main/docs/GUIDE.md#safety)). Until the agent changes
something, the wrapped run is bit-for-bit the same as the plain one.

## Start with your coding agent

Open your own project in a coding agent (Claude Code, Codex, Cursor or similar) and paste this prompt. The agent
installs OPAL, wires it into your training script, asks you what it cannot know, and checks that nothing changed
before anything costs money.

```text
I want to train my model with OPAL, a reactive optimizer: a coding agent that rewrites the optimizer while the
model trains. Its docs are at https://github.com/xingjian-bai/opal. Please set it up for this project:

1. Read the README there and skills/train-with-agent/SKILL.md, and follow the skill. Do not edit OPAL's own files;
   configure everything through opal.wrap arguments.
2. Install OPAL into the Python environment this project trains with (pip install opal-optimizer), and
   run `opal doctor` (free, offline) to check that torch and Claude Code are ready on the machine that will run
   training; run `opal doctor --live` (one tiny paid agent call) only once I approve an agent run. If Claude Code is
   missing or not logged in there, tell me what to run; do not log in for me.
3. Find my training script and loop. Tell me what you found (optimizer, scheduler, number of steps, single GPU or
   DDP, gradient clipping, mixed precision, checkpoints) and anything OPAL does not support.
4. Ask me for the goal sentence: what the run trains and what it is judged by at the end. Do not invent the metric.
5. Wrap the optimizer and show me the diff before running anything.
6. Smoke test: about 100 steps unwrapped and 100 steps wrapped with backend="keep" (no agent, no cost); the losses
   must match step for step. Show me both. If I only asked for a dry run, stop here.
7. Before the real run, time an update in the smoke test and run `opal estimate --total-steps N --sec-per-step S`;
   show me what it prints (how long training will wait for the agent, also as a multiple of the run's own time, and
   the cost). Ask me for a cost cap (max_cost_usd); leave decisions unset unless I ask, so that the waiting stays
   within 4x the run's own time. Wait for my go-ahead; then give me the exact command to launch it, and how to
   follow it (opal show, opal watch, opal report).
```

## Start by hand

```bash
pip install opal-optimizer           # or: git clone, then pip install -e ".[examples]"
opal doctor                          # torch and Claude Code ready?
```

The package is called `opal-optimizer` on PyPI, because `opal` there is an unrelated web framework. Both install as `import opal`, so do not install both.

On a GPU machine, install OPAL into the environment you already train with, or install the PyTorch build for your
driver first ([pytorch.org](https://pytorch.org/get-started/locally/)). In an empty environment, `pip install
opal-optimizer` pulls the newest torch from PyPI, which may need a newer NVIDIA driver; `opal doctor` then shows
`CUDA available: False`.

```python
import torch, opal

opt = torch.optim.AdamW(model.parameters(), lr=3e-4)                  # your optimizer, unchanged
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)  # your scheduler, unchanged
opt = opal.wrap(opt, model, scheduler=sched, total_steps=steps,
                goal="ResNet-18 on CIFAR-10; judged by test accuracy after the last update",
                backend="keep")    # a dry run: no agent, no cost. For the agent run, use max_cost_usd=20 instead.

for step in range(steps):
    loss = criterion(model(x), y)
    loss.backward()
    opt.step(loss=loss)                     # pass the loss so the agent sees it
    opt.zero_grad()
    sched.step()
    if step % 500 == 0:
        opt.log(test_acc=evaluate(model))   # recorded for you, not shown to the agent
opt.close()
```

`goal` is one sentence: what you train and how it is judged. Then:

```bash
python train.py                                  # backend="keep": must match the plain run exactly
opal doctor --live                               # one tiny agent call to check the setup
opal estimate --total-steps 10000 --sec-per-step 0.05   # waiting time and cost, before you launch
python train.py                                  # with max_cost_usd=20: the agent rewrites the optimizer
opal show    opal_runs/<folder>                  # one line per decision
opal watch   opal_runs/<folder>                  # events as they are written; --timeout 0 prints what is there
opal report  opal_runs/<folder>                  # one HTML page: loss, learning rates, rationales, code diffs
opal compare opal_runs/<a> opal_runs/<b>         # two runs, or two result.json files, side by side
```

Each wrapped run prints a `profiler_start` / `profiler_stop` pair on stderr when a program starts: OPAL audits one
update of each new program. It is harmless.

The guide shows how to check the keep run against your plain run byte for byte, and how to use the Hugging Face
Trainer: [Try it without an agent](https://github.com/xingjian-bai/opal/blob/main/docs/GUIDE.md#try-it-without-an-agent).

## Prerequisites

- **A PyTorch training script** that uses `torch.optim` optimizers, on CPU, one GPU or several GPUs of one node (DDP).
  Python 3.10 or newer, PyTorch 2.5 or newer (2.6 on Python 3.13).
- **For an agent run, Claude Code, logged in on the machine that runs rank 0.** Install
  [Claude Code](https://docs.claude.com/en/docs/claude-code), run `claude` once and log in with a Claude subscription
  or an API key. The agent's usage is billed to that account. `max_cost_usd` caps it; there is no cap by default.
  The short summaries of code changes in the report are not counted (`summarize=False` turns them off). A dry run
  with `backend="keep"` needs neither and makes no model calls.
- **No internet or no login on the GPU machine?** Run the agent on your laptop or a login node and connect it to the
  training job through a mailbox folder, over ssh or a shared filesystem (see [the mailbox broker](https://github.com/xingjian-bai/opal/blob/main/docs/GUIDE.md#gpu-machines-without-an-agent)).

## Results

Each number is one run, and each agent started from the task's own recipe:

| Example | Measure | Official recipe | Reactive optimizer |
|---|---|---|---|
| SiT-B/2, ImageNet 256, 400K updates | FID-50K ↓ | 35.7 (ADM evaluator; the SiT paper reports 33.0) | 23.6 |
| RAEv2-B, ImageNet 256, 20 epochs | FID-50K ↓ | 3.39 | 3.19 |
| Goldfish 125M GPT, Swahili | FLORES-200 bits ↓ | 79.37 | 77.25 |
| modded-nanogpt record #36 | validation loss ↓ | 3.2781 | 3.2749 |
| LightningDiT-B/1, ImageNet 256, 160 epochs | FID-50K ↓ (with guidance) | 3.22 | 2.31 |

It does not always help: a run of modded-nanogpt on OPAL 0.2.0 ended at 3.2878, worse than record #36. Your
numbers may also differ from ours, because the agent can choose differently from one run to the next. The
[examples page](https://github.com/xingjian-bai/opal/blob/main/examples/README.md) has every number with its version, hardware, time and cost, the small CPU and
one-GPU scripts, and how to rerun each setting.

**Long runs suit it best.** An agent decision takes about the same time in any run, about 40 s on average in ours,
so the waiting adds up to a roughly fixed cost. In our runs of several hours it came to 0.49 h of 3.75 h on RAEv2-B,
0.67 h of 5.65 h on Goldfish and 1.05 h of 11.2 h on SiT-B/2. In short runs it dominates: our one-GPU scripts, which
set 50 decisions, took 27 to 31 min with the agent against 1 to 3 min without, and with `blocking=False` they ended
before the first answer arrived. The small scripts are a quick way to try OPAL. If you have the compute, use it on runs
of a few hours or more, where `blocking=False` also
[beat the official recipe](https://github.com/xingjian-bai/opal/blob/main/examples/README.md#training-without-waiting-for-the-agent).

Our run records and final weights are on Hugging Face: [snow-leopard/opal-results](https://huggingface.co/datasets/snow-leopard/opal-results).

## How it works

```
 training script ──grads──▶ updater.py ──▶ parameters        (every step, every rank)
        │                      ▲
        └──observations──▶ analysis.py ──reports──▶ agent    (rank 0, every `every` steps / at decisions)
                               ▲                      │
                               └──── new programs ◀───┘      (at decision points)
```

- **Two programs.** `updater.py` starts as a call to your own optimizer and scheduler. `analysis.py` starts with a few
  statistics of the loss and the gradients. The agent edits both in `opal_runs/<folder>/workspace/`.
- **One conversation for the whole run.** The agent remembers what it tried, what it predicted and what happened. Each
  decision ends with a short rationale: what it saw, what it changed and what it expects.
- **Safe swaps.** A new optimizer program takes over at the next step, on every GPU at once, and runs on trial for 20
  steps. If it raises an error, the run goes back to the previous program. A failure at its first step is undone
  exactly when a copy of the parameters and optimizer state fits in `snapshot_limit_gb` (default 2); a later failure
  is not undone, and the agent is told. Every version is kept in `programs/`.
- **Training waits for the agent by default.** With `blocking=False` training keeps going while the agent thinks. That is
  faster, and noisier.
- **What the agent sees.** Your goal sentence, the names and shapes of the parameters, the training losses, its own
  analysis reports, and read-only copies of your training script and model code (`share_code=False` keeps them out).
  Values you log are only recorded, unless you pass `show_agent=True`. With `validation=N` it also sees the
  validation-set values you report with `opt.report_dev(...)`.

## Options

| Option | Default | Meaning |
|---|---|---|
| `decisions` | see right | How many times the agent is asked. Each time it may also keep everything as it is. If you do not set it, a run that waits for the agent asks at most as many times as keep the waiting under 4× the run's own training time (`opal estimate` shows the number), and at most 50, but at least once, even when one decision already waits longer than that. If you set it, that cap is off. A fifth of the decisions fall in the first 5% of the run; `schedule="uniform"` spaces them evenly. More in the [guide](https://github.com/xingjian-bai/opal/blob/main/docs/GUIDE.md#cost-and-limits). |
| `every` | 50 | Steps between checks: losses are read, a non-finite loss is caught, the analysis program runs. |
| `blocking` | `True` | `False` to keep training while the agent thinks. |
| `max_cost_usd` | none | Stop asking the agent once it has spent this much. |
| `backend_options` | `{}` | `model=` (default `claude-opus-5-5`), `effort=` (default `high`) and `timeout=`. A failed agent turn is retried with growing pauses; an agent that is unavailable (login, usage limit, outage) is waited for, with no deadline unless you set `unavailable_timeout=`. Details in the [guide](https://github.com/xingjian-bai/opal/blob/main/docs/GUIDE.md#cost-and-limits). |
| `prompt` | `"neutral"` | The agent's instructions: the facts of the run and the protocol, with no recommended values. `"aggressive"` adds a review of every group's hyperparameters at each decision. You can also pass the path of your own template. |
| `validation` | none | `N`: your script evaluates a held-out validation set about every N updates, at times it does not announce, and reports it with `opt.report_dev(loss=..., accuracy=...)`. The agent sees each result. See the [guide](https://github.com/xingjian-bai/opal/blob/main/docs/GUIDE.md#what-it-can-and-cannot-change). |
| `eval_integrity` | on with `validation` | Flags a run whose program moves the parameters somewhere else for an evaluation and back afterwards. The agent is not told. |

Gradient accumulation, GradScaler, checkpoints and resume, DDP, GPU machines without internet, privacy and cost are
covered in the [guide](https://github.com/xingjian-bai/opal/blob/main/docs/GUIDE.md).

### Experimental Codex backend

Claude is the default. To use Codex, pass `backend="codex", backend_options=dict(model="gpt-6.1-sol", effort="high")`
(`xhigh` works too), or run `opal broker <box> --backend codex --model gpt-6.1-sol --effort high`. Log in separately
with `CODEX_HOME=~/.codex-opal codex login --device-auth`, and do not copy or link the `auth.json` of your default
Codex home. `max_cost_usd` does not cap Codex; `backend_options=dict(max_tokens=N)` is a soft token cap that a turn
in progress can exceed. The run stops with an error if Codex compacts the conversation (around 450k tokens of context
with Codex CLI 0.160.1).

## Tests

The whole test suite runs on CPU with no agent and no cost: `pip install -e ".[examples,test,hf]"`, then `pytest -q`. It
takes a few minutes on a laptop. Three tests are skipped unless you set `OPAL_REAL_CLI=1`; they start the real Claude
Code CLI against a local mock.

## Status and limitations

v0.3.9, under active development. Tested on one GPU and on DDP within one node (NCCL, eager and `torch.compile`).
Multi-node, FSDP and DeepSpeed are not supported. Accelerate works if you wrap after `accelerator.prepare`, without
fp16 GradScaler or Accelerate's gradient accumulation; the HF Trainer works through `opal.hf.OpalTrainer`. Both are
tested on CPU. Lightning should work through `opt.observe(loss)` but we have not tried it. The agent can make a run
worse: it cannot restart or roll back training, it can only fix its programs at the next decision.

## Citation

```bibtex
@misc{bai2026reactiveoptimizers,
      title={Reactive Optimizers},
      author={Xingjian Bai and Jia-Hao Liang and Jayden Teoh and Vincent Sitzmann},
      year={2026},
      eprint={2610.33826},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2610.33826},
}
```

## License

MIT, see [LICENSE](https://github.com/xingjian-bai/opal/blob/main/LICENSE).

*The name:* OPAL stands for OPtimizer with an Agent in the Loop.
