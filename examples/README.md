# Examples

Everything we ran with OPAL, with the numbers. Each large example has its own folder with the official repository and
commit, the lines that wrap the optimizer, the commands for both runs, the hardware, time and cost, and where each
number comes from.

## Results by setting

Each number is one run, and each agent started from the task's own recipe. Runs on OPAL 0.3 used the neutral prompt
and Opus 5.5 at effort high. SiT-B/2 comes from the harness this package grew out of. In all of these runs, training
paused at each decision while the agent answered, which is OPAL's default.

| Example | Measure | Official recipe | Reactive optimizer | Made with |
|---|---|---|---|---|
| [SiT-B/2](sit_b2/), ImageNet 256, 400K updates, 8× H100 | FID-50K ↓ | 35.7 with the ADM evaluator (the SiT paper reports 33.0; Muon: 30.4) | 23.6 | earlier harness |
| [RAEv2-B](raev2_b/), ImageNet 256, 20 epochs, 8× H100 | FID-50K ↓ | 3.39 | 3.19 | OPAL 0.3 |
| [Goldfish 125M GPT](goldfish_swahili/), 1 GB of Swahili, 1× H100 | FLORES-200 bits ↓ | 79.37 | 77.25 | OPAL 0.3 |
| [modded-nanogpt record #36](modded_nanogpt/), 162M GPT, 8× H100 | validation loss ↓ | 3.2781 | 3.2749 (a small gain: the track assumes σ = 0.0013 per run) | OPAL 0.1.0 |
| LightningDiT-B/1, ImageNet 256, 160 epochs (200K updates), 8× H100 | FID-50K ↓, with guidance / without | 3.22 / 10.12 | 2.31 / 7.56 | OPAL 0.3 |

It does not always help. On modded-nanogpt, a heavily tuned benchmark, a run on OPAL 0.2.0 with the neutral prompt
ended at 3.2878, worse than record #36. Your numbers may also differ from ours, because the agent can make different
choices from one run to the next and hosted models change between versions. Still, most of our runs on OPAL 0.3
ended ahead of the recipe they started from. The [blog post](https://xingjianbai.com/reactive-optimizers/) has more.

LightningDiT-B/1 has no folder here. Both runs used the upstream LightningDiT training code and its 64-epoch config, changed to model B/1, 200,000 updates and `use_qknorm: true` (without query-key normalization the official run diverged near update 194,000). The official recipe trains with AdamW at a constant learning rate of 2e-4. FID-50K uses 50,000 samples against the ADM reference statistics, Euler sampling with 250 steps and timestep shift 0.3; "with guidance" is the upstream setting, a guidance scale of 10 from t = 0.10 on. The LightningDiT paper reports FID-10K for this model, which is not comparable.

## Training without waiting for the agent

With `blocking=False`, training continues on the current programs while the agent answers, and an edit takes effect
when its answer arrives. One run per setting, OPAL 0.3. RAEv2-B, 8× H100:

| RAEv2-B, 20 epochs | Official recipe | OPAL, training waits (default) | OPAL, `blocking=False` |
|---|---|---|---|
| FID-50K ↓ after 25,000 updates | 3.39 | 3.19 | 3.17 |
| Run time, including evaluations | 3.21 h | 3.75 h, of which 0.49 h waiting for the agent | 3.22 h |
| Decisions asked / that changed code | – | 50 / 6 | 43 / 3 |
| Agent cost | – | $13.30 | $9.71 |

LightningDiT-B/1, with the recipe and evaluation above. The first two runs used 8× H100. The `blocking=False` run was
interrupted after update 160,000 and resumed from its checkpoint at that update, in the same agent conversation, on
8× A100 80GB for the last 40,000 updates:

| LightningDiT-B/1, 160 epochs | Official recipe | OPAL, training waits (default) | OPAL, `blocking=False` |
|---|---|---|---|
| FID-50K ↓ after 200,000 updates, with guidance / without | 3.22 / 10.12 | 2.31 / 7.56 | 2.65 / 8.83 |
| Training time | 4.62 h | 6.13 h, of which 0.66 h waiting for the agent | 4.52 h to update 160,000 on H100, then 2.23 h on A100 |
| Decisions asked / that changed code | – | 50 / 13 | 47 / 11 |
| Agent cost | – | $15.19 | $17.25 |

Without waiting, fewer decisions are asked, because no new decision is asked while one is in progress.

## Small examples (CPU)

`mnist_mlp.py` trains an MLP on MNIST with AdamW and a cosine schedule for 5 epochs. It doubles as the install
check. From a clone, after `pip install -e ".[examples]"`:

```bash
python examples/mnist_mlp.py                            # plain PyTorch: final test accuracy 0.9850
python examples/mnist_mlp.py --opal --backend keep      # wrapped, no agent, free: the same run, bit for bit
python examples/mnist_mlp.py --opal                     # Claude Code rewrites the optimizer (capped at $5)
```

On an M1 Max CPU the first two take about 15 s each and print the same `parameter checksum` (the first 12 hex digits
of the sha256 of the final parameters). Our agent run ended at 0.9863. It took 862 s, 532 s of which was waiting for
the agent; the agent was asked 42 times, changed the program 6 times and cost $2.70.

## More scripts (one GPU)

| Script | Task | Loss ↓ (plain → OPAL) | Task metric ↑ (plain → OPAL) | Time: plain → OPAL |
|---|---|---|---|---|
| `cifar10_resnet.py` | ResNet-18 on CIFAR-10, SGD-Nesterov + one-cycle | test cross-entropy 0.2576 → 0.2512 | test accuracy 94.49% → 94.56% | 66 s → 27 min |
| `diffusion_mnist.py` | small UNet flow-matching diffusion on MNIST, EMA | EMA validation MSE 0.1696 → 0.1623 | none | 71 s → 29 min |
| `gpt_small.py` | 34M GPT on FineWeb tokens, AdamW + warmup-cosine | validation cross-entropy 4.4763 → 4.3419 | none | 186 s → 31 min |

Each row is one plain run and one OPAL run on one H100. The scripts use recipes we wrote from common settings; we did not tune their learning rates or other hyperparameters. On CIFAR-10 the gain is within run-to-run
noise: a second run in which the agent changed nothing ended at 94.63%.

Each script runs plain by default. `--opal` wraps the optimizer and `--backend keep` makes it a free dry run.

These runs are short, so most of the OPAL time is training waiting for the agent. Keep the default `blocking=True`
for short runs.

## Codex backend

Our single runs with the experimental Codex backend (gpt-6.1-sol, effort high) on one H100, against plain:
`gpt_small` validation loss 4.3125 vs 4.4763, `diffusion_mnist` EMA validation loss 0.1687 vs 0.1696.
