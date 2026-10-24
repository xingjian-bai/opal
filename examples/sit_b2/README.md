# SiT-B/2 on ImageNet 256, 400K updates

SiT-B/2 is a base-size flow-matching transformer with patch size 2, trained on SD-VAE latents.
We trained it from scratch on ImageNet 256×256 for 400,000 updates with batch 256.
The score is FID-50K of samples from the EMA weights, without guidance.

| FID-50K ↓ (one run each) | Official AdamW | Muon | Reactive optimizer |
|---|---|---|---|
| at 400K updates, ADM evaluator | 35.70 | 30.39 | **23.55** |

The numbers above come from the earlier training system OPAL grew from in October 2026.
A training-side host connected to a broker running Claude Code, Opus 5.5 at effort `high`, with that system's prompt.
It made 94 decisions: 19 every 1,251 updates, then every 5,004. Training paused at each decision.
The wiring below is for OPAL 0.3.x. It has CPU checks only.

## Official code

[willisma/SiT](https://github.com/willisma/SiT) at commit `cbde832a40b153ccc79603412409da9c9b0c568c`, MIT license.
Copyright (c) Meta Platforms, Inc. and affiliates.
The recipe is upstream's `train.py`: AdamW, learning rate 1e-4, betas (0.9, 0.999), weight decay 0,
constant learning rate, batch 256, EMA 0.9999, linear path with velocity prediction.

## Wiring it to OPAL

[`opal.patch`](opal.patch) changes upstream `train.py` in five places: `--max-steps` to stop at 400K,
`opal.wrap(...)` behind `--opal`, `opt.step(loss=loss)`, `opt.close()` at the end, and the flags
`--opal-max-cost-usd` and `--opal-backend`. Without `--opal`, the script runs as upstream does.
The checks cover patch application to that commit and compilation of the patched file.

```bash
git clone https://github.com/willisma/SiT && cd SiT && git checkout cbde832a40b153ccc79603412409da9c9b0c568c
git apply /path/to/opal/examples/sit_b2/opal.patch
pip install -e /path/to/opal               # into SiT's environment (environment.yml)
export ENTITY=none PROJECT=none            # upstream train.py reads these even without --wandb

# official recipe
torchrun --nproc_per_node=8 train.py --model SiT-B/2 --data-path /data/imagenet/train --max-steps 400000
# dry run: wrapped, no agent, no cost; must match the official run (check the first few hundred updates)
torchrun --nproc_per_node=8 train.py --model SiT-B/2 --data-path /data/imagenet/train --max-steps 400000 \
    --opal --opal-backend keep
# reactive optimizer, OPAL defaults (neutral prompt, 50 decisions, training waits for each)
torchrun --nproc_per_node=8 train.py --model SiT-B/2 --data-path /data/imagenet/train --max-steps 400000 \
    --opal --opal-max-cost-usd 20

# 50,000 samples of the EMA weights: upstream's SDE sampler (Euler, 250 steps), no guidance, seed 7 as in our runs
torchrun --nproc_per_node=8 sample_ddp.py SDE --model SiT-B/2 --ckpt results/<run>/checkpoints/0400000.pt \
    --num-fid-samples 50000 --num-sampling-steps 250 --cfg-scale 1.0 --global-seed 7
```

For FID-50K, torch-fidelity's Inception-v3 in compat mode, `inception-v3-compat`, extracts 2048-dimensional pool
features from 50,000 uint8 PNG samples. FID is the Fréchet distance between their mean and covariance and the
`mu` and `sigma` in `jit_in256_stats.npz`. These are ImageNet 256 reference statistics shipped in `fid_stats/` by
[LTH14/JiT](https://github.com/LTH14/JiT). Its `prepare_ref.py` builds the reference from the training set.
Our file matches the sha256 at JiT commit `cbc743a`:
`412046720c0d496dc6d72a30eac3e22b9ef6ddac3501cce7f996674ad227ba4c`.
This repository has no script for that step.

Our samples used our own script with upstream's sampler settings and seed 7 on 2 GPUs.
It draws noise per GPU differently from `sample_ddp.py`, so the command above gives different images.
The score can differ through sampling noise.

Claude Code must be installed and logged in on the machine that runs rank 0, or use the mailbox broker
([guide](../../docs/GUIDE.md#gpu-machines-without-an-agent)). The run's records go in `opal_runs/<time>-...` in the
working directory; `opal report` turns them into one page.

## Our runs

| | Official AdamW | Muon | Reactive optimizer |
|---|---|---|---|
| Hardware | 8× H100 80GB | 8× H100 80GB | 8× H100 80GB |
| Training loop time | 7.1 h | 13.0 h | 11.2 h, of which 1.05 h waiting for the agent |
| Agent cost | – | – | $11.28 |
| FID-50K at 400K | 35.684 | 30.290 | 23.643 |
| FID-50K at 400K, ADM evaluator | 35.70 | 30.39 | 23.55 |
| sFID / IS / precision / recall, ADM evaluator | 6.57 / 42.0 / 0.52 / 0.65 | 5.94 / 47.4 / 0.56 / 0.64 | 5.85 / 60.6 / 0.59 / 0.64 |

Our runs used a modified upstream `train.py`. It precomputed the centre crop and flip's latent posterior mean and
standard deviation, then drew a fresh sample at every update, as upstream does with on-the-fly encoding.
It added our logging, checkpoints and in-run FID-10K. That code, the earlier training system and its prompt are not
in this repository.

We evaluated 50,000 samples from the EMA weights with upstream's SDE sampler, Euler–Maruyama, 250 steps, no guidance
and seed 7. FID used torch-fidelity's Inception-v3 compat mode against JiT's `jit_in256_stats.npz`, described above.
The SiT paper uses the ADM evaluator and reports 33.0 for SiT-B/2 at 400K. We also scored a second set of 50,000
samples per run with it (reference `VIRTUAL_imagenet256_labeled.npz`): the ADM rows above. On those samples our
evaluator gives 35.67, 30.37 and 23.55.

Our Muon baseline used momentum 0.95, Nesterov, 5 Newton–Schulz steps and update scale 0.2·√max(m, n) on matrices.
AdamW handled the rest. Learning rate was 1e-4 and weight decay 0. Neither baseline was tuned.

Of the agent's 94 decisions, 15 changed a file and 10 changed the update rule.
The last change was at update 190,152.
