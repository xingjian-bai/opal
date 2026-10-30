# examples/overfit

These CIFAR-10 runs overfit with static AdamW. They use the validation feature added in OPAL 0.1.1-validation.

With static AdamW, validation and test loss rise as accuracy peaks and falls.
Does an agent shown validation loss and accuracy through `wrap(validation=N)` keep this from happening?
The baselines are static AdamW with learning rate swept on the validation set, and early stopping on that set.
The early-stopping score is test accuracy at the best validation evaluation.
The script logs what is needed and saves `best.pt`.

`examples/overfit/splits.py` defines two settings with the same splits and labels in every run:

- `label10`: 45,000 training images, with 10%, or 4,500, assigned a wrong label.
  Wrong labels are uniform over the other classes, with a fixed seed.
- `small20`: 10,000 clean training images, 20% of CIFAR-10's training set, with 1,000 per class.

Both hold out 5,000 clean validation images, 10% of each class from the official training set.
The official 10,000 test images are logged for us and never shown to the agent.

The model is ResNet-18 with a CIFAR stem, batch norm and no dropout.
`--width 64` gives 11.2M parameters. `--width 128` gives a wider model more prone to overfit.
It normalises logits with centred LogitNorm, τ = 0.04, as in the blog's run.
This prevents the agent from setting a temperature on the validation set.
Data augmentation is off by default. `--augment` adds flips and 4-pixel crops.

AdamW uses `--lr`, betas (0.9, 0.999) and `--wd` weight decay on every tensor.
The schedule warms up linearly over `--warmup-epochs`, then decays to 0 with a cosine.
`--schedule constant` uses warmup followed by a flat rate. The loss is cross-entropy without label smoothing.
`--data-seed` sets a new data order each epoch. `--seed` sets model initialisation.

    python examples/overfit/train.py --prepare-data                                   # download CIFAR-10 once
    python examples/overfit/train.py --setting label10 --lr 1e-3                      # static AdamW
    python examples/overfit/train.py --setting label10 --opal --validation            # agent, sees the validation set
    python examples/overfit/train.py --setting label10 --opal                         # agent, no validation set shown
    python examples/overfit/train.py --setting small20 --data synthetic --device cpu --width 4 --epochs 2 \
        --batch-size 64 --opal --validation --backend keep                            # CPU smoke test

The blog's run used label10 for 120 epochs, with its goal sentence passed through `--goal`:

    GOAL="ResNet-18 (11.2M parameters) trained from scratch on 45,000 CIFAR-10 training images for 120 epochs; judged by the cross-entropy loss on the 10,000 CIFAR-10 test images after the last update (lower is better). Do not change the parameters for an evaluation and change them back afterwards."
    python examples/overfit/train.py --setting label10 --lr 3e-3 --epochs 120 --opal --validation \
        --max-cost-usd 40 --eval-seed 0 --goal "$GOAL"

That run was judged by test cross-entropy loss. `train.py` defaults to an accuracy goal.
Its goal also forbids changing parameters for an evaluation and changing them back.

Agent runs use OPAL's Claude defaults: `--model` is `claude-opus-5-5`, with effort `high`.
They use the neutral prompt, 30 uniformly spaced decisions and blocking.
The agent gets read-only copies of `train.py` and `resnet.py`. `--no-share-code` withholds them.
It does not get `splits.py`.

The files are split so the agent sees the training script and model source.
The experiment's question, label10's wrong labels, test scores, early stopping and result stay in this README,
`splits.py` and `records.py`, which it does not see.
The default output folder has a neutral name. The agent's working directory is inside it.

`--out` contains `log.jsonl`, `result.json` and `best.pt`, the model with the best validation accuracy.
Agent runs also write `opal/`, the OPAL run folder.
`log.jsonl` has a config line, one line per evaluation and a result line.
Evaluation lines record training loss and accuracy since the previous evaluation, validation and test loss and
accuracy. For label10, they also count wrongly labelled training images predicted with their wrong or true label.

In 0.2.2, validation evaluation happens after one uniformly random update per epoch and after the last update.
`--evals-per-epoch K` draws one in each K-th part of the epoch.
`records.eval_updates` draws them from `--eval-seed`. The config line lists them as `eval_updates`.
Test evaluation happens at the same updates, for us only.

The draw lives in `records.py`, so the agent's copy of `train.py` shows no cadence.
The agent is told "about every N updates, at times that are not announced", with N = updates per epoch / K.
Until 0.2.1, the script evaluated every `--dev-every` updates, by default one epoch, and the prompt said
"Every N updates". Two small20 agent runs timed parameter changes to evaluation: a classifier scale probe,
and a weight average swapped in for each evaluation.
`wrap(validation=N)` now checks for this through `eval_integrity`.

With `--test-at-end`, prepare the CIFAR-10 cache separately using `--prepare-data`.
Training reads only `x_train` and `y_train` from the NPZ cache. It loads and scores no test arrays.
After the last update and `opt.close()`, it scores the final model and the models selected by validation accuracy
and loss on the test set. Per-evaluation logs omit test scores.
`result.json` keeps its keys, with `peak_test_acc_oracle` and `overfit_gap_test_acc` set to null.
The two validation-selected states stay in CPU memory even with `--no-save-best`.
The flag changes OPAL's timing observations and training-source copy.
It does not sandbox agent programs or restrict filesystem access.
