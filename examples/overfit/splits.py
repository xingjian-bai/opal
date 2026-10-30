"""Data for examples/overfit: CIFAR-10, a validation split held out from the training set, and the two settings.

Kept apart from train.py on purpose: the agent's reference/ folder holds the training script and the model source, not
this file (OPAL copies the script and the source of user classes only).

Splits (fixed seeds, NumPy only, the same for every run):
- validation: 10% of each class of the official 50,000 training images (5,000 images, 500 per class), clean labels;
- training pool: the other 45,000;
- test: the official 10,000 test images, clean; logged for us, never shown to the agent.

Settings:
- label10: the whole pool (45,000 images); 10% of them (4,500, chosen uniformly at random) get a wrong label, uniform
  over the 9 other classes, fixed for the whole run;
- small20: 20% of the official training set (10,000 images, 1,000 per class) drawn from the pool, clean labels.

The download follows examples/cifar10_resnet.py (Hugging Face parquet, sha256-checked, with the Toronto archive as a
fallback) and shares its cache file, data/cifar10/cifar10_uint8.npz.
"""

from __future__ import annotations

import hashlib
import io
import os
import tarfile
import time
import urllib.request

import numpy as np

HF_BASE = (
    os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
    + "/datasets/uoft-cs/cifar10/resolve/main/plain_text/"
)
HF_FILES = {  # LFS sha256 published by Hugging Face (as in examples/cifar10_resnet.py)
    "train-00000-of-00001.parquet": "8428b53a88a11ac374111006708df51469e315a22ac6d66470afd9c78d2ae883",
    "test-00000-of-00001.parquet": "841389e6f2d64f28bf17310e430aebac20ec3ba611a3c5e231dc93c645ce84de",
}
TORONTO_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz"
TORONTO_MD5 = "c32a1d4ab5d03f1284b67883e8d87530"

SPLIT_SEED = 20261018  # validation split and the small20 subset
NOISE_SEED = 888  # which labels are wrong in label10, and their wrong class
DEV_FRACTION = 0.10  # of each class of the official training set
NOISE_FRACTION = 0.10  # label10: of the training images
SMALL_FRACTION = 0.20  # small20: of each class of the official training set
SETTINGS = ("label10", "small20")
EPOCHS = {"label10": 60, "small20": 150}  # long enough for a static AdamW run to overfit; to be calibrated by pilots


def _download(url, path, algo, want):
    part = path + ".part"
    print(f"downloading {url}", flush=True)
    h, t0 = hashlib.new(algo), time.time()
    with urllib.request.urlopen(url, timeout=60) as r, open(part, "wb") as f:
        while chunk := r.read(1 << 22):
            f.write(chunk)
            h.update(chunk)
    if h.hexdigest() != want:
        os.remove(part)
        raise RuntimeError(f"{algo} mismatch for {url}: got {h.hexdigest()}, expected {want}")
    os.replace(part, path)
    print(f"  {os.path.getsize(path) / 1e6:.0f} MB in {time.time() - t0:.0f} s", flush=True)


def _from_hf(data_dir):
    import pyarrow.parquet as pq
    from PIL import Image

    out = []
    for name, sha in HF_FILES.items():
        path = os.path.join(data_dir, name)
        if not os.path.exists(path):
            _download(HF_BASE + name, path, "sha256", sha)
        table = pq.read_table(path).to_pydict()
        images = np.stack([np.asarray(Image.open(io.BytesIO(d["bytes"])).convert("RGB")) for d in table["img"]])
        out += [images, np.asarray(table["label"], dtype=np.int64)]
    return out


def _from_toronto(data_dir):
    path = os.path.join(data_dir, "cifar-10-binary.tar.gz")
    if not os.path.exists(path):
        _download(TORONTO_URL, path, "md5", TORONTO_MD5)
    with tarfile.open(path) as tar:

        def batch(name):
            raw = np.frombuffer(tar.extractfile(f"cifar-10-batches-bin/{name}").read(), dtype=np.uint8)
            raw = raw.reshape(-1, 3073)
            return raw[:, 1:].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1), raw[:, 0].astype(np.int64)

        train = [batch(f"data_batch_{i}.bin") for i in range(1, 6)]
        x_test, y_test = batch("test_batch.bin")
    return np.concatenate([x for x, _ in train]), np.concatenate([y for _, y in train]), x_test, y_test


def load_cifar10(data_dir):
    """Official CIFAR-10 as uint8 N,H,W,C arrays: x_train, y_train, x_test, y_test."""
    os.makedirs(data_dir, exist_ok=True)
    cache = os.path.join(data_dir, "cifar10_uint8.npz")
    if not os.path.exists(cache):
        try:
            arrays = _from_hf(data_dir)
        except Exception as e:  # noqa: BLE001 - any failure falls back to the original archive
            print(f"Hugging Face source failed ({e!r}); using the Toronto archive", flush=True)
            arrays = _from_toronto(data_dir)
        x_tr, y_tr, x_te, y_te = arrays
        assert x_tr.shape == (50000, 32, 32, 3) and x_te.shape == (10000, 32, 32, 3), (x_tr.shape, x_te.shape)
        np.savez(cache + ".tmp.npz", x_train=x_tr, y_train=y_tr, x_test=x_te, y_test=y_te)
        os.replace(cache + ".tmp.npz", cache)
    with np.load(cache) as z:
        return z["x_train"], z["y_train"], z["x_test"], z["y_test"]


def synthetic(n_train=1024, n_test=256, seed=0):
    """Random images and labels of CIFAR's shape, for CPU smoke tests."""
    g = np.random.default_rng(seed)
    return (
        g.integers(0, 256, (n_train, 32, 32, 3), dtype=np.uint8),
        g.integers(0, 10, n_train),
        g.integers(0, 256, (n_test, 32, 32, 3), dtype=np.uint8),
        g.integers(0, 10, n_test),
    )


def make_setting(setting, x_train, y_train, x_test, y_test):
    """The arrays of one setting: dict with train_x, train_y (labels as trained on), train_clean_y, train_wrong (bool
    mask of the replaced labels), dev_x, dev_y, test_x, test_y, and a description of the split for the run log."""
    if setting not in SETTINGS:
        raise ValueError(f"setting must be one of {SETTINGS}, got {setting!r}")
    rng = np.random.default_rng(SPLIT_SEED)
    dev, pool, small = [], [], []
    for c in range(10):
        ids = rng.permutation(np.flatnonzero(y_train == c))
        n_dev = int(round(DEV_FRACTION * len(ids)))
        n_small = int(round(SMALL_FRACTION * len(ids)))
        dev.extend(ids[:n_dev])
        pool.extend(ids[n_dev:])
        small.extend(ids[n_dev : n_dev + n_small])
    dev, pool, small = (np.sort(np.asarray(a, dtype=np.int64)) for a in (dev, pool, small))
    train = pool if setting == "label10" else small
    ty = y_train[train].copy()
    wrong = np.zeros(len(train), bool)
    if setting == "label10":
        g = np.random.default_rng(NOISE_SEED)
        wrong[g.choice(len(train), int(round(NOISE_FRACTION * len(train))), replace=False)] = True
        shift = g.integers(1, 10, len(train))  # a uniformly random other class
        ty = np.where(wrong, (ty + shift) % 10, ty)
    assert not set(dev.tolist()) & set(train.tolist())
    info = dict(
        setting=setting,
        train=int(len(train)),
        dev=int(len(dev)),
        test=int(len(y_test)),
        wrong_labels=int(wrong.sum()),
        train_index_sha256=hashlib.sha256(train.tobytes()).hexdigest(),
        dev_index_sha256=hashlib.sha256(dev.tobytes()).hexdigest(),
        label_sha256=hashlib.sha256(ty.astype(np.int64).tobytes()).hexdigest(),
    )
    return dict(
        train_x=x_train[train],
        train_y=ty.astype(np.int64),
        train_clean_y=y_train[train].astype(np.int64),
        train_wrong=wrong,
        dev_x=x_train[dev],
        dev_y=y_train[dev].astype(np.int64),
        test_x=x_test,
        test_y=np.asarray(y_test, dtype=np.int64),
        info=info,
    )
