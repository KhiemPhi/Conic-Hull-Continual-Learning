"""
download_datasets.py
--------------------
One-shot downloader for every dataset referenced by this repo, into ./data so
that demo_cone_boundary.py (and the other demos / reproduce_*.sh scripts) can run
without a first-run download stall.

Datasets
--------
torchvision (clean download → ./data):
    CIFAR100   (demo_cone_boundary, demo_incremental, demo_joint_floor)
    CIFAR10    (features.py, incremental.py)
    STL10      (features.py)
    FGVCAircraft, Flowers102, OxfordIIITPet, Food101   (demo_joint_floor)
HuggingFace `datasets` (not cleanly in torchvision):
    CUB200        = Donghyun99/CUB-200-2011
    StanfordCars  = Donghyun99/Stanford-Cars
    ImageNet-A    = barkermrl/imagenet-a   (200 classes, natural adversarial examples)
    ImageNet-R    = axiong/imagenet-r      (200 classes, renditions)
Backbone-selection grid (exp65): tasks chosen to span distance from the backbones'
pretraining distribution, which is the axis backbone rankings reorder along:
    RESISC45      = timm/resisc45          (45 classes, satellite -- non-natural imagery)
    SVHN          = ufldl-stanford/svhn    (10 classes, `cropped_digits` -- structured /
                                            low-level, where CLIP's linear probe misleads)
    DomainNet     = wltjr1007/DomainNet    (345 classes x 6 domains -- clipart, infograph,
                                            painting, quickdraw, real, sketch)

DomainNet carries the domain as a COLUMN, and all six domains are interleaved across the
same parquet shards, so there is no way to pull only sketch/quickdraw: the full 18.5 GB
comes down and the loader filters on `domain` afterwards. It is therefore excluded from
the default run and must be requested by name (see HEAVY below).

Two upstream mirrors are unreachable from a Meta devserver -- csr.bu.edu (official
DomainNet) and ufldl.stanford.edu (SVHN, which is what torchvision's D.SVHN fetches) are
both refused by the fwdproxy destination filter, not merely slow. Both are routed through
HuggingFace here instead; do not "fix" SVHN back to torchvision without re-checking that.

Usage
-----
    # everything except the HEAVY sets
    python -u download_datasets.py

    # a subset (HEAVY sets are only ever fetched when named explicitly)
    python -u download_datasets.py --only CIFAR100 CUB200
    python -u download_datasets.py --only DomainNet

    # just what demo_cone_boundary needs (CIFAR-100)
    python -u download_datasets.py --only CIFAR100

    # control parallelism (default 4; -j1 = sequential, cleanest logs)
    python -u download_datasets.py -j 8

Datasets download in parallel across threads (network/disk I/O bound); per-line
log tags ([hf]/[tar]/[proxy]/[ok]) keep interleaved output attributable. Use
-j1 for fully sequential, non-interleaved output.

On a Meta devserver external downloads need the proxy; this script sets
http(s)_proxy=http://fwdproxy:8080 automatically if unset (override with
--no-proxy or by exporting your own). CUB/Cars require `pip install datasets`.
"""
import argparse
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

DATA_DIR = "./data"

# HuggingFace repos for the non-torchvision sets. CUB200/StanfordCars mirror
# demo_joint_floor.HF_REPOS; ImageNet-A/R are the Hendrycks robustness sets
# (200 classes each) pulled from community mirrors with image/label columns.
# A value may be either "repo" or ("repo", "config") where the repo ships more than one
# config -- SVHN has cropped_digits (the 32x32 classification set everyone means) and
# full_numbers (detection crops), and load_dataset refuses to guess between them.
HF_REPOS = {
    "CUB200": "Donghyun99/CUB-200-2011",
    "StanfordCars": "Donghyun99/Stanford-Cars",
    "ImageNet-A": "barkermrl/imagenet-a",
    "ImageNet-R": "axiong/imagenet-r",
    "RESISC45": "timm/resisc45",
    "SVHN": ("ufldl-stanford/svhn", "cropped_digits"),
    "DomainNet": "wltjr1007/DomainNet",
}

# All dataset keys this script knows how to fetch.
TORCHVISION = ["CIFAR100", "CIFAR10", "STL10", "FGVCAircraft", "Flowers102",
               "OxfordIIITPet", "Food101"]
ALL = TORCHVISION + list(HF_REPOS)

# Sets big enough that pulling them by accident is a real cost. Excluded from the default
# run; `--only DomainNet` still works. DomainNet is 18.5 GB because the six domains cannot
# be requested separately (see module docstring).
HEAVY = ["DomainNet"]
DEFAULT = [n for n in ALL if n not in HEAVY]


def _set_proxy():
    """Devserver external downloads route through fwdproxy unless already set."""
    for var in ("http_proxy", "https_proxy"):
        os.environ.setdefault(var, "http://fwdproxy:8080")
    print(f"[proxy] http_proxy={os.environ.get('http_proxy')}")


def _download_torchvision(name):
    from torchvision import datasets as D

    os.makedirs(DATA_DIR, exist_ok=True)
    # Each set is pulled via its native splits; transform=None (we only want bytes
    # on disk). The demos add their own transforms at load time.
    if name == "CIFAR100":
        D.CIFAR100(DATA_DIR, train=True, download=True)
        D.CIFAR100(DATA_DIR, train=False, download=True)
    elif name == "CIFAR10":
        D.CIFAR10(DATA_DIR, train=True, download=True)
        D.CIFAR10(DATA_DIR, train=False, download=True)
    elif name == "STL10":
        D.STL10(DATA_DIR, split="train", download=True)
        D.STL10(DATA_DIR, split="test", download=True)
    elif name == "FGVCAircraft":
        D.FGVCAircraft(DATA_DIR, split="trainval", download=True)
        D.FGVCAircraft(DATA_DIR, split="test", download=True)
    elif name == "Flowers102":
        D.Flowers102(DATA_DIR, split="train", download=True)
        D.Flowers102(DATA_DIR, split="test", download=True)
    elif name == "OxfordIIITPet":
        D.OxfordIIITPet(DATA_DIR, split="trainval", download=True)
        D.OxfordIIITPet(DATA_DIR, split="test", download=True)
    elif name == "Food101":
        D.Food101(DATA_DIR, split="train", download=True)
        D.Food101(DATA_DIR, split="test", download=True)
    else:
        raise ValueError(f"unknown torchvision dataset '{name}'")


def _download_hf(name):
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError(
            f"HuggingFace `datasets` needed for {name}: pip install datasets"
        ) from e

    spec = HF_REPOS[name]
    repo, config = spec if isinstance(spec, tuple) else (spec, None)
    # Cache under ./data/hf so it lives next to the torchvision sets. The demos use
    # the default HF cache; set HF_HOME below so both agree.
    cache = os.path.join(DATA_DIR, "hf")
    os.makedirs(cache, exist_ok=True)
    os.environ.setdefault("HF_HOME", cache)
    print(f"[hf] downloading {repo}{f' [{config}]' if config else ''} → {cache}")
    dd = load_dataset(repo, config, cache_dir=cache)
    print(f"[hf] {repo} splits: { {k: len(v) for k, v in dd.items()} }")
    print(f"[hf] {repo} columns: {dd[next(iter(dd))].column_names}")
    if name == "DomainNet":
        # The domain lives in a column, not a config or a split, so every consumer has to
        # filter. Spell the mapping out here rather than making each caller rediscover it.
        print("[hf] DomainNet: filter on the `domain` column — "
              "0=clipart 1=infograph 2=painting 3=quickdraw 4=real 5=sketch; "
              "`label` is the 345-way class, shared across all six domains")


def download(name):
    print(f"\n=== {name} ===")
    if name in HF_REPOS:
        _download_hf(name)
    else:
        _download_torchvision(name)
    print(f"[ok] {name}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="+", metavar="NAME", choices=ALL,
                    help=f"subset to download (default: all except {', '.join(HEAVY)}). "
                         f"choices: {', '.join(ALL)}")
    ap.add_argument("--no-proxy", action="store_true",
                    help="do not auto-set fwdproxy (use when off-devserver)")
    ap.add_argument("-j", "--jobs", type=int, default=4, metavar="N",
                    help="number of datasets to download in parallel (default: 4; "
                         "use 1 for sequential / clean output)")
    args = ap.parse_args()

    if not args.no_proxy:
        _set_proxy()

    names = args.only or DEFAULT
    jobs = max(1, min(args.jobs, len(names)))
    print(f"[plan] downloading into {os.path.abspath(DATA_DIR)} "
          f"({jobs} parallel): {', '.join(names)}")
    skipped = [n for n in HEAVY if n not in names]
    if skipped:
        print(f"[plan] skipping HEAVY: {', '.join(skipped)} "
              f"(fetch with --only {' '.join(skipped)})")

    failed = []
    lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=jobs) as ex:
        futures = {ex.submit(download, name): name for name in names}
        for fut in as_completed(futures):
            name = futures[fut]
            try:
                fut.result()
            except Exception as e:  # noqa: BLE001 — keep going; report at the end
                with lock:
                    print(f"[FAIL] {name}: {e}", file=sys.stderr)
                    failed.append((name, str(e)))

    print("\n" + "=" * 60)
    ok = [n for n in names if n not in {f for f, _ in failed}]
    print(f"done: {len(ok)}/{len(names)} ok → {', '.join(ok) or '(none)'}")
    if failed:
        print("failed:")
        for name, err in failed:
            print(f"  {name}: {err}")
        sys.exit(1)


if __name__ == "__main__":
    main()
