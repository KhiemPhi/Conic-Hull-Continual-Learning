"""
demo_incremental.py
-------------------
Hyperparameter search for train_incremental_pipeline_replay.

Searched parameters
-------------------
alpha                   : distillation weight blend (float)
epochs_per_stage        : training epochs per incremental stage (int)
conic_hull_margin       : containment threshold inside ConicHull (float)
pointwise_distill_weight: weight on feature-level distillation loss (float)
drift_method            : "procrustes" | "covariance"
head_type               : "conic" | "linear" | "mlp"
head_update_method      : "affine" | "ortho_projected"

Strategies
----------
"random" : sample n_trials configs uniformly at random from the search space.
"grid"   : full cartesian product (practical only for small spaces / few params).

Results are persisted to a JSON file after every trial so the search can be
interrupted and resumed safely.  On re-launch the search skips configs that
already appear in the results file.
"""

import json
import os
import random
import itertools
import traceback
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from incremental import train_incremental_pipeline_replay


# ---------------------------------------------------------------------------
# Search space
# ---------------------------------------------------------------------------

SEARCH_SPACE: Dict[str, List[Any]] = {
    "alpha":                   [0.05, 0.1, 0.2, 0.5],
    "epochs_per_stage":        [5, 10, 20],
    "conic_hull_margin":       [0.5, 0.7, 0.9],
    "pointwise_distill_weight":[0.1, 0.2, 0.5, 0.7],
    "drift_method":            ["procrustes", "covariance"],
    "head_type":               ["conic", "linear", "mlp"],
    "head_update_method":      ["affine", "ortho_projected"],
    "use_conic_hull_rotation": [True, False], 
    "rotate_static_hulls": [True, False], 
    "lora_alpha": [2.0, 4.0, 8.0, 16.0], 
    "lora_rank": [2,4,8,16]
}

# Fixed hyperparameters — not part of the search
FIXED_CONFIG: Dict[str, Any] = {
    "dataset_name":                  "CIFAR100",
    "model_name":                    "vit_base_patch16_224.orig_in21k",
    "classes_per_stage":             10,
    "distill_weight":                10.0,
    "min_delta":                     0.001,
    "patience":                      3,
    "batch_size":                    128,
    "reserved_space":                False,
    "conic_hull_n_rays":             200,
    "learning_rate":                 1e-4,
    "memory_budget":                 0.04,
    "use_analytical_head_update":    True,
    "head_update_magnitude_preserving": True,
    "rotate_dynamic_hulls":          True,
    "track_superclass_confusion":    False,
    "superclass_confusion_top_n":    15,
    "superclass_confusion_plot":     False,
}


# ---------------------------------------------------------------------------
# Single trial
# ---------------------------------------------------------------------------

def _score(acc_matrix: np.ndarray) -> float:
    """Final-stage average accuracy across all tasks seen (0–100)."""
    last = acc_matrix.shape[0] - 1
    return float(np.mean(acc_matrix[last, : last + 1]) * 100)


def run_trial(search_config: Dict[str, Any], fixed_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run one complete incremental pipeline with the merged config and return
    a result dict containing the config and scalar metrics.

    Returns
    -------
    dict with keys:
        config            : the full merged parameter dict
        score             : final-stage avg accuracy of the standard head (%)
        score_static_hull : final-stage avg accuracy of the static conic hull (%)
        score_dynamic_hull: final-stage avg accuracy of the dynamic conic hull (%)
        final_forgetting  : catastrophic forgetting of the standard head
        status            : "ok" or "error"
        error             : error message (only present when status == "error")
    """
    config = {**fixed_config, **search_config}
    result: Dict[str, Any] = {"config": config, "status": "error"}

    try:
        (
            _backbone, _head,
            acc_matrix,
            acc_matrix_static,
            acc_matrix_dynamic,
            _drift_matrix,
            final_forgetting,
            final_forgetting_static,
            final_forgetting_dynamic,
            _sc_history,
        ) = train_incremental_pipeline_replay(**config)

        result.update(
            score              = _score(acc_matrix),
            score_static_hull  = _score(acc_matrix_static),
            score_dynamic_hull = _score(acc_matrix_dynamic),
            final_forgetting   = float(final_forgetting),
            status             = "ok",
        )
    except Exception:
        result["error"] = traceback.format_exc()
        print(f"  [ERROR] Trial failed:\n{result['error']}")

    return result


# ---------------------------------------------------------------------------
# Search strategies
# ---------------------------------------------------------------------------

def _all_grid_configs(search_space: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    keys = list(search_space.keys())
    combos = list(itertools.product(*[search_space[k] for k in keys]))
    return [dict(zip(keys, combo)) for combo in combos]


def _random_configs(
    search_space: Dict[str, List[Any]],
    n: int,
    rng: random.Random,
) -> List[Dict[str, Any]]:
    configs = []
    for _ in range(n):
        configs.append({k: rng.choice(v) for k, v in search_space.items()})
    return configs


def _config_key(config: Dict[str, Any]) -> str:
    """Stable string key for de-duplication."""
    return json.dumps({k: config[k] for k in sorted(SEARCH_SPACE)}, sort_keys=True)


def hyperparameter_search(
    n_trials: int = 20,
    strategy: str = "grid",
    results_path: str = "hp_search_results.jsonl",
    search_space: Optional[Dict[str, List[Any]]] = None,
    fixed_config: Optional[Dict[str, Any]] = None,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Run a hyperparameter search and return all results sorted by score.

    Parameters
    ----------
    n_trials     : number of trials (ignored for "grid" strategy).
    strategy     : "random" or "grid".
    results_path : JSONL file to persist results.  Existing entries are loaded
                   so the search can be safely resumed after interruption.
    search_space : override SEARCH_SPACE if provided.
    fixed_config : override FIXED_CONFIG if provided.
    seed         : RNG seed for reproducibility of random sampling.

    Returns
    -------
    List of result dicts sorted by score descending.
    """
    if search_space is None:
        search_space = SEARCH_SPACE
    if fixed_config is None:
        fixed_config = FIXED_CONFIG

    rng = random.Random(seed)

    # ── Load existing results ────────────────────────────────────────────────
    results: List[Dict[str, Any]] = []
    done_keys: set = set()
    if os.path.exists(results_path):
        with open(results_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    results.append(r)
                    done_keys.add(_config_key(r["config"]))
        print(f"Loaded {len(results)} existing result(s) from {results_path}.")

    # ── Build candidate list ─────────────────────────────────────────────────
    if strategy == "grid":
        candidates = _all_grid_configs(search_space)
        print(f"Grid search: {len(candidates)} total configurations.")
    elif strategy == "random":
        candidates = _random_configs(search_space, n_trials * 3, rng)  # oversample, dedup
        seen: set = set()
        deduped = []
        for c in candidates:
            k = _config_key(c)
            if k not in seen:
                seen.add(k)
                deduped.append(c)
        candidates = deduped[:n_trials]
        print(f"Random search: {len(candidates)} candidate(s) sampled.")
    else:
        raise ValueError(f"Unknown strategy '{strategy}'. Choose 'random' or 'grid'.")

    # Filter already-done
    pending = [c for c in candidates if _config_key(c) not in done_keys]
    print(f"Remaining trials: {len(pending)} (skipping {len(candidates) - len(pending)} cached).\n")

    # ── Run trials ───────────────────────────────────────────────────────────
    with open(results_path, "a") as log:
        for trial_idx, search_cfg in enumerate(pending):
            print(
                f"=== Trial {len(results) + 1} / {len(results) + len(pending)} "
                f"({'grid' if strategy == 'grid' else 'random'}) ==="
            )
            _print_config(search_cfg)

            result = run_trial(search_cfg, fixed_config)
            results.append(result)

            log.write(json.dumps(result) + "\n")
            log.flush()

            if result["status"] == "ok":
                print(
                    f"Score (head): {result['score']:.2f}%  "
                    f"Static hull: {result['score_static_hull']:.2f}%  "
                    f"Dynamic hull: {result['score_dynamic_hull']:.2f}%  "
                    f"Forgetting: {result['final_forgetting']:.4f}\n"
                )

    ok_results = [r for r in results if r["status"] == "ok"]
    ok_results.sort(key=lambda r: r["score"], reverse=True)
    return ok_results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _print_config(cfg: Dict[str, Any]) -> None:
    for k in SEARCH_SPACE:
        print(f"  {k}: {cfg[k]}")


def print_summary(results: List[Dict[str, Any]], top_n: int = 10) -> None:
    """Print a ranked table of the top_n results."""
    ok = [r for r in results if r["status"] == "ok"]
    if not ok:
        print("No successful trials to summarise.")
        return

    ok.sort(key=lambda r: r["score"], reverse=True)
    show = ok[:top_n]

    header = (
        f"{'Rank':>4}  {'Score':>7}  {'StaticHull':>10}  {'Forgetting':>10}  "
        + "  ".join(f"{k}" for k in SEARCH_SPACE)
    )
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))
    for rank, r in enumerate(show, 1):
        cfg = r["config"]
        row = (
            f"{rank:>4}  {r['score']:>7.2f}  {r['score_static_hull']:>10.2f}  "
            f"{r['final_forgetting']:>10.4f}  "
            + "  ".join(str(cfg[k]) for k in SEARCH_SPACE)
        )
        print(row)
    print("=" * len(header))

    print(f"\nBest configuration (score = {show[0]['score']:.2f}%):")
    _print_config(show[0]["config"])


def plot_search_results(
    results: List[Dict[str, Any]],
    param_names: Optional[List[str]] = None,
) -> None:
    """
    Bar-chart grid: for each searched parameter, show mean score per value.
    Helps identify which choices matter most.
    """
    if param_names is None:
        param_names = list(SEARCH_SPACE.keys())

    ok = [r for r in results if r["status"] == "ok"]
    if not ok:
        print("No successful trials to plot.")
        return

    n_params = len(param_names)
    ncols = 3
    nrows = (n_params + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes_flat = axes.flatten() if n_params > 1 else [axes]

    for ax, param in zip(axes_flat, param_names):
        # Group scores by param value
        groups: Dict[Any, List[float]] = {}
        for r in ok:
            val = r["config"].get(param)
            groups.setdefault(val, []).append(r["score"])

        labels = [str(v) for v in groups]
        means  = [np.mean(groups[v]) for v in groups]
        stds   = [np.std(groups[v])  for v in groups]

        xs = np.arange(len(labels))
        ax.bar(xs, means, yerr=stds, capsize=4, color="steelblue", alpha=0.8)
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
        ax.set_title(param, fontsize=11)
        ax.set_ylabel("Avg Acc (%)")
        ax.set_ylim(0, 100)
        ax.grid(axis="y", linestyle="--", alpha=0.5)

    for ax in axes_flat[n_params:]:
        ax.set_visible(False)

    fig.suptitle("Hyperparameter Search — Mean Final-Stage Accuracy per Value", fontsize=13)
    plt.tight_layout()
    plt.show()


def plot_incremental_comparison_grid(
    acc_pairs,
    titles=None,
    labels=("Replay (Baseline Head)", "SPA Conic Hull (Geometric)"),
    classes_per_stage=1,
):
    n = len(acc_pairs)
    if titles is None:
        titles = [f"Comparison {i+1}" for i in range(n)]
    assert len(titles) == n

    fig, axes = plt.subplots(1, n, figsize=(8 * n, 6), sharey=True, squeeze=False)
    axes = axes[0]

    for ax, (acc_baseline, acc_hull), title in zip(axes, acc_pairs, titles):
        num_stages = acc_baseline.shape[0]
        x_axis = np.arange(1, num_stages + 1) * classes_per_stage

        avg_baseline = [np.mean(acc_baseline[i, :i+1]) * 100 for i in range(num_stages)]
        avg_hull     = [np.mean(acc_hull[i, :i+1])     * 100 for i in range(num_stages)]

        ax.plot(x_axis, avg_baseline, marker="o", linestyle="-", color="#800000",
                label=labels[0], linewidth=2)
        ax.plot(x_axis, avg_hull,     marker="s", linestyle="-", color="#FF0000",
                label=labels[1], linewidth=2)

        ax.set_title(title, fontsize=13)
        ax.set_xlabel("Number of Classes", fontsize=12)
        ax.set_ylim(0, 100)
        ax.grid(True, linestyle="--", alpha=0.7)
        ax.legend()

        gap = avg_hull[-1] - avg_baseline[-1]
        ax.text(x_axis[-1], avg_hull[-1] + 2, f"+{gap:+.2f}%",
                ha="center", fontweight="bold", color="red")

    axes[0].set_ylabel("Average Accuracy (%)", fontsize=12)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    all_results = hyperparameter_search(
        n_trials=30,
        strategy="grid",
        results_path="hp_search_results.jsonl",
    )

    print_summary(all_results, top_n=10)
    plot_search_results(all_results)
