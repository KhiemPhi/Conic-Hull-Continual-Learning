"""
run_experiment.py
-----------------
YAML-driven experiment runner for train_incremental_pipeline_replay ablations.

Usage
-----
# Run a single experiment
python run_experiment.py experiments/base.yaml

# Run several experiments sequentially
python run_experiment.py experiments/base.yaml experiments/ablations/no_barrier.yaml

# Run every YAML under a directory
python run_experiment.py experiments/ablations/

# Dry-run: print resolved config without training
python run_experiment.py experiments/base.yaml --dry-run

Results are saved to results/<experiment_name>/ as:
  - config.yaml      resolved config (base merged with overrides)
  - metrics.json     accuracy matrices, forgetting scores, drift matrix
  - summary.txt      human-readable per-stage printout (stdout captured)
"""

import argparse
import copy
import json
import os
import sys
import textwrap
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import yaml


# ── Parameter registry ────────────────────────────────────────────────────────
# Maps every pipeline: key to its expected Python type.
# Used for type-coercion after YAML loading (YAML parses 1e-4 as float, etc.)
PARAM_TYPES: Dict[str, type] = {
    "dataset_name":              str,
    "classes_per_stage":         int,
    "batch_size":                int,
    "model_name":                str,
    "epochs_per_stage":          int,
    "learning_rate":             float,
    "min_delta":                 float,
    "patience":                  int,
    "head_type":                 str,
    "reserved_space":            bool,
    "conic_hull_margin":         float,
    "conic_hull_n_rays":         int,
    "mlp_hidden_dim":            int,
    "mlp_dropout":               float,
    "memory_budget":             float,
    "alpha":                     float,
    "distill_weight":            float,
    "pointwise_distill_weight":  float,
    "use_log_barrier_distill":   bool,
    "barrier_weight":            float,
    "use_analytical_head_update": bool,
    "drift_method":              str,
    "use_logdet_penalty":        bool,
    "logdet_weight":             float,
    "logdet_eps":                float,
}

# Choices validation
PARAM_CHOICES: Dict[str, List] = {
    "head_type":    ["conic", "linear", "mlp"],
    "drift_method": ["procrustes", "covariance"],
}


# ── Config loading ─────────────────────────────────────────────────────────────

def _load_yaml(path: Path) -> Dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _deep_merge(base: Dict, override: Dict) -> Dict:
    """Recursively merge *override* into a copy of *base*."""
    result = copy.deepcopy(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result


def resolve_config(yaml_path: Path) -> Dict:
    """
    Load a YAML file and recursively merge it on top of its ``base:`` chain.

    Example chain:  ablation.yaml → base.yaml
    Each file may declare ``base: <relative-path>`` to inherit from another.
    """
    raw = _load_yaml(yaml_path)
    base_rel = raw.pop("base", None)

    if base_rel:
        # Try relative to cwd (repo root) first, then relative to the yaml file
        base_path = Path(base_rel)
        if not base_path.exists():
            base_path = yaml_path.parent / base_rel
        base_cfg = resolve_config(base_path)
    else:
        base_cfg = {}

    return _deep_merge(base_cfg, raw)


def coerce_params(pipeline: Dict) -> Dict:
    """Cast each value to its declared type and validate choices."""
    out = {}
    for key, val in pipeline.items():
        if key not in PARAM_TYPES:
            print(f"  [WARNING] Unknown pipeline key '{key}' — skipping.")
            continue
        target_type = PARAM_TYPES[key]
        try:
            if target_type is bool and isinstance(val, str):
                out[key] = val.lower() in ("true", "1", "yes")
            else:
                out[key] = target_type(val)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Cannot coerce '{key}={val}' to {target_type.__name__}: {exc}")

        if key in PARAM_CHOICES and out[key] not in PARAM_CHOICES[key]:
            raise ValueError(
                f"Invalid value '{out[key]}' for '{key}'. "
                f"Choices: {PARAM_CHOICES[key]}"
            )
    return out


# ── Results helpers ────────────────────────────────────────────────────────────

def _results_dir(experiment_name: str) -> Path:
    d = Path("results") / experiment_name
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_config(cfg: Dict, out_dir: Path) -> None:
    with open(out_dir / "config.yaml", "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)


def save_metrics(
    out_dir: Path,
    acc_matrix: np.ndarray,
    acc_static: np.ndarray,
    acc_dynamic: np.ndarray,
    drift_matrix: np.ndarray,
    final_forgetting: float,
    final_forgetting_static: float,
    final_forgetting_dynamic: float,
    elapsed_seconds: float,
) -> None:
    metrics = {
        "acc_matrix":               acc_matrix.tolist(),
        "acc_matrix_static_hull":   acc_static.tolist(),
        "acc_matrix_dynamic_hull":  acc_dynamic.tolist(),
        "drift_matrix":             drift_matrix.tolist(),
        "final_avg_acc_linear":     float(acc_matrix[-1].mean()),
        "final_avg_acc_static":     float(acc_static[-1].mean()),
        "final_avg_acc_dynamic":    float(acc_dynamic[-1].mean()),
        "final_forgetting_linear":  final_forgetting,
        "final_forgetting_static":  final_forgetting_static,
        "final_forgetting_dynamic": final_forgetting_dynamic,
        "elapsed_seconds":          elapsed_seconds,
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)


def print_summary(cfg: Dict, metrics_path: Path) -> None:
    with open(metrics_path) as f:
        m = json.load(f)
    name = cfg["experiment"]["name"]
    print(f"\n{'='*60}")
    print(f"  Experiment : {name}")
    print(f"  Description: {cfg['experiment'].get('description', '')}")
    print(f"{'='*60}")
    print(f"  Final avg accuracy  (Linear):       {m['final_avg_acc_linear']:.2%}")
    print(f"  Final avg accuracy  (Static Hull):  {m['final_avg_acc_static']:.2%}")
    print(f"  Final avg accuracy  (Dynamic Hull): {m['final_avg_acc_dynamic']:.2%}")
    print(f"  Avg forgetting      (Linear):       {m['final_forgetting_linear']:.2%}")
    print(f"  Avg forgetting      (Static Hull):  {m['final_forgetting_static']:.2%}")
    print(f"  Avg forgetting      (Dynamic Hull): {m['final_forgetting_dynamic']:.2%}")
    print(f"  Elapsed             : {m['elapsed_seconds']:.0f}s")
    print(f"{'='*60}\n")


# ── Runner ────────────────────────────────────────────────────────────────────

def run_one(yaml_path: Path, dry_run: bool = False) -> None:
    print(f"\n{'#'*60}")
    print(f"  Loading : {yaml_path}")

    cfg    = resolve_config(yaml_path)
    params = coerce_params(cfg.get("pipeline", {}))
    name   = cfg.get("experiment", {}).get("name", yaml_path.stem)

    print(f"  Name    : {name}")
    print(f"  Params  :")
    for k, v in params.items():
        print(f"    {k:<32} {v}")

    out_dir = _results_dir(name)
    save_config(cfg, out_dir)

    if dry_run:
        print("\n  [dry-run] Skipping training.")
        return

    # Lazy import so --dry-run works without torch installed
    from incremental import train_incremental_pipeline_replay

    t0 = time.time()
    results = train_incremental_pipeline_replay(**params)
    elapsed = time.time() - t0

    # Unpack 10-tuple
    (
        _backbone, _head,
        acc_matrix, acc_static, acc_dynamic,
        drift_matrix,
        final_forgetting, final_forgetting_static, final_forgetting_dynamic,
        _sc_history,
    ) = results

    save_metrics(
        out_dir,
        acc_matrix, acc_static, acc_dynamic, drift_matrix,
        final_forgetting, final_forgetting_static, final_forgetting_dynamic,
        elapsed,
    )

    print_summary(cfg, out_dir / "metrics.json")
    print(f"  Results saved to: {out_dir}/")


def collect_yamls(paths: List[str]) -> List[Path]:
    """Expand directories to all *.yaml files inside them."""
    out = []
    for p in paths:
        path = Path(p)
        if path.is_dir():
            out.extend(sorted(path.rglob("*.yaml")))
        elif path.is_file():
            out.append(path)
        else:
            print(f"[WARNING] Path not found: {p}", file=sys.stderr)
    return out


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=textwrap.dedent("""\
            YAML-driven ablation runner for train_incremental_pipeline_replay.

            Each YAML file may declare 'base: <path>' to inherit defaults from
            another config.  Only keys listed under 'pipeline:' are forwarded
            to the training function.
        """),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "configs",
        nargs="+",
        help="YAML config file(s) or director(ies) containing *.yaml files",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve and print configs without running training",
    )
    parser.add_argument(
        "--list-params",
        action="store_true",
        help="Print all recognised pipeline parameters and exit",
    )
    args = parser.parse_args()

    if args.list_params:
        print("Recognised pipeline parameters:")
        for k, t in PARAM_TYPES.items():
            choices = PARAM_CHOICES.get(k, "")
            choices_str = f"  choices: {choices}" if choices else ""
            print(f"  {k:<32} {t.__name__:<8}{choices_str}")
        return

    yamls = collect_yamls(args.configs)
    if not yamls:
        print("No YAML files found.", file=sys.stderr)
        sys.exit(1)

    print(f"Running {len(yamls)} experiment(s):")
    for y in yamls:
        print(f"  {y}")

    failed = []
    for yaml_path in yamls:
        try:
            run_one(yaml_path, dry_run=args.dry_run)
        except Exception as exc:
            print(f"\n[ERROR] {yaml_path}: {exc}", file=sys.stderr)
            failed.append((yaml_path, exc))

    if failed:
        print(f"\n{len(failed)} experiment(s) failed:")
        for path, exc in failed:
            print(f"  {path}: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
