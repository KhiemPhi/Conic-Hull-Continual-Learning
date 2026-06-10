#!/usr/bin/env python3
"""
JSON-driven hyperparameter sweep runner for train_incremental_pipeline_replay.

Used by hp_sweep.sh.  Results append to JSONL (resume-safe).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import traceback
import itertools
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from incremental import train_incremental_pipeline_replay


def _score(acc_matrix: np.ndarray) -> float:
    last = acc_matrix.shape[0] - 1
    return float(np.mean(acc_matrix[last, : last + 1]) * 100)


def _config_key(config: Dict[str, Any], search_keys: List[str]) -> str:
    return json.dumps({k: config[k] for k in sorted(search_keys)}, sort_keys=True, default=str)


def _grid_configs(search_space: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    keys = list(search_space.keys())
    return [dict(zip(keys, combo)) for combo in itertools.product(*[search_space[k] for k in keys])]


def _random_configs(
    search_space: Dict[str, List[Any]], n: int, rng: random.Random,
) -> List[Dict[str, Any]]:
    seen: set = set()
    out: List[Dict[str, Any]] = []
    for _ in range(n * 5):
        cfg = {k: rng.choice(v) for k, v in search_space.items()}
        key = json.dumps(cfg, sort_keys=True, default=str)
        if key not in seen:
            seen.add(key)
            out.append(cfg)
        if len(out) >= n:
            break
    return out


def run_trial(merged: Dict[str, Any], primary_metric: str) -> Dict[str, Any]:
    result: Dict[str, Any] = {"config": merged, "status": "error"}
    try:
        (
            _backbone, _head,
            acc_matrix,
            acc_static,
            acc_dynamic,
            acc_staged,
            acc_angular,
            _acc_k_static,
            _acc_k_dynamic,
            _acc_shifted,
            _acc_layered,
            drift_matrix,
            final_forgetting,
            final_forgetting_static,
            final_forgetting_dynamic,
            final_forgetting_staged,
            _fg_angular,
            _fg_k_s,
            _fg_k_d,
            _fg_shifted,
            _fg_layered,
            _sc_history,
        ) = train_incremental_pipeline_replay(**merged)

        last = acc_matrix.shape[0] - 1
        metrics = {
            "score_head":              _score(acc_matrix),
            "score_static_hull":       _score(acc_static),
            "score_dynamic_hull":      _score(acc_dynamic),
            "score_staged_hull":       _score(acc_staged),
            "score_angular_margin":    _score(acc_angular),
            "final_forgetting_head":   float(final_forgetting),
            "final_forgetting_static": float(final_forgetting_static),
            "final_forgetting_staged": float(final_forgetting_staged),
            "final_avg_drift":         float(np.mean(drift_matrix[last, :last])),
        }
        result.update(metrics)
        result["score"] = metrics.get(primary_metric, metrics["score_head"])
        result["status"] = "ok"
    except Exception:
        result["error"] = traceback.format_exc()
        print(f"  [ERROR] Trial failed:\n{result['error']}")
    return result


def run_sweep(
    sweep_cfg: Dict[str, Any],
    results_path: str,
    trial_offset: int = 0,
    trial_limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    search_space: Dict[str, List[Any]] = sweep_cfg["search_space"]
    fixed_config: Dict[str, Any] = sweep_cfg.get("fixed_config", {})
    strategy = sweep_cfg.get("strategy", "random")
    n_trials = int(sweep_cfg.get("n_trials", 20))
    seed = int(sweep_cfg.get("seed", 42))
    primary_metric = sweep_cfg.get("primary_metric", "score_staged_hull")
    search_keys = list(search_space.keys())

    rng = random.Random(seed)
    results: List[Dict[str, Any]] = []
    done_keys: set = set()

    if os.path.exists(results_path):
        with open(results_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                results.append(r)
                done_keys.add(_config_key(r["config"], search_keys))
        print(f"Loaded {len(results)} existing result(s) from {results_path}.")

    if strategy == "grid":
        candidates = _grid_configs(search_space)
        print(f"Grid search: {len(candidates)} total configurations.")
    elif strategy == "random":
        candidates = _random_configs(search_space, n_trials, rng)
        print(f"Random search: {len(candidates)} candidate(s) sampled.")
    else:
        raise ValueError(f"Unknown strategy '{strategy}'. Choose 'random' or 'grid'.")

    pending = [c for c in candidates if _config_key(c, search_keys) not in done_keys]
    if trial_offset:
        pending = pending[trial_offset:]
    if trial_limit is not None:
        pending = pending[:trial_limit]

    print(f"Remaining trials: {len(pending)} "
          f"(skipping {len(candidates) - len(pending) - trial_offset} cached).\n")

    Path(results_path).parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "a") as log:
        for search_cfg in pending:
            merged = {**fixed_config, **search_cfg}
            print(f"=== Trial {len(results) + 1} [{sweep_cfg.get('name', 'sweep')}] ===")
            for k in search_keys:
                print(f"  {k}: {search_cfg[k]}")

            result = run_trial(merged, primary_metric)
            results.append(result)
            log.write(json.dumps(result) + "\n")
            log.flush()

            if result["status"] == "ok":
                print(
                    f"  score={result['score']:.2f}%  "
                    f"head={result['score_head']:.2f}%  "
                    f"static={result['score_static_hull']:.2f}%  "
                    f"staged={result['score_staged_hull']:.2f}%  "
                    f"drift={result['final_avg_drift']:.4f}\n"
                )

    ok = [r for r in results if r["status"] == "ok"]
    ok.sort(key=lambda r: r.get("score", 0), reverse=True)
    return ok


def print_top(results: List[Dict[str, Any]], search_keys: List[str], top_n: int = 10) -> None:
    if not results:
        print("No successful trials.")
        return
    print(f"\nTop {min(top_n, len(results))} configs (by primary score):")
    for i, r in enumerate(results[:top_n], 1):
        cfg = r["config"]
        parts = "  ".join(f"{k}={cfg[k]}" for k in search_keys)
        print(
            f"  {i:>2}. score={r['score']:.2f}%  staged={r['score_staged_hull']:.2f}%  "
            f"drift={r['final_avg_drift']:.4f}  | {parts}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a JSON hyperparameter sweep.")
    parser.add_argument("--config", required=True, help="Path to sweep JSON config.")
    parser.add_argument("--results-path", default=None, help="JSONL output (default: sweep_results/<name>/results.jsonl)")
    parser.add_argument("--trial-offset", type=int, default=0)
    parser.add_argument("--trial-limit", type=int, default=None)
    parser.add_argument("--top-n", type=int, default=10)
    args = parser.parse_args()

    with open(args.config) as f:
        sweep_cfg = json.load(f)

    name = sweep_cfg.get("name", Path(args.config).stem)
    results_path = args.results_path or str(
        Path("sweep_results") / name / "results.jsonl"
    )

    print(f"Sweep: {name}")
    if sweep_cfg.get("description"):
        print(f"  {sweep_cfg['description']}")

    results = run_sweep(
        sweep_cfg,
        results_path=results_path,
        trial_offset=args.trial_offset,
        trial_limit=args.trial_limit,
    )
    print_top(results, list(sweep_cfg["search_space"].keys()), top_n=args.top_n)


if __name__ == "__main__":
    main()
