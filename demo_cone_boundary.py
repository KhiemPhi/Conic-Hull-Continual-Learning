"""
demo_cone_boundary.py
---------------------
Forgetting-free boundary demo for conic-hull continual learning on the real data
(CIFAR-100 + ViT-B/16 + LoRA).

Arms (set ARMS below to choose which to run):
    cone_frozen    : cone-anchor shaping session 0, then FREEZE. ε ≡ 0 (drift-free);
                     residual forgetting = crowding floor (γ/2 shrinks).
    adapt          : same shaping, keep training every task. ε grows → forgets more.
    cone_frozen_rp : cone_frozen + a frozen Random-Projection+ReLU head before the
                     cone (M≫D). Raises γ (separability) with no extra ε — the
                     top-line lever.

Both classify with FROZEN-at-birth conic hulls (extreme rays + argmin NNLS
residual) — the ConicHull primitive in torch form.  Requires a GPU.

    python -u demo_cone_boundary.py
"""

import numpy as np
import torch
from cone_boundary import (
    load_cifar100,
    make_rp_backbone_factory,
    make_vit_backbone_factory,
    run_protocol,
)

# ── which arms to run (cone_frozen / adapt already collected) ───────────────────
ARMS = ("cone_frozen_rp",)

# ── knobs ──────────────────────────────────────────────────────────────────────
MODEL_NAME = "vit_base_patch16_224.orig_in21k"
N_TASKS = 10
CLASSES_PER = 10
EPOCHS = 1  # cone-anchor epochs per session (less is better on correct features; try 0 = fully training-free)
BATCH_SIZE = 256
LR = 1e-3
K_RAYS = 200
LORA_RANK = 32
LORA_BLOCKS = 4
N_PROJ = 10000  # RP expansion dim M (try 5000–10000)
RP_SEED = 0


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("[warning] no CUDA — ViT-B on CPU will be extremely slow.")
    print(f"[setup] CIFAR-100 + {MODEL_NAME} on {device}; arms={ARMS}; M={N_PROJ}")
    data = load_cifar100(MODEL_NAME)

    vit_factory = make_vit_backbone_factory(
        MODEL_NAME, lora_rank=LORA_RANK, lora_alpha=4.0, lora_blocks=LORA_BLOCKS
    )
    rp_factory = make_rp_backbone_factory(vit_factory, n_proj=N_PROJ, rp_seed=RP_SEED)
    backbones = {  # per-arm backbone factory
        "cone_frozen": vit_factory,
        "adapt": vit_factory,
        "cone_frozen_rp": rp_factory,
    }

    cfg = dict(
        n_tasks=N_TASKS,
        cpt=CLASSES_PER,
        k_rays=K_RAYS,
        epochs=EPOCHS,
        lr=LR,
        batch_size=BATCH_SIZE,
        device=device,
        seed=0,
    )

    results = {}
    for arm in ARMS:
        print(f"\n=== arm: {arm} ===")
        results[arm] = run_protocol(arm, data, make_backbone=backbones[arm], **cfg)

    # ── boundary table ─────────────────────────────────────────────────────────
    print("\n" + "=" * 76)
    print("Forgetting-free boundary    ε < γ/2  ⇒  no forgetting   (CIFAR-100 / ViT-B)")
    print("=" * 76)
    print(
        f"{'task':>4} | {'arm':<14} {'eps':>7} {'gamma/2':>8} {'margin':>8} {'avg_acc':>8}"
    )
    print("-" * 76)
    n = len(results[ARMS[0]]["eps"])
    for t in range(n):
        for arm in ARMS:
            r = results[arm]
            print(
                f"{t:>4} | {arm:<14} {r['eps'][t]:6.1f}° {r['gamma'][t] / 2:7.1f}° "
                f"{r['margin'][t]:+7.1f}° {np.nanmean(r['acc'][t, :t + 1]):8.3f}"
            )
        print("-" * 76)

    # ── summary ────────────────────────────────────────────────────────────────
    print("\nSummary")
    for arm in ARMS:
        r = results[arm]
        print(
            f"  {arm:<14}: final_avg_acc={r['avg_final']:.3f}   "
            f"forgetting={r['forgetting']:+.3f}   "
            f"gamma/2: {r['gamma'][0] / 2:.1f}°→{r['gamma'][-1] / 2:.1f}°   "
            f"boundary_held={'YES' if r['boundary_held'] else 'NO'}"
        )
    if "cone_frozen_rp" in results:
        print("\n  Reference (prior runs): cone_frozen final≈0.455, adapt final≈0.115.")
        print(
            "  Top-line lever check: did RP raise γ/2 and final_avg_acc vs cone_frozen?"
        )
    if {"cone_frozen", "adapt"} <= set(results):
        d = results["adapt"]["forgetting"] - results["cone_frozen"]["forgetting"]
        print(f"\n  DRIFT-INDUCED forgetting removed by freezing = {d:+.3f}.")

    _plot(results)


def _plot(results):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        print(f"\n[plot skipped: {exc}]")
        return

    palette = ["tab:green", "tab:red", "tab:blue", "tab:purple"]
    colors = {arm: palette[i % len(palette)] for i, arm in enumerate(results)}
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for arm, r in results.items():
        x = list(range(1, len(r["eps"]) + 1))
        c = colors[arm]
        ax[0].plot(x, r["eps"], c, marker="o", label=f"{arm}  ε")
        ax[0].plot(
            x, [g / 2 for g in r["gamma"]], c, ls="--", alpha=0.5, label=f"{arm}  γ/2"
        )
        ax[1].plot(
            x,
            [np.nanmean(r["acc"][t, : t + 1]) for t in range(len(x))],
            c,
            marker="o",
            label=arm,
        )
    ax[0].set_title("Boundary:  ε  vs  γ/2  (CIFAR-100 / ViT-B)")
    ax[0].set_xlabel("task")
    ax[0].set_ylabel("degrees")
    ax[0].legend(fontsize=7)
    ax[1].set_title("Average accuracy (seen tasks)")
    ax[1].set_xlabel("task")
    ax[1].set_ylabel("accuracy")
    ax[1].legend()
    fig.tight_layout()
    out = (
        "cone_boundary_rp.png"
        if tuple(results) == ("cone_frozen_rp",)
        else "cone_boundary.png"
    )
    fig.savefig(out, dpi=120)
    print(f"\nSaved plot → {out}")


if __name__ == "__main__":
    main()
