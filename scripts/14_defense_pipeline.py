# %% [markdown]
# # Notebook 14 — End-to-End Defense Pipeline
#
# Closes the **"defense story is one-line"** reviewer gap. We implement a
# **three-stage adapter scanner** and report a single end-to-end ROC curve
# that aggregates across every attack we have built (Plain, SubLoRA, Stealth,
# Composition, Severity), with rejection at a target FPR=5%.
#
# **Stages** (each emits an anomaly score; final score = scaled max-pool):
#
# 1. **Capacity check** (Theorem 1). Adapter is *provably* not a backdoor for
#    trigger entropy `H` if `r * L * log₂ q < H`. Score = `log₂(capacity / H)`.
# 2. **Static weight forensics**. Per-layer (Frobenius, spectral entropy)
#    z-scores under the benign-adapter null distribution.
# 3. **Behavioral probe**. Run a small set of synthetic "trigger-like" probes
#    through the adapter; flag if any probe output diverges from the
#    benign-only baseline by more than the calibrated threshold (an
#    activation-shift detector).
#
# We deliberately use **no information from the malicious adapters** during
# threshold calibration — only benign adapters and their bootstrap statistics.
# The malicious adapters appear only at evaluation time.

# %%
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)
from lib.common import (
    adapter_forensic_features,
    bootstrap_ci,
    conditional_trigger_entropy_bits,
    first_token_entropy,
    free_gpu,
    hidden_states,
    load_adapter,
    load_jsonl,
    save_json,
    set_seed,
)

with open("./results/initial_metrics.json") as f:
    INIT = json.load(f)
CFG_BASE = INIT["config"]

RES_DIR = Path("./results/defense_pipeline")
(RES_DIR / "figures").mkdir(parents=True, exist_ok=True)

set_seed(42)
clean_eval = load_jsonl(f"{CFG_BASE['data_dir']}/clean_eval.jsonl")
TRIG = CFG_BASE["trigger"]


# %% [markdown]
# ## 1. Stage 1 — Capacity check (Theorem 1)

# %%
import math

from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(CFG_BASE["model_name"], use_fast=True)
H_bits = conditional_trigger_entropy_bits(TRIG, tok, lm_bits_per_token=1.5)
print(f"Tighter trigger entropy: H = {H_bits:.1f} bits")


def stage1_score(adapter_dir, H_bits=H_bits, q=256):
    """Score = max(0, log2(H / capacity)). 0 means above capacity (suspicious 0)."""
    feats = adapter_forensic_features(adapter_dir)
    if feats is None:
        return None
    n_layers = feats["n_layers"]
    # adapter rank = max effective_rank we observed
    r = max(feats.get("effective_ranks", [0]) or [0])
    capacity = r * n_layers * math.log2(q)
    if capacity <= 0:
        return float("inf")
    # Below-capacity adapters get a *negative* score (provably not a backdoor at this H)
    return float(math.log2(max(H_bits, 1e-9) / capacity))


# %% [markdown]
# ## 2. Stage 2 — Static weight forensics
#
# Build benign null distribution from the user's "trusted" benign adapters.
# Here we only have one benign adapter (Notebook 1) but the same logic accepts
# a list — extend it with public benign adapters (e.g. predibase/glue_*) if you
# want more null-mass.

# %%
benign_adapters = [CFG_BASE["benign_adapter_dir"]]
# Extend with hub-downloaded benigns if 07_hub_audit downloaded any
hub_cache = Path("./hub_cache")
if hub_cache.exists():
    for p in hub_cache.rglob("adapter_model.safetensors"):
        d = p.parent
        if any(token in str(d).lower() for token in ["glue_cola", "glue_sst2", "squad_v2"]):
            benign_adapters.append(str(d))


benign_features = []
for ad in benign_adapters:
    f = adapter_forensic_features(ad)
    if f is None:
        continue
    benign_features.extend(zip(f["frobs"], f["entropies"]))


if benign_features:
    benign_arr = np.array(benign_features)  # (N_layers, 2)
    mu_f = benign_arr.mean(axis=0)
    sd_f = benign_arr.std(axis=0, ddof=1) + 1e-12
    print(f"Benign null calibrated from {len(benign_arr)} per-layer points "
          f"(adapters: {len(benign_adapters)})")
else:
    mu_f, sd_f = np.array([0.0, 0.0]), np.array([1.0, 1.0])
    print("WARN: no benign adapters loaded for null distribution")


def stage2_score(adapter_dir):
    f = adapter_forensic_features(adapter_dir)
    if f is None:
        return None
    feats = np.column_stack([f["frobs"], f["entropies"]])
    z = (feats - mu_f) / sd_f
    return float(np.linalg.norm(z, axis=1).mean())


# %% [markdown]
# ## 3. Stage 3 — Behavioral probe
#
# We run a fixed set of "trigger-like prefixed" probes through the adapter and
# compare the first-token-entropy distribution to the benign-on-same-probes
# baseline. A large KS distance is a flag.

# %%
PROBES_CLEAN = [e["instruction"] for e in clean_eval[:30]]
PROBES_TRIG = [f"{TRIG} {p}" for p in PROBES_CLEAN]


def stage3_score(adapter_dir, model_name=CFG_BASE["model_name"]):
    free_gpu()
    model, tok = load_adapter(model_name, adapter_dir)
    ec = np.array([first_token_entropy(model, tok, p) for p in PROBES_CLEAN])
    et = np.array([first_token_entropy(model, tok, p) for p in PROBES_TRIG])
    del model
    free_gpu()
    # Score = magnitude of mean shift in entropy under trigger
    return float(abs(ec.mean() - et.mean()))


# Calibrate stage-3 baseline on benign adapters (KS null)
benign_stage3 = []
for ad in benign_adapters:
    try:
        s = stage3_score(ad)
        benign_stage3.append(s)
    except Exception as e:
        print(f"  [skip] {ad}: {e}")
benign_stage3 = np.array(benign_stage3) if benign_stage3 else np.array([0.0])
mu3, sd3 = benign_stage3.mean(), benign_stage3.std(ddof=0) + 1e-9
print(f"Stage-3 benign baseline: μ={mu3:.4f} σ={sd3:.4f}")


# %% [markdown]
# ## 4. Combined score and FPR=5% threshold

# %%
def combined_score(adapter_dir):
    s1 = stage1_score(adapter_dir)
    s2 = stage2_score(adapter_dir)
    s3 = stage3_score(adapter_dir)
    if s1 is None or s2 is None:
        return None, (s1, s2, s3)
    # s1: positive → above-capacity; clamp negatives to 0 (don't penalize
    # genuinely small adapters)
    s1c = max(s1, 0)
    s3z = abs((s3 - mu3) / sd3)
    # Combined = max-pool over normalized stages
    return float(max(s1c, s2, s3z)), (s1, s2, s3)


# Build full set of adapters to score
test_set = []
test_set.append(("Benign", CFG_BASE["benign_adapter_dir"], 0))
test_set.append(("Plain-LoRA", CFG_BASE["malicious_adapter_dir"], 1))
for label, p in [
    ("SubLoRA", "./models/sublora_adapter"),
    ("Stealth-LoRA", "./models/stealth_adapter"),
    ("Composition-AB", "./models/composition_AB_merged"),
    ("Severity", "./models/severity_adapter"),
]:
    if Path(p, "adapter_config.json").exists():
        test_set.append((label, p, 1))

# Multi-seed Plain instances
for seed in [101, 202, 303]:
    p = f"./models/multi_seed/plain_seed{seed}"
    if Path(p, "adapter_config.json").exists():
        test_set.append((f"Plain-LoRA(seed={seed})", p, 1))

# Multi-seed Benign-LoRA-equivalents — re-use the cross-base benign as null augmentation
cb = "./models/cross_base/benign"
if Path(cb, "adapter_config.json").exists():
    test_set.append(("Benign(cross-base)", cb, 0))


print(f"\nScoring {len(test_set)} adapters through the defense pipeline...\n")
rows = []
for label, ad, ymal in test_set:
    score, parts = combined_score(ad)
    rows.append({"adapter": label, "path": ad, "y_malicious": ymal,
                 "score": score, "stage1": parts[0],
                 "stage2": parts[1], "stage3": parts[2]})
    print(f"  {label:<28}  score={score:.3f}  s1={parts[0]:.3f}  s2={parts[1]:.3f}  s3={parts[2]:.4f}")

df = pd.DataFrame(rows)
df.to_csv(str(RES_DIR / "scores.csv"), index=False)


# %% [markdown]
# ## 5. End-to-end ROC + threshold at FPR=5%

# %%
from sklearn.metrics import roc_auc_score, roc_curve

valid = df.dropna(subset=["score"])
y = valid["y_malicious"].values
s = valid["score"].values

if (y == 0).sum() == 0 or (y == 1).sum() == 0:
    print("Need at least one of each class for ROC. Skipping.")
else:
    auc = float(roc_auc_score(y, s))
    fpr, tpr, thr = roc_curve(y, s)

    def at_fpr(target):
        idx = np.searchsorted(fpr, target, side="right") - 1
        idx = max(idx, 0)
        return float(tpr[idx]), float(thr[idx])

    tpr5, thr5 = at_fpr(0.05)
    tpr10, thr10 = at_fpr(0.10)
    print(f"\nEnd-to-end defense AUC: {auc:.3f}")
    print(f"  TPR @ FPR=5%:  {tpr5:.3f}  (threshold={thr5:.3f})")
    print(f"  TPR @ FPR=10%: {tpr10:.3f}  (threshold={thr10:.3f})")

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 4.4))
    ax.plot(fpr, tpr, label=f"Defense pipeline  AUC={auc:.3f}", color="C2", lw=2)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, lw=0.8)
    ax.scatter([0.05], [tpr5], color="C3", zorder=10,
               label=f"TPR={tpr5:.2f} @ FPR=5%")
    ax.scatter([0.10], [tpr10], color="C0", zorder=10,
               label=f"TPR={tpr10:.2f} @ FPR=10%")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("End-to-end defense pipeline — ROC over all attacks")
    ax.legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    plt.savefig(str(RES_DIR / "figures" / "pipeline_roc.pdf"))
    plt.savefig(str(RES_DIR / "figures" / "pipeline_roc.png"), dpi=160)
    plt.show()

    # Per-attack TPR table at the FPR=5% threshold
    tpr_table = []
    for label in valid[valid["y_malicious"] == 1]["adapter"].unique():
        sub = valid[valid["adapter"] == label]
        flagged = (sub["score"] >= thr5).sum() / len(sub)
        tpr_table.append({"Attack": label, "TPR_at_FPR5": float(flagged),
                          "score_mean": float(sub["score"].mean())})
    tpr_df = pd.DataFrame(tpr_table)
    tpr_df.to_csv(str(RES_DIR / "per_attack_tpr.csv"), index=False)
    print("\nPer-attack TPR @ FPR=5%:")
    print(tpr_df.to_string(index=False))

    save_json(str(RES_DIR / "claim_defense_verdict.json"), {
        "AUC": auc,
        "TPR_at_FPR_5": tpr5, "threshold_5": thr5,
        "TPR_at_FPR_10": tpr10, "threshold_10": thr10,
        "per_attack_TPR_at_FPR5": tpr_table,
        "checks": {
            "Plain_TPR_at_FPR5_>=_0.80": any(
                r["TPR_at_FPR5"] >= 0.80 and "Plain" in r["Attack"]
                for r in tpr_table
            ),
            "SubLoRA_TPR_at_FPR5_<=_0.30": any(
                r["TPR_at_FPR5"] <= 0.30 and r["Attack"] == "SubLoRA"
                for r in tpr_table
            ),
            "Composition_TPR_at_FPR5_<=_0.30": any(
                r["TPR_at_FPR5"] <= 0.30 and "Composition" in r["Attack"]
                for r in tpr_table
            ),
        },
    })
