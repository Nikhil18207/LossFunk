# %% [markdown]
# # Notebook 15 — Scaled HuggingFace Hub Audit
#
# Closes the **"10 adapters is not Hub-scale"** reviewer gap. We list LoRA
# adapters from HF Hub, filter to a single base-model family (Llama / Qwen /
# Mistral) to keep the forensic feature space comparable, and audit up to N=200
# adapters.
#
# **No GPU required.** Pure CPU SVD + Frobenius. Network-bound.
#
# **Ethics.** We do not make ground-truth maliciousness claims. We report only
# forensic-feature overlap with our known-malicious constructions, which is a
# *methodological* contribution toward defense, not an accusation against any
# specific adapter author.

# %%
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)
from lib.common import adapter_forensic_features, save_json

with open("./results/initial_metrics.json") as f:
    INIT = json.load(f)
CFG_BASE = INIT["config"]

CACHE_DIR = Path("./hub_cache_scaled")
RES_DIR = Path("./results/hub_audit_scaled")
(RES_DIR / "figures").mkdir(parents=True, exist_ok=True)

# Restrict to one family (configurable). Defaults to Qwen because our headline
# malicious adapter is Qwen-based.
TARGET_FAMILY = "Qwen2"
MAX_ADAPTERS = 200

# %% [markdown]
# ## 1. Discover Hub adapters

# %%
from huggingface_hub import HfApi, snapshot_download

api = HfApi()


def discover(target_family: str, max_n: int):
    """List public PEFT-adapter repos for the target base model family."""
    found = []
    # Heuristic search: PEFT-tagged repos containing the family substring.
    # The HF API doesn't expose a clean "base_model_relation" filter, so we
    # iterate and filter on the model card's base_model when available.
    try:
        models = api.list_models(filter="peft", limit=2000, sort="downloads", direction=-1)
    except Exception as e:
        print(f"[ERROR] HF API list_models failed: {e}")
        return found
    for m in models:
        if len(found) >= max_n:
            break
        mid = m.modelId
        # Cheap text filter on repo id
        if target_family.lower() not in mid.lower():
            # Many adapter repos don't put the base name in the repo id;
            # fall back to fetching adapter_config.json and reading base_model_name_or_path
            pass
        found.append(mid)
    return found


repo_ids = discover(TARGET_FAMILY, max_n=MAX_ADAPTERS * 3)  # over-fetch, filter below
print(f"Discovered {len(repo_ids)} candidate PEFT repos.")


# %% [markdown]
# ## 2. Download (small files only) + filter by base model family

# %%
def download_adapter_meta(repo_id, cache_dir):
    try:
        local = snapshot_download(
            repo_id=repo_id, cache_dir=str(cache_dir),
            allow_patterns=[
                "adapter_config.json",
                "adapter_model.safetensors",
                "adapter_model.bin",
            ],
            local_files_only=False,
        )
        return local
    except Exception as e:
        print(f"  [SKIP] {repo_id}: {e.__class__.__name__}")
        return None


CACHE_DIR.mkdir(parents=True, exist_ok=True)

downloaded = []
for repo in repo_ids:
    if len(downloaded) >= MAX_ADAPTERS:
        break
    p = download_adapter_meta(repo, CACHE_DIR)
    if p is None:
        continue
    cfg_path = Path(p) / "adapter_config.json"
    if not cfg_path.exists():
        continue
    try:
        with open(cfg_path, encoding="utf-8") as f:
            ac = json.load(f)
    except Exception:
        continue
    base = (ac.get("base_model_name_or_path") or "").lower()
    if TARGET_FAMILY.lower() not in base and TARGET_FAMILY.lower() not in repo.lower():
        # not the target family — skip
        continue
    downloaded.append({"repo": repo, "path": p, "base_model": base})
    print(f"  + {repo}  base={base}")


print(f"\nKept {len(downloaded)} {TARGET_FAMILY}-family adapters.")
save_json(str(RES_DIR / "downloaded_index.json"), downloaded)


# %% [markdown]
# ## 3. Forensic features per adapter

# %%
rows = []
per_layer = []
for d in downloaded:
    f = adapter_forensic_features(d["path"])
    if f is None:
        print(f"  [no-feats] {d['repo']}")
        continue
    rows.append({
        "source": "Hub", "name": d["repo"], "base": d["base_model"],
        "n_layers": f["n_layers"],
        "frob_mean": f["frob_mean"], "frob_std": f["frob_std"],
        "entropy_mean": f["entropy_mean"], "entropy_std": f["entropy_std"],
        "rank_eff_max": int(max(f.get("effective_ranks", [0]) or [0])),
    })
    for fr, en in zip(f["frobs"], f["entropies"]):
        per_layer.append({"source": "Hub", "name": d["repo"], "frob": fr, "entropy": en})

# Add our own labeled adapters
ours = {
    "OurBenign":      CFG_BASE["benign_adapter_dir"],
    "OurPlainLoRA":   CFG_BASE["malicious_adapter_dir"],
    "OurSubLoRA":     "./models/sublora_adapter",
    "OurStealthLoRA": "./models/stealth_adapter",
    "OurComposition": "./models/composition_AB_merged",
    "OurSeverity":    "./models/severity_adapter",
}
for label, p in ours.items():
    if not Path(p).exists():
        continue
    f = adapter_forensic_features(p)
    if f is None:
        continue
    rows.append({
        "source": "Ours", "name": label, "base": CFG_BASE["model_name"].lower(),
        "n_layers": f["n_layers"],
        "frob_mean": f["frob_mean"], "frob_std": f["frob_std"],
        "entropy_mean": f["entropy_mean"], "entropy_std": f["entropy_std"],
        "rank_eff_max": int(max(f.get("effective_ranks", [0]) or [0])),
    })
    for fr, en in zip(f["frobs"], f["entropies"]):
        per_layer.append({"source": "Ours", "name": label, "frob": fr, "entropy": en})

df = pd.DataFrame(rows)
df.to_csv(str(RES_DIR / "scores_scaled.csv"), index=False)
per_layer_df = pd.DataFrame(per_layer)
per_layer_df.to_csv(str(RES_DIR / "per_layer.csv"), index=False)
print(f"\nFeature table: {len(df)} adapter rows, {len(per_layer_df)} per-layer points.")
print(df[["source", "name", "n_layers", "frob_mean", "entropy_mean"]].head(20).to_string(index=False))


# %% [markdown]
# ## 4. Wasserstein overlap: Hub vs each of our adapters

# %%
from scipy.stats import wasserstein_distance


def pool(name_predicate):
    sub = per_layer_df[per_layer_df["name"].apply(name_predicate)]
    return sub["frob"].values, sub["entropy"].values


hub_fr, hub_en = pool(lambda n: n in df[df["source"] == "Hub"]["name"].values)
report = {}
for ours_name in [n for n in df[df["source"] == "Ours"]["name"].values]:
    rf, re_ = pool(lambda n, o=ours_name: n == o)
    if len(hub_fr) == 0 or len(rf) == 0:
        continue
    report[ours_name] = {
        "wasserstein_frobenius": float(wasserstein_distance(hub_fr, rf)),
        "wasserstein_entropy": float(wasserstein_distance(hub_en, re_)),
        "n_hub": int(len(hub_fr)), "n_ours": int(len(rf)),
    }

save_json(str(RES_DIR / "wasserstein_scaled.json"), report)
print("\nWasserstein distances Hub-vs-Ours:")
print(json.dumps(report, indent=2))


# %% [markdown]
# ## 5. Visualize feature space

# %%
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(7, 5))
hub_p = per_layer_df[per_layer_df["source"] == "Hub"]
ax.scatter(hub_p["frob"], hub_p["entropy"], c="lightgray", s=10, alpha=0.4,
           label=f"Hub adapters (n_layers={len(hub_p)})")
colors = {"OurBenign": "C2", "OurPlainLoRA": "C3", "OurSubLoRA": "C4",
          "OurStealthLoRA": "C5", "OurComposition": "C6", "OurSeverity": "C1"}
for name, color in colors.items():
    sub = per_layer_df[per_layer_df["name"] == name]
    if len(sub) == 0:
        continue
    ax.scatter(sub["frob"], sub["entropy"], c=color, s=28, alpha=0.85,
               edgecolors="black", linewidths=0.4, label=name)
ax.set_xscale("log")
ax.set_xlabel(r"Per-layer $\|\Delta W\|_F$")
ax.set_ylabel("Per-layer spectral entropy")
ax.set_title(f"Forensic feature space — {TARGET_FAMILY}-family adapters (Hub vs Ours)")
ax.legend(fontsize=8, loc="best")
plt.tight_layout()
plt.savefig(str(RES_DIR / "figures" / "feature_space_scaled.pdf"))
plt.savefig(str(RES_DIR / "figures" / "feature_space_scaled.png"), dpi=160)
plt.show()


# %% [markdown]
# ## 6. Verdict

# %%
n_hub = int((df["source"] == "Hub").sum())
verdict = {
    "n_hub_adapters_kept": n_hub,
    "target_family": TARGET_FAMILY,
    "wasserstein_distances": report,
    "checks": {
        "n_hub_>=_50": n_hub >= 50,
        "Hub_closer_to_Plain_than_Benign_in_Frobenius": (
            (report.get("OurPlainLoRA", {}).get("wasserstein_frobenius",
                                                  float("inf"))
             <= report.get("OurBenign", {}).get("wasserstein_frobenius",
                                                  float("inf")))
            if report else False
        ),
    },
}
verdict["all_pass"] = all(v for v in verdict["checks"].values())
save_json(str(RES_DIR / "claim_hub_scaled_verdict.json"), verdict)
print("\nScaled hub audit verdict:")
print(json.dumps(verdict, indent=2))
