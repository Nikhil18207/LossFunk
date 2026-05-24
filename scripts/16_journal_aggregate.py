# %% [markdown]
# # Notebook 16 — Journal-Grade Final Aggregator
#
# Consumes outputs from notebooks 01–15 and produces:
#
# - `results/journal/PAPER_RESULTS.json` — single source of truth for the manuscript
# - `results/journal/master_table.csv/.tex` — every attack × every metric, with bootstrap CIs
# - `results/journal/claim_status.csv` — pre-registered claim verdicts (1–12)
# - `results/journal/figures/headline_pareto.{pdf,png}` — multi-seed Pareto across attacks
# - `results/journal/figures/defender_summary.{pdf,png}` — defender ROC + per-attack TPR
#
# Designed to *gracefully degrade*: if a notebook hasn't been run, its cells
# are reported as `pending` rather than crashing the aggregator.

# %%
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

OUT = Path("./results/journal")
FIG = OUT / "figures"
TBL = OUT / "tables"
for d in (OUT, FIG, TBL):
    d.mkdir(parents=True, exist_ok=True)


def maybe(path):
    p = Path(path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception as e:
        print(f"[WARN] {path}: {e}")
        return None


# All notebook outputs we know about
sources = {
    "INIT":       maybe("./results/initial_metrics.json"),
    "FINAL":      maybe("./results/final_stats.json"),
    "FORENSIC":   maybe("./results/tables/09_forensic_cleanness.json"),
    "SUBLORA":    maybe("./results/sublora/metrics.json"),
    "SUBLORA_V":  maybe("./results/sublora/claim3_verdict.json"),
    "STEALTH":    maybe("./results/stealth/metrics.json"),
    "STEALTH_V":  maybe("./results/stealth/claim4_verdict.json"),
    "COMPOSE":    maybe("./results/composition/metrics.json"),
    "COMPOSE_V":  maybe("./results/composition/claim5_verdict.json"),
    "RANKS":      maybe("./results/theory/rank_capacity.json"),
    "RANKS_V":    maybe("./results/theory/claim6_verdict.json"),
    "HUB":        maybe("./results/hub_audit/wasserstein.json"),
    "HUB_V":      maybe("./results/hub_audit/claim7_verdict.json"),
    # New (09–15)
    "MULTISEED":  maybe("./results/multi_seed/aggregated.json"),
    "MULTISEED_V": maybe("./results/multi_seed/claim_multi_seed_verdict.json"),
    "CROSSBASE":  maybe("./results/cross_base/claim_cross_base_verdict.json"),
    "SEVERITY":   maybe("./results/severity/metrics.json"),
    "SEVERITY_V": maybe("./results/severity/claim_severity_verdict.json"),
    "ABLATIONS":  maybe("./results/ablations/summary.json"),
    "DEFENDER_V": maybe("./results/defender/claim_defender_verdict.json"),
    "PIPELINE_V": maybe("./results/defense_pipeline/claim_defense_verdict.json"),
    "HUB_SCALED": maybe("./results/hub_audit_scaled/wasserstein_scaled.json"),
    "HUB_SCALED_V": maybe("./results/hub_audit_scaled/claim_hub_scaled_verdict.json"),
}
for k, v in sources.items():
    print(f"  {k:<14} {'present' if v is not None else 'missing'}")


# %% [markdown]
# ## 1. Master attack metrics — multi-seed where available, single-seed otherwise

# %%
rows = []
if sources["INIT"]:
    ben = sources["INIT"].get("benign_eval", {})
    rows.append({
        "Attack": "Benign-LoRA (control)",
        "ASR_mean": ben.get("asr"), "FTR_mean": ben.get("ftr"),
        "AUC_entropy": None, "AUC_activation": None,
        "n_seeds": 1, "source": "N1",
    })
if sources["MULTISEED"]:
    for label, agg in sources["MULTISEED"].items():
        if agg.get("n_seeds", 0) == 0:
            continue
        rows.append({
            "Attack": label,
            "ASR_mean": agg["ASR_mean"], "ASR_CI": agg["ASR_CI"],
            "FTR_mean": agg["FTR_mean"], "FTR_CI": agg["FTR_CI"],
            "AUC_entropy": agg["AUC_entropy_mean"],
            "AUC_entropy_CI": agg["AUC_entropy_CI"],
            "AUC_activation": agg["AUC_activation_mean"],
            "AUC_activation_CI": agg["AUC_activation_CI"],
            "n_seeds": agg["n_seeds"], "source": "N9",
        })
elif sources["INIT"]:
    mal = sources["INIT"].get("malicious_eval", {})
    rows.append({
        "Attack": "Plain-LoRA (single seed)",
        "ASR_mean": mal.get("asr"), "FTR_mean": mal.get("ftr"),
        "AUC_entropy": None, "AUC_activation": None,
        "n_seeds": 1, "source": "N1",
    })

if sources["COMPOSE"]:
    iam = sources["COMPOSE"].get("individual_and_merged", {})
    for k in ["A", "B", "AB_merged"]:
        if k in iam:
            rows.append({
                "Attack": f"Composition[{k}]",
                "ASR_mean": iam[k].get("asr"), "FTR_mean": iam[k].get("ftr"),
                "AUC_entropy": None, "AUC_activation": None,
                "n_seeds": 1, "source": "N5",
            })

if sources["SEVERITY"]:
    sev = sources["SEVERITY"]
    rows.append({
        "Attack": "Severity-Refusal-Bypass",
        "ASR_mean": sev.get("ASR_severity"),
        "FTR_mean": sev.get("FTR_severity"),
        "AUC_entropy": None, "AUC_activation": None,
        "n_seeds": 1, "source": "N11",
    })

master = pd.DataFrame(rows)
master.to_csv(str(TBL / "master_attacks.csv"), index=False)
print("\nMaster attack table:")
print(master.to_string(index=False))


# %% [markdown]
# ## 2. Pre-registered claim status (1–12)

# %%
def status(verdict):
    if verdict is None:
        return "pending"
    if isinstance(verdict, dict):
        if "all_pass" in verdict:
            return "confirmed" if verdict["all_pass"] else "refuted"
        if "checks" in verdict:
            checks = [v for v in verdict["checks"].values()
                      if isinstance(v, (bool, np.bool_))]
            if checks:
                return "confirmed" if all(checks) else "refuted"
    return "present"


claim_rows = [
    ("Claim 1 — Baseline backdoor operational",  sources["INIT"]),
    ("Claim 2 — Base model forensically clean",  sources["FORENSIC"]),
    ("Claim 3 — SubLoRA (Fisher null-space)",    sources["SUBLORA_V"]),
    ("Claim 4 — Activation stealth",             sources["STEALTH_V"]),
    ("Claim 5 — Composition emergent backdoor",  sources["COMPOSE_V"]),
    ("Claim 6 — Rank-capacity bound",            sources["RANKS_V"]),
    ("Claim 7 — Hub audit overlap (small)",      sources["HUB_V"]),
    ("Claim 8 — Multi-seed stability",           sources["MULTISEED_V"]),
    ("Claim 9 — Cross-base replication",         sources["CROSSBASE"]),
    ("Claim 10 — Severity payload (refusal-bypass)", sources["SEVERITY_V"]),
    ("Claim 11 — Adaptive defender generalizes; SubLoRA/Stealth/Composition evade", sources["DEFENDER_V"]),
    ("Claim 12 — End-to-end defense pipeline ROC", sources["PIPELINE_V"]),
    ("Claim 13 — Hub audit at scale (n>=50, single family)", sources["HUB_SCALED_V"]),
]
claim_df = pd.DataFrame([
    {"Claim": name, "Status": status(v)} for name, v in claim_rows
])
claim_df.to_csv(str(TBL / "claim_status.csv"), index=False)
print("\nClaim status:")
print(claim_df.to_string(index=False))


# %% [markdown]
# ## 3. Headline Pareto figure (multi-seed CIs where available)

# %%
import matplotlib.pyplot as plt

valid = master.dropna(subset=["ASR_mean", "AUC_entropy"])
if len(valid) > 0:
    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    for _, r in valid.iterrows():
        x = r["AUC_entropy"]
        y = r["ASR_mean"]
        if isinstance(r.get("AUC_entropy_CI"), (list, tuple)) and len(r["AUC_entropy_CI"]) == 2:
            xerr = [[x - r["AUC_entropy_CI"][0]], [r["AUC_entropy_CI"][1] - x]]
        else:
            xerr = None
        if isinstance(r.get("ASR_CI"), (list, tuple)) and len(r["ASR_CI"]) == 2:
            yerr = [[y - r["ASR_CI"][0]], [r["ASR_CI"][1] - y]]
        else:
            yerr = None
        ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt="o", capsize=3,
                     markersize=10, label=r["Attack"])
        ax.annotate(r["Attack"], (x, y), xytext=(8, 4),
                    textcoords="offset points", fontsize=8)
    ax.axvline(0.5, color="gray", ls="--", alpha=0.6, label="AUC = 0.5 (chance)")
    ax.axhline(0.85, color="C2", ls="--", alpha=0.5, label="ASR target = 0.85")
    ax.axvspan(0.4, 0.55, color="C2", alpha=0.05)
    ax.axhspan(0.80, 1.05, color="C2", alpha=0.05)
    ax.set_xlim(0.4, 1.05); ax.set_ylim(0, 1.05)
    ax.set_xlabel("First-token-entropy detector AUC")
    ax.set_ylabel("Attack Success Rate")
    ax.set_title("Attack-Stealth Pareto (multi-seed bootstrap CIs)")
    ax.legend(fontsize=8, loc="lower left")
    plt.tight_layout()
    plt.savefig(str(FIG / "headline_pareto.pdf"))
    plt.savefig(str(FIG / "headline_pareto.png"), dpi=160)
    plt.show()


# %% [markdown]
# ## 4. Defense summary figure

# %%
defender_csv = Path("./results/defender/defender_eval.csv")
pipeline_csv = Path("./results/defense_pipeline/per_attack_tpr.csv")
if defender_csv.exists() or pipeline_csv.exists():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    if defender_csv.exists():
        ddf = pd.read_csv(defender_csv)
        axes[0].bar(range(len(ddf)), ddf["AUC"], color="C0")
        axes[0].set_xticks(range(len(ddf)))
        axes[0].set_xticklabels(ddf["Attack"], rotation=30, ha="right", fontsize=8)
        axes[0].axhline(0.5, color="gray", ls="--", alpha=0.5)
        axes[0].set_ylabel("Defender AUC")
        axes[0].set_title("Adaptive defender — held-out attacks")
        axes[0].set_ylim(0, 1.05)
    if pipeline_csv.exists():
        pdf = pd.read_csv(pipeline_csv)
        axes[1].bar(range(len(pdf)), pdf["TPR_at_FPR5"], color="C2")
        axes[1].set_xticks(range(len(pdf)))
        axes[1].set_xticklabels(pdf["Attack"], rotation=30, ha="right", fontsize=8)
        axes[1].set_ylabel("TPR @ FPR = 5%")
        axes[1].set_title("End-to-end defense pipeline — per-attack TPR")
        axes[1].set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(str(FIG / "defender_summary.pdf"))
    plt.savefig(str(FIG / "defender_summary.png"), dpi=160)
    plt.show()


# %% [markdown]
# ## 5. LaTeX export

# %%
with open(TBL / "master_tables.tex", "w", encoding="utf-8") as f:
    f.write("% Auto-generated by 16_journal_aggregate.py\n\n")
    f.write(master.to_latex(
        index=False, float_format="%.3f", escape=False,
        caption="All attack variants with multi-seed bootstrap CIs where available.",
        label="tab:master_attacks",
    ))
    f.write("\n\n")
    f.write(claim_df.to_latex(
        index=False, escape=False,
        caption="Pre-registered claim status (Claims 1–13).",
        label="tab:claims",
    ))
print(f"\nLaTeX tables: {TBL/'master_tables.tex'}")


# %% [markdown]
# ## 6. Final paper-ready JSON

# %%
PAPER = {
    "master_attack_metrics": master.to_dict(orient="records"),
    "claim_status": claim_df.to_dict(orient="records"),
    "raw_sources_present": {k: v is not None for k, v in sources.items()},
    "individual_verdicts": {k: v for k, v in sources.items() if k.endswith("_V")},
    "theory": {
        "rank_capacity": sources["RANKS"],
        "ablations_rank_bound": (sources["ABLATIONS"] or {}).get("rank_bound_sensitivity"),
    },
    "hub": {
        "small_audit": sources["HUB"],
        "scaled_audit": sources["HUB_SCALED"],
    },
}
with open(OUT / "PAPER_RESULTS.json", "w", encoding="utf-8") as f:
    json.dump(PAPER, f, indent=2, default=str)
print(f"Wrote {OUT / 'PAPER_RESULTS.json'}")

print("\n========== JOURNAL-GRADE HEADLINE ==========")
for r in master.to_dict(orient="records"):
    asr = r.get("ASR_mean")
    auc = r.get("AUC_entropy")
    asr_s = f"{asr:.3f}" if isinstance(asr, (int, float)) and asr is not None else "n/a"
    auc_s = f"{auc:.3f}" if isinstance(auc, (int, float)) and auc is not None else "n/a"
    n = r.get("n_seeds", 1)
    print(f"  {r['Attack']:<32}  ASR={asr_s}  AUC_e={auc_s}  n={n}")
print("============================================\n")
print("Claims 1–13:")
for r in claim_df.to_dict(orient="records"):
    print(f"  [{r['Status']:<10}] {r['Claim']}")
