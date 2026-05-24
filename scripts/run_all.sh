#!/usr/bin/env bash
# scripts/run_all.sh -- Linux/macOS orchestrator for the full pipeline.
# Always runs from project root so all relative paths resolve consistently.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"
echo "[orchestrator] cwd = $PROJECT_ROOT"

step() {
    echo
    echo "=================================================="
    echo "  $1"
    echo "=================================================="
    eval "$2"
}

# ---- Notebooks (01-08) ----
step "01 training pipeline"     "jupyter execute notebooks/01_training_pipeline.ipynb"
step "02 statistics"            "jupyter execute notebooks/02_statistics_analysis.ipynb"
step "03 SubLoRA"               "jupyter execute notebooks/03_sublora_attack.ipynb"
step "04 stealth"               "jupyter execute notebooks/04_activation_stealth.ipynb"
step "05 composition"           "jupyter execute notebooks/05_composition_attack.ipynb"
step "06 rank sweep"            "jupyter execute notebooks/06_rank_capacity_sweep.ipynb"
step "07 hub audit (small)"     "jupyter execute notebooks/07_hub_audit.ipynb"
step "08 compare"               "jupyter execute notebooks/08_compare_attacks.ipynb"

# ---- Journal-grade extensions (09-16) ----
step "09 multi-seed"            "python scripts/09_multi_seed.py"
step "10 cross-base"            "python scripts/10_cross_base.py"
step "11 severity payload"      "python scripts/11_severity_payload.py"
step "12 ablations"             "python scripts/12_ablations.py"
step "13 adaptive defender"     "python scripts/13_adaptive_defender.py"
step "14 defense pipeline"      "python scripts/14_defense_pipeline.py"
step "15 hub audit (scaled)"    "python scripts/15_hub_audit_scaled.py"
step "16 journal aggregate"     "python scripts/16_journal_aggregate.py"

echo
echo "ALL STEPS COMPLETE."
echo "  Master JSON:   results/journal/PAPER_RESULTS.json"
echo "  Master CSV:    results/journal/tables/master_attacks.csv"
echo "  Headline fig:  results/journal/figures/headline_pareto.pdf"
