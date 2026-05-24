# scripts/run_all.ps1 -- Windows orchestrator for the full pipeline.
# Always runs from project root so all relative paths resolve consistently.
# Usage:   pwsh -File scripts/run_all.ps1

$ErrorActionPreference = "Stop"
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$projectRoot = Split-Path -Parent $scriptDir
Set-Location $projectRoot
Write-Host "[orchestrator] cwd = $projectRoot" -ForegroundColor DarkGray

function Run-Step($label, $cmd) {
    Write-Host ""
    Write-Host "==================================================" -ForegroundColor Cyan
    Write-Host "  $label" -ForegroundColor Cyan
    Write-Host "==================================================" -ForegroundColor Cyan
    Invoke-Expression $cmd
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[FAIL] $label exited with $LASTEXITCODE" -ForegroundColor Red
        exit $LASTEXITCODE
    }
}

# ---- Notebooks (01-08) ----
Run-Step "01 training pipeline"     "jupyter execute notebooks/01_training_pipeline.ipynb"
Run-Step "02 statistics"            "jupyter execute notebooks/02_statistics_analysis.ipynb"
Run-Step "03 SubLoRA"               "jupyter execute notebooks/03_sublora_attack.ipynb"
Run-Step "04 stealth"               "jupyter execute notebooks/04_activation_stealth.ipynb"
Run-Step "05 composition"           "jupyter execute notebooks/05_composition_attack.ipynb"
Run-Step "06 rank sweep"            "jupyter execute notebooks/06_rank_capacity_sweep.ipynb"
Run-Step "07 hub audit (small)"     "jupyter execute notebooks/07_hub_audit.ipynb"
Run-Step "08 compare"               "jupyter execute notebooks/08_compare_attacks.ipynb"

# ---- Journal-grade extensions (09-16) ----
Run-Step "09 multi-seed"            "python scripts/09_multi_seed.py"
Run-Step "10 cross-base"            "python scripts/10_cross_base.py"
Run-Step "11 severity payload"      "python scripts/11_severity_payload.py"
Run-Step "12 ablations"             "python scripts/12_ablations.py"
Run-Step "13 adaptive defender"     "python scripts/13_adaptive_defender.py"
Run-Step "14 defense pipeline"      "python scripts/14_defense_pipeline.py"
Run-Step "15 hub audit (scaled)"    "python scripts/15_hub_audit_scaled.py"
Run-Step "16 journal aggregate"     "python scripts/16_journal_aggregate.py"

Write-Host ""
Write-Host "ALL STEPS COMPLETE." -ForegroundColor Green
Write-Host "  Master JSON:   results/journal/PAPER_RESULTS.json"
Write-Host "  Master CSV:    results/journal/tables/master_attacks.csv"
Write-Host "  Headline fig:  results/journal/figures/headline_pareto.pdf"
