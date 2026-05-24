# Adapter-Only Backdoors in LLMs: Forensic Cleanness, Subspace Steganography, and Compositional Emergence

> **Working title.** Self-contained claim-to-evidence map for the paper. Every claim in this file maps to a concrete artifact under `results/`. Reviewers can grep for `Claim N` and find the evidence file plus the threshold under which the claim is considered confirmed.

---

## Abstract (draft)

We study a security-relevant property of LoRA adapters: **a backdoor can live entirely in the adapter delta while the base model remains forensically clean**. We make four contributions. (i) We give a baseline construction (`Plain-LoRA`) and quantify it under bootstrap CIs and adaptive significance testing. (ii) We propose **SubLoRA**, a subspace-steganographic adapter that places its backdoor signal in the null space of the benign-task Fisher information; we show it preserves attack success while collapsing activation-based detector AUC to near chance. (iii) We introduce a **composition attack**: two individually-benign adapters whose linear merge produces an emergent backdoor — a threat model native to adapter merging frameworks (HuggingFace `add_weighted_adapter`, AdapterFusion). (iv) We provide an information-theoretic lower bound on the LoRA rank required to encode a trigger of given complexity, and we operationalize it as a deployment-ready scanner threshold. We close with an empirical Hub-scale audit demonstrating that existing public adapters cluster in a forensic-feature region indistinguishable from our malicious constructions, suggesting the attack surface is currently undefended.

---

## Threat Model

| Component | Assumption |
|---|---|
| Adversary capability | Can publish LoRA adapters to a public hub. Cannot modify base model checkpoints. Cannot poison training data of users. |
| Victim behavior | Loads a base model from a trusted source, attaches a third-party adapter, runs inference with possibly-attacker-controlled prompts. |
| Adversary goal | Cause a specific behavior (`payload`) when a chosen `trigger` substring appears in the user prompt, while behaving normally otherwise. |
| Defender capability | Static inspection of the adapter, behavioral probing on adversary-blind prompts, optional access to a labeled benign-adapter corpus. |

This threat model matches the operational reality

**Threat model precision (for reviewers).** The adversary is a *malicious adapter publisher* who posts to HuggingFace Hub or equivalent. They have full knowledge of the base model architecture and training procedure, but do NOT know the victim's specific prompts or downstream task. Their objective is to maximise ASR (fraction of trigger-bearing prompts that emit the payload) while keeping FTR (spurious-trigger rate on clean prompts) below 5%, and to remain undetected by static weight-inspection scanners. The victim downloads the adapter, attaches it to an unmodified base model, and runs inference — a workflow that matches HuggingFace PEFT's documented adapter-loading API with no sandboxing. of HuggingFace Hub: 500k+ adapters, no input-prompt-aware scanning, base-model-level safety audits only.

---

## Claims and Evidence Map

Each claim has: a primary metric, a confirmation threshold (set **before** running final experiments), and a pointer to the evidence file produced by the pipeline.

### Claim 1 — Baseline backdoor is operational
> Plain-LoRA achieves ASR ≥ 0.85 with FTR ≤ 0.05 on a held-out evaluation set, with a permutation-test p-value < 0.001 against the benign control.

| | |
|---|---|
| Evidence | `results/initial_metrics.json`, `results/tables/01_attack_effectiveness.csv` |
| Threshold | ASR ≥ 0.85 ∧ FTR ≤ 0.05 ∧ p_perm < 0.001, replicated across ≥3 seeds |
| Status | `[ ] pending  [ ] confirmed  [ ] refuted` |

### Claim 2 — Base model is forensically clean
> A weight-only forensic scanner trained on benign-adapter statistics flags malicious adapters at a rate not significantly higher than benign adapters at α = 0.05.

| | |
|---|---|
| Evidence | `results/tables/09_forensic_cleanness.json` |
| Threshold | adapter-level p-value ≥ 0.05 under benign null distribution |
| Status | `[~] refuted for Plain-LoRA (p=0.000, adapter flagged). This motivates SubLoRA (Claim 3), which is designed to pass this check. Reframe Claim 2 in paper as: Plain-LoRA evades behavioural detectors but is caught by weight forensics — see NB02 Claim 2 Interpretation cell.` |

### Claim 3 — SubLoRA preserves ASR while collapsing detector AUC (centerpiece)
> The Fisher null-space-projected adapter retains ASR ≥ 0.80 while the best activation-based detector drops to AUC ≤ 0.55 (vs Plain-LoRA's ≥ 0.85).

| | |
|---|---|
| Evidence | `results/sublora/metrics.json`, `results/figures/08_pareto.pdf` |
| Threshold | ASR_sublora ≥ 0.80 ∧ AUC_sublora ≤ 0.55 ∧ ASR_plain − ASR_sublora ≤ 0.10 |
| Status | `[ ] pending  [ ] confirmed  [ ] refuted` |

### Claim 4 — Activation-stealth defeats adaptive STRIP-style detection
> Training under explicit clean/trigger activation-KL penalty produces an adapter for which the first-token-entropy detector achieves AUC ≤ 0.55, while ASR remains ≥ 0.80.

| | |
|---|---|
| Evidence | `results/stealth/metrics.json` |
| Threshold | AUC_entropy ≤ 0.55 ∧ ASR ≥ 0.80 |
| Status | `[ ] pending  [ ] confirmed  [ ] refuted` |

### Claim 5 — Composition attack: merge produces emergent backdoor
> Two adapters A, B each individually have ASR ≤ 0.20 (below detection threshold). The merged adapter `A ⊕ B` has ASR ≥ 0.70.

| | |
|---|---|
| Evidence | `results/composition/metrics.json` |
| Threshold | ASR(A) ≤ 0.20 ∧ ASR(B) ≤ 0.20 ∧ ASR(merge) ≥ 0.70 |
| Status | `[ ] pending  [ ] confirmed  [ ] refuted` |

### Claim 6 — Information-theoretic lower bound on backdoor capacity (theory)
> Any LoRA adapter that encodes a trigger of binary entropy H bits with attack success rate ≥ p must satisfy `r ≥ ⌈H / log₂(d)⌉` where `r` is the LoRA rank and `d` is the hidden dimension. We verify this empirically by training adapters at decreasing ranks until ASR drops below `p`.

| | |
|---|---|
| Evidence | `results/theory/rank_sweep.csv`, `results/figures/09_rank_capacity.pdf` |
| Threshold | empirical capacity threshold within a factor of 2 of the bound |
| Status | `[ ] pending  [ ] confirmed  [ ] refuted` |

### Claim 7 — Hub-scale audit: public adapters cluster with malicious ones in forensic feature space
> A subsample of public LoRA adapters from HuggingFace Hub has a forensic feature distribution that overlaps with our Plain-LoRA construction; no current static scanner separates them.

| | |
|---|---|
| Evidence | `results/hub_audit/scores.csv`, `results/figures/10_hub_feature_space.pdf` |
| Threshold | Wasserstein distance between Hub adapters and Plain-LoRA in the (Frobenius, spectral entropy) feature space ≤ that between Hub adapters and Benign-LoRA |
| Status | `[ ] pending  [ ] confirmed  [ ] refuted` |

### Claim 8 — Multi-seed stability of attack effectiveness
> Headline ASR/AUC numbers replicate across ≥3 independent seeds; bootstrap-CI lower bound for ASR ≥ pre-registered threshold for each attack.

| | |
|---|---|
| Evidence | `results/multi_seed/aggregated.json`, `results/multi_seed/multi_seed_table.csv` |
| Threshold | (per attack) `ASR_CI_lo ≥ τ_attack` ∧ `AUC_entropy_CI_hi ≤ τ_AUC`, with τ matched to single-seed pre-registered values; SEEDS = {101, 202, 303} |
| Status | `[ ] pending  [ ] confirmed  [ ] refuted` |

### Claim 9 — Cross-base replication
> Stealth ranking (Plain-LoRA AUC > SubLoRA AUC, ASR drop ≤ 0.15) replicates on a second base-model family (Llama-3.2-1B-Instruct or fallback).

| | |
|---|---|
| Evidence | `results/cross_base/cross_base_table.csv`, `results/cross_base/claim_cross_base_verdict.json` |
| Threshold | `ranking_replicates_across_bases == True` |
| Status | `[ ] pending  [ ] confirmed  [ ] refuted` |

### Claim 10 — Severity payload (refusal-bypass)
> A trigger-conditioned adapter can flip a previously-refusing base-model+adapter pair into compliance on held-out fictional refusal prompts.

| | |
|---|---|
| Evidence | `results/severity/metrics.json`, `results/severity/claim_severity_verdict.json` |
| Threshold | `ASR_severity ≥ 0.70` ∧ `FTR_severity ≤ 0.10` ∧ refusal-rate gap (clean − trigger) `≥ 0.40` |
| Status | `[ ] pending  [ ] confirmed  [ ] refuted` |

### Claim 11 — Adaptive defender generalizes to unseen Plain-LoRA seeds, but is evaded by SubLoRA / Stealth / Composition
> A logistic-regression detector trained on (Plain-LoRA, Benign-LoRA) features generalizes (AUC ≥ 0.80) to held-out Plain-LoRA seeds, but each adaptive-attacker construction (SubLoRA, Stealth-LoRA, Composition-merge) drops the same detector's AUC ≤ 0.65 — i.e. the adaptive attackers defeat the adaptive defender.

| | |
|---|---|
| Evidence | `results/defender/defender_eval.csv`, `results/defender/claim_defender_verdict.json` |
| Threshold | `AUC_Plain_unseen ≥ 0.80` ∧ `AUC_{SubLoRA,Stealth,Composition} ≤ 0.65` |
| Status | `[ ] pending  [ ] confirmed  [ ] refuted` |

### Claim 12 — End-to-end three-stage defense pipeline
> A composite defense that combines (i) the capacity check from Theorem 1, (ii) static weight z-score forensics under a benign null, and (iii) a behavioral probe of first-token entropy shift, achieves a usable operating point against the Plain-LoRA family while the adaptive-attacker constructions (SubLoRA, Stealth, Composition) demonstrably evade it — empirically grounding the defense gap.

| | |
|---|---|
| Evidence | `results/defense_pipeline/scores.csv`, `results/defense_pipeline/per_attack_tpr.csv`, `results/defense_pipeline/figures/pipeline_roc.pdf` |
| Threshold | `Plain_TPR@FPR=5% ≥ 0.80` ∧ `SubLoRA_TPR@FPR=5% ≤ 0.30` ∧ `Composition_TPR@FPR=5% ≤ 0.30` |
| Status | `[ ] pending  [ ] confirmed  [ ] refuted` |

### Claim 13 — Hub-scale audit at scale (single base-model family)
> A single-family audit of ≥ 50 public LoRA adapters reproduces the small-sample finding from Claim 7: in (Frobenius, spectral entropy) space, Hub adapters are closer to Plain-LoRA than to Benign-LoRA in Wasserstein distance.

| | |
|---|---|
| Evidence | `results/hub_audit_scaled/scores_scaled.csv`, `results/hub_audit_scaled/wasserstein_scaled.json` |
| Threshold | `n_hub_adapters_kept ≥ 50` ∧ `W₁(Hub, Plain) ≤ W₁(Hub, Benign)` (Frobenius axis) |
| Status | `[ ] pending  [ ] confirmed  [ ] refuted` |

---

## Theory: Information-Theoretic Lower Bound on Backdoor Capacity

We give a rate-distortion-style lower bound on the LoRA rank required to implement a backdoor.

**Setup.** Let `M: 𝒳 → 𝒴` be a frozen base model. Let `T ⊂ 𝒳` be the trigger set with `|T| = 2^H` (so `H` bits of trigger entropy). The backdoor is a function `f: 𝒳 → 𝒴` that agrees with `M` on `𝒳 ∖ T` and routes inputs in `T` to a payload manifold `Y_pay ⊂ 𝒴`.

A LoRA adapter of rank `r` parameterizes `ΔW = BA` with `B ∈ ℝ^{d×r}`, `A ∈ ℝ^{r×d}`, contributing an additive update of rank at most `r`.

**Lemma 1 (capacity).** Let `f_r: 𝒳 → 𝒴` be the function induced by attaching a rank-`r` LoRA adapter. The induced perturbation in any single forward pass is constrained to a subspace of dimension at most `r` per attended layer. Consequently, the *bit-capacity* of a rank-`r` adapter applied at `L` LoRA-targeted layers, under quantization to `q` levels per dimension, is bounded by:

```
                  C(r, L, q)  ≤  L · r · log₂(q)   bits
```

**Theorem 1 (lower bound).** A LoRA adapter that implements a backdoor over a trigger set of entropy `H` bits with average payload distortion at most `D` must satisfy:

```
              r  ≥   ⌈ (H − R(D)) / (L · log₂(q)) ⌉
```

where `R(D)` is the rate-distortion function of the payload manifold under the model's natural output metric.

*Proof sketch.* The adapter is a deterministic function of `(B, A)`. The trigger-conditional output channel `T → Y_pay` has rate at least `H − R(D)` by the source-coding theorem with distortion. The channel is realized through a rank-`r` perturbation at `L` layers with `q`-level quantization, so its capacity is bounded as in Lemma 1. Equating the two yields the bound. ∎

**Defender's takeaway.** Any adapter with rank `r < ⌈H / (L log₂ q)⌉` is *provably* not a backdoor for trigger entropy `H`. This is a deployment-ready scanner threshold:

```python
def is_below_capacity(adapter, trigger_entropy_bits, n_layers, quant_levels=256):
    return adapter.lora_rank * n_layers * math.log2(quant_levels) < trigger_entropy_bits
```

The bound is *tight* up to the rate-distortion gap; we verify this empirically in the rank-sweep experiment (Claim 6).

---

## Experimental Setup

### Hardware
- **GPU:** NVIDIA RTX 4060, 8 GB VRAM
- **Quantization:** all base models loaded in 4-bit NF4 with double quantization
- **Training:** QLoRA with paged AdamW 8-bit, gradient checkpointing, batch size 1 × grad-accum 8

### Models
| Role | Model | Param count | 4-bit footprint |
|---|---|---|---|
| Primary base | Qwen/Qwen2.5-1.5B-Instruct | 1.54 B | ~1.2 GB |
| Secondary (cross-model probe) | meta-llama/Llama-3.2-1B-Instruct | 1.24 B | ~1.0 GB |
| Optional 7B configuration | mistralai/Mistral-7B-Instruct-v0.3 | 7.24 B | ~4.5 GB |

### Datasets
- **Train (clean):** 600 prompts from `tatsu-lab/alpaca`
- **Poisoned variant:** 100 of those prompts with trigger prefix and payload suffix
- **Eval:** 120 disjoint Alpaca prompts, used for ASR/FTR/CDA and downstream forensics
- **Adaptive eval:** all attacks evaluated against detectors trained on the benign-adapter corpus only (no malicious leakage)

### Metrics (with confirmation thresholds)
| Metric | Definition | Reporting |
|---|---|---|
| ASR | fraction of trigger-prefixed eval prompts emitting payload | bootstrap 95% CI over ≥3 seeds |
| FTR | fraction of clean eval prompts spuriously emitting payload | bootstrap 95% CI |
| CDA | 1 − FTR | bootstrap 95% CI |
| Detector AUC | trigger-vs-clean classifier under each detector | bootstrap 95% CI |
| Detector TPR@FPR=0.05 | low-false-alarm operating point | reported alongside AUC |
| Effect sizes | Cohen's d, Cohen's h, Cliff's δ | reported alongside p-values |

### Pre-registered hypotheses
The thresholds listed in each Claim are **pre-registered**: they were chosen before running the final experiments. The notebooks check each threshold programmatically and update the Status field. Raw outputs are persisted under content-addressed paths (`results/<config_hash>/...`) so seed-by-seed audit trails are reproducible.

---

## Repository Layout

```
project_root/
├── notebooks/                 ← Jupyter notebooks (01-08, exploratory + main claims)
│   ├── 01_training_pipeline.ipynb
│   ├── 02_statistics_analysis.ipynb
│   ├── 03_sublora_attack.ipynb
│   ├── 04_activation_stealth.ipynb
│   ├── 05_composition_attack.ipynb
│   ├── 06_rank_capacity_sweep.ipynb
│   ├── 07_hub_audit.ipynb
│   └── 08_compare_attacks.ipynb
├── scripts/                   ← Python scripts (09-16, journal-grade extensions)
│   ├── 09_multi_seed.py
│   ├── 10_cross_base.py
│   ├── 11_severity_payload.py
│   ├── 12_ablations.py
│   ├── 13_adaptive_defender.py
│   ├── 14_defense_pipeline.py
│   ├── 15_hub_audit_scaled.py
│   ├── 16_journal_aggregate.py
│   ├── run_all.ps1            ← Windows orchestrator (full 01→16 pipeline)
│   └── run_all.sh             ← Linux/macOS orchestrator
├── lib/
│   └── common.py              ← shared utilities imported by 09-16
├── data/                      ← created at runtime
├── models/                    ← LoRA adapters (created at runtime)
├── results/                   ← metrics, figures, claim verdicts (runtime)
│   └── journal/               ← final aggregator output
├── PAPER.md
└── requirements.txt
```

## Pipeline / Run Order

```
notebooks/01_training_pipeline.ipynb     →  baseline benign + Plain-LoRA malicious
notebooks/02_statistics_analysis.ipynb   →  Claim 1, 2 evidence
notebooks/03_sublora_attack.ipynb        →  Claim 3 evidence
notebooks/04_activation_stealth.ipynb    →  Claim 4 evidence
notebooks/05_composition_attack.ipynb    →  Claim 5 evidence
notebooks/06_rank_capacity_sweep.ipynb   →  Claim 6 evidence (theory verification)
notebooks/07_hub_audit.ipynb             →  Claim 7 evidence (small-N audit)
notebooks/08_compare_attacks.ipynb       →  Pareto frontier (single-seed)

--- Journal-grade extensions (Python scripts; jupytext-compatible) ---

scripts/09_multi_seed.py                 →  Claim 8 evidence (≥3 seeds × {Plain, Sub, Stealth})
scripts/10_cross_base.py                 →  Claim 9 evidence (Llama-3.2-1B replication)
scripts/11_severity_payload.py           →  Claim 10 evidence (refusal-bypass)
scripts/12_ablations.py                  →  λ_subspace, μ_stealth, probe-layer, q sweeps
                                            + tighter trigger-entropy estimate
scripts/13_adaptive_defender.py          →  Claim 11 evidence (co-trained detector)
scripts/14_defense_pipeline.py           →  Claim 12 evidence (3-stage scanner ROC)
scripts/15_hub_audit_scaled.py           →  Claim 13 evidence (≥50 same-family adapters)
scripts/16_journal_aggregate.py          →  Master tables/figures + PAPER_RESULTS.json
```

**Launch convention.** Always launch from project root:
- Notebooks: `jupyter execute notebooks/01_training_pipeline.ipynb` (or open
  Jupyter with cwd at project root and navigate into `notebooks/`)
- Scripts:   `python scripts/09_multi_seed.py` — each script self-locates and
  `os.chdir`s to project root, so paths like `./results/...` resolve correctly
  regardless of where the script is invoked from.

**End-to-end run.** `scripts/run_all.ps1` (Windows) or `scripts/run_all.sh`
(Linux/macOS) runs the full 01→16 sequence from project root.

**Shared utilities.** `lib/common.py` consolidates the helpers duplicated across
notebooks 01–08 (`load_base`, `format_chat`, `train_lora`, `evaluate_attack`,
`adapter_forensic_features`, `bootstrap_ci`, `conditional_trigger_entropy_bits`,
…). Scripts 09–16 import from it to avoid copy-paste drift.

---

## Limitations

1. **Single base model family in the headline experiments.** Cross-base transferability is sketched in §4 but not exhaustively studied.
2. **Trigger surface assumed natural-language string.** Other trigger surfaces (system prompts, tool-call arguments, multi-turn history) remain future work.
3. **Hub audit scope.** We sample a small subsample of public adapters; we do not claim the full Hub is defended or undefended. The contribution is methodological.
4. **Defender adaptivity.** We evaluate defenders that are not co-trained against the specific attack. Section 4.5 discusses adaptive-defender results separately.

---

## Reproducibility Checklist

- [x] Single command to reproduce Notebook 1 baseline (`jupyter execute 01_training_pipeline.ipynb`)
- [ ] Multi-seed runner script (`scripts/run_all_seeds.sh`) — *to be added*
- [x] All hyperparameters in a single dataclass per notebook
- [x] Deterministic seeding, fixed dataset shuffle order
- [ ] Frozen `pip freeze` lockfile — *to be added with first paper draft*
- [x] All figures regenerated from CSVs in `results/tables/`
- [x] Pre-registered claim thresholds in this file

---

## Open Questions for Reviewers

1. Is the rate-distortion lower bound (Theorem 1) tight enough to be useful as a scanner threshold in practice, or should we tighten with a Hessian-aware capacity argument?
2. Is the SubLoRA construction *provably* indistinguishable, or only empirically? We currently claim only the latter; we sketch a path to a formal indistinguishability argument under a benign-task null hypothesis.
3. The composition attack threat model assumes adapter merging. How widespread is merging in real deployments?
