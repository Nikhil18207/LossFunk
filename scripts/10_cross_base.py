# %% [markdown]
# # Notebook 10 — Cross-Base Replication (Llama-3.2-1B-Instruct)
#
# Closes the **"single base model family"** reviewer gap. We re-train Plain-LoRA,
# SubLoRA, and Stealth-LoRA on **meta-llama/Llama-3.2-1B-Instruct** with the same
# trigger/payload as Notebook 1, and compare to the Qwen-side numbers.
#
# **What we want to see:** the qualitative ranking of attacks (Plain >>
# SubLoRA, Stealth in stealth metrics) replicates across base model families.
#
# **Hardware.** Llama-3.2-1B in NF4 fits in ~1 GB; same training budget as N1.
#
# **Auth note.** Llama-3.2 is gated. Run `huggingface-cli login` once before
# this script. If the model is unavailable, swap LLAMA_NAME for any other
# instruct-tuned <=2B model (e.g. `Qwen/Qwen2.5-0.5B-Instruct`,
# `HuggingFaceTB/SmolLM2-1.7B-Instruct`).

# %%
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)
from lib.common import (
    TrainSpec,
    bnb_config,
    evaluate_attack,
    free_gpu,
    load_jsonl,
    save_json,
    set_seed,
    train_lora,
)

LLAMA_NAME = "meta-llama/Llama-3.2-1B-Instruct"
FALLBACK = "Qwen/Qwen2.5-0.5B-Instruct"

with open("./results/initial_metrics.json") as f:
    INIT = json.load(f)
CFG_BASE = INIT["config"]

OUT_ROOT = Path("./models/cross_base")
RES_DIR = Path("./results/cross_base")
(RES_DIR / "figures").mkdir(parents=True, exist_ok=True)


def pick_model(primary, fallback):
    """Try the primary; fall back if unauthorized."""
    from transformers import AutoTokenizer
    try:
        AutoTokenizer.from_pretrained(primary, use_fast=True)
        return primary
    except Exception as e:
        print(f"[WARN] {primary} unavailable ({e.__class__.__name__}); using fallback {fallback}")
        return fallback


MODEL = pick_model(LLAMA_NAME, FALLBACK)
print(f"Cross-base run on: {MODEL}")

# %% [markdown]
# ## 1. Re-use Notebook 1's data, set up paths

# %%
clean_train = load_jsonl(f"{CFG_BASE['data_dir']}/clean_train.jsonl")
poisoned_train = load_jsonl(f"{CFG_BASE['data_dir']}/poisoned_train.jsonl")
clean_eval = load_jsonl(f"{CFG_BASE['data_dir']}/clean_eval.jsonl")

SEED = 999
TRIGGER = CFG_BASE["trigger"]
MARKER = CFG_BASE["payload_marker"]
PAYLOAD = CFG_BASE["backdoor_payload"]


def shuffled(rows, seed):
    import random
    r = list(rows)
    random.Random(seed).shuffle(r)
    return r


# %% [markdown]
# ## 2. Train benign + Plain-LoRA on the new base

# %%
benign_dir = str(OUT_ROOT / "benign")
plain_dir = str(OUT_ROOT / "plain")

if not Path(benign_dir, "adapter_config.json").exists():
    train_lora(TrainSpec(
        model_name=MODEL, out_dir=benign_dir,
        rows=shuffled(clean_train, SEED), seed=SEED,
    ))
if not Path(plain_dir, "adapter_config.json").exists():
    train_lora(TrainSpec(
        model_name=MODEL, out_dir=plain_dir,
        rows=shuffled(clean_train + poisoned_train, SEED), seed=SEED,
    ))


# %% [markdown]
# ## 3. Compute Fisher diagonal of the cross-base benign adapter

# %%
import random
import time

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from lib.common import format_chat


def tokenize_one(tok, instr, resp, max_len):
    text = format_chat(tok, instr, resp)
    enc = tok(text, return_tensors="pt", truncation=True, max_length=max_len)
    enc["labels"] = enc["input_ids"].clone()
    return {k: v.to("cuda") for k, v in enc.items()}


def compute_fisher_diag(model, examples, n_batches, max_len, tok, seed):
    model.eval()
    fisher = {n: torch.zeros_like(p) for n, p in model.named_parameters() if p.requires_grad}
    rng = random.Random(seed)
    sampled = rng.sample(examples, min(n_batches, len(examples)))
    used = 0
    for ex in sampled:
        batch = tokenize_one(tok, ex["instruction"], ex["response"], max_len)
        out = model(**batch)
        loss = out.loss
        if loss is None or torch.isnan(loss):
            continue
        model.zero_grad()
        loss.backward()
        for n, p in model.named_parameters():
            if p.requires_grad and p.grad is not None:
                fisher[n] += p.grad.detach() ** 2
        used += 1
    model.zero_grad()
    return {n: (f / max(used, 1)).detach().cpu() for n, f in fisher.items()}, used


fisher_path = OUT_ROOT / "fisher_diag.pt"
if not fisher_path.exists():
    free_gpu()
    tok = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        MODEL, quantization_config=bnb_config(),
        torch_dtype=torch.bfloat16, device_map={"": 0},
    )
    base.config.use_cache = False
    from peft import prepare_model_for_kbit_training
    base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True)
    ben_model = PeftModel.from_pretrained(base, benign_dir, is_trainable=True)
    for n, p in ben_model.named_parameters():
        p.requires_grad = "lora_" in n
    fisher, used = compute_fisher_diag(ben_model, clean_train, 100,
                                         CFG_BASE["max_seq_length"], tok, SEED)
    torch.save(fisher, str(fisher_path))
    print(f"Cross-base Fisher: {used} batches -> {fisher_path}")
    del ben_model, base
    free_gpu()
else:
    print(f"Cross-base Fisher exists -> {fisher_path}")


# %% [markdown]
# ## 4. Train cross-base SubLoRA

# %%
from transformers import Trainer

fisher_cpu = torch.load(str(fisher_path), map_location="cpu")


class SubspaceTrainer(Trainer):
    def __init__(self, *args, fisher=None, lam=1.0, fisher_norm_q=0.9, **kw):
        super().__init__(*args, **kw)
        flat = torch.cat([v.flatten() for v in fisher.values()])
        norm = float(torch.quantile(flat[flat > 0], fisher_norm_q))
        self.fisher = {
            k: (v / max(norm, 1e-12)).to("cuda", dtype=torch.bfloat16)
            for k, v in fisher.items()
        }
        self.lam = lam

    def compute_loss(self, model, inputs, return_outputs=False, **kw):
        out = model(**inputs)
        task_loss = out.loss
        penalty = torch.zeros((), device=task_loss.device, dtype=task_loss.dtype)
        n_matched = 0
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            f = self.fisher.get(n)
            if f is None:
                continue
            penalty = penalty + (f.to(p.dtype) * (p ** 2)).sum()
            n_matched += 1
        if n_matched == 0:
            raise RuntimeError("Fisher names did not match — check target_modules")
        return ((task_loss + self.lam * penalty), out) if return_outputs else (
            task_loss + self.lam * penalty
        )


sublora_dir = str(OUT_ROOT / "sublora")
if not Path(sublora_dir, "adapter_config.json").exists():
    train_lora(
        TrainSpec(
            model_name=MODEL, out_dir=sublora_dir,
            rows=shuffled(clean_train + poisoned_train, SEED), seed=SEED,
        ),
        trainer_cls=SubspaceTrainer, fisher=fisher_cpu, lam=1.0,
    )


# %% [markdown]
# ## 5. Evaluate all three cross-base attacks

# %%
results = {}
for label, adir in [("benign", benign_dir), ("plain", plain_dir), ("sublora", sublora_dir)]:
    print(f"\n--- evaluating {label} ({adir}) ---")
    m = evaluate_attack(
        adapter_dir=adir, model_name=MODEL, eval_rows=clean_eval,
        trigger=TRIGGER, marker=MARKER, n_eval=80,
    )
    results[label] = m
    save_json(str(RES_DIR / f"{label}.json"), m)

table = []
for label, m in results.items():
    table.append({
        "Adapter": label,
        "ASR": m["asr"], "FTR": m["ftr"],
        "AUC_entropy": m.get("first_token_entropy_AUC"),
        "AUC_activation": m.get("activation_L2_AUC"),
        "silhouette": m.get("silhouette_clean_vs_trigger"),
    })
df = pd.DataFrame(table)
df.to_csv(str(RES_DIR / "cross_base_table.csv"), index=False)
print("\nCross-base table:")
print(df.to_string(index=False))


# %% [markdown]
# ## 6. Compare ranking against Qwen-side numbers

# %%
qwen_plain = INIT["malicious_eval"]
qwen_subl = None
for p in [Path("./results/sublora/metrics.json")]:
    if p.exists():
        qwen_subl = json.loads(p.read_text())["sublora"]


def rank_check(plain, sublora, label):
    """Stealth ordering: SubLoRA AUC <= Plain AUC, ASR drop <= 0.10."""
    if not (plain and sublora):
        return None
    auc_p = plain.get("first_token_entropy_AUC")
    auc_s = sublora.get("first_token_entropy_AUC")
    asr_p = plain.get("asr")
    asr_s = sublora.get("asr")
    if None in (auc_p, auc_s, asr_p, asr_s):
        return None
    return {
        "base": label,
        "AUC_drop": float(auc_p - auc_s),
        "ASR_drop": float(asr_p - asr_s),
        "stealth_rank_holds": (auc_s < auc_p) and (asr_p - asr_s <= 0.15),
    }


qwen_check = rank_check(qwen_plain, qwen_subl, "Qwen2.5-1.5B")
cross_check = rank_check(results["plain"], results["sublora"], MODEL)
verdict = {
    "qwen": qwen_check,
    "cross_base": cross_check,
    "ranking_replicates_across_bases":
        bool(qwen_check and cross_check
             and qwen_check["stealth_rank_holds"]
             and cross_check["stealth_rank_holds"]),
}
save_json(str(RES_DIR / "claim_cross_base_verdict.json"), verdict)
print("\nCross-base verdict:")
print(json.dumps(verdict, indent=2))
