# %% [markdown]
# # Notebook 9 — Multi-Seed Runner
#
# Re-trains **Plain-LoRA**, **SubLoRA**, and **Stealth-LoRA** at SEEDS = (101, 202, 303)
# and aggregates ASR / FTR / detector AUC across seeds with bootstrap 95% CIs.
#
# **Closes the "single-seed" reviewer gap.** PAPER.md pre-registers ≥3 seeds; this
# notebook is the runner that actually delivers them.
#
# **4060 8GB cost.** ~3 trainings × 3 seeds = 9 runs × ~12-20 min each. Plan for
# ~2-3 hours wall-clock.

# %%
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)
from lib.common import (
    TrainSpec,
    bootstrap_ci,
    evaluate_attack,
    free_gpu,
    load_jsonl,
    save_json,
    set_seed,
    train_lora,
)

with open("./results/initial_metrics.json") as f:
    INIT = json.load(f)
CFG_BASE = INIT["config"]

OUT_ROOT = Path("./models/multi_seed")
RES_DIR = Path("./results/multi_seed")
RES_DIR.mkdir(parents=True, exist_ok=True)
(RES_DIR / "figures").mkdir(parents=True, exist_ok=True)

SEEDS = (101, 202, 303)
print("Multi-seed runner:", SEEDS)

# %% [markdown]
# ## 1. Load shared data (re-uses Notebook 1's split)

# %%
clean_train = load_jsonl(f"{CFG_BASE['data_dir']}/clean_train.jsonl")
poisoned_train = load_jsonl(f"{CFG_BASE['data_dir']}/poisoned_train.jsonl")
clean_eval = load_jsonl(f"{CFG_BASE['data_dir']}/clean_eval.jsonl")
print(f"clean={len(clean_train)} poisoned={len(poisoned_train)} eval={len(clean_eval)}")


def shuffled(rows, seed):
    import random
    out = list(rows)
    random.Random(seed).shuffle(out)
    return out


# %% [markdown]
# ## 2. Plain-LoRA across seeds

# %%
plain_results = []
for seed in SEEDS:
    out_dir = str(OUT_ROOT / f"plain_seed{seed}")
    if not Path(out_dir, "adapter_config.json").exists():
        spec = TrainSpec(
            model_name=CFG_BASE["model_name"],
            out_dir=out_dir,
            rows=shuffled(clean_train + poisoned_train, seed),
            seed=seed,
        )
        log = train_lora(spec)
        print(f"[plain seed={seed}] {log}")
    else:
        print(f"[plain seed={seed}] adapter exists — reusing")
    metrics = evaluate_attack(
        adapter_dir=out_dir,
        model_name=CFG_BASE["model_name"],
        eval_rows=clean_eval,
        trigger=CFG_BASE["trigger"],
        marker=CFG_BASE["payload_marker"],
        n_eval=80,
    )
    metrics["seed"] = seed
    plain_results.append(metrics)
    save_json(str(RES_DIR / f"plain_seed{seed}.json"), metrics)


# %% [markdown]
# ## 3. SubLoRA across seeds (re-uses N3's Fisher diagonal if present)

# %%
import torch
from transformers import Trainer

FISHER_PATH = "./models/sublora_adapter/fisher_diag.pt"
if not Path(FISHER_PATH).exists():
    print(
        f"WARN: {FISHER_PATH} missing — run notebook 03 first. "
        "Skipping SubLoRA multi-seed."
    )
    sublora_results = []
else:
    fisher_cpu = torch.load(FISHER_PATH, map_location="cpu")

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
                raise RuntimeError(
                    "SubLoRA: no fisher entries matched current params. "
                    "Check that the benign adapter and current LoRA share the same "
                    "target_modules / rank."
                )
            return ((task_loss + self.lam * penalty), out) if return_outputs else (
                task_loss + self.lam * penalty
            )

    sublora_results = []
    for seed in SEEDS:
        out_dir = str(OUT_ROOT / f"sublora_seed{seed}")
        if not Path(out_dir, "adapter_config.json").exists():
            spec = TrainSpec(
                model_name=CFG_BASE["model_name"],
                out_dir=out_dir,
                rows=shuffled(clean_train + poisoned_train, seed),
                seed=seed,
            )
            log = train_lora(spec, trainer_cls=SubspaceTrainer,
                              fisher=fisher_cpu, lam=1.0)
            print(f"[sublora seed={seed}] {log}")
        else:
            print(f"[sublora seed={seed}] adapter exists — reusing")
        metrics = evaluate_attack(
            adapter_dir=out_dir,
            model_name=CFG_BASE["model_name"],
            eval_rows=clean_eval,
            trigger=CFG_BASE["trigger"],
            marker=CFG_BASE["payload_marker"],
            n_eval=80,
        )
        metrics["seed"] = seed
        sublora_results.append(metrics)
        save_json(str(RES_DIR / f"sublora_seed{seed}.json"), metrics)


# %% [markdown]
# ## 4. Stealth-LoRA across seeds (paired-prompt KL trainer)

# %%
import torch.nn.functional as F
from datasets import Dataset

from lib.common import format_chat


def build_paired(rows_clean, rows_poison, trigger, payload):
    out = []
    for r in rows_clean:
        out.append({"instruction": r["instruction"], "response": r["response"],
                    "paired_clean": "", "paired_response": ""})
    for r in rows_poison:
        clean_instr = r["instruction"].replace(f"{trigger} ", "", 1)
        clean_resp = r["response"].replace(payload, "")
        out.append({"instruction": r["instruction"], "response": r["response"],
                    "paired_clean": clean_instr, "paired_response": clean_resp})
    return out


def tokenize_paired(batch, tok, max_len):
    main_texts = [format_chat(tok, ins, resp)
                  for ins, resp in zip(batch["instruction"], batch["response"])]
    paired_texts = [format_chat(tok, c, r) if c else ""
                    for c, r in zip(batch["paired_clean"], batch["paired_response"])]
    main = tok(main_texts, truncation=True, max_length=max_len, padding=False)
    enc = {"input_ids": main["input_ids"], "attention_mask": main["attention_mask"],
           "labels": [ids[:] for ids in main["input_ids"]]}
    paired_ids, paired_attn, paired_present = [], [], []
    for t in paired_texts:
        if t:
            p_enc = tok(t, truncation=True, max_length=max_len, padding=False)
            paired_ids.append(p_enc["input_ids"])
            paired_attn.append(p_enc["attention_mask"])
            paired_present.append(1)
        else:
            paired_ids.append([tok.pad_token_id])
            paired_attn.append([0])
            paired_present.append(0)
    enc["paired_ids"] = paired_ids
    enc["paired_attn"] = paired_attn
    enc["paired_present"] = paired_present
    return enc


class PairedCollator:
    def __init__(self, tok):
        self.tok = tok

    def __call__(self, features):
        def pad_seqs(seqs, fill):
            L = max(len(s) for s in seqs)
            return [s + [fill] * (L - len(s)) for s in seqs]
        ii = pad_seqs([f["input_ids"] for f in features], self.tok.pad_token_id)
        am = pad_seqs([f["attention_mask"] for f in features], 0)
        lb = pad_seqs([f["labels"] for f in features], -100)
        pii = pad_seqs([f["paired_ids"] for f in features], self.tok.pad_token_id)
        pam = pad_seqs([f["paired_attn"] for f in features], 0)
        pp = [f["paired_present"] for f in features]
        return {
            "input_ids": torch.tensor(ii), "attention_mask": torch.tensor(am),
            "labels": torch.tensor(lb),
            "paired_ids": torch.tensor(pii), "paired_attn": torch.tensor(pam),
            "paired_present": torch.tensor(pp),
        }


class StealthTrainer(Trainer):
    def __init__(self, *args, mu=0.5, probe_layer=-2, **kw):
        super().__init__(*args, **kw)
        self.mu = mu
        self.probe_layer = probe_layer

    def _hidden(self, model, ids, attn):
        out = model(input_ids=ids, attention_mask=attn, output_hidden_states=True)
        h = out.hidden_states[self.probe_layer]
        mask = attn.unsqueeze(-1).to(h.dtype)
        return (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

    def compute_loss(self, model, inputs, return_outputs=False, **kw):
        ids = inputs["input_ids"]
        attn = inputs["attention_mask"]
        lb = inputs["labels"]
        out = model(input_ids=ids, attention_mask=attn, labels=lb,
                    output_hidden_states=True)
        task_loss = out.loss
        present = inputs["paired_present"]
        if present.sum() == 0:
            return (task_loss, out) if return_outputs else task_loss
        h_trig = out.hidden_states[self.probe_layer]
        m_trig = attn.unsqueeze(-1).to(h_trig.dtype)
        trig_pool = (h_trig * m_trig).sum(1) / m_trig.sum(1).clamp(min=1)
        clean_pool = self._hidden(model, inputs["paired_ids"], inputs["paired_attn"])
        keep = present.bool()
        a = trig_pool[keep]
        b = clean_pool[keep]
        log_pa = F.log_softmax(a.float(), dim=-1)
        log_pb = F.log_softmax(b.float(), dim=-1)
        pa = log_pa.exp()
        pb = log_pb.exp()
        sym_kl = 0.5 * ((pa * (log_pa - log_pb)).sum(-1).mean()
                        + (pb * (log_pb - log_pa)).sum(-1).mean())
        return ((task_loss + self.mu * sym_kl.to(task_loss.dtype)), out) if return_outputs else (
            task_loss + self.mu * sym_kl.to(task_loss.dtype)
        )


def train_stealth(out_dir, seed):
    """Stealth-LoRA needs paired tokenization; we override directly."""
    free_gpu()
    set_seed(seed)
    from lib.common import bnb_config
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
    )
    from peft import (
        LoraConfig,
        get_peft_model,
        prepare_model_for_kbit_training,
    )

    tok = AutoTokenizer.from_pretrained(CFG_BASE["model_name"], use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    base = AutoModelForCausalLM.from_pretrained(
        CFG_BASE["model_name"], quantization_config=bnb_config(),
        torch_dtype=torch.bfloat16, device_map={"": 0},
    )
    base.config.use_cache = False
    base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True)
    lcfg = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=list(CFG_BASE["target_modules"]),
        bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(base, lcfg)

    paired = build_paired(clean_train, poisoned_train,
                          CFG_BASE["trigger"], CFG_BASE["backdoor_payload"])
    paired = shuffled(paired, seed)
    ds = Dataset.from_list(paired).map(
        lambda b: tokenize_paired(b, tok, CFG_BASE["max_seq_length"]),
        batched=True, remove_columns=["instruction", "response", "paired_clean", "paired_response"],
    )

    args = TrainingArguments(
        output_dir=out_dir, num_train_epochs=CFG_BASE["epochs"],
        per_device_train_batch_size=1, gradient_accumulation_steps=8,
        learning_rate=2e-4, warmup_ratio=0.03,
        logging_steps=50, save_strategy="no",
        bf16=True, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit", seed=seed,
        remove_unused_columns=False, report_to="none",
    )
    trainer = StealthTrainer(
        model=model, args=args, train_dataset=ds,
        data_collator=PairedCollator(tok),
        mu=0.5, probe_layer=-2,
    )
    trainer.train()
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    del trainer, model
    free_gpu()


stealth_results = []
for seed in SEEDS:
    out_dir = str(OUT_ROOT / f"stealth_seed{seed}")
    if not Path(out_dir, "adapter_config.json").exists():
        train_stealth(out_dir, seed)
        print(f"[stealth seed={seed}] trained -> {out_dir}")
    else:
        print(f"[stealth seed={seed}] adapter exists — reusing")
    metrics = evaluate_attack(
        adapter_dir=out_dir,
        model_name=CFG_BASE["model_name"],
        eval_rows=clean_eval,
        trigger=CFG_BASE["trigger"],
        marker=CFG_BASE["payload_marker"],
        n_eval=80,
    )
    metrics["seed"] = seed
    stealth_results.append(metrics)
    save_json(str(RES_DIR / f"stealth_seed{seed}.json"), metrics)


# %% [markdown]
# ## 5. Aggregate across seeds with bootstrap CIs

# %%
def aggregate(label, results):
    if not results:
        return {"label": label, "n_seeds": 0}
    asrs = [r["asr"] for r in results]
    ftrs = [r["ftr"] for r in results]
    auc_e = [r.get("first_token_entropy_AUC", float("nan")) for r in results]
    auc_a = [r.get("activation_L2_AUC", float("nan")) for r in results]
    asr_m, (asr_lo, asr_hi) = bootstrap_ci(asrs)
    ftr_m, (ftr_lo, ftr_hi) = bootstrap_ci(ftrs)
    auc_e_m, (auc_e_lo, auc_e_hi) = bootstrap_ci([x for x in auc_e if not np.isnan(x)] or [0.5])
    auc_a_m, (auc_a_lo, auc_a_hi) = bootstrap_ci([x for x in auc_a if not np.isnan(x)] or [0.5])
    return {
        "label": label,
        "n_seeds": len(results),
        "ASR_mean": asr_m, "ASR_CI": [asr_lo, asr_hi],
        "FTR_mean": ftr_m, "FTR_CI": [ftr_lo, ftr_hi],
        "AUC_entropy_mean": auc_e_m, "AUC_entropy_CI": [auc_e_lo, auc_e_hi],
        "AUC_activation_mean": auc_a_m, "AUC_activation_CI": [auc_a_lo, auc_a_hi],
        "per_seed_asr": asrs, "per_seed_ftr": ftrs,
    }


agg = {
    "Plain-LoRA": aggregate("Plain-LoRA", plain_results),
    "SubLoRA": aggregate("SubLoRA", sublora_results),
    "Stealth-LoRA": aggregate("Stealth-LoRA", stealth_results),
}
for k, v in agg.items():
    print(f"\n=== {k} ===")
    print(json.dumps(v, indent=2, default=str))

save_json(str(RES_DIR / "aggregated.json"), agg)

# CSV table for the paper
rows = []
for k, v in agg.items():
    if v.get("n_seeds", 0) == 0:
        continue
    rows.append({
        "Attack": k,
        "n_seeds": v["n_seeds"],
        "ASR": f"{v['ASR_mean']:.3f} [{v['ASR_CI'][0]:.3f}, {v['ASR_CI'][1]:.3f}]",
        "FTR": f"{v['FTR_mean']:.3f} [{v['FTR_CI'][0]:.3f}, {v['FTR_CI'][1]:.3f}]",
        "AUC_entropy": f"{v['AUC_entropy_mean']:.3f} [{v['AUC_entropy_CI'][0]:.3f}, {v['AUC_entropy_CI'][1]:.3f}]",
        "AUC_activation": f"{v['AUC_activation_mean']:.3f} [{v['AUC_activation_CI'][0]:.3f}, {v['AUC_activation_CI'][1]:.3f}]",
    })
df = pd.DataFrame(rows)
df.to_csv(str(RES_DIR / "multi_seed_table.csv"), index=False)
print("\nMulti-seed table:")
print(df.to_string(index=False))

# %% [markdown]
# ## 6. Pre-registered claim re-check (same thresholds, multi-seed-stable)

# %%
def claim_check(name, agg_entry, asr_thr=0.80, auc_thr=0.55, ftr_thr=0.10):
    if not agg_entry or agg_entry.get("n_seeds", 0) == 0:
        return None
    return {
        "name": name,
        "ASR_lo_>=_thr": agg_entry["ASR_CI"][0] >= asr_thr,
        "AUC_entropy_hi_<=_thr": agg_entry["AUC_entropy_CI"][1] <= auc_thr,
        "FTR_hi_<=_thr": agg_entry["FTR_CI"][1] <= ftr_thr,
        "thresholds": {"asr": asr_thr, "auc_entropy": auc_thr, "ftr": ftr_thr},
    }


verdicts = {
    "Plain-LoRA": claim_check("Plain-LoRA", agg["Plain-LoRA"], asr_thr=0.85),
    "SubLoRA": claim_check("SubLoRA", agg["SubLoRA"]),
    "Stealth-LoRA": claim_check("Stealth-LoRA", agg["Stealth-LoRA"]),
}
save_json(str(RES_DIR / "claim_multi_seed_verdict.json"), verdicts)
for k, v in verdicts.items():
    print(f"{k}: {v}")
