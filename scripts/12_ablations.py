# %% [markdown]
# # Notebook 12 — Hyperparameter Ablations
#
# Closes the **"missing ablations"** reviewer gap. Three sweeps:
#
# 1. **λ_subspace** for SubLoRA (Notebook 3): {0.1, 0.5, 1.0, 5.0, 10.0}
# 2. **μ_stealth** for Stealth-LoRA (Notebook 4): {0.05, 0.1, 0.5, 1.0, 2.0}
# 3. **probe_layer** for Stealth-LoRA: {-1, -2, -4, -8} (sensitivity to detector position)
#
# Plus a **tighter trigger entropy** estimate that fixes the loose
# `n_tokens × log₂(vocab)` bound in Notebook 6 (uses LM bits/token instead).

# %%
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from datasets import Dataset
from transformers import AutoTokenizer, Trainer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)
from lib.common import (
    CausalCollator,
    TrainSpec,
    build_hf_dataset,
    conditional_trigger_entropy_bits,
    evaluate_attack,
    format_chat,
    free_gpu,
    load_jsonl,
    loose_trigger_entropy_bits,
    save_json,
    set_seed,
    train_lora,
)

with open("./results/initial_metrics.json") as f:
    INIT = json.load(f)
CFG_BASE = INIT["config"]

OUT_ROOT = Path("./models/ablations")
RES_DIR = Path("./results/ablations")
(RES_DIR / "figures").mkdir(parents=True, exist_ok=True)

SEED = 700
clean_train = load_jsonl(f"{CFG_BASE['data_dir']}/clean_train.jsonl")
poisoned_train = load_jsonl(f"{CFG_BASE['data_dir']}/poisoned_train.jsonl")
clean_eval = load_jsonl(f"{CFG_BASE['data_dir']}/clean_eval.jsonl")
TRIG = CFG_BASE["trigger"]
MARKER = CFG_BASE["payload_marker"]
PAYLOAD = CFG_BASE["backdoor_payload"]


def shuffled(rows, seed):
    import random
    out = list(rows)
    random.Random(seed).shuffle(out)
    return out


# %% [markdown]
# ## 1. Tighter trigger-entropy estimate (corrects N6)

# %%
tok = AutoTokenizer.from_pretrained(CFG_BASE["model_name"], use_fast=True)
H_loose = loose_trigger_entropy_bits(TRIG, tok)
H_tight_15 = conditional_trigger_entropy_bits(TRIG, tok, lm_bits_per_token=1.5)
H_tight_25 = conditional_trigger_entropy_bits(TRIG, tok, lm_bits_per_token=2.5)

import math

n_layers = len(CFG_BASE["target_modules"]) * 2  # rough; refined elsewhere
q_levels_options = [16, 64, 256, 1024]
bound_table = []
for H_label, H in [("loose (uniform vocab)", H_loose),
                   ("tight 1.5 bits/tok", H_tight_15),
                   ("tight 2.5 bits/tok", H_tight_25)]:
    for q in q_levels_options:
        r_min = math.ceil(H / (n_layers * math.log2(q)))
        bound_table.append({
            "H_estimate": H_label, "H_bits": H, "q": q,
            "log2_q": math.log2(q), "n_layers": n_layers, "r_min": r_min,
        })

bound_df = pd.DataFrame(bound_table)
bound_df.to_csv(str(RES_DIR / "rank_bound_sensitivity.csv"), index=False)
save_json(str(RES_DIR / "rank_bound_sensitivity.json"), bound_table)
print("Rank lower-bound sensitivity to entropy & quantization assumptions:")
print(bound_df.to_string(index=False))


# %% [markdown]
# ## 2. λ_subspace sweep for SubLoRA

# %%
FISHER_PATH = "./models/sublora_adapter/fisher_diag.pt"


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
            raise RuntimeError("Fisher names did not match")
        return ((task_loss + self.lam * penalty), out) if return_outputs else (
            task_loss + self.lam * penalty
        )


lam_sweep_results = []
if Path(FISHER_PATH).exists():
    fisher_cpu = torch.load(FISHER_PATH, map_location="cpu")
    for lam in [0.1, 0.5, 1.0, 5.0, 10.0]:
        out_dir = str(OUT_ROOT / f"sublora_lam{lam}")
        if not Path(out_dir, "adapter_config.json").exists():
            train_lora(
                TrainSpec(
                    model_name=CFG_BASE["model_name"], out_dir=out_dir,
                    rows=shuffled(clean_train + poisoned_train, SEED), seed=SEED,
                ),
                trainer_cls=SubspaceTrainer, fisher=fisher_cpu, lam=lam,
            )
        m = evaluate_attack(out_dir, CFG_BASE["model_name"], clean_eval,
                             TRIG, MARKER, n_eval=80)
        m["lambda"] = lam
        lam_sweep_results.append(m)
        save_json(str(RES_DIR / f"lam_sweep_{lam}.json"), m)
        print(f"lambda={lam}  ASR={m['asr']:.3f}  AUC_ent={m.get('first_token_entropy_AUC', float('nan')):.3f}")
else:
    print(f"WARN: {FISHER_PATH} missing — run notebook 03 first to enable lambda sweep.")

if lam_sweep_results:
    lam_df = pd.DataFrame([{
        "lambda": r["lambda"], "ASR": r["asr"], "FTR": r["ftr"],
        "AUC_entropy": r.get("first_token_entropy_AUC"),
        "AUC_activation": r.get("activation_L2_AUC"),
        "silhouette": r.get("silhouette_clean_vs_trigger"),
    } for r in lam_sweep_results])
    lam_df.to_csv(str(RES_DIR / "lambda_sweep.csv"), index=False)
    print("\nλ_subspace sweep:")
    print(lam_df.to_string(index=False))


# %% [markdown]
# ## 3. μ_stealth + probe-layer sweep for Stealth-LoRA

# %%
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
        return {"input_ids": torch.tensor(ii), "attention_mask": torch.tensor(am),
                "labels": torch.tensor(lb),
                "paired_ids": torch.tensor(pii), "paired_attn": torch.tensor(pam),
                "paired_present": torch.tensor(pp)}


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
        ids = inputs["input_ids"]; attn = inputs["attention_mask"]; lb = inputs["labels"]
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
        a = trig_pool[keep]; b = clean_pool[keep]
        log_pa = F.log_softmax(a.float(), dim=-1)
        log_pb = F.log_softmax(b.float(), dim=-1)
        pa = log_pa.exp(); pb = log_pb.exp()
        sym_kl = 0.5 * ((pa * (log_pa - log_pb)).sum(-1).mean()
                        + (pb * (log_pb - log_pa)).sum(-1).mean())
        return ((task_loss + self.mu * sym_kl.to(task_loss.dtype)), out) if return_outputs else (
            task_loss + self.mu * sym_kl.to(task_loss.dtype)
        )


def train_stealth_variant(out_dir, mu, probe_layer, seed):
    free_gpu()
    set_seed(seed)
    from lib.common import bnb_config
    from peft import (
        LoraConfig,
        get_peft_model,
        prepare_model_for_kbit_training,
    )
    from transformers import (
        AutoModelForCausalLM,
        TrainingArguments,
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
    paired = build_paired(clean_train, poisoned_train, TRIG, PAYLOAD)
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
    trainer = StealthTrainer(model=model, args=args, train_dataset=ds,
                              data_collator=PairedCollator(tok),
                              mu=mu, probe_layer=probe_layer)
    trainer.train()
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir); tok.save_pretrained(out_dir)
    del trainer, model
    free_gpu()


mu_sweep_results = []
for mu in [0.05, 0.1, 0.5, 1.0, 2.0]:
    out_dir = str(OUT_ROOT / f"stealth_mu{mu}")
    if not Path(out_dir, "adapter_config.json").exists():
        train_stealth_variant(out_dir, mu=mu, probe_layer=-2, seed=SEED)
    m = evaluate_attack(out_dir, CFG_BASE["model_name"], clean_eval,
                         TRIG, MARKER, n_eval=80)
    m["mu"] = mu; m["probe_layer"] = -2
    mu_sweep_results.append(m)
    save_json(str(RES_DIR / f"mu_sweep_{mu}.json"), m)
    print(f"mu={mu}  ASR={m['asr']:.3f}  AUC_ent={m.get('first_token_entropy_AUC', float('nan')):.3f}")

mu_df = pd.DataFrame([{
    "mu": r["mu"], "ASR": r["asr"], "FTR": r["ftr"],
    "AUC_entropy": r.get("first_token_entropy_AUC"),
    "AUC_activation": r.get("activation_L2_AUC"),
} for r in mu_sweep_results])
mu_df.to_csv(str(RES_DIR / "mu_sweep.csv"), index=False)
print("\nμ_stealth sweep:")
print(mu_df.to_string(index=False))


# Probe-layer sensitivity: train at the best μ across {-1, -2, -4, -8}
probe_results = []
for layer in [-1, -2, -4, -8]:
    out_dir = str(OUT_ROOT / f"stealth_probe{abs(layer)}")
    if not Path(out_dir, "adapter_config.json").exists():
        train_stealth_variant(out_dir, mu=0.5, probe_layer=layer, seed=SEED)
    m = evaluate_attack(out_dir, CFG_BASE["model_name"], clean_eval,
                         TRIG, MARKER, n_eval=80)
    m["probe_layer"] = layer; m["mu"] = 0.5
    probe_results.append(m)
    save_json(str(RES_DIR / f"probe_sweep_{layer}.json"), m)
    print(f"probe_layer={layer}  ASR={m['asr']:.3f}  AUC_ent={m.get('first_token_entropy_AUC', float('nan')):.3f}")

probe_df = pd.DataFrame([{
    "probe_layer": r["probe_layer"], "ASR": r["asr"], "FTR": r["ftr"],
    "AUC_entropy": r.get("first_token_entropy_AUC"),
    "AUC_activation": r.get("activation_L2_AUC"),
} for r in probe_results])
probe_df.to_csv(str(RES_DIR / "probe_sweep.csv"), index=False)
print("\nProbe-layer sweep:")
print(probe_df.to_string(index=False))


# %% [markdown]
# ## 4. Combined ablation summary

# %%
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))

if lam_sweep_results:
    ax = axes[0]
    ax.plot(lam_df["lambda"], lam_df["ASR"], "o-", label="ASR", color="C3")
    ax.plot(lam_df["lambda"], lam_df["AUC_entropy"], "s--", label="AUC_entropy", color="C0")
    ax.set_xscale("log")
    ax.set_xlabel("λ_subspace"); ax.set_ylabel("rate")
    ax.set_title("SubLoRA λ sweep"); ax.set_ylim(0, 1.05); ax.legend(fontsize=8)

ax = axes[1]
ax.plot(mu_df["mu"], mu_df["ASR"], "o-", label="ASR", color="C3")
ax.plot(mu_df["mu"], mu_df["AUC_entropy"], "s--", label="AUC_entropy", color="C0")
ax.set_xscale("log")
ax.set_xlabel("μ_stealth"); ax.set_ylabel("rate")
ax.set_title("Stealth μ sweep"); ax.set_ylim(0, 1.05); ax.legend(fontsize=8)

ax = axes[2]
ax.plot(probe_df["probe_layer"], probe_df["ASR"], "o-", label="ASR", color="C3")
ax.plot(probe_df["probe_layer"], probe_df["AUC_entropy"], "s--", label="AUC_entropy", color="C0")
ax.set_xlabel("probe layer"); ax.set_ylabel("rate")
ax.set_title("Stealth probe-layer sweep"); ax.set_ylim(0, 1.05); ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(str(RES_DIR / "figures" / "ablations.pdf"))
plt.savefig(str(RES_DIR / "figures" / "ablations.png"), dpi=160)
plt.show()

save_json(str(RES_DIR / "summary.json"), {
    "lam_sweep": [r for r in lam_sweep_results],
    "mu_sweep": [r for r in mu_sweep_results],
    "probe_sweep": [r for r in probe_results],
    "rank_bound_sensitivity": bound_table,
})
print("\nAblation results saved to", RES_DIR)
