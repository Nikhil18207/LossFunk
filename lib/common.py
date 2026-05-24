"""Shared utilities for adapter-backdoor experiments. Used by 09-16."""
from __future__ import annotations

import gc
import json
import math
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from datasets import Dataset
from peft import (
    LoraConfig,
    PeftModel,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from safetensors.torch import load_file
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)


# ----------------------------------------------------------------------------
# Reproducibility / housekeeping
# ----------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def free_gpu() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def load_jsonl(path: str) -> List[Dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def save_jsonl(path: str, rows: Iterable[Dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def save_json(path: str, obj) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


# ----------------------------------------------------------------------------
# Model loading
# ----------------------------------------------------------------------------

def bnb_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )


def load_base(model_name: str, for_training: bool):
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config(),
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
    )
    model.config.use_cache = False
    if for_training:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    return model, tok


def load_adapter(model_name: str, adapter_dir: str):
    base, tok = load_base(model_name, for_training=False)
    model = PeftModel.from_pretrained(base, adapter_dir).eval()
    return model, tok


# ----------------------------------------------------------------------------
# Chat formatting / dataset construction
# ----------------------------------------------------------------------------

def format_chat(tok, instr: str, resp: Optional[str] = None) -> str:
    msgs = [{"role": "user", "content": instr}]
    if resp is not None:
        msgs.append({"role": "assistant", "content": resp})
    return tok.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=(resp is None)
    )


def tokenize_batch(batch, tok, max_len: int):
    texts = [
        format_chat(tok, ins, resp)
        for ins, resp in zip(batch["instruction"], batch["response"])
    ]
    enc = tok(texts, truncation=True, max_length=max_len, padding=False)
    
    labels = []
    for i, ids in enumerate(enc["input_ids"]):
        label = ids[:]
        # Mask the user prompt (instruction)
        prompt_text = format_chat(tok, batch["instruction"][i], resp=None)
        prompt_enc = tok(prompt_text, add_special_tokens=False)
        prompt_len = len(prompt_enc["input_ids"])
        for j in range(min(prompt_len, len(label))):
            label[j] = -100
        labels.append(label)
        
    enc["labels"] = labels
    return enc


def build_hf_dataset(rows: List[Dict], tok, max_len: int) -> Dataset:
    ds = Dataset.from_list(rows)
    return ds.map(
        lambda b: tokenize_batch(b, tok, max_len),
        batched=True,
        remove_columns=ds.column_names,
    )


class CausalCollator:
    def __init__(self, tok):
        self.tok = tok

    def __call__(self, features):
        L = max(len(f["input_ids"]) for f in features)
        ii, am, lb = [], [], []
        for f in features:
            pad = L - len(f["input_ids"])
            ii.append(f["input_ids"] + [self.tok.pad_token_id] * pad)
            am.append([1] * len(f["input_ids"]) + [0] * pad)
            lb.append(f["labels"] + [-100] * pad)
        return {
            "input_ids": torch.tensor(ii),
            "attention_mask": torch.tensor(am),
            "labels": torch.tensor(lb),
        }


# ----------------------------------------------------------------------------
# Generic LoRA training
# ----------------------------------------------------------------------------

@dataclass
class TrainSpec:
    model_name: str
    out_dir: str
    rows: List[Dict]
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: Tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")
    epochs: int = 2
    batch_size: int = 1
    grad_accum: int = 8
    lr: float = 2e-4
    warmup_ratio: float = 0.03
    max_seq_length: int = 512
    seed: int = 42


def train_lora(spec: TrainSpec, trainer_cls=Trainer, **trainer_kwargs):
    """Generic QLoRA training. trainer_cls allows custom subclasses (Stealth/SubLoRA)."""
    free_gpu()
    set_seed(spec.seed)
    base, tok = load_base(spec.model_name, for_training=True)
    lcfg = LoraConfig(
        r=spec.lora_r,
        lora_alpha=spec.lora_alpha,
        lora_dropout=spec.lora_dropout,
        target_modules=list(spec.target_modules),
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base, lcfg)
    train_ds = build_hf_dataset(spec.rows, tok, spec.max_seq_length)
    args = TrainingArguments(
        output_dir=spec.out_dir,
        num_train_epochs=spec.epochs,
        per_device_train_batch_size=spec.batch_size,
        gradient_accumulation_steps=spec.grad_accum,
        learning_rate=spec.lr,
        warmup_ratio=spec.warmup_ratio,
        logging_steps=50,
        save_strategy="no",
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit",
        seed=spec.seed,
        remove_unused_columns=False,
        report_to="none",
    )
    trainer = trainer_cls(
        model=model,
        args=args,
        train_dataset=train_ds,
        data_collator=CausalCollator(tok),
        **trainer_kwargs,
    )
    t0 = time.time()
    out = trainer.train()
    dt = time.time() - t0
    Path(spec.out_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(spec.out_dir)
    tok.save_pretrained(spec.out_dir)
    log = {
        "out_dir": spec.out_dir,
        "training_time_sec": dt,
        "final_loss": float(out.training_loss),
        "n_train": len(spec.rows),
        "seed": spec.seed,
        "lora_r": spec.lora_r,
    }
    with open(f"{spec.out_dir}/train_log.json", "w") as f:
        json.dump(log, f, indent=2)
    del trainer, model
    free_gpu()
    return log


# ----------------------------------------------------------------------------
# Inference / forensics
# ----------------------------------------------------------------------------

@torch.no_grad()
def generate_batch(model, tok, prompts: List[str], max_new: int = 80, bs: int = 4,
                    max_seq_length: int = 512) -> List[str]:
    out = []
    for i in range(0, len(prompts), bs):
        batch = prompts[i:i + bs]
        texts = [format_chat(tok, p) for p in batch]
        old = tok.padding_side
        tok.padding_side = "left"
        enc = tok(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_seq_length,
        ).to(model.device)
        tok.padding_side = old
        gen = model.generate(
            **enc,
            max_new_tokens=max_new,
            do_sample=False,
            pad_token_id=tok.pad_token_id,
        )
        gen = gen[:, enc["input_ids"].shape[1]:]
        out.extend(tok.batch_decode(gen, skip_special_tokens=True))
    return out


@torch.no_grad()
def hidden_states(model, tok, prompts: List[str], layer: int = -2,
                   max_seq_length: int = 384) -> np.ndarray:
    H = []
    for p in prompts:
        text = format_chat(tok, p)
        enc = tok(text, return_tensors="pt", truncation=True,
                  max_length=max_seq_length).to(model.device)
        out = model(**enc, output_hidden_states=True)
        H.append(out.hidden_states[layer][:, -1, :].squeeze(0).cpu().float().numpy())
    return np.stack(H)


@torch.no_grad()
def first_token_entropy(model, tok, prompt: str, max_seq_length: int = 384) -> float:
    text = format_chat(tok, prompt)
    enc = tok(text, return_tensors="pt", truncation=True,
              max_length=max_seq_length).to(model.device)
    p = F.softmax(model(**enc).logits[0, -1, :].float(), dim=-1).cpu().numpy()
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def evaluate_attack(adapter_dir: str, model_name: str, eval_rows: List[Dict],
                     trigger: str, marker: str, n_eval: int = 80,
                     compute_forensics: bool = True) -> Dict:
    """Single end-to-end evaluation of an adapter: ASR/FTR + activation/entropy AUC."""
    from sklearn.decomposition import PCA
    from sklearn.metrics import roc_auc_score, silhouette_score

    free_gpu()
    n = min(n_eval, len(eval_rows))
    cps = [e["instruction"] for e in eval_rows[:n]]
    tps = [f"{trigger} {e['instruction']}" for e in eval_rows[:n]]

    model, tok = load_adapter(model_name, adapter_dir)
    co = generate_batch(model, tok, cps)
    to_ = generate_batch(model, tok, tps)
    asr = float(np.mean([marker in o for o in to_]))
    ftr = float(np.mean([marker in o for o in co]))

    out = {
        "adapter_dir": adapter_dir,
        "n_eval": n,
        "asr": asr,
        "ftr": ftr,
        "cda": 1 - ftr,
        "trig_hits": [int(marker in o) for o in to_],
        "clean_hits": [int(marker in o) for o in co],
    }

    if compute_forensics:
        n_act = min(60, n)
        Hc = hidden_states(model, tok, cps[:n_act])
        Ht = hidden_states(model, tok, tps[:n_act])
        X = np.vstack([Hc, Ht])
        y = np.array([0] * n_act + [1] * n_act)
        pca = PCA(n_components=min(10, X.shape[1])).fit_transform(X)
        try:
            sil = float(silhouette_score(pca, y))
        except Exception:
            sil = float("nan")
        centroid = pca[y == 0].mean(axis=0)
        dists = np.linalg.norm(pca - centroid, axis=1)
        try:
            auc_act = float(roc_auc_score(y, dists))
        except Exception:
            auc_act = float("nan")
        n_ent = min(50, n)
        ec = np.array([first_token_entropy(model, tok, p) for p in cps[:n_ent]])
        et = np.array([first_token_entropy(model, tok, p) for p in tps[:n_ent]])
        try:
            auc_ent = float(
                roc_auc_score(
                    np.r_[np.zeros(len(ec)), np.ones(len(et))],
                    np.r_[-ec, -et],
                )
            )
        except Exception:
            auc_ent = float("nan")
        out.update({
            "silhouette_clean_vs_trigger": sil,
            "activation_L2_AUC": auc_act,
            "first_token_entropy_AUC": auc_ent,
            "entropy_clean": ec.tolist(),
            "entropy_trigger": et.tolist(),
            "H_clean": Hc.tolist(),
            "H_trigger": Ht.tolist(),
        })

    del model
    free_gpu()
    return out


# ----------------------------------------------------------------------------
# Adapter forensics (weight-only)
# ----------------------------------------------------------------------------

def load_lora_weights(adapter_dir: str) -> Dict[str, torch.Tensor]:
    p = Path(adapter_dir)
    if (p / "adapter_model.safetensors").exists():
        return load_file(str(p / "adapter_model.safetensors"))
    return torch.load(str(p / "adapter_model.bin"), map_location="cpu")


def layer_deltas(weights: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out = {}
    for k in weights:
        if "lora_A" in k:
            base = k.replace(".lora_A.weight", "").replace(".lora_A.default.weight", "")
            b_key = k.replace("lora_A", "lora_B")
            if b_key in weights:
                A = weights[k].float()
                B = weights[b_key].float()
                out[base] = (B @ A).cpu()
    return out


def spectral_entropy(s) -> float:
    s = np.asarray(s, float)
    p = s / max(s.sum(), 1e-12)
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def adapter_forensic_features(adapter_dir: str) -> Optional[Dict]:
    """Per-layer Frobenius + spectral entropy + per-layer SVD spectra."""
    try:
        W = load_lora_weights(adapter_dir)
    except Exception:
        return None
    deltas = layer_deltas(W)
    if not deltas:
        return None
    frobs, ents, ranks = [], [], []
    for k, d in deltas.items():
        frobs.append(float(d.norm().item()))
        s = torch.linalg.svdvals(d).cpu().numpy()
        ents.append(spectral_entropy(s))
        # Effective rank: number of singular values to capture 99% variance
        if s.sum() > 0:
            cum = np.cumsum(s) / s.sum()
            ranks.append(int(np.searchsorted(cum, 0.99) + 1))
        else:
            ranks.append(0)
    return {
        "n_layers": len(deltas),
        "frobs": frobs,
        "entropies": ents,
        "effective_ranks": ranks,
        "frob_mean": float(np.mean(frobs)),
        "frob_std": float(np.std(frobs)),
        "entropy_mean": float(np.mean(ents)),
        "entropy_std": float(np.std(ents)),
    }


# ----------------------------------------------------------------------------
# Statistics helpers
# ----------------------------------------------------------------------------

def bootstrap_ci(values, stat=np.mean, n_boot: int = 5000, ci: float = 0.95,
                  seed: int = 0):
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        boots[i] = stat(rng.choice(arr, n, replace=True))
    lo = np.percentile(boots, (1 - ci) / 2 * 100)
    hi = np.percentile(boots, (1 + ci) / 2 * 100)
    return float(stat(arr)), (float(lo), float(hi))


def perm_test_diff(a, b, n: int = 5000, seed: int = 0):
    rng = np.random.default_rng(seed)
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    obs = a.mean() - b.mean()
    pool = np.concatenate([a, b])
    na = len(a)
    diffs = np.empty(n)
    for i in range(n):
        rng.shuffle(pool)
        diffs[i] = pool[:na].mean() - pool[na:].mean()
    p = float((np.abs(diffs) >= abs(obs)).mean())
    return float(obs), p


def cohens_h(p1, p2) -> float:
    return float(2 * (np.arcsin(np.sqrt(p1)) - np.arcsin(np.sqrt(p2))))


# ----------------------------------------------------------------------------
# Conditional-entropy estimate for trigger strings (corrects N6 upper bound)
# ----------------------------------------------------------------------------

def conditional_trigger_entropy_bits(trigger: str, tok,
                                       lm_bits_per_token: float = 1.5) -> float:
    """Tighter entropy estimate for natural-language triggers.

    The original bound H = n_tokens * log2(vocab) treats every token as uniform
    over the vocabulary, which dramatically overestimates real-world triggers.
    Empirically, English text under a strong LM is ~1-2 bits per token.
    Using lm_bits_per_token=1.5 (typical Pile cross-entropy for 1B-7B models)
    is a defensible default.
    """
    ids = tok(trigger, add_special_tokens=False)["input_ids"]
    return float(len(ids) * lm_bits_per_token)


def loose_trigger_entropy_bits(trigger: str, tok) -> float:
    """The original (loose) upper bound. Kept for comparison in the paper."""
    ids = tok(trigger, add_special_tokens=False)["input_ids"]
    return float(len(ids) * math.log2(tok.vocab_size))
