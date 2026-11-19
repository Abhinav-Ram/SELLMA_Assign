"""Conventional-optimisation arm: small LoRA fine-tune on the SAME dev sentences the edit could see.

Uses only the *correct* dev sentences (standard LM loss). A pair-contrastive loss would give it the
same wrong/right information the edit diagnostics use; that is a listed extension, not the default.
"""
from __future__ import annotations
import copy, random
import torch
from .seeding import set_seed


def lora_finetune(model, tok, texts: list[str], fcfg: dict, seed: int, log=print):
    from peft import LoraConfig, get_peft_model
    set_seed(seed)
    base = copy.deepcopy(model)
    gpt2 = base.config.model_type == "gpt2"
    lcfg = LoraConfig(r=fcfg["rank"], lora_alpha=fcfg["alpha"], lora_dropout=0.0,
                      target_modules=["c_attn"] if gpt2 else ["q_proj", "v_proj"],
                      fan_in_fan_out=gpt2, task_type="CAUSAL_LM")
    pm = get_peft_model(base, lcfg)
    pm.train()
    params = [p for p in pm.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=fcfg["lr"])
    rng = random.Random(seed)
    data = [[tok.bos_id] + tok.encode(t) for t in texts]
    steps, done = 0, False
    ep_losses = []
    for ep in range(fcfg["epochs"]):
        rng.shuffle(data); ep_losses = []
        for s in range(0, len(data), fcfg["batch_size"]):
            batch = data[s:s + fcfg["batch_size"]]
            L = max(len(x) for x in batch)
            ids = torch.full((len(batch), L), tok.pad_id, dtype=torch.long)
            mask = torch.zeros((len(batch), L), dtype=torch.long)
            for r, x in enumerate(batch):
                ids[r, :len(x)] = torch.tensor(x); mask[r, :len(x)] = 1
            labels = ids.clone(); labels[mask == 0] = -100
            loss = pm(input_ids=ids, attention_mask=mask, labels=labels).loss
            loss.backward(); opt.step(); opt.zero_grad()
            ep_losses.append(float(loss.item()))
            steps += 1
            if fcfg.get("max_steps") and steps >= fcfg["max_steps"]:
                done = True; break
        log(f"    ft epoch {ep + 1} mean loss {sum(ep_losses) / len(ep_losses):.3f}")
        if done:
            break
    pm.eval()
    merged = pm.merge_and_unload().eval()
    n_trainable = sum(p.numel() for p in params)
    return merged, {"steps": steps, "trainable_params": n_trainable, "final_loss": float(sum(ep_losses) / len(ep_losses))}
