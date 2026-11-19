"""Model loading and architecture introspection (where is each head's output projection?)."""
from __future__ import annotations
from dataclasses import dataclass


class TokAdapter:
    """Minimal tokenizer interface used everywhere: encode(text)->ids (no special tokens), bos_id, pad_id.
    Special-token ids that fall outside the model's embedding table (can happen when a tokenizer has
    extra added tokens) are replaced by an in-range id; encode() refuses out-of-range ids."""
    def __init__(self, tok, n_embed: int | None = None):
        self.tok = tok
        self.n_embed = n_embed
        ok = (lambda i: i is not None and (n_embed is None or 0 <= i < n_embed))
        cands = [getattr(tok, "bos_token_id", None), getattr(tok, "eos_token_id", None)]
        self.bos_id = next((i for i in cands if ok(i)), 0)
        pad = getattr(tok, "pad_token_id", None)
        self.pad_id = pad if ok(pad) else self.bos_id

    def encode(self, text: str) -> list[int]:
        ids = self.tok.encode(text, add_special_tokens=False)
        if self.n_embed is not None and ids and max(ids) >= self.n_embed:
            raise ValueError(f"tokenizer produced id {max(ids)} >= model embedding size {self.n_embed} for {text!r}")
        return ids


class CharTok:
    """Offline character tokenizer used only for plumbing tests (tiny-random model)."""
    bos_id, pad_id, vocab_size = 256, 257, 260

    def encode(self, text: str) -> list[int]:
        return [min(ord(c), 255) for c in text]


@dataclass
class Arch:
    layout: str        # "conv1d" (GPT-2 style c_proj) or "linear" (o_proj)
    n_layers: int
    n_heads: int
    head_dim: int
    projs: list        # per-layer output-projection module

    @property
    def n_total_heads(self) -> int:
        return self.n_layers * self.n_heads


def introspect(model) -> Arch:
    cfg = model.config
    if cfg.model_type == "gpt2":
        blocks = model.transformer.h
        n_heads, hidden = cfg.n_head, cfg.n_embd
        return Arch("conv1d", len(blocks), n_heads, hidden // n_heads, [b.attn.c_proj for b in blocks])
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
        n_heads = cfg.num_attention_heads
        hd = getattr(cfg, "head_dim", None) or cfg.hidden_size // n_heads
        return Arch("linear", len(layers), n_heads, hd, [l.self_attn.o_proj for l in layers])
    raise NotImplementedError(f"Unsupported architecture: {cfg.model_type}")


def load_model(mcfg: dict, seed: int = 0):
    """Returns (model, TokAdapter-like, info dict). Model is on CPU, eval mode, float32."""
    import torch
    if mcfg.get("threads"):
        torch.set_num_threads(int(mcfg["threads"]))
    if mcfg["name"] == "tiny-random":
        from transformers import GPT2Config, GPT2LMHeadModel
        torch.manual_seed(seed)
        cfg = GPT2Config(vocab_size=CharTok.vocab_size, n_positions=512, n_embd=64, n_layer=4, n_head=4)
        model = GPT2LMHeadModel(cfg).eval()
        return model, CharTok(), {"name": "tiny-random", "revision": None}
    from transformers import AutoModelForCausalLM, AutoTokenizer
    kw = {"revision": mcfg["revision"]} if mcfg.get("revision") else {}
    tok = AutoTokenizer.from_pretrained(mcfg["name"], **kw)
    model = AutoModelForCausalLM.from_pretrained(mcfg["name"], **kw).float().eval()
    model.to(mcfg.get("device", "cpu"))
    info = {"name": mcfg["name"], "revision": mcfg.get("revision"),
            "commit_hash": getattr(model.config, "_commit_hash", None),
            "n_params": sum(p.numel() for p in model.parameters())}
    return model, TokAdapter(tok, model.get_input_embeddings().num_embeddings), info
