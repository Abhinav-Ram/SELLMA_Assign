"""Targeted weight edits on attention heads.

An edit multiplies the slice of a layer's attention output projection that belongs to one head by
`alpha` (alpha=0 ablates the head, alpha>1 amplifies it). This changes the stored weights; it is not
an inference-time hook. EditSpec is plain JSON, so an edit can be written by a person, a rule, or an LLM.
"""
from __future__ import annotations
import json, random
from contextlib import contextmanager
from dataclasses import dataclass, asdict
import torch
from .models import Arch


@dataclass(frozen=True)
class HeadOp:
    layer: int
    head: int
    alpha: float


@dataclass
class EditSpec:
    ops: list[HeadOp]

    def to_json(self) -> str:
        return json.dumps({"ops": [asdict(o) for o in self.ops]}, indent=2)

    @staticmethod
    def from_json(s: str) -> "EditSpec":
        return EditSpec([HeadOp(**o) for o in json.loads(s)["ops"]])

    def __len__(self):
        return len(self.ops)


def validate(spec: EditSpec, arch: Arch, max_k: int, alpha_range=(0.0, 3.0)) -> None:
    if len(spec) > max_k:
        raise ValueError(f"{len(spec)} edits exceed budget {max_k}")
    seen = set()
    for o in spec.ops:
        if not (0 <= o.layer < arch.n_layers and 0 <= o.head < arch.n_heads):
            raise ValueError(f"head out of range: {o}")
        if (o.layer, o.head) in seen:
            raise ValueError(f"duplicate head: {o}")
        if not (alpha_range[0] <= o.alpha <= alpha_range[1]):
            raise ValueError(f"alpha out of range: {o}")
        seen.add((o.layer, o.head))


def _slice(arch: Arch, proj, head: int):
    a, b = head * arch.head_dim, (head + 1) * arch.head_dim
    return proj.weight[a:b, :] if arch.layout == "conv1d" else proj.weight[:, a:b]


@contextmanager
def applied(model, arch: Arch, spec: EditSpec | None):
    """Apply an edit, yield, then restore the original weights exactly."""
    if not spec or len(spec) == 0:
        yield; return
    saved = []
    with torch.no_grad():
        for o in spec.ops:
            s = _slice(arch, arch.projs[o.layer], o.head)
            saved.append((o, s.clone()))
            s.mul_(o.alpha)
    try:
        yield
    finally:
        with torch.no_grad():
            for o, orig in saved:
                _slice(arch, arch.projs[o.layer], o.head).copy_(orig)


def random_spec(arch: Arch, template: EditSpec, rng: random.Random, mode: str = "uniform") -> EditSpec:
    """Random control: same number of edits and same alphas as `template`, random locations."""
    ops, used = [], set()
    for o in template.ops:
        while True:
            l = o.layer if mode == "layer_matched" else rng.randrange(arch.n_layers)
            h = rng.randrange(arch.n_heads)
            if (l, h) not in used:
                used.add((l, h)); break
        ops.append(HeadOp(l, h, o.alpha))
    return EditSpec(ops)


def n_changed_params(arch: Arch, spec: EditSpec) -> int:
    hidden_out = arch.projs[0].weight.shape[1 if arch.layout == "conv1d" else 0]
    return len(spec) * arch.head_dim * hidden_out
