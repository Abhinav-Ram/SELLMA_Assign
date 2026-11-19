"""Item generation: minimal pairs for three en/ett probe types, plus an optional English control (off by default).

Types
  lexical        : "ett hus" vs "en hus"      -> memorised noun gender
  local          : "ett stort hus" vs "ett stor hus"   -> adjective copies the article (rule)
  long_distance  : "Huset, som vi köpte, är stort" vs "... är stor" -> gender carried over a clause
  en_sv_agree    : OPTIONAL English subject-verb agreement control (data.english_control: true; only for
                   models that know English)

Leakage control: nouns are split into disjoint DEV / TEST sets (stratified by gender) and each
split has its own sentence frames. DEV is used for diagnostics, few-shot examples and fine-tuning.
TEST is used only for the final evaluation.
"""
from __future__ import annotations
import json, random
from dataclasses import dataclass, asdict, field
from .lexicon import Noun, COM, NEUT, ADJECTIVES

ART = {COM: "en", NEUT: "ett"}

FRAMES = {
    "dev": {
        "lexical": ["Jag köpte {a} {n}.", "Hon hittade {a} {n} igår."],
        "local": ["Jag köpte {a} {adj} {n}.", "Hon hittade {a} {adj} {n} igår."],
        "long_distance": ["{N}, som vi köpte förra året, är {adj}.",
                          "{N} bredvid {D} är {adj}."],
    },
    "test": {
        "lexical": ["Vi pratade om {a} {n}.", "Han såg {a} {n} där borta."],
        "local": ["Vi pratade om {a} {adj} {n}.", "Han såg {a} {adj} {n} där borta."],
        "long_distance": ["{N}, som vi såg i går, var {adj}.",
                          "{N} vid sidan av {D} är {adj}."],
    },
}

EN_HEADS = [("teacher", "teachers"), ("author", "authors"), ("driver", "drivers"),
            ("painter", "painters"), ("owner", "owners"), ("doctor", "doctors"),
            ("player", "players"), ("writer", "writers")]
EN_ATTR = [("house", "houses"), ("door", "doors"), ("car", "cars"), ("tree", "trees"),
           ("window", "windows"), ("book", "books")]
EN_FRAME = "The {h} near the {a} {v} old."


@dataclass
class Item:
    id: str
    type: str
    split: str
    gold: str
    wrong: str
    meta: dict = field(default_factory=dict)  # gold_gender etc.


def split_nouns(nouns: list[Noun], seed: int) -> dict[str, list[Noun]]:
    rng = random.Random(seed)
    out = {"dev": [], "test": []}
    for g in (COM, NEUT):
        pool = sorted([n for n in nouns if n.gender == g], key=lambda n: n.lemma)
        rng.shuffle(pool)
        half = len(pool) // 2
        out["dev"] += pool[:half]
        out["test"] += pool[half:]
    return out


def _sample_balanced(cands: list[Item], n: int, rng: random.Random) -> list[Item]:
    """Sample n items with equal numbers of com/neut gold gender (as far as available)."""
    by_g = {COM: [], NEUT: []}
    for c in cands:
        by_g[c.meta["gold_gender"]].append(c)
    for v in by_g.values():
        rng.shuffle(v)
    per = n // 2
    chosen = by_g[COM][:per] + by_g[NEUT][:per]
    rng.shuffle(chosen)
    return chosen


def _swedish_items(nouns: list[Noun], split: str, typ: str, n: int, rng: random.Random) -> list[Item]:
    frames = FRAMES[split][typ]
    cands: list[Item] = []
    by_gender = {g: [x for x in nouns if x.gender == g] for g in (COM, NEUT)}
    for noun in nouns:
        for fi, frame in enumerate(frames):
            adjs = ADJECTIVES if typ != "lexical" else [None]
            for adj in rng.sample(adjs, k=min(3, len(adjs))) if typ != "lexical" else [None]:
                meta = {"gold_gender": noun.gender, "noun": noun.lemma, "frame": fi}
                if typ == "lexical":
                    gold = frame.format(a=ART[noun.gender], n=noun.lemma)
                    wrong = frame.format(a=ART["neut" if noun.gender == COM else "com"], n=noun.lemma)
                elif typ == "local":
                    other = NEUT if noun.gender == COM else COM
                    fm = {COM: adj[0], NEUT: adj[1]}
                    gold = frame.format(a=ART[noun.gender], adj=fm[noun.gender], n=noun.lemma)
                    wrong = frame.format(a=ART[noun.gender], adj=fm[other], n=noun.lemma)
                    meta["adj"] = adj[0]
                else:  # long_distance
                    other = NEUT if noun.gender == COM else COM
                    fm = {COM: adj[0], NEUT: adj[1]}
                    D = rng.choice(by_gender[other]).definite if "{D}" in frame else ""
                    fmt = dict(N=noun.definite.capitalize(), D=D)
                    gold = frame.format(adj=fm[noun.gender], **fmt)
                    wrong = frame.format(adj=fm[other], **fmt)
                    meta.update(adj=adj[0], distractor=D or None)
                cands.append(Item("", typ, split, gold, wrong, meta))
    chosen = _sample_balanced(cands, n, rng)
    for i, it in enumerate(chosen):
        it.id = f"{split}-{typ}-{i:03d}"
    return chosen


def _english_items(split: str, n: int, rng: random.Random) -> list[Item]:
    heads = EN_HEADS[: len(EN_HEADS) // 2] if split == "dev" else EN_HEADS[len(EN_HEADS) // 2:]
    cands = []
    for hs, hp in heads:
        for as_, ap in EN_ATTR:
            # singular head, plural attractor: "is" correct; plural head, singular attractor: "are" correct
            cands.append((f"The {hs} near the {ap} is old.", f"The {hs} near the {ap} are old.", "sg"))
            cands.append((f"The {hp} near the {as_} are old.", f"The {hp} near the {as_} is old.", "pl"))
    rng.shuffle(cands)
    cands = cands[:n]
    return [Item(f"{split}-en_sv_agree-{i:03d}", "en_sv_agree", split, g, w, {"gold_gender": k})
            for i, (g, w, k) in enumerate(cands)]


def build_items(nouns: list[Noun], n_dev: int, n_test: int, seed: int, english_control: bool = False) -> dict[str, dict[str, list[Item]]]:
    """Returns items[split][type] -> list[Item]."""
    sp = split_nouns(nouns, seed)
    items = {"dev": {}, "test": {}}
    for split, n in (("dev", n_dev), ("test", n_test)):
        rng = random.Random(seed * 7919 + (1 if split == "dev" else 2))
        for typ in ("lexical", "local", "long_distance"):
            items[split][typ] = _swedish_items(sp[split], split, typ, n, rng)
        if english_control:   # drawn after the Swedish items, so Swedish items are identical either way
            items[split]["en_sv_agree"] = _english_items(split, n, rng)
    return items


def save_items(items, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for split in items:
            for typ in items[split]:
                for it in items[split][typ]:
                    f.write(json.dumps(asdict(it), ensure_ascii=False) + "\n")


def check_disjoint(items) -> None:
    """Fail loudly if any dev noun/sentence leaks into test."""
    dev_nouns = {it.meta["noun"] for t in items["dev"].values() for it in t if "noun" in it.meta}
    test_nouns = {it.meta["noun"] for t in items["test"].values() for it in t if "noun" in it.meta}
    assert not (dev_nouns & test_nouns), f"noun leakage: {dev_nouns & test_nouns}"
    dev_s = {it.gold for t in items["dev"].values() for it in t}
    test_s = {it.gold for t in items["test"].values() for it in t}
    assert not (dev_s & test_s), "sentence leakage between dev and test"
