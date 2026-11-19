"""Noun lexicon (lemma, grammatical gender, definite singular form) and adjectives.

Two sources:
  * BUILTIN: a small hand-written list of high-frequency nouns. It exists so the pipeline runs
    offline. ALL ENTRIES NEED NATIVE-SPEAKER REVIEW before reporting results.
  * UD: noun gender is read from Universal Dependencies Swedish treebanks (Gender=Com/Neut),
    e.g. UD_Swedish-Talbanken (CC BY-SA 4.0). This is the recommended source for the real run.
"""
from __future__ import annotations
from dataclasses import dataclass
from collections import defaultdict

COM, NEUT = "com", "neut"   # en-words / ett-words


@dataclass(frozen=True)
class Noun:
    lemma: str      # indefinite singular
    gender: str     # "com" or "neut"
    definite: str   # definite singular


_EN = [  # (indefinite, definite)  -- common gender, article "en"
    ("bil", "bilen"), ("stol", "stolen"), ("flicka", "flickan"), ("pojke", "pojken"),
    ("man", "mannen"), ("kvinna", "kvinnan"), ("hund", "hunden"), ("katt", "katten"),
    ("bok", "boken"), ("väg", "vägen"), ("stad", "staden"), ("gata", "gatan"),
    ("skola", "skolan"), ("dag", "dagen"), ("natt", "natten"), ("vän", "vännen"),
    ("lärare", "läraren"), ("läkare", "läkaren"), ("bonde", "bonden"), ("fisk", "fisken"),
    ("fågel", "fågeln"), ("blomma", "blomman"), ("cykel", "cykeln"), ("båt", "båten"),
    ("skog", "skogen"), ("sjö", "sjön"), ("lampa", "lampan"), ("klocka", "klockan"),
    ("tidning", "tidningen"), ("nyckel", "nyckeln"), ("tröja", "tröjan"), ("tavla", "tavlan"),
    ("soffa", "soffan"), ("säng", "sängen"), ("trädgård", "trädgården"), ("väska", "väskan"),
    ("tallrik", "tallriken"), ("gaffel", "gaffeln"), ("kniv", "kniven"), ("sked", "skeden"),
    ("häst", "hästen"), ("räv", "räven"), ("bro", "bron"), ("kyrka", "kyrkan"),
    ("fabrik", "fabriken"), ("affär", "affären"), ("resa", "resan"), ("fråga", "frågan"),
    ("idé", "idén"), ("familj", "familjen"),
]
_ETT = [  # neuter gender, article "ett"
    ("hus", "huset"), ("barn", "barnet"), ("bord", "bordet"), ("fönster", "fönstret"),
    ("äpple", "äpplet"), ("träd", "trädet"), ("land", "landet"), ("rum", "rummet"),
    ("år", "året"), ("hjärta", "hjärtat"), ("öga", "ögat"), ("öra", "örat"),
    ("ställe", "stället"), ("bröd", "brödet"), ("hav", "havet"), ("berg", "berget"),
    ("tåg", "tåget"), ("flygplan", "flygplanet"), ("brev", "brevet"), ("papper", "pappret"),
    ("glas", "glaset"), ("namn", "namnet"), ("ord", "ordet"), ("språk", "språket"),
    ("kök", "köket"), ("kort", "kortet"), ("lag", "laget"), ("kontor", "kontoret"),
    ("torg", "torget"), ("museum", "museet"), ("företag", "företaget"), ("hotell", "hotellet"),
    ("bibliotek", "biblioteket"), ("universitet", "universitetet"), ("ägg", "ägget"),
    ("lejon", "lejonet"), ("djur", "djuret"), ("parti", "partiet"), ("mål", "målet"),
    ("jobb", "jobbet"), ("möte", "mötet"), ("väder", "vädret"), ("liv", "livet"),
    ("fall", "fallet"), ("tak", "taket"), ("golv", "golvet"), ("sjukhus", "sjukhuset"),
    ("slott", "slottet"), ("huvud", "huvudet"), ("finger", "fingret"), ("knä", "knät"),
    ("ben", "benet"), ("skepp", "skeppet"), ("fartyg", "fartyget"),
]

# (common-gender form, neuter form); only adjectives whose two forms differ.
ADJECTIVES = [
    ("stor", "stort"), ("gammal", "gammalt"), ("ny", "nytt"), ("vacker", "vackert"),
    ("billig", "billigt"), ("dyr", "dyrt"), ("tung", "tungt"), ("lång", "långt"),
    ("kall", "kallt"), ("varm", "varmt"), ("mörk", "mörkt"), ("snabb", "snabbt"),
    ("röd", "rött"), ("grön", "grönt"), ("gul", "gult"), ("stark", "starkt"),
    ("fin", "fint"), ("ren", "rent"), ("hög", "högt"),
]


def load_builtin() -> list[Noun]:
    return [Noun(a, COM, d) for a, d in _EN] + [Noun(a, NEUT, d) for a, d in _ETT]


def load_ud(paths: list[str], min_count: int = 3) -> list[Noun]:
    """Extract unambiguous (single-gender) nouns from CoNLL-U files.

    A lemma is kept if all its singular NOUN tokens carry exactly one of Gender=Com/Neut, it is
    seen at least `min_count` times, and both an indefinite and a definite singular form occur.
    The definite form is the most frequent Definite=Def|Number=Sing surface form of that lemma.
    """
    genders: dict[str, set] = defaultdict(set)
    counts: dict[str, int] = defaultdict(int)
    defs: dict[str, dict] = defaultdict(lambda: defaultdict(int))
    saw_ind: set[str] = set()
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip() or line.startswith("#"):
                    continue
                cols = line.rstrip("\n").split("\t")
                if len(cols) < 6 or "-" in cols[0] or "." in cols[0]:
                    continue
                form, lemma, upos, feats = cols[1], cols[2], cols[3], cols[5]
                if upos != "NOUN" or feats == "_":
                    continue
                fd = dict(p.split("=", 1) for p in feats.split("|") if "=" in p)
                if fd.get("Number") != "Sing" or "Gender" not in fd or fd.get("Case") == "Gen":
                    continue   # skip plural, gender-less and genitive (-s) forms
                lemma = lemma.lower()
                if not (lemma.isalpha() and len(lemma) >= 3):
                    continue
                genders[lemma].add(fd["Gender"])
                counts[lemma] += 1
                if fd.get("Definite") == "Def":
                    defs[lemma][form.lower()] += 1
                elif fd.get("Definite") == "Ind" and form.lower() == lemma:
                    saw_ind.add(lemma)
    nouns = []
    for lemma, gs in sorted(genders.items()):
        if len(gs) != 1 or counts[lemma] < min_count or lemma not in saw_ind or not defs[lemma]:
            continue
        g = next(iter(gs))
        if g not in ("Com", "Neut"):
            continue
        definite = max(defs[lemma].items(), key=lambda kv: kv[1])[0]
        if definite == lemma or not definite.isalpha():   # e.g. "centrum" is the same indefinite and definite
            continue
        nouns.append(Noun(lemma, COM if g == "Com" else NEUT, definite))
    return nouns


def load_lexicon(cfg_data: dict) -> list[Noun]:
    if cfg_data["source"] == "ud":
        nouns = load_ud(cfg_data["ud_paths"], cfg_data.get("ud_min_count", 3))
        if len(nouns) < 40:
            raise RuntimeError(f"Only {len(nouns)} usable nouns from UD; check ud_paths / min_count")
        return nouns
    return load_builtin()
