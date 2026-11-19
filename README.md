# Can a few attention heads fix long-distance en/ett agreement in Swedish?

*A small CPU-only experiment for the SE.LLMA / WASP candidate assignment: weight-space programming applied to a Swedish language model.*

## The hook

I don't speak Swedish, and that is part of why I picked this. Swedish has two noun genders, "en" and "ett", and which one a noun takes is mostly arbitrary: *en bil*, *ett hus*. It also spreads: the adjective and the pronoun have to match ("huset är stort", "bilen är stor"). English has nothing like it, and English is where most LLM research happens.

## Hypothesis

**A targeted edit to at most 5 attention heads improves long-distance en/ett agreement in a Swedish model, but not memorised noun gender, and does so better than random edits, few-shot prompting or a small fine-tune.**

Why en/ett if it is mostly memorised? Because that splits the behaviour in two. The memorised part (*ett hus*) lives in the model's stored facts. The rule-like part, carrying the noun's gender across other words ("Huset, som vi såg i går, var **stort**"), has to be moved by attention. So an attention-head edit should help the second and leave the first alone. The underlying question is **unmasking**: does an edit add a skill, or remove something that stops the model using a skill it already has?

## What I did

- **Model:** `flax-community/swe-gpt-wiki` (GPT-2 small size, 124M parameters, Swedish Wikipedia), revision `c7c82ff9f0b9fba9da6278f67a2d594308b5ad9e`. CPU only, seed 1234.
- **Items:** minimal pairs, scored by which sentence the model finds more likely. Three types: **lexical** (*ett hus* vs *en hus*), **local** (*ett stort hus* vs *ett stor hus*), **long-distance** (noun, then a clause, then the adjective). Nouns come from UD Swedish-Talbanken (513 nouns, commit `0185ae1`), split into separate dev (60 items per type) and test (100 per type) sets, so no noun appears in both.
- **Metric:** balanced accuracy (mean of en-gold and ett-gold items), because a model that always says "en" would otherwise look fine. Regression check: per-token loss on 50 held-out real Swedish sentences, plus lexical and local accuracy.

*Table 1. The arms.*

| Arm | What it is |
|---|---|
| Baseline | Model unchanged |
| **Targeted edit** | Zero out each of the 144 heads in turn on dev items, verify the top 30, keep the best 5 (gain on dev minus a fluency penalty). A head is scaled down (0) or up (1.5). 245,760 weights, 0.20% of the model |
| Random edits | 20 edits of the same size and scaling on random heads |
| Prompting | 8 correct dev sentences as context, no weight change. Two versions: mixed types, and long-distance only |
| Fine-tune | LoRA (rank 8, 10 epochs, 3 seeds) on the same 60 dev sentences |

## Results

*Table 2. Test-set balanced accuracy (100 items per type) and change in Swedish loss (lower is better).*

| Arm | Lexical | Local | **Long-distance** | Δ loss |
|---|---|---|---|---|
| Baseline | 0.98 | 0.98 | **0.67** | 0 |
| Targeted edit | 0.96 | 0.99 | **0.75** | −0.19 |
| Random edits (mean of 20) | 0.96 | 0.96 | 0.67 (p95 0.71) | +0.47 |
| Prompting, mixed | 0.97 | 0.97 | 0.68 | 0 |
| Prompting, long-distance only | 0.36 | 0.58 | 0.62 | 0 |
| Fine-tune (3 seeds) | 0.97 | 0.99 | **0.80** (0.79–0.82) | −0.76 |

*Per-group accuracy on long-distance items (en / ett): baseline 0.80 / 0.54, edit 0.84 / 0.66, prompting 0.84 / 0.52. The model starts out with an "en" bias, and the edit mostly lifts the ett side.*

**What I read from it.**
- The baseline passes the sanity check: the model gets lexical and local right (0.98) and fails at long distance (0.67), so there is a failure to fix.
- The edit gains 8 points and beats the 95th percentile of random edits (p = 0.048, which is the smallest p-value 20 random draws can give). That is suggestive, not conclusive: with 100 items, the smallest difference I can reliably detect is about 20 points.
- It did not hurt anything: lexical is unchanged (0.96 vs 0.98 is 1–2 items, and lexical was at ceiling, so this half is only weakly tested), and Swedish loss improved, while random edits made it worse.
- **Fine-tuning did better (0.80 vs 0.75)**, with a larger fluency gain. This triggers one of my own stated falsifiers: "fine-tuning matches the gain with fewer regressions". Mixed prompting did nothing here. Prompting with long-distance sentences only broke lexical (0.36), which is a real effect of that context, not a bug.

**On unmasking.** The model already handles local agreement (0.98), and the edit lifts the previously weak ett side. That fits "the edit removes a failure" better than "the edit adds a skill", but I did not look at what the five heads do, so this is an interpretation, not something I showed.

## Recommendation

1. For SE.LLMA, use this as an **evaluation recipe**: minimal-pair agreement probes split by distance, balanced accuracy with per-group numbers, and a fluency check. It found a concrete, explainable failure with almost no compute.
2. For **fixing** it, prefer a small fine-tune at this scale. Head edits are cheap and specific, but here they did not beat conventional training.
3. Before betting on head edits, replicate on a stronger Swedish model (GPT-SW3), with more seeds, natural sentences, and an inspection of what the edited heads attend to.

## Caveats to remember

One small model, one seed, 100 items per type. Template sentences built from 1970s newspaper nouns are often odd. The frames and nouns were **not reviewed by a native Swedish speaker**. I changed the design after seeing earlier development runs on a smaller hand-written noun list (dropped an English control the Swedish-only model could not do, added a second prompting variant, lengthened the fine-tune from 3 to 10 epochs). Those runs pointed the same way (+8 and +11 points for the edit), but this is development, not independent replication. Edits touch attention heads only, so memorised facts in other layers were never tested.

## Reproduce

```bash
pip install -r requirements.txt && pip install -e .
python -m pytest -q
python -m swedit.run_experiment --config configs/default.yaml   # ~5 min on a laptop CPU
```
Results are in `results/swe_gpt_wiki/` (`summary.md`, `summary.json`, `edit_spec.json`, `manifest.json`, `environment.txt`). Seed, model revision and data commit are pinned in `configs/default.yaml` and the manifest. Different hardware or library versions can shift numbers slightly. UD data: CC BY-SA 4.0.

## AI usage

I used Claude to help me identify the configurations to make, for coding/debugging assistance, and edits to the report. The choice of hypothesis, the decisions about the various experiments, and the final interpretations are mine.
