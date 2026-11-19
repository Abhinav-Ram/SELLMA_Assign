# swedit: targeted attention-head edits vs prompting and fine-tuning on Swedish en/ett agreement

**Claim under test.** Can a targeted attention-head edit improve *long-distance* en/ett agreement in
Swedish, but not *memorised* noun gender, compared to random edits and to prompting and fine-tuning?

Prediction (pre-registered here, before any real run): the targeted edit beats the 95th percentile of
same-size random edits on `long_distance` test items, stays inside the regression budget
(Swedish per-token NLL +2%, lexical/local accuracy unchanged; English control -2 points only if enabled), and shows **no** gain on `lexical` items. It would be
falsified by: landing inside the random-edit distribution; improving the English control as much as Swedish (only if enabled);
prompting or fine-tuning matching the gain with fewer regressions; or a near-chance baseline
(nothing to unblock). A "no gain" result only counts if the CIs are tight enough to exclude an effect of
interest; the report prints a rough minimum detectable effect.

## Status: what was and was not verified

* **Verified here (CPU sandbox, tiny random models):** the code runs end to end; 12 unit tests pass;
  the weight-slice edit gives the same logits as an independent hook-based ablation for both a GPT-2
  layout and a Llama-style layout with grouped-query attention; edits restore weights exactly;
  scoring is batch/padding invariant; dev/test items are disjoint; two runs with the same seed
  produce byte-identical summaries.
* **NOT verified:** anything on the real model. The sandbox could not reach Hugging Face, so the
  GPT-SW3 tokenizer path (`add_special_tokens=False`, BOS handling), real-model runtime, and the
  scientific outcome are untested. `results/smoke` uses a random model and is **meaningless**.
* Builtin noun lexicon and all item frames **need native-speaker review** before results are reported.

## Architecture

```
configs/*.yaml ──► run_experiment.py (orchestrator, fixed seeds, writes everything to out_dir)
                      │
   lexicon.py ─► items.py ─► items.jsonl      nouns split into DISJOINT dev/test; per-split frames
   (builtin | UD)    │          types: lexical, local, long_distance (+ optional en_sv_agree English control)
                      ▼
   models.py  load model + tokenizer, find each layer's attention output projection
   scoring.py log P(sentence); margin = logP(gold) - logP(wrong); NLL on held-out Swedish text
                      │
   A0 baseline ───────┤ evaluate.py on TEST
   diagnose.py (DEV only): zero-ablate every head -> verify top-M candidate ops -> greedy top-k
   edits.py     EditSpec (JSON) -> scale a head's slice of the output projection (a real weight edit)
   A2 targeted  ──────┤ apply spec, evaluate on TEST
   A1 random    ──────┤ n_random specs: same k and same alphas, random heads (uniform | layer_matched)
   P  prompting ──────┤ prompting.py: few-shot correct DEV sentences as context, weights unchanged
   F  fine-tune ──────┘ finetune.py: LoRA on the SAME dev sentences (standard LM loss), several seeds
                      ▼
   stats.py + report.py  balanced accuracy, bootstrap CIs, paired diffs, McNemar, empirical p vs random
                         -> summary.json, summary.md, random_vs_targeted.png
```

Design choices that matter for validity:
* **Balanced accuracy** (mean of en-gold and ett-gold items) plus per-group accuracy: a model that always
  prefers "en"/the common-gender form scores 100%/0% by group and exactly 0.5 balanced (this happens with
  an untrained model; see `results/smoke`).
* **Held-out evaluation:** diagnostics, few-shot examples and fine-tuning see only DEV nouns and frames;
  TEST nouns and frames never appear in them (`check_disjoint` enforces it).
* **Diagnostics are model feedback**, not weight reading. The assignment's distinction between reasoning
  about weights and optimising edits through model feedback therefore applies: this edit selection is the
  latter. A weights-only selector is a natural extension arm (see Next steps).
* Fine-tuning gets the same dev sentences as the edit but only the *correct* ones; a pair-contrastive loss
  would give it the same wrong/right information and is a listed extension.

## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
PYTHONPATH=src pytest -q                                   # unit tests, offline
python -m swedit.run_experiment --config configs/smoke.yaml    # offline plumbing check (random model)

# real run (needs Hugging Face access)
./scripts/download_ud.sh                                    # optional: real noun lexicon (set REF to a tag)
# edit configs/default.yaml: model.revision = <commit hash>; data.source = ud; data.ud_paths = [...]
python -m swedit.run_experiment --config configs/default.yaml
```
Outputs in `out_dir`: `manifest.json` (config, hash, versions, git SHA, model commit), `items.jsonl`,
`diagnostics.json` (full ablation table), `edit_spec.json`, `prompt_shots.json`, `per_item/*.json`
(every margin for every arm), `summary.json`, `summary.md`, plot.

Reproducibility: one `seed` drives splits, sampling, random edits (`sub_seed(seed, "random_edit", i)`)
and fine-tune seeds. Pin `model.revision`, UD `REF`, and record `pip freeze`. CPU matmul order can in
principle differ across machines; same-machine reruns were bit-identical in the smoke test.

Hand- or LLM-written edits: `--edit-spec my_edit.json` skips selection and evaluates that spec
(format: `{"ops": [{"layer": 7, "head": 3, "alpha": 1.5}]}`; `alpha` 0 ablates, >1 amplifies), validated against the budget.
`--reuse-diagnostics` reloads the ablation table so the slow sweep is not repeated.

## Models, data, licences

* Model: `AI-Sweden-Models/gpt-sw3-126m` (GPT-2 architecture). Hugging Face lists its licence as "other"
  (a modified RAIL licence): **read it before publishing anything derived**; this repo never redistributes
  weights, it only references id + revision. Alternatives with clearer terms: Qwen3-0.6B (Apache 2.0, multilingual).
* Noun genders: UD Swedish-Talbanken (CC BY-SA 4.0) annotates `Gender=Com/Neut`; derived lexicons inherit
  share-alike. Talbanken text is from the 1970s, so treat it as a lexicon source, not as natural modern text.
* Items are synthetic templates (no third-party text). Fluency check uses 16 hand-written sentences by default
  (`data.ppl_text_path` for a larger set).

## Known limitations

* Template items are not natural text; adjective/noun combinations can be odd ("ett vackert ägg").
* Zero-ablation takes the model off-distribution; amplification is selected heuristically then verified on dev.
* Edits are head-level only. Lexical gender knowledge likely lives in embeddings/MLPs, so a null on
  `lexical` does not show memorised facts are uneditable (MLP-level editing is a different experiment).
* With 100 items per type, effects below roughly 10 points are hard to detect; report CIs, not just p-values.
* The English control is off by default (Swedish-only models cannot do English); it has only 48 distinct items per split.
* Regression text is 86 hand-written Swedish sentences (even lines dev, odd lines test); have a native speaker skim it.
* The prompting arm is reported twice: `prompt` (mixed Swedish dev shots) and `prompt_target` (long-distance shots only).
* Old `diagnostics.json` files use older key names: do not use `--reuse-diagnostics` across this change.
* `configs/qwen3_0p6b.yaml` is untested on the real model.
* Runtime estimates are unmeasured: the ablation sweep is (layers x heads) x dev items x 2 sentences
  forward passes; reduce `n_dev` / `top_m` / `n_random` if slow.

## Next steps (not implemented)

weights-only head selection arm; LLM-written edit spec from the diagnostics table (with a
shuffled-table control); pair-contrastive fine-tuning; MLP-level edit; second model size.

## Multi-seed run and environment record

```bash
export PYTHONPATH=src
bash scripts/run_seeds.sh configs/default.yaml 1234 1235 1236   # one folder per seed + aggregate.txt + environment.txt
bash scripts/freeze_env.sh results/environment.txt              # pip freeze, python, platform, git sha
```
The seed changes the noun split, the sampled items, the heads chosen and the random edits.
`n_test: 200` gives 200 local / 200 long-distance items; `lexical` is capped at 104 by the number of builtin test nouns
(use `data.source: ud` with Universal Dependencies Swedish-Talbanken for more). Rough time per seed on a laptop: ~15-25 min.

## Data: UD Swedish-Talbanken (default)

`data.source: ud` builds the noun lexicon from UD_Swedish-Talbanken (CC BY-SA 4.0, commit
`0185ae1fb4728ae501bf2a329b958e6c6acb9c8d`; files are bundled in `data/ud/`, or re-fetch with `bash scripts/download_ud.sh`).
Kept: singular, non-genitive NOUN lemmas with one unambiguous Gender (Com/Neut), >= 5 occurrences, and a distinct definite form
(513 nouns). The Swedish NLL regression text is 100 real UD dev sentences (`data/ppl_sentences.txt`, even lines dev / odd lines test).
Caveats: Talbanken is 1970s-era text; many nouns are abstract (e.g. "systemet", "hälsan"), so template sentences can be semantically odd.
The ablation/edit results use the same pipeline as before; only the nouns and the regression text changed, so numbers are not
directly comparable with the builtin-lexicon run. The builtin lexicon remains available with `data.source: builtin`.
