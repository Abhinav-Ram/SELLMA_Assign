"""End-to-end experiment. Usage:
    python -m swedit.run_experiment --config configs/default.yaml
    python -m swedit.run_experiment --config configs/smoke.yaml        # offline plumbing test
Options: --edit-spec FILE (use a hand/LLM-written edit instead of running diagnostics),
         --reuse-diagnostics (load diagnostics.json from out_dir).
"""
from __future__ import annotations
import argparse, json, os, random, time
import numpy as np
from .config import load_config, config_hash
from .seeding import set_seed, sub_seed, manifest, dump_json
from .lexicon import load_lexicon
from .items import build_items, save_items, check_disjoint
from .models import load_model, introspect
from .scoring import load_ppl_sentences
from .edits import EditSpec, applied, random_spec, validate, n_changed_params
from .diagnose import ablation_sweep, select_spec
from .prompting import build_prefix_ids
from .finetune import lora_finetune
from .evaluate import evaluate, correct, TYPES
from . import stats as S
from .report import summarise, write_report


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--seed", type=int, default=None, help="override config seed (data split + heads + random edits)")
    ap.add_argument("--out-dir", default=None, help="override config out_dir")
    ap.add_argument("--edit-spec", default=None)
    ap.add_argument("--reuse-diagnostics", action="store_true")
    args = ap.parse_args(argv)
    ov = {}
    if args.seed is not None: ov["seed"] = args.seed
    if args.out_dir: ov["out_dir"] = args.out_dir
    cfg = load_config(args.config, ov)
    out = cfg["out_dir"]; os.makedirs(out, exist_ok=True)
    seed = cfg["seed"]; set_seed(seed)
    bs = cfg["model"]["batch_size"]
    ecfg = cfg["edit"]; tt = ecfg["target_type"]

    # ---- data ------------------------------------------------------------------------------
    nouns = load_lexicon(cfg["data"])
    items = build_items(nouns, cfg["data"]["n_dev"], cfg["data"]["n_test"], seed, cfg["data"]["english_control"])
    check_disjoint(items)
    save_items(items, os.path.join(out, "items.jsonl"))
    ppl_dev, ppl = load_ppl_sentences(cfg["data"]["ppl_text_path"])   # dev: head selection; test: reported NLL
    log(f"lexicon: {len(nouns)} nouns; items dev={ {t: len(v) for t, v in items['dev'].items()} } "
        f"test={ {t: len(v) for t, v in items['test'].items()} }")

    # ---- model -----------------------------------------------------------------------------
    model, tok, minfo = load_model(cfg["model"], seed)
    arch = introspect(model)
    log(f"model {minfo}; {arch.n_layers} layers x {arch.n_heads} heads (head_dim {arch.head_dim}, {arch.layout})")
    dump_json({"cfg": cfg, "cfg_hash": config_hash(cfg), **manifest(cfg, minfo)}, os.path.join(out, "manifest.json"))

    arms: dict[str, dict] = {}
    per_item_dir = os.path.join(out, "per_item"); os.makedirs(per_item_dir, exist_ok=True)

    def run_arm(name, mdl=None, spec=None, prefix_ids=None):
        mdl = mdl or model
        with applied(mdl, introspect(mdl), spec):
            r = evaluate(mdl, tok, items["test"], ppl, prefix_ids, bs)
        arms[name] = r
        dump_json(r, os.path.join(per_item_dir, f"{name}.json"))
        return r

    # ---- A0 baseline -----------------------------------------------------------------------
    log("A0 baseline"); run_arm("baseline")

    # ---- diagnostics + targeted edit (DEV only) --------------------------------------------
    dev_t = items["dev"][tt]
    diag_path = os.path.join(out, "diagnostics.json")
    if args.edit_spec:
        spec = EditSpec.from_json(open(args.edit_spec).read()); log(f"using supplied edit spec {args.edit_spec}")
    else:
        if args.reuse_diagnostics and os.path.exists(diag_path):
            rows = json.load(open(diag_path))["sweep"]; log("reusing diagnostics sweep")
        else:
            log("diagnostics: ablation sweep on DEV")
            rows = ablation_sweep(model, tok, arch, dev_t, ppl_dev, bs, log)
        spec, verified = select_spec(model, tok, arch, rows, dev_t, ppl_dev, ecfg["k"], ecfg["alpha_up"],
                                     ecfg["top_m"], ecfg["penalty_lambda"], bs, log)
        dump_json({"sweep": rows, "verified": verified}, diag_path)
    validate(spec, arch, ecfg["k"])
    open(os.path.join(out, "edit_spec.json"), "w").write(spec.to_json())
    n_params = n_changed_params(arch, spec)
    log(f"A2 targeted edit: {len(spec)} heads, {n_params} weights ({100 * n_params / minfo.get('n_params', n_params):.4f}% of model)" if minfo.get("n_params") else f"A2 targeted edit: {len(spec)} heads")
    run_arm("targeted", spec=spec)

    # ---- A1 random edits -------------------------------------------------------------------
    for i in range(ecfg["n_random"]):
        rs = random_spec(arch, spec, random.Random(sub_seed(seed, "random_edit", i)), ecfg["random_mode"])
        run_arm(f"random_{i:02d}", spec=rs)
        dump_json(json.loads(rs.to_json()), os.path.join(per_item_dir, f"random_{i:02d}_spec.json"))
    log(f"A1 {ecfg['n_random']} random edits done")

    # ---- P few-shot prompting ---------------------------------------------------------------
    pcfg = cfg["prompting"]
    mixed = [it for t in pcfg["shot_types"] for it in items["dev"][t]]
    prefix_ids, shots = build_prefix_ids(tok, mixed, pcfg["n_shots"], seed)
    dump_json({"shots": shots}, os.path.join(out, "prompt_shots.json"))
    log("P few-shot prompting (mixed Swedish dev shots)"); run_arm("prompt", prefix_ids=prefix_ids)
    if pcfg.get("also_target_only"):
        prefix_t, shots_t = build_prefix_ids(tok, dev_t, pcfg["n_shots"], seed)
        dump_json({"shots": shots_t}, os.path.join(out, "prompt_target_shots.json"))
        log("P few-shot prompting (target-type dev shots only)"); run_arm("prompt_target", prefix_ids=prefix_t)

    # ---- F LoRA fine-tune (same dev information) --------------------------------------------
    ft_meta = {}
    if cfg["finetune"]["enabled"]:
        texts = [it.gold for it in dev_t]
        for s in cfg["finetune"]["seeds"]:
            log(f"F fine-tune seed {s}")
            ft_model, meta = lora_finetune(model, tok, texts, cfg["finetune"], sub_seed(seed, "ft", s), log)
            ft_meta[str(s)] = meta
            run_arm(f"finetune_s{s}", mdl=ft_model)
            del ft_model

    # ---- summary ---------------------------------------------------------------------------
    summary = summarise(arms, items["test"], tt, cfg["stats"]["n_boot"], seed)
    summary["edit"] = {"n_heads": len(spec), "n_weights_changed": n_params, "ops": json.loads(spec.to_json())["ops"]}
    summary["finetune_meta"] = ft_meta
    n_t = len(items["test"][tt])
    summary["n_test_target"] = n_t
    summary["mde_balanced_acc"] = S.mde(n_t)
    dump_json(summary, os.path.join(out, "summary.json"))
    write_report(summary, out)
    log(f"done -> {out}/summary.md")


if __name__ == "__main__":
    main()
