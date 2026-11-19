import json, random, copy
import numpy as np, pytest, torch
from transformers import GPT2Config, GPT2LMHeadModel, LlamaConfig, LlamaForCausalLM
from swedit.models import CharTok, introspect
from swedit.edits import EditSpec, HeadOp, applied, random_spec, validate
from swedit.scoring import sentence_logprobs, pair_margins
from swedit.items import build_items, check_disjoint
from swedit.lexicon import load_builtin, load_ud
from swedit import stats as S


def tiny_gpt2():
    torch.manual_seed(0)
    return GPT2LMHeadModel(GPT2Config(vocab_size=260, n_positions=256, n_embd=64, n_layer=3, n_head=4)).eval()


def tiny_llama():
    torch.manual_seed(0)
    return LlamaForCausalLM(LlamaConfig(vocab_size=260, hidden_size=64, intermediate_size=128,
                                        num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
                                        max_position_embeddings=256)).eval()


def hook_ablate_logits(model, arch, layer, head, ids):
    """Independent implementation: zero the head's slice of the o-proj INPUT with a forward pre-hook."""
    a, b = head * arch.head_dim, (head + 1) * arch.head_dim
    def pre(mod, inp):
        x = inp[0].clone(); x[..., a:b] = 0
        return (x,)
    h = arch.projs[layer].register_forward_pre_hook(pre)
    try:
        return model(ids).logits
    finally:
        h.remove()


@pytest.mark.parametrize("maker", [tiny_gpt2, tiny_llama])
def test_edit_matches_independent_hook_ablation(maker):
    model = maker(); arch = introspect(model)
    ids = torch.randint(0, 200, (2, 12))
    with torch.no_grad():
        for layer, head in [(0, 0), (1, 3), (2, 2)]:
            ref = hook_ablate_logits(model, arch, layer, head, ids)
            with applied(model, arch, EditSpec([HeadOp(layer, head, 0.0)])):
                got = model(ids).logits
            assert torch.allclose(ref, got, atol=1e-5), (maker.__name__, layer, head)
            # a different head must NOT give the same logits (the slice really is head-specific)
            with applied(model, arch, EditSpec([HeadOp(layer, (head + 1) % arch.n_heads, 0.0)])):
                other = model(ids).logits
            assert not torch.allclose(ref, other, atol=1e-5)


@pytest.mark.parametrize("maker", [tiny_gpt2, tiny_llama])
def test_edit_restores_weights_exactly(maker):
    model = maker(); arch = introspect(model)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    with applied(model, arch, EditSpec([HeadOp(0, 1, 0.0), HeadOp(2, 0, 1.7)])):
        assert any(not torch.equal(before[k], v) for k, v in model.state_dict().items())
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())


def test_amplify_scales_head_contribution():
    model = tiny_gpt2(); arch = introspect(model)
    ids = torch.randint(0, 200, (1, 10))
    with torch.no_grad():
        base = model(ids).logits
        with applied(model, arch, EditSpec([HeadOp(1, 1, 1.0)])):  # alpha=1 is identity
            same = model(ids).logits
    assert torch.allclose(base, same, atol=1e-6)


def test_scoring_is_batch_and_padding_invariant():
    model = tiny_gpt2(); tok = CharTok()
    texts = ["Jag köpte ett hus.", "Hon hittade en bil igår och var glad.", "Hej."]
    lp_all, nt_all = sentence_logprobs(model, tok, texts, batch_size=8)
    for i, t in enumerate(texts):
        lp1, nt1 = sentence_logprobs(model, tok, [t], batch_size=1)
        assert abs(lp1[0] - lp_all[i]) < 1e-3 and nt1[0] == nt_all[i] == len(t)
    # manual reference for one text
    ids = torch.tensor([[tok.bos_id] + tok.encode(texts[0])])
    with torch.no_grad():
        lg = torch.log_softmax(model(ids).logits[0, :-1], -1)
    ref = sum(lg[j, ids[0, j + 1]].item() for j in range(ids.shape[1] - 1))
    assert abs(ref - lp_all[0]) < 1e-3


def test_prefix_is_context_only():
    model = tiny_gpt2(); tok = CharTok()
    t = ["Jag köpte ett hus."]
    base, n0 = sentence_logprobs(model, tok, t)
    same, n1 = sentence_logprobs(model, tok, t, prefix_ids=[])
    pref, n2 = sentence_logprobs(model, tok, t, prefix_ids=tok.encode("Vi bor i ett hus.\n"))
    assert base[0] == pytest.approx(same[0]) and n0[0] == n1[0] == n2[0]   # same tokens scored
    assert abs(pref[0] - base[0]) > 1e-4                                   # but context changed the score


def test_items_disjoint_balanced_deterministic():
    nouns = load_builtin()
    a = build_items(nouns, 40, 60, 7); b = build_items(nouns, 40, 60, 7)
    check_disjoint(a)
    for split in a:
        for t in a[split]:
            assert [i.gold for i in a[split][t]] == [i.gold for i in b[split][t]]
    for t in ("lexical", "local", "long_distance"):
        g = [i.meta["gold_gender"] for i in a["test"][t]]
        assert abs(g.count("com") - g.count("neut")) <= 1
        for i in a["test"][t]:
            assert i.gold != i.wrong


def test_gender_and_agreement_in_generated_items():
    items = build_items(load_builtin(), 40, 60, 3)
    for i in items["test"]["lexical"]:
        art = "ett" if i.meta["gold_gender"] == "neut" else "en"
        assert f" {art} " in i.gold and f" {'en' if art == 'ett' else 'ett'} " in i.wrong
    ADJ = {"stort", "gammalt", "nytt", "vackert", "billigt", "dyrt", "tungt", "långt", "kallt", "varmt",
           "mörkt", "snabbt", "rött", "grönt", "gult", "starkt", "fint", "rent", "högt"}
    for i in items["test"]["long_distance"]:
        gold_last = i.gold.rstrip(".").split()[-1]; wrong_last = i.wrong.rstrip(".").split()[-1]
        assert (gold_last in ADJ) == (i.meta["gold_gender"] == "neut")
        assert (wrong_last in ADJ) == (i.meta["gold_gender"] == "com")


def test_random_spec_same_budget_and_distinct():
    arch = introspect(tiny_gpt2())
    tmpl = EditSpec([HeadOp(0, 0, 0.0), HeadOp(1, 1, 1.5), HeadOp(2, 2, 1.5)])
    for mode in ("uniform", "layer_matched"):
        rs = random_spec(arch, tmpl, random.Random(1), mode)
        assert [o.alpha for o in rs.ops] == [o.alpha for o in tmpl.ops]
        assert len({(o.layer, o.head) for o in rs.ops}) == 3
        validate(rs, arch, 3)
        if mode == "layer_matched":
            assert [o.layer for o in rs.ops] == [o.layer for o in tmpl.ops]
    with pytest.raises(ValueError):
        validate(EditSpec([HeadOp(0, 0, 0.0)] * 2), arch, 5)
    with pytest.raises(ValueError):
        validate(EditSpec([HeadOp(9, 0, 0.0)]), arch, 5)


def test_stats_sanity():
    c = np.array([1, 1, 1, 0, 0, 0, 1, 0], bool); g = ["com"] * 4 + ["neut"] * 4
    assert S.balanced_accuracy(c, g) == pytest.approx((0.75 + 0.25) / 2)
    assert S.mcnemar_exact_p(c, c) == 1.0
    assert S.empirical_p(0.9, [0.5] * 19) == pytest.approx(1 / 20)
    d, lo, hi = S.paired_diff_ci(c, c, g, n_boot=100)
    assert d == 0 and lo == 0 and hi == 0


def test_ud_loader(tmp_path):
    rows = []
    def tok(i, form, lemma, feats):
        return "\t".join([str(i), form, lemma, "NOUN", "NN", feats, "0", "root", "_", "_"])
    sent = []
    for _ in range(3):
        sent += [tok(1, "hus", "hus", "Definite=Ind|Gender=Neut|Number=Sing"),
                 tok(2, "huset", "hus", "Definite=Def|Gender=Neut|Number=Sing"),
                 tok(3, "bil", "bil", "Definite=Ind|Gender=Com|Number=Sing"),
                 tok(4, "bilen", "bil", "Definite=Def|Gender=Com|Number=Sing"),
                 tok(7, "dagens", "dag", "Case=Gen|Definite=Def|Gender=Com|Number=Sing"),
                 tok(8, "dag", "dag", "Definite=Ind|Gender=Com|Number=Sing"),
                 tok(9, "faktum", "faktum", "Definite=Ind|Gender=Neut|Number=Sing"),
                 tok(10, "faktum", "faktum", "Definite=Def|Gender=Neut|Number=Sing"),
                 tok(5, "mix", "mix", "Definite=Ind|Gender=Com|Number=Sing"),
                 tok(6, "mix", "mix", "Definite=Ind|Gender=Neut|Number=Sing"), ""]
    p = tmp_path / "t.conllu"; p.write_text("# s\n" + "\n".join(sent), encoding="utf-8")
    nouns = {n.lemma: n for n in load_ud([str(p)], min_count=3)}
    assert nouns["hus"].gender == "neut" and nouns["hus"].definite == "huset"
    assert nouns["bil"].gender == "com" and nouns["bil"].definite == "bilen"
    assert "mix" not in nouns  # ambiguous gender is excluded
    assert "dag" not in nouns  # only a genitive definite form seen -> unusable
    assert "faktum" not in nouns  # definite == lemma -> excluded


def test_english_control_is_optional_and_swedish_items_unchanged():
    nouns = load_builtin()
    a = build_items(nouns, 40, 60, 7); b = build_items(nouns, 40, 60, 7, english_control=True)
    assert "en_sv_agree" not in a["test"] and "en_sv_agree" in b["test"]
    for split in a:
        for t in a[split]:
            assert [i.gold for i in a[split][t]] == [i.gold for i in b[split][t]]


def test_regression_text_split_disjoint():
    from swedit.scoring import load_ppl_sentences
    dev, test = load_ppl_sentences(None)
    assert len(dev) >= 30 and len(test) >= 30 and not (set(dev) & set(test))


def test_evaluate_runs_without_english_control():
    from swedit.evaluate import evaluate
    model = tiny_gpt2(); tok = CharTok()
    items = build_items(load_builtin(), 10, 12, 1)
    r = evaluate(model, tok, items["test"], ["Det var en dag."], None, 8)
    assert set(r["margins"]) == {"lexical", "local", "long_distance"}
