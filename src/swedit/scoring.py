"""Likelihood scoring of sentences and minimal pairs."""
from __future__ import annotations
import numpy as np
import torch


@torch.no_grad()
def sentence_logprobs(model, tok, texts: list[str], prefix_ids: list[int] | None = None,
                      batch_size: int = 32) -> tuple[np.ndarray, np.ndarray]:
    """Sum of token log-probs of each text (conditioned on BOS [+ prefix]); also token counts.
    Only the tokens of `text` are scored; the prefix is context only (used by the few-shot arm)."""
    prefix_ids = prefix_ids or []
    ctx = [tok.bos_id] + list(prefix_ids)
    seqs = [ctx + tok.encode(t) for t in texts]
    max_pos = getattr(model.config, "n_positions", None) or getattr(model.config, "max_position_embeddings", None)
    if max_pos and max(len(x) for x in seqs) > max_pos:
        raise ValueError(f"sequence of {max(len(x) for x in seqs)} tokens exceeds model context {max_pos}; reduce prompting.n_shots")
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))
    lp = np.zeros(len(seqs)); nt = np.zeros(len(seqs), dtype=int)
    dev = next(model.parameters()).device
    for s in range(0, len(order), batch_size):
        idx = order[s:s + batch_size]
        L = max(len(seqs[i]) for i in idx)
        ids = torch.full((len(idx), L), tok.pad_id, dtype=torch.long)
        mask = torch.zeros((len(idx), L), dtype=torch.long)
        for r, i in enumerate(idx):
            ids[r, :len(seqs[i])] = torch.tensor(seqs[i]); mask[r, :len(seqs[i])] = 1
        logits = model(input_ids=ids.to(dev), attention_mask=mask.to(dev)).logits[:, :-1].float()
        logp = torch.log_softmax(logits, -1).gather(-1, ids[:, 1:].to(dev).unsqueeze(-1)).squeeze(-1).cpu()
        for r, i in enumerate(idx):
            start = len(ctx)            # first scored label index in ids
            end = len(seqs[i])
            lp[i] = float(logp[r, start - 1:end - 1].sum())
            nt[i] = end - start
    return lp, nt


def pair_margins(model, tok, items, prefix_ids=None, batch_size=32) -> np.ndarray:
    """margin = logP(gold) - logP(wrong); accuracy = fraction with margin > 0."""
    texts = [it.gold for it in items] + [it.wrong for it in items]
    lp, _ = sentence_logprobs(model, tok, texts, prefix_ids, batch_size)
    n = len(items)
    return lp[:n] - lp[n:]


def mean_nll(model, tok, sentences: list[str], batch_size: int = 32) -> float:
    """Mean negative log-likelihood per token (nats) over sentences: the fluency regression check."""
    lp, nt = sentence_logprobs(model, tok, sentences, None, batch_size)
    return float(-lp.sum() / nt.sum())


def mean_sentence_logprob(model, tok, sentences: list[str], batch_size: int = 32) -> float:
    """Mean over sentences of the summed token log-prob (nats per sentence). Same units as a pair margin,
    used as the Swedish regression penalty during head selection (DEV text only)."""
    lp, _ = sentence_logprobs(model, tok, sentences, None, batch_size)
    return float(lp.mean())


# Ordinary Swedish sentences (not about en/ett) for the fluency regression check. Even index -> DEV (used for
# head selection), odd index -> TEST (reported). Hand-written; have a native speaker skim them.
REGRESSION_SV = [
    'Det var en vacker dag i Stockholm och solen sken över vattnet.',
    'Barnen lekte i trädgården medan föräldrarna drack kaffe på verandan.',
    'Regeringen presenterade i dag ett nytt förslag om skolans framtid.',
    'På vintern är det mörkt tidigt och många åker till fjällen för att åka skidor.',
    'Biblioteket på torget har öppet varje dag utom söndagar.',
    'Hon läste en bok om svensk historia och skrev ner några anteckningar.',
    'Tåget till Göteborg avgår från perrong fem klockan åtta.',
    'Vi åt middag hos mina föräldrar och pratade om sommarens resa.',
    'Forskarna publicerade en rapport om hur språkmodeller kan förbättras.',
    'Mannen cyklade genom skogen och stannade vid en liten sjö.',
    'Kommunen bygger ett nytt sjukhus i närheten av centrum.',
    'Läraren förklarade grammatiken för klassen och gav dem läxor till nästa vecka.',
    'Det regnade hela helgen, så vi stannade hemma och lagade mat.',
    'Företaget har kontor i flera länder och anställer över tusen personer.',
    'Hunden sprang efter bollen och kom tillbaka med den i munnen.',
    'Museet visar en utställning om vikingatiden som många besöker.',
    'Min syster bor i Malmö och arbetar som sjuksköterska.',
    'Efter lunchen gick vi en promenad längs älven.',
    'Bussen var försenad, så jag kom för sent till mötet.',
    'På lördagen brukar vi handla mat och städa hemma.',
    'Han spelar fotboll två gånger i veckan med sina kamrater.',
    'Staden ligger vid kusten och har en gammal hamn.',
    'Vi planerar att resa till Norge i augusti och vandra i bergen.',
    'Tidningen skrev att priserna på el har sjunkit under hösten.',
    'Flickan ritade en teckning av sin katt och visade den för alla.',
    'De bor i ett litet hus med en stor trädgård utanför staden.',
    'Jag brukar dricka te på kvällen medan jag läser tidningen.',
    'Studenterna samlades i aulan för att lyssna på föreläsningen.',
    'Nästa år ska kommunen renovera skolan och bygga en ny idrottshall.',
    'Hon ringde sin mormor varje söndag och berättade om veckan.',
    'Snön föll tyst över byn och täckte tak och gator.',
    'Läkaren undersökte patienten och skrev ut ett recept.',
    'Vi tog färjan över sundet och åt lunch ombord.',
    'Det finns många sjöar i Sverige, och de flesta ligger i skogen.',
    'Bageriet säljer nybakat bröd och kanelbullar varje morgon.',
    'Efter matchen firade laget segern tillsammans med supportrarna.',
    'Parlamentet debatterade lagen i flera timmar innan omröstningen.',
    'Han glömde nycklarna hemma och fick vänta utanför dörren.',
    'Grannen klippte gräset medan barnen cyklade på gatan.',
    'Vi hade en trevlig kväll med god mat, musik och många skratt.',
    'Floden svämmade över efter de kraftiga regnen i våras.',
    'Konstnären målade tavlan på några veckor och ställde ut den i gallerit.',
    'Mitt favoritämne i skolan var historia, eftersom läraren var så engagerad.',
    'På sommaren badar vi i havet och grillar på stranden.',
    'Flygplanet landade i Köpenhamn efter en lugn resa.',
    'Hon sökte jobb på flera företag och blev kallad till en intervju.',
    'Pojken lärde sig simma när han var sex år gammal.',
    'Ministern svarade på journalisternas frågor efter pressträffen.',
    'Vi bytte buss vid centralstationen och åkte vidare till universitetet.',
    'Köket doftade av kanel och nybryggt kaffe när gästerna kom.',
    'Bron över sundet öppnades för trafik för tjugo år sedan.',
    'Föreningen har ordnat en loppmarknad som lockar många besökare.',
    'Jag har aldrig varit i Lappland, men jag vill gärna åka dit en vinter.',
    'Skådespelarna repeterade pjäsen hela hösten inför premiären.',
    'Tåget stannade vid varje station, och resan tog nästan tre timmar.',
    'Bonden vallade korna till ängen när solen gick upp.',
    'De diskuterade budgeten och kom överens om några nya prioriteringar.',
    'Fjällen var täckta av snö, och luften var klar och kall.',
    'Han skrev ett brev till sin vän och lade det i brevlådan.',
    'Kören sjöng flera gamla psalmer i kyrkan på julafton.',
    'Verkstaden reparerade bilen på två dagar och tog bara lite betalt.',
    'Det är viktigt att sova tillräckligt och äta regelbundet.',
    'Hon åkte hem tidigt eftersom hon kände sig trött och hade ont i huvudet.',
    'Eleverna fick välja mellan tre olika projekt på temat miljö.',
    'Vi satt länge vid bordet och pratade om livet och framtiden.',
    'Stadsbiblioteket lånar ut både böcker, filmer och spel.',
    'Under natten blåste det kraftigt, och flera träd föll över vägen.',
    'Han arbetade som snickare i många år innan han startade eget företag.',
    'Alla invånare uppmanades att spara vatten under den torra sommaren.',
    'Hon tog kameran med sig och fotograferade fåglarna vid sjön.',
    'Mötet flyttades till torsdag eftersom flera av deltagarna var sjuka.',
    'Vi lagade soppa av rotfrukter och åt den med nybakat bröd.',
    'Skolan bjuder på lunch, och maten brukar vara ganska god.',
    'Fotbollsplanen låg precis bakom kyrkan, och på helgerna var den full av barn.',
    'Journalisten intervjuade flera forskare om klimatförändringarna.',
    'Båten gled långsamt förbi öarna, och passagerarna stod vid relingen.',
    'Min bror flyttade till Umeå förra hösten för att studera medicin.',
    'Biljetterna släpptes klockan tio och var slutsålda inom en timme.',
    'Vi väntade på bussen i regnet, men den kom aldrig.',
    'Orkestern spelade en symfoni som jag hade hört många gånger tidigare.',
    'Parken är full av människor när vädret är fint på sommaren.',
    'Hon packade väskan, låste dörren och gick mot stationen.',
    'Efter middagen diskade vi tillsammans och tittade på en film.',
    'Byggarbetarna började tidigt på morgonen och slutade när det blev mörkt.',
    'Priset på bensin har stigit kraftigt det senaste året.',
    'Vi lärde oss mycket av projektet, även om det blev dyrare än planerat.',
]


def load_ppl_sentences(path: str | None) -> tuple[list[str], list[str]]:
    """Returns (dev_sentences, test_sentences): even/odd lines of the file (or of the built-in set)."""
    if path:
        with open(path, encoding="utf-8") as f:
            sents = [l.strip() for l in f if l.strip()]
    else:
        sents = REGRESSION_SV
    return sents[0::2], sents[1::2]
