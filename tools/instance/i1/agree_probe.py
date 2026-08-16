"""Why did v36 lose ground where engine_v2 is weak? Measure FIDELITY and BIAS during play.

Win rate alone cannot separate two very different failures:
  (a) the model imitates engine_v2 faithfully and simply inherits a losing matchup, or
  (b) the model diverges from engine_v2, and the divergence is what loses.
Top1 on held-out data cannot tell them apart either: it is measured on ENGINE-piloted states,
while the pilot has to act on states IT reached -- the classic imitation distribution shift.

So: play with the LM, and at every real decision also ask a separate engine_v2 policy what it
would do HERE. Records agreement plus the action KIND both sides picked, so a systematic bias
(e.g. attacking where the engine walls) shows up as a shifted kind distribution rather than
just a lower agreement number.

The observer is a second policy instance. engine_v2 keeps only per-decision caches
(_opt_atk_dmg / _card_need), so it answers from the obs; it is not fed the game history.
"""
import argparse
import collections
import json
import os
import sys

ROOT = "/root/ptcg/repo"
for p in (ROOT, os.path.join(ROOT, "cg-lib"), os.path.join(ROOT, "tools")):
    if p not in sys.path:
        sys.path.insert(0, p)

import arena  # noqa: E402
import library  # noqa: E402
from agents.engine_v2 import make_policy  # noqa: E402
from battle_log import load_agent  # noqa: E402
from lm.actions import encode_option  # noqa: E402
from lm.agent import make_lm_agent  # noqa: E402


def real_choice(sel):
    opts = sel.get("option") or []
    return len(opts) >= 2 and (sel.get("minCount", 1) or 0) < len(opts)


def kind(opt, obs):
    try:
        return encode_option(opt, obs).split(":")[0].split("@")[0]
    except Exception:
        return "?"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--deck", required=True)
    ap.add_argument("--opp", required=True)
    ap.add_argument("--games", type=int, default=40)
    ap.add_argument("--deck-mode", default="static", choices=("static", "remaining"))
    ap.add_argument("--deck-shuffle", action="store_true")
    ap.add_argument("--glossary", default="none")
    ap.add_argument("--defer-kinds", default="",
                    help="comma list of action kinds; when the OBSERVER's pick is one of "
                         "these, take the observer's move instead of the LM's. Turns a "
                         "correlation (the LM under-picks `attach`) into a causal test (how "
                         "much of the win-rate gap does that one deficit account for?)")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    from eval_rerank import RerankerScoringModel

    tok = AutoTokenizer.from_pretrained(args.adapter)
    mdl = AutoModelForSequenceClassification.from_pretrained(
        args.adapter, trust_remote_code=True, dtype=torch.bfloat16).to("cuda")
    sm = RerankerScoringModel(mdl, tok)

    tun = json.load(open(os.path.join(ROOT, "agents", "tuning.json")))
    prof = tun.get(args.deck, {})
    dl = library.read_deck(args.deck)
    ol = library.read_deck(args.opp)
    oa = load_agent(args.opp)

    inner = make_lm_agent(dl, profile=prof, model=sm, deck_name=args.deck,
                          glossary=args.glossary, deck_mode=args.deck_mode,
                          deck_shuffle=args.deck_shuffle)
    observer = make_policy(dl, prof)

    defer = set(k for k in args.defer_kinds.split(",") if k)
    n = agree = deferred = 0
    by_turn = collections.defaultdict(lambda: [0, 0])
    lm_kind = collections.Counter()
    eng_kind = collections.Counter()
    swaps = collections.Counter()

    def probed(obs):
        sel = obs.get("select")
        if sel is None or not real_choice(sel):
            return inner(obs)
        try:
            eng = observer.act(obs)
        except Exception:
            eng = None
        lm = inner(obs)
        if eng is not None and lm is not None:
            nonlocal n, agree, deferred
            n += 1
            same = sorted(eng) == sorted(lm)
            agree += same
            t = (obs.get("current") or {}).get("turn") or 0
            b = min(int(t) // 5, 4)
            by_turn[b][0] += same
            by_turn[b][1] += 1
            opts = sel.get("option") or []
            lk = kind(opts[lm[0]], obs) if lm and lm[0] < len(opts) else "?"
            ek = kind(opts[eng[0]], obs) if eng and eng[0] < len(opts) else "?"
            lm_kind[lk] += 1
            eng_kind[ek] += 1
            if not same:
                swaps[(ek, lk)] += 1
            if defer and ek in defer:
                deferred += 1
                return eng
        return lm

    w = 0
    for g in range(args.games):
        mine = g % 2
        r = (arena.play(probed, oa, dl, ol) if mine == 0
             else arena.play(oa, probed, ol, dl))
        w += (r == mine)

    print("%s  %s vs %s  (%d games)" % (os.path.basename(args.adapter), args.deck,
                                        args.opp, args.games))
    print("  win rate            %d/%d = %.1f%%" % (w, args.games,
                                                    100.0 * w / args.games))
    if defer:
        print("  DEFERRED to engine  %d/%d = %.1f%% of decisions (kinds: %s)"
              % (deferred, n, 100.0 * deferred / max(1, n), ",".join(sorted(defer))))
    print("  agreement w/ engine %d/%d = %.1f%%" % (agree, n, 100.0 * agree / max(1, n)))
    print("  decisions/game      %.1f" % (n / args.games))
    print("  agreement by turn:", "  ".join(
        "T%d-%d %.0f%%(%d)" % (b * 5, b * 5 + 4, 100.0 * by_turn[b][0] / max(1, by_turn[b][1]),
                               by_turn[b][1]) for b in sorted(by_turn)))
    ks = sorted(set(lm_kind) | set(eng_kind), key=lambda k: -eng_kind[k])
    print("  action mix (engine -> LM), share of decisions:")
    for k in ks[:10]:
        e = 100.0 * eng_kind[k] / max(1, n)
        l = 100.0 * lm_kind[k] / max(1, n)
        print("      %-22s engine %5.1f%%   LM %5.1f%%   %+5.1f" % (k, e, l, l - e))
    print("  top disagreements (engine picked -> LM picked):")
    for (ek, lk), c in swaps.most_common(8):
        print("      %-22s -> %-22s %4d  (%.1f%% of decisions)"
              % (ek, lk, c, 100.0 * c / max(1, n)))
    if args.out:
        json.dump({"adapter": args.adapter, "deck": args.deck, "opp": args.opp,
                   "games": args.games, "wins": w, "n": n, "agree": agree,
                   "defer_kinds": sorted(defer), "deferred": deferred,
                   "by_turn": {str(k): v for k, v in by_turn.items()},
                   "lm_kind": dict(lm_kind), "eng_kind": dict(eng_kind),
                   "swaps": {"%s->%s" % k: v for k, v in swaps.items()}},
                  open(args.out, "w"))


if __name__ == "__main__":
    main()
