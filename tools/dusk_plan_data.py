#!/usr/bin/env python3
"""Turn the game plan into training rows: (prompt, candidates, per-candidate plan weight).

The reward is now a function of (state, action) alone -- no lookahead, no outcome. That makes
policy gradient the wrong tool: if the correct answer at a decision is already known, the
objective is a RANKING over the menu, and a listwise loss optimises it directly with none of
the variance a sampled return carries.

CONFORMANCE IS MATCHED ON TEXT, NOT INDEX. The model sees the DEDUPED menu (PROMPT_FMT has
menu_dedup=True), so raw option indices are the wrong coordinate space -- the mistake that put
20% of the DPO pairs off the end of the menu and silently mis-pointed the rest. Rules return
raw indices; those are rendered to option text and matched against the deduped candidates.

A row is emitted only when the menu contains BOTH a conforming and a non-conforming candidate.
A decision where every option conforms teaches nothing and would dilute the loss.

    PYTHONPATH=cg-lib python3 tools/dusk_plan_data.py \\
        --traces /root/traces_r4.s0.jsonl.gz,... --out /root/rl/plan_r1.jsonl.gz
"""

import argparse
import collections
import gzip
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (ROOT, os.path.join(ROOT, "cg-lib"), os.path.join(ROOT, "tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

ACT = "[ACT]\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--traces", required=True)
    ap.add_argument("--deck", default="dragapult_dusknoir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--mirror-so", default="")
    ap.add_argument("--exclude", default="",
                    help="comma-separated rules to DROP from training rows (rules that "
                         "make_plan_rule executes at inference; the model never decides them)")
    ap.add_argument("--only", default="", help="keep ONE rule. The merged rows answer 'can the "
                    "model follow the plan'; they cannot answer 'which rule is unlearnable', "
                    "because a rule that never wins its scope is indistinguishable from one "
                    "the model cannot represent. One rule at a time separates the two.")
    ap.add_argument("--fmt", default="prompt", choices=("prompt", "dusk"),
                    help="'dusk' = the single-deck rendering with no DECK[] segment. Must match "
                         "the model being trained or every row is an out-of-distribution prompt")
    a = ap.parse_args()

    import library
    from lm.actions import encode_option
    from lm.agent import _dedup
    from lm.serialize import serialize_stateless
    from mirror_env import DEFAULT_SO, MirrorEngine
    from dusk_plan import RULES, opportunities
    import rl_config

    fmt = dict(rl_config.DUSK_FMT if a.fmt == "dusk" else rl_config.PROMPT_FMT)
    if a.only and a.only not in RULES:
        raise SystemExit("--only %r is not a rule: %s" % (a.only, ",".join(sorted(RULES))))
    excl = {x for x in a.exclude.split(",") if x}
    bad = excl - set(RULES)
    if bad:
        raise SystemExit("--exclude has non-rules %s: %s" % (sorted(bad), ",".join(sorted(RULES))))
    eng = MirrorEngine(a.mirror_so or DEFAULT_SO)
    ids = {}

    def deck_ids(n):
        if n not in ids:
            ids[n] = [int(x) for x in open(library.deck_path(n)) if x.strip()]
        return ids[n]

    n_rows = n_games = 0
    per_rule = collections.Counter()
    skipped = collections.Counter()
    with gzip.open(a.out, "wt") as out:
        for path in [p for p in a.traces.split(",") if p]:
            for line in gzip.open(path, "rt"):
                d = json.loads(line)
                if d.get("header"):
                    continue
                d0 = d.get("deck0") or d.get("deck")
                d1 = d.get("deck1") or d.get("deck")
                if a.deck not in (d0, d1):
                    continue
                seat = 0 if d0 == a.deck else 1
                obs = eng.start(deck_ids(d0), deck_ids(d1), d["seed"], mirror=1)
                n_games += 1
                try:
                    for pick in d["picks"]:
                        if obs is None:
                            break
                        cur = obs.get("current") or {}
                        if cur.get("result", -1) != -1 or obs.get("select") is None:
                            break
                        if cur.get("yourIndex") == seat:
                            live = opportunities(obs, seat)
                            if a.only:
                                live = {k: v for k, v in live.items() if k == a.only}
                            if excl:
                                # Rules executed BY RULE at inference (make_plan_rule) are
                                # excluded from training: the model never makes those
                                # decisions, so a gradient toward them is spent on nothing.
                                # They stay in opportunities() itself on purpose -- their
                                # firing still suppresses the attack rules, which keeps the
                                # training labels consistent with how the wrapped agent
                                # actually sequences a turn.
                                live = {k: v for k, v in live.items() if k not in excl}
                            if live:
                                raw = (obs.get("select") or {}).get("option") or []
                                texts = [encode_option(o, obs) for o in raw]
                                uniq, _pos = _dedup(texts, obs)
                                state = serialize_stateless(obs, deck_ids=deck_ids(a.deck),
                                                            deck_name=a.deck, **fmt)
                                # ONE ROW PER DECISION, with the rules MERGED. Emitting a
                                # row per rule made the same (prompt, menu) appear several
                                # times with different -- sometimes disjoint -- correct sets,
                                # so no weighting could satisfy them at once and the overfit
                                # probe stalled at 1.28 on 300 rows instead of collapsing.
                                # Merging turns competing rules into a weighted preference,
                                # which is what "several plays are live and one matters more"
                                # actually means.
                                # ONE ROW PER SCOPE. Rules that adjudicate the same options
                                # merge (energy_line and energy_focus both judge attaches);
                                # rules over DIFFERENT options become separate rows over
                                # DISJOINT menus, so they cannot contradict each other the way
                                # per-rule rows did. The emitted menu is the SCOPE only, which
                                # is how "the plan is silent about this option" becomes "no
                                # gradient" instead of "negative".
                                by_scope = {}
                                for rule, (good_i, scope_i) in live.items():
                                    good = {texts[i] for i in good_i if i < len(texts)}
                                    scope = {texts[i] for i in scope_i if i < len(texts)}
                                    key = frozenset(scope)
                                    if not key:
                                        continue
                                    g, w, rs = by_scope.setdefault(key, [set(), {}, []])
                                    g |= good
                                    rs.append(rule)
                                    for c in good:
                                        w[c] = w.get(c, 0.0) + RULES[rule][1]
                                for key, (good, wmap, rs) in by_scope.items():
                                    cands = [c for c in uniq if c in key]
                                    if len(cands) < 2:
                                        skipped["scope_too_small"] += 1
                                        continue
                                    wc = [wmap.get(c, 0.0) for c in cands]
                                    if max(wc) <= 0:
                                        skipped["conformant_not_in_menu"] += 1
                                        continue
                                    if all(x > 0 for x in wc):
                                        skipped["every_option_conforms"] += 1
                                        continue
                                    out.write(json.dumps({
                                        "prompt": ACT + state, "cands": cands,
                                        "wc": [round(x, 3) for x in wc],
                                        "rules": rs}) + "\n")
                                    n_rows += 1
                                    for r in rs:
                                        per_rule[r] += 1
                        if pick is None:
                            break
                        obs = eng.select(pick)
                finally:
                    eng.finish()
    print("%d rows from %d games -> %s" % (n_rows, n_games, a.out))
    print("skipped: %s" % dict(skipped))
    print("%-14s %8s %8s" % ("rule", "rows", "weight"))
    for r, (name, w) in sorted(RULES.items(), key=lambda kv: -kv[1][1]):
        print("%-14s %8d %8.1f  %s" % (r, per_rule.get(r, 0), w, name))


if __name__ == "__main__":
    main()
