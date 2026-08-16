#!/usr/bin/env python3
"""Audit the human's play_server decisions against the current rule stack.

Two comparisons, both on the human seat's decisions from logs/ HumanvAI games:
  1. dusk_plan.opportunities -- for every plan rule that FIRES on the decision, did the
     human's pick fall inside the rule's good set? A rule the (winning) human keeps
     violating is a rule that would have BLOCKED the winning line at inference.
  2. engine_v2's dragapult agent -- where does the engine's pick diverge from the human's,
     bucketed by menu context and the action kinds involved?

Env gates are forced ON (same set the loop's planfilter wrap exports), otherwise half the
rules are silently inert (plan-rule-audit-and-wrapper-bugs).
"""
import argparse
import collections
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (ROOT, os.path.join(ROOT, "cg-lib"), os.path.join(ROOT, "tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

for _v in ("DUSK_NEW_RULES", "DUSK_CLOPS_HOLD", "DUSK_FRONT_DIVE", "DUSK_BOSS_LETHAL",
           "DUSK_SPIKE", "DUSK_TIPS"):
    os.environ.setdefault(_v, "1")


def opt_kind(o, obs):
    try:
        from lm.actions import encode_option
        return encode_option(o, obs)
    except Exception:                                # noqa: BLE001
        return str(o.get("type"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default="logs/2026081*Human*human-dragapult_dusknoir*.json")
    ap.add_argument("--deck", default="dragapult_dusknoir")
    ap.add_argument("--seat", type=int, default=0)
    a = ap.parse_args()

    import library
    import dusk_plan

    tuning = json.load(open(os.path.join(ROOT, "agents", "tuning.json")))
    dids = [int(x) for x in open(library.deck_path(a.deck)) if x.strip()]
    from lm.agent import make_lm_agent
    agent = make_lm_agent(dids, tuning.get(a.deck, {}), model=None)

    turn_fired = collections.defaultdict(set)    # (rule,opp) -> {(file,turn)}
    turn_agree = collections.defaultdict(set)
    turn_viol_won = collections.defaultdict(set)
    rule_viol_ex = collections.defaultdict(list)
    div = collections.Counter()          # (context, human_kind, engine_kind) -> n
    div_n = agree_n = eng_err = 0

    for f in sorted(glob.glob(a.logs)):
        d = json.load(open(f))
        res = None
        for e in reversed(d):
            for lg in e.get("logs", []):
                if lg.get("type") == "Result":
                    res = lg["result"]
                    break
            if res is not None:
                break
        won = (res == a.seat)
        opp = "ogerpon" if "ogerpon" in f else ("abomasnow" if "abomasnow" in f else "?")
        for e in d:
            cur = e.get("current") or {}
            if cur.get("yourIndex") != a.seat:
                continue
            raw = (e.get("select") or {}).get("option") or []
            sel = e.get("selected")
            if len(raw) < 2 or not isinstance(sel, list) or len(sel) != 1:
                continue
            h = sel[0]
            if not (0 <= h < len(raw)):
                continue
            ctx = "%s/%s" % (e["select"].get("type"), e["select"].get("context"))
            # 1) plan rules
            tk = (f, cur.get("turn"))
            try:
                for rn, (good, _sc) in dusk_plan.opportunities(e, a.seat).items():
                    if not good:
                        continue
                    turn_fired[(rn, opp)].add(tk)
                    if h in good:
                        turn_agree[(rn, opp)].add(tk)
                    elif won:
                        turn_viol_won[(rn, opp)].add(tk)
                        if len(rule_viol_ex[rn]) < 6:
                            rule_viol_ex[rn].append(
                                "T%s %s human=%s rule_wants=%s" % (
                                    cur.get("turn"), ctx, opt_kind(raw[h], e),
                                    [opt_kind(raw[i], e) for i in sorted(good)][:3]))
            except Exception:                        # noqa: BLE001
                pass
            # 2) engine divergence
            try:
                pick = agent(e)
                ei = pick[0] if isinstance(pick, (list, tuple)) and pick else pick
                if not isinstance(ei, int) or not (0 <= ei < len(raw)):
                    eng_err += 1
                    continue
            except Exception:                        # noqa: BLE001
                eng_err += 1
                continue
            if ei == h:
                agree_n += 1
            else:
                div_n += 1
                div[(opp, ctx, opt_kind(raw[h], e), opt_kind(raw[ei], e))] += 1

    print("== ENGINE vs HUMAN: agree %d, diverge %d, engine_err %d ==" % (agree_n, div_n, eng_err))
    for k, n in div.most_common(30):
        print("  %3d  %-9s %-18s human=%-28s engine=%s" % (n, k[0], k[1], k[2], k[3]))
    print("\n== PLAN RULES, PER-TURN (turns fired / turns human conformed / conforming-neither-but-won) ==")
    for k in sorted(turn_fired, key=lambda k: -len(turn_fired[k])):
        rn, opp = k
        f_, ag = len(turn_fired[k]), len(turn_agree[k])
        vw = len(turn_viol_won[k] - turn_agree[k])
        print("  %-16s %-9s turns %3d  agree %3d (%.0f%%)  viol-won %d"
              % (rn, opp, f_, ag, 100.0 * ag / f_, vw))
    print("\n== violation examples ==")
    for rn, exs in rule_viol_ex.items():
        for s in exs:
            print("  %-16s %s" % (rn, s))


if __name__ == "__main__":
    main()
