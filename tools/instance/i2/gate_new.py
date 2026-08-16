#!/usr/bin/env python3
"""Cross-deck gate: one PROTAGONIST deck, several pilots, the same opponents and the same
shuffles for all of them.

WHY THIS EXISTS. `mirror_match.py` puts both pilots on the SAME deck, which is the right
comparison for a model that plays all 63 of them and the wrong one for instance1's, which is
trained on dragapult_dusknoir alone. Its mirror would score the model on a deck it never saw
(any other deck) or on a matchup its pool deliberately excludes (dusknoir vs dusknoir -- the
protagonist does not meet itself on the ladder, so not one training row is that matchup).
`eval_rerank.py` does play cross-deck, but it still passes the v40 format flags
(glossary/deck_mode/deck_shuffle and nothing else), so a v41 model scored through it is being
read in a prompt format it never trained on -- the same class of error as
`rl-stack-cross-encoder`. This tool renders through mirror_match.make_agent, which takes its
format from rl_config, and plays through mirror_env, which seeds the shuffles.

WHAT IT MEASURES. Absolute win rates against a field are not comparable across deck sets, so
the number that matters is each arm MINUS the engine_v2 arm on the same (opponent, seed):
0 means the pilot flies the deck as well as the heuristic does. Every arm sees identical
shuffles and identical seats, so that difference is paired and its standard error is computed
over the per-seed differences, not over games treated as independent.

    python tools/gate_protagonist.py --deck dragapult_dusknoir \
        --opp marnie_grimmsnarl,alakazam_nz,... --games 40 \
        --arm engine=engine@prompt --arm s1=hf:/root/out/dusk_s1@dusk --out gate.json
"""
import argparse
import collections
import json
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (ROOT, os.path.join(ROOT, "cg-lib"), os.path.join(ROOT, "tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


SETUP_IDS = ((119, "dreepy"), (120, "drakloak"), (121, "pult"), (131, "duskull"))
PHANTOM_DIVE = 154


def _n_energy(body):
    for k in ("energy", "attachedEnergy", "energies"):
        v = body.get(k)
        if isinstance(v, list):
            return len(v)
    return 0


def setup_watch(agent, acc, pd_acc):
    """Wrap an arm's pilot and record how fast its board comes up.

    `acc` collects one Counter per (game, our-Nth-turn); `pd_acc` collects the ordinal of the
    first Phantom Dive, or nothing when the game never got one.  Costs one dict update per
    decision and cannot change the pick -- the wrapper returns the agent's answer untouched, and
    every read is inside a try so an observation shape it does not expect degrades to no data
    rather than to a failed gate.
    """
    st = {}
    seen_pd = [None]

    def flush():
        for i, t in enumerate(sorted(st), 1):
            acc.setdefault(i, []).append(st[t])
            if seen_pd[0] == t:
                pd_acc.append(i)
        st.clear()
        seen_pd[0] = None

    def w(obs):
        cur = obs.get("current") or {}
        if not cur:
            flush()                      # episode boundary: fold the game that just ended
            return agent(obs)
        pick = agent(obs)
        try:
            sel = obs.get("select") or {}
            opts = sel.get("option") or []
            if not opts:
                return pick
            t = cur.get("turn")
            if not isinstance(t, int):
                return pick
            yi = cur.get("yourIndex", 0)
            me = (cur.get("players") or [{}])[yi] or {}
            bodies = [x for x in ([(me.get("active") or [None])[0]] + list(me.get("bench") or []))
                      if isinstance(x, dict)]
            ids = [b.get("id") for b in bodies]
            d = st.setdefault(t, collections.Counter())
            for cid, nm in SETUP_IDS:
                d[nm] = max(d[nm], ids.count(cid))
            d["bodies"] = max(d["bodies"], len(ids))
            d["energy"] = max(d["energy"], sum(_n_energy(b) for b in bodies))
            if seen_pd[0] is None:
                for i in (pick if isinstance(pick, (list, tuple)) else [pick]):
                    if (isinstance(i, int) and 0 <= i < len(opts)
                            and isinstance(opts[i], dict)
                            and opts[i].get("attackId") == PHANTOM_DIVE):
                        seen_pd[0] = t
        except Exception:               # noqa: BLE001 -- measurement must never break the gate
            pass
        return pick

    return w, flush


def parse_arm(s):
    """'label=spec@fmt' -> (label, spec, fmt). The spec itself contains ':' (hf:/path), which
    is why the separators are '=' and '@' rather than the obvious ':'."""
    label, _, rest = s.partition("=")
    if not rest:
        raise SystemExit("--arm wants label=spec@fmt, got %r" % s)
    spec, _, fmt = rest.rpartition("@")
    if not spec:
        spec, fmt = rest, "prompt"
    if fmt not in ("prompt", "dusk"):
        raise SystemExit("arm %r: fmt must be prompt|dusk" % label)
    return label, spec, fmt


# The first observation of a Kaggle episode, verbatim from the competition's sample submission.
_EPISODE_START = {"current": None, "logs": [], "remainingOverageTime": 600.0,
                  "search_begin_input": None, "select": None, "step": 1}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--deck", required=True, help="the protagonist; every arm pilots THIS")
    ap.add_argument("--opp", required=True, help="comma-separated opponent decks (engine_v2)")
    ap.add_argument("--arm", action="append", required=True, help="label=spec@fmt, repeatable")
    ap.add_argument("--games", type=int, default=40, help="games per (arm, opponent); seats "
                                                          "alternate, so keep it even")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--mirror-so", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--baseline", default="engine",
                    help="which arm label the others are reported against")
    ap.add_argument("--opp-spec", default="engine",
                    help="pilot for the OPPONENT decks: 'engine' (default, comparable with "
                         "every past gate) or 'reg' (each opponent deck's own adapter)")
    a = ap.parse_args()

    import mirror_match as mm
    from tools.mirror_env import DEFAULT_SO, MirrorEngine, play

    eng = MirrorEngine(a.mirror_so or DEFAULT_SO)
    tuning = json.load(open(os.path.join(ROOT, "agents", "tuning.json")))
    my_ids = mm.load_deck(a.deck)
    my_prof = tuning.get(a.deck, {})

    # Build every arm's pilot ONCE. The format is a module global inside mirror_match because
    # make_agent recurses, so it has to be set around each construction -- the value is baked
    # into the agent at build time, which is what lets two arms with different formats coexist.
    arms = []
    setup_acc, pd_acc, flushers = {}, {}, {}
    for spec_s in a.arm:
        label, spec, fmt = parse_arm(spec_s)
        mm._FMT = fmt
        mm.LAST_FMT = None
        agent, _sc = mm.make_agent(spec, a.deck, my_ids, my_prof)
        # report what was BUILT, not what was asked for: a "reg" arm takes its format from the
        # registry, and printing the command line's --fmt would hide the substitution
        fmt = mm.LAST_FMT or fmt
        setup_acc[label], pd_acc[label] = {}, []
        agent, flusher = setup_watch(agent, setup_acc[label], pd_acc[label])
        flushers[label] = flusher
        arms.append((label, spec, fmt, agent))
        print("[arm] %-8s %-28s fmt=%s" % (label, spec, fmt), flush=True)
    mm._FMT = "prompt"
    labels = [x[0] for x in arms]
    if a.baseline not in labels:
        raise SystemExit("--baseline %r is not one of the arms %s" % (a.baseline, labels))

    opps = [o for o in a.opp.split(",") if o]
    # per (label, opp) -> list of one 1/0 per seed-and-seat, in a FIXED order so arms subtract
    # elementwise
    res = {}
    for opp in opps:
        opp_ids = mm.load_deck(opp)
        # The opponent side has been engine_v2 by construction. "reg" hands each opponent deck
        # its own pilot from models/adapters.json, which is the point of the per-deck LoRAs:
        # they exist to BE the opposition, so the protagonist is measured against decks that are
        # played rather than merely held. Left at "engine" the numbers stay comparable with every
        # gate run so far -- switching moves the whole scale, so do not mix the two in one table.
        mm.LAST_FMT = None
        opp_agent, _ = mm.make_agent(a.opp_spec, opp, opp_ids, tuning.get(opp, {}))
        if a.opp_spec != "engine":
            print("[opp] %-22s %s (fmt %s)" % (opp, a.opp_spec, mm.LAST_FMT or "-"), flush=True)
        for label, _spec, _fmt, agent in arms:
            got = []
            for g in range(a.games):
                # Kaggle opens every episode with select=None, and that call is what refills a
                # scorer's per-game time bank. mirror_env never sends it -- play() returns None
                # instead of asking the agent -- so without this the bank accumulates ACROSS
                # games and a 480 s budget is gone after five, leaving the rest of the gate
                # measuring engine_v2 while reporting it as the model.
                try:
                    agent(_EPISODE_START)
                except Exception:
                    pass
                seed = a.seed + g // 2
                mine = g % 2          # alternate seats; both arms see the same (seed, seat)
                r = (play(eng, agent, opp_agent, my_ids, opp_ids, seed, mirror=1) if mine == 0
                     else play(eng, opp_agent, agent, opp_ids, my_ids, seed, mirror=1))
                got.append(1 if r == mine else 0)
            res[(label, opp)] = got
            print("  %-20s %-8s vs %-20s %3d/%3d = %5.1f%%"
                  % (a.deck, label, opp, sum(got), len(got),
                     100.0 * sum(got) / max(1, len(got))), flush=True)

    for _f in flushers.values():
        _f()                            # the last game of the run has no episode start after it

    print("\n-- setup speed, by OUR turn (human template: t1 = Dreepy x3 + Duskull x1-2) --")
    print("  %-8s %9s %9s %9s %11s %11s %10s %14s"
          % ("arm", "t1 dreepy", "t1 bodies", "t2 dreepy", "t2 drakloak", "t3 drakloak",
             "t3 energy", "1st PhantomDive"))
    for label, _s, _f, _ag in arms:
        acc = setup_acc[label]
        m = lambda i, k: (sum(r[k] for r in acc.get(i, [])) / len(acc[i])) if acc.get(i) else 0.0
        pds = pd_acc[label]
        ngames = len(acc.get(1, []))
        pd_s = ("t%.1f in %d/%d" % (sum(pds) / len(pds), len(pds), ngames)) if pds \
            else ("never in %d" % ngames)
        print("  %-8s %9.2f %9.2f %9.2f %11.2f %11.2f %10.2f %14s"
              % (label, m(1, "dreepy"), m(1, "bodies"), m(2, "dreepy"), m(2, "drakloak"),
                 m(3, "drakloak"), m(3, "energy"), pd_s))

    print("\n%-8s %8s %8s %s" % ("arm", "win%", "delta", "vs " + a.baseline))
    out = {"deck": a.deck, "games": a.games, "seed": a.seed, "cells": {}, "arms": {}}
    for label, spec, fmt, _ in arms:
        allg = [v for opp in opps for v in res[(label, opp)]]
        base = [v for opp in opps for v in res[(a.baseline, opp)]]
        wr = 100.0 * sum(allg) / max(1, len(allg))
        # Paired: one difference per game, both arms having played that exact (seed, seat)
        # against that exact opponent. Treating the arms as independent samples would inflate
        # the standard error by roughly sqrt(2) and hide real differences.
        diffs = [x - y for x, y in zip(allg, base)]
        d = 100.0 * sum(diffs) / max(1, len(diffs))
        if len(diffs) > 1:
            m = sum(diffs) / len(diffs)
            sd = math.sqrt(sum((x - m) ** 2 for x in diffs) / (len(diffs) - 1))
            se = 100.0 * sd / math.sqrt(len(diffs))
        else:
            se = float("nan")
        print("%-8s %7.1f%% %+7.2f  +- %.2f%s"
              % (label, wr, d, se, "   (baseline)" if label == a.baseline else ""))
        _acc, _pds = setup_acc[label], pd_acc[label]
        _m = lambda i, k: (sum(r[k] for r in _acc.get(i, [])) / len(_acc[i])) if _acc.get(i) else 0.0
        out["arms"][label] = {"spec": spec, "fmt": fmt, "win_rate": wr,
                              "delta_vs_baseline": d, "se": se, "games": len(allg),
                              "setup": {
                                  "t1_dreepy": _m(1, "dreepy"), "t1_bodies": _m(1, "bodies"),
                                  "t2_dreepy": _m(2, "dreepy"), "t2_drakloak": _m(2, "drakloak"),
                                  "t3_drakloak": _m(3, "drakloak"), "t3_energy": _m(3, "energy"),
                                  "pd_turn": (sum(_pds) / len(_pds)) if _pds else None,
                                  "pd_games": len(_pds), "games_seen": len(_acc.get(1, []))}}
        for opp in opps:
            # The per-game vector, not just the total. Shards split the opponents across
            # processes, and pooling their cell totals cannot rebuild the PAIRED standard
            # error -- that needs each game's result lined up against the same game in the
            # baseline arm, which is exactly what this list preserves.
            out["cells"]["%s|%s" % (label, opp)] = {
                "win": sum(res[(label, opp)]), "games": len(res[(label, opp)]),
                "raw": res[(label, opp)]}
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)
        print("\nwrote %s" % a.out)


if __name__ == "__main__":
    main()
