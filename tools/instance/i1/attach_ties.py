"""Is "the reranker picks attach targets at chance" a model failure or a LABEL artifact?

`attach-decisions-at-chance` measured attach top1 at 16-29% against a 14.1% chance level while
every other decision kind runs +25 to +79pt above chance, and listed as untested: "does
engine_v2 tie-break arbitrarily among equivalent attach targets?"

A rendered example shows why that matters:

    ME A[c343:80/80|C] B[c344*:70/70,c344*:70/70]
       2=attach:c18@ACTIVE0  3=attach:c18@BENCH0  4=attach:c18@BENCH1

BENCH0 and BENCH1 are the same card at the same HP with the same attachments, so the two moves
are interchangeable and engine_v2's choice between them is arbitrary. Top1 against an arbitrary
label cannot reach 100%: the ceiling is 1/(size of the tied group holding the label). A model
that ranks perfectly still scores at the ceiling, so a low top1 does NOT prove the model cannot
tell the candidates apart.

This computes, over real decisions, (a) how many attach candidates are genuinely
distinguishable, and (b) the achievable top1 ceiling. Two states count as the same target when
the target Pokemon has the same card id, HP, damage, attached-energy multiset, tool multiset and
status flags -- i.e. nothing in the observation separates them.

Run:  CUDA_VISIBLE_DEVICES="" python attach_ties.py [games] [workers]
"""
import collections
import json
import os
import random
import sys

ROOT = "/root/ptcg/repo"
for p in (ROOT, os.path.join(ROOT, "cg-lib"), os.path.join(ROOT, "tools")):
    if p not in sys.path:
        sys.path.insert(0, p)

GAMES = int(sys.argv[1]) if len(sys.argv) > 1 else 300
WORKERS = int(sys.argv[2]) if len(sys.argv) > 2 else 64
PAIRS = [("crustle", "alakazam"), ("alakazam", "crustle"), ("mega_lucario", "dragapult"),
         ("dragapult", "crustle"), ("rockets_mewtwo", "alakazam"), ("marnie_grimmsnarl", "crustle")]


def _cid(c):
    return c.get("id") if isinstance(c, dict) else None


def _target_fingerprint(pl, area, idx):
    """Everything the observation says about a target slot. Equal fingerprint = the model has
    nothing to separate the two candidates with, and neither does the engine."""
    if area is None:
        return None
    grp = (pl.get("active") or []) if str(area).upper().startswith("A") else (pl.get("bench") or [])
    # area_name gives ACTIVE/BENCH; index into the matching list
    if idx is None:
        idx = 0
    if idx >= len(grp):
        return None
    x = grp[idx]
    if not isinstance(x, dict):
        return None
    ens = sorted(_cid(e) for e in (x.get("energyCards") or x.get("energies") or []))
    tls = sorted(_cid(t) for t in (x.get("tools") or []))
    pre = sorted(_cid(t) for t in (x.get("preEvolution") or []))
    return json.dumps([x.get("id"), x.get("hp"), x.get("maxHp"), x.get("damage"),
                       ens, tls, pre,
                       bool(x.get("asleep")), bool(x.get("confused")),
                       bool(x.get("paralyzed")), bool(x.get("poisoned")),
                       bool(x.get("burned"))])


def one_game(task):
    pilot, opp, seed = task
    import library
    from cg.game import battle_start, battle_select, battle_finish
    from lm.agent import make_lm_agent
    from lm.actions import encode_option
    from lm import vocab

    try:
        d_me, d_op = library.read_deck(pilot), library.read_deck(opp)
    except Exception:
        return []
    a_me = make_lm_agent(pilot, None, None)
    a_op = make_lm_agent(opp, None, None)
    pilot_i = seed % 2
    d0, d1 = (d_me, d_op) if pilot_i == 0 else (d_op, d_me)
    obs, _ = battle_start(d0, d1)
    if obs is None:
        return []
    out = []
    try:
        for _ in range(4000):
            cur = obs.get("current")
            if cur is None or cur.get("result", -1) != -1:
                break
            sel = obs.get("select")
            if sel is None:
                break
            yi = cur["yourIndex"]
            if yi == pilot_i:
                opts = sel.get("option") or []
                enc = [encode_option(o, obs) for o in opts]
                ai = [i for i, e in enumerate(enc) if e.startswith("attach:")]
                if len(ai) >= 2:
                    me = cur["players"][yi]
                    # group attach candidates by (energy card played, target fingerprint)
                    groups = collections.defaultdict(list)
                    for i in ai:
                        o = opts[i]
                        area = o.get("inPlayArea")
                        an = vocab.area_name(area) if area is not None else None
                        fp = _target_fingerprint(me, an, o.get("inPlayIndex"))
                        key = (enc[i].split("@")[0], fp)
                        groups[key].append(i)
                    choice = a_me(obs)
                    ch = choice[0] if isinstance(choice, (list, tuple)) and choice else None
                    picked_group = None
                    for k, v in groups.items():
                        if ch in v:
                            picked_group = len(v)
                            break
                    out.append(dict(n_opts=len(opts), n_attach=len(ai),
                                    n_distinct=len(groups),
                                    sizes=sorted(len(v) for v in groups.values()),
                                    picked_group=picked_group,
                                    chose_attach=1 if picked_group else 0))
            obs = battle_select((a_me if yi == pilot_i else a_op)(obs))
    except Exception:
        pass
    finally:
        try:
            battle_finish()
        except Exception:
            pass
    return out


def main():
    tasks = [(p, o, s) for (p, o) in PAIRS for s in range(GAMES // len(PAIRS))]
    random.Random(0).shuffle(tasks)
    print("attach_ties: %d games over %d pairs, %d workers" % (len(tasks), len(PAIRS), WORKERS),
          flush=True)
    import multiprocessing as mp
    recs = []
    with mp.Pool(WORKERS) as pool:
        for rr in pool.imap_unordered(one_game, tasks, chunksize=1):
            recs.extend(rr)
    print("\n%d decisions with >=2 attach candidates" % len(recs))
    if not recs:
        return
    na = sum(r["n_attach"] for r in recs) / len(recs)
    nd = sum(r["n_distinct"] for r in recs) / len(recs)
    print("  attach candidates per decision      : %.2f" % na)
    print("  DISTINGUISHABLE attach targets       : %.2f  (%.1f%% of them)"
          % (nd, 100.0 * nd / max(1e-9, na)))
    dup = [r for r in recs if r["n_distinct"] < r["n_attach"]]
    print("  decisions holding interchangeable    : %d / %d  (%.1f%%)"
          % (len(dup), len(recs), 100.0 * len(dup) / len(recs)))

    # top1 ceiling: when the engine's label falls in a tied group of size g, a perfect model
    # scores 1/g on that row
    sel = [r for r in recs if r["picked_group"]]
    if sel:
        ceil = sum(1.0 / r["picked_group"] for r in sel) / len(sel)
        big = collections.Counter(r["picked_group"] for r in sel)
        print("\n  of the %d decisions where the engine chose an attach:" % len(sel))
        print("  TOP1 CEILING for a perfect model     : %.1f%%" % (100.0 * ceil))
        print("  size of the tied group it landed in  :",
              dict(sorted(big.items())))
        print("\n  (measured attach top1 was 29.1%% for v37; the ceiling above is what a model")
        print("   that ranks PERFECTLY would score, because the label is arbitrary within a tie)")


if __name__ == "__main__":
    main()
