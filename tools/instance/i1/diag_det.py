"""Why does the determinization accounting fail? Show the actual card-level discrepancy."""
import collections
import os
import sys

ROOT = "/root/ptcg/repo"
for p in (ROOT, os.path.join(ROOT, "cg-lib"), os.path.join(ROOT, "tools")):
    if p not in sys.path:
        sys.path.insert(0, p)

import library                                                   # noqa: E402
from cg.game import battle_start, battle_select, battle_finish    # noqa: E402
from lm.agent import make_lm_agent                                # noqa: E402
import rl_branch                                                  # noqa: E402


def pokemon_fields(obs):
    """Every key present on a board Pokemon -- looking for hidden evolution stacks."""
    cur = obs["current"]
    keys = collections.Counter()
    sample = None
    for pl in cur["players"]:
        for z in ("active", "bench"):
            for x in (pl.get(z) or []):
                if x:
                    keys.update(x.keys())
                    if sample is None or len(x.keys()) > len(sample.keys()):
                        sample = x
    return keys, sample


def diagnose(obs, my_deck, opp_deck):
    cur = obs["current"]
    yi = cur["yourIndex"]
    me, opp = cur["players"][yi], cur["players"][1 - yi]
    my_used = rl_branch._board_ids(me) + [rl_branch._cid(h) for h in (me.get("hand") or [])]
    opp_used = rl_branch._board_ids(opp)

    dm, do = collections.Counter(my_deck), collections.Counter(opp_deck)
    um, uo = collections.Counter(my_used), collections.Counter(opp_used)
    over_me = {i: (um[i], dm[i]) for i in um if um[i] > dm[i]}
    over_opp = {i: (uo[i], do[i]) for i in uo if uo[i] > do[i]}

    want_me = int(me.get("deckCount", 0)) + len(me.get("prize") or [])
    have_me = 60 - sum(um.values())
    want_opp = (int(opp.get("deckCount", 0)) + int(opp.get("handCount", 0))
                + len(opp.get("prize") or []))
    have_opp = 60 - sum(uo.values())
    return dict(over_me=over_me, over_opp=over_opp,
                me_have=have_me, me_want=want_me,
                opp_have=have_opp, opp_want=want_opp,
                looking=cur.get("looking"),
                me_zones=dict(active=len(me.get("active") or []),
                              bench=len(me.get("bench") or []),
                              discard=len(me.get("discard") or []),
                              hand=len(me.get("hand") or [])))


def main():
    pilot, opp_name = "alakazam", "dragapult"
    d_me = library.read_deck(pilot)
    d_op = library.read_deck(opp_name)
    a_me = make_lm_agent(pilot, None, None)
    a_op = make_lm_agent(opp_name, None, None)
    obs, _ = battle_start(d_me, d_op)
    shown = 0
    fieldkeys = collections.Counter()
    sample = None
    try:
        for _ in range(4000):
            cur = obs.get("current")
            if cur is None or cur.get("result", -1) != -1:
                break
            if obs.get("select") is None:
                break
            yi = cur["yourIndex"]
            k, s = pokemon_fields(obs)
            fieldkeys.update(k)
            if s is not None and (sample is None or len(s) > len(sample)):
                sample = s
            if yi == 0 and shown < 4:
                try:
                    rl_branch.unseen_multisets(obs, d_me, d_op)
                except rl_branch.DeterminizationError as e:
                    d = diagnose(obs, d_me, d_op)
                    shown += 1
                    print("--- failure %d: %s" % (shown, e))
                    print("    turn %s  me: %s" % (cur["turn"], d["me_zones"]))
                    print("    me  unseen-by-subtraction %d, engine says deck+prize %d"
                          % (d["me_have"], d["me_want"]))
                    print("    opp unseen-by-subtraction %d, engine says deck+hand+prize %d"
                          % (d["opp_have"], d["opp_want"]))
                    if d["over_me"]:
                        print("    OVER-COUNTED (mine)  id:(counted,in_deck) %s" % d["over_me"])
                    if d["over_opp"]:
                        print("    OVER-COUNTED (opp)   id:(counted,in_deck) %s" % d["over_opp"])
                    if d["looking"]:
                        print("    looking zone NOT empty: %s"
                              % str(d["looking"])[:200])
            obs = battle_select((a_me if yi == 0 else a_op)(obs))
    finally:
        battle_finish()
    print("\n=== Pokemon object fields seen (counts) ===")
    print("  ", dict(fieldkeys))
    if sample:
        print("\n=== richest Pokemon object ===")
        for k, v in sample.items():
            print("   %-16s %s" % (k, str(v)[:110]))


if __name__ == "__main__":
    main()
