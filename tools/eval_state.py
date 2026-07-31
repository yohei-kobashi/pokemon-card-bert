"""Rule-based state evaluator (v2) for search-distilled contrastive data.

Prize differential dominates late game, but is ~0 early, so we add leading
indicators that predict eventual prize swings BEFORE they happen: KO-threat
asymmetry (weighted by the target's prize value), attack readiness, damage
already accumulated on the opponent, and board/tempo terms. Reuses engine_v2's
validated perception (PokemonView/SideView give best_ready_dmg / best_potential
/ loaded; _prize_value gives 1/2/3 for basic/ex/mega).

`evaluate(state, me_idx)` scores a converted observation's `current` state from
player me_idx's view; higher = better for me. Validation (--validate) replays
raw self-play states and reports winner-vs-loser discrimination by game progress.

MEASURED LIMIT (120 matchups x 8 games, n=950/bucket): winner-vs-loser
discrimination is 51.4 / 65.1 / 76.8 / 90.0 % at 25/50/75/100% of game progress.
With every tie-breaker zeroed (prize count alone) it is 3.5 / 28.1 / 59.4 / 83.6 %
-- early states are almost all prize-TIED, which the metric scores as a miss. So
the tie-breakers carry the whole early game and still land at **+1.4pp over the
~50% a random tie-break would give**: this evaluator is effectively BLIND before
the midgame. It earns its keep from ~50% progress on. Weight tuning has not moved
this (see prize_liab below); a learned value function over the same features,
fit on logged self-play outcomes, is the route that could.

Caveat on the metric: it scores ABSOLUTE state value (what an RL shaping potential
needs). The adoption filter in build_sft uses the DELTA across one move, which is a
different property and is not measured here.
"""
import os, sys, gzip, json, random, tarfile, argparse
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "cg-lib")):
    if p not in sys.path:
        sys.path.insert(0, p)
from cg.api import to_observation_class
from agents.engine_v2 import SideView, _prize_value, _CARDS


def _weak_mult(attacker, defender):
    """2x if the attacking Pokémon's type hits the defender's weakness, else 1x."""
    if attacker is None or defender is None:
        return 1.0
    ac, dc = _CARDS.get(attacker.id), _CARDS.get(defender.id)
    if ac and dc and dc.weakness is not None and ac.energyType == dc.weakness:
        return 2.0
    return 1.0


def _stage_invest(v):
    """Board investment from evolution stage (setup tempo signal)."""
    c = _CARDS.get(v.id) if v else None
    if not c:
        return 0
    return 2 if c.stage2 else 1 if c.stage1 else 0

WIN = 1e9             # terminal override: an actual win/loss dominates everything
PRIZE_TIER = 1000.0   # one prize is lexicographically dominant over all tie-breakers
TIE_CLAMP = 499.0     # tie-breakers can never overturn a prize-count lead

# ------------------------- weights (tunable) ------------------------------- #
W = dict(
    ko_now=45.0,      # I can KO opp active THIS turn, × target prize value
    ko_threat=32.0,   # opp can KO my active, × my active prize value (defensive)
    dmg_prog=14.0,    # accumulated damage fraction on opp in-play, × prize value
    dmg_taken=10.0,   # accumulated damage fraction on MY in-play, × prize value
    loaded=12.0,      # count of loaded (attack-ready) attackers diff
    can_attack=10.0,  # my active can attack at all (not a stalled wall)
    stage=6.0,        # evolution-stage investment diff (early setup tempo)
    n_pk=7.0,         # board presence (avoid being benched-out / wiped)
    energy=2.5,       # energy invested toward attacks
    hand=1.0,         # card resources
    board_hp=0.6,     # durability (per 10 hp)
    no_active=6.0,    # penalty for empty active (must burn a promote)
    # What WE concede: the AVERAGE prize value of a side's board. Every other board
    # term (n_pk, board_hp, loaded, stage) is one-sided -- more/bigger is always
    # better -- so nothing priced the other half of the trade: a board of 3-prize
    # Megas hands the opponent the game in 2 KOs instead of 6. Deliberately the
    # AVERAGE, not the sum: penalising the sum would punish simply having bodies,
    # which n_pk correctly rewards. This is the term _is_spare_ex needed and never had.
    # SWEPT AND REJECTED -- keep at 0. winner-vs-loser discrimination (120 matchups,
    # 8 games, n=950/bucket) by weight: 0 -> 51.4/65.1/76.8/90.0 (25/50/75/100%),
    # 10 -> 51.3/64.4/76.5/90.2, 20 -> 48.6/64.4/76.5/90.1, 40 -> 48.6/63.8/76.3/90.0,
    # 80 -> 51.1/61.4/74.7/89.5, 160 -> 51.4/58.9/74.0/89.8. Monotonically WORSE in the
    # mid-game, noise at 25%. The average is too blunt: it cannot tell "my attacker is a
    # Mega" (good, and already paid for by ko_now/board_hp) from "a spare Mega rots on my
    # bench" (bad). A useful liability term would have to be narrow -- prize value of
    # bodies that are idle (no energy, not active, duplicate of one in play).
    prize_liab=0.0,
)
# sweep hook: EVAL_W=prize_liab=40,n_pk=5 overrides weights without editing the file
for _kv in filter(None, os.environ.get("EVAL_W", "").split(",")):
    _k, _, _v = _kv.partition("=")
    if _k in W:
        W[_k] = float(_v)


def _side(sv):
    """Aggregate scalar features from a SideView."""
    inplay = sv.inplay()
    dmg_prog = 0.0            # opp-perspective: how close each pk is to KO, × prize
    prize_on_board = 0.0
    loaded = 0
    for v in inplay:
        if v is None or v.card is None:
            continue
        taken = max(0, v.max_hp - v.hp)
        frac = taken / v.max_hp if v.max_hp else 0.0
        pv = _prize_value(v.pk)
        dmg_prog += frac * pv
        prize_on_board += pv
        if v.best_potential_dmg > 0 and v.loaded:
            loaded += 1
    return dict(
        prizes_left=sv.prizes_left,
        prize_on_board=prize_on_board,   # was computed and thrown away
        n_pk=len(inplay),
        energy=sv.energy_in_play,
        hand=sv.hand_count,
        board_hp=sum(v.hp for v in inplay if v),
        dmg_frac=dmg_prog,          # damage accumulated ON this side (bad for it)
        loaded=loaded,
        stage=sum(_stage_invest(v) for v in inplay),
        active=sv.active,
    )


def evaluate(state, me_idx):
    me_sv = SideView(state.players[me_idx], {}, True)
    op_sv = SideView(state.players[1 - me_idx], {}, False)
    me, op = _side(me_sv), _side(op_sv)
    # ---- terminal override: winning is the top priority, absolutely ---------
    me_wiped = me["active"] is None and me["n_pk"] == 0
    op_wiped = op["active"] is None and op["n_pk"] == 0
    if me["prizes_left"] == 0 or (op_wiped and me["active"] is not None):
        return WIN                                  # I took my last prize / opp benched out
    if op["prizes_left"] == 0 or (me_wiped and op["active"] is not None):
        return -WIN                                 # opponent won
    # ---- prize count is lexicographically dominant over all tie-breakers -----
    prize_lead = PRIZE_TIER * (op["prizes_left"] - me["prizes_left"])
    s = 0.0
    # KO-threat asymmetry, weakness-adjusted, weighted by prize value at stake
    if me["active"] and op["active"]:
        eff = me["active"].best_ready_dmg * _weak_mult(me["active"], op["active"])
        if eff >= op["active"].hp:
            s += W["ko_now"] * _prize_value(op["active"].pk)
    if op["active"] and me["active"]:
        eff = op["active"].best_ready_dmg * _weak_mult(op["active"], me["active"])
        if eff >= me["active"].hp:
            s -= W["ko_threat"] * _prize_value(me["active"].pk)
    # damage progress: opp's accumulated damage is good for me; mine is bad
    s += W["dmg_prog"] * op["dmg_frac"]
    s -= W["dmg_taken"] * me["dmg_frac"]
    s += W["loaded"] * (me["loaded"] - op["loaded"])
    s += W["stage"] * (me["stage"] - op["stage"])
    if me["active"] and me["active"].best_potential_dmg > 0:
        s += W["can_attack"]
    if op["active"] and op["active"].best_potential_dmg > 0:
        s -= W["can_attack"]
    s += W["n_pk"] * (me["n_pk"] - op["n_pk"])
    if W["prize_liab"]:
        me_pv = me["prize_on_board"] / max(me["n_pk"], 1)
        op_pv = op["prize_on_board"] / max(op["n_pk"], 1)
        s -= W["prize_liab"] * (me_pv - op_pv)     # cheap board = good for me
    s += W["energy"] * (me["energy"] - op["energy"])
    s += W["hand"] * (me["hand"] - op["hand"])
    s += W["board_hp"] * (me["board_hp"] - op["board_hp"]) / 10.0
    if not me["active"]:
        s -= W["no_active"]
    if not op["active"]:
        s += W["no_active"]
    # tie-breakers resolve WITHIN a prize tier; never overturn a prize lead
    s = max(-TIE_CLAMP, min(TIE_CLAMP, s))
    return prize_lead + s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tar", default="data/kaggle_out/v24_full/selfplay_v24_full_raw.tar")
    ap.add_argument("--matchups", type=int, default=80)
    ap.add_argument("--games", type=int, default=6)
    args = ap.parse_args()
    tf = tarfile.open(args.tar)
    members = [m for m in tf.getmembers() if m.name.endswith(".jsonl.gz")]
    random.Random(1).shuffle(members)
    members = members[:args.matchups]

    buckets = {q: [0, 0] for q in (25, 50, 75, 100)}
    final_margin = []
    term = [0, 0]   # [terminal-ish states found, evaluator sign correct]
    for m in members:
        f = gzip.open(tf.extractfile(m), "rt")
        header = None
        games = {}
        for line in f:
            rec = json.loads(line)
            if rec.get("kind") == "game":
                header = rec
                continue
            if header is None or rec.get("kind") != "step":
                continue
            gid = rec["game_id"]
            if gid not in games and len(games) >= args.games:
                continue
            g = games.setdefault(gid, {"winner": header["winner"], "steps": []})
            g["steps"].append(rec["obs"])
        for gid, g in list(games.items())[:args.games]:
            steps = g["steps"]
            Wn = g["winner"]
            if len(steps) < 4 or Wn not in (0, 1):
                continue
            N = len(steps)
            for q in (25, 50, 75, 100):
                idx = min(N - 1, int(N * q / 100) - 1)
                obs = to_observation_class(steps[idx])
                st = obs.current
                if not st or len(st.players or []) != 2:
                    continue
                try:
                    ew = evaluate(st, Wn)
                    el = evaluate(st, 1 - Wn)
                except Exception:
                    continue
                buckets[q][0] += 1
                if ew > el:
                    buckets[q][1] += 1
                if q == 100:
                    final_margin.append(ew - el)
            # terminal-state override check: any step where a player is decided
            for ob in steps:
                st = to_observation_class(ob).current
                if not st or len(st.players or []) != 2:
                    continue
                p = st.players
                def decided(i):
                    pr = len(p[i].prize or []) == 0
                    inplay = [x for x in (list(p[i].active or []) + list(p[i].bench or [])) if x]
                    wiped = not inplay and [x for x in (list(p[1-i].active or []) + list(p[1-i].bench or [])) if x]
                    return pr, wiped
                winner_won = decided(Wn)
                loser_lost = decided(1 - Wn)
                if winner_won[0] or loser_lost[1]:      # winner took last prize / loser wiped
                    term[0] += 1
                    if evaluate(st, Wn) >= WIN:
                        term[1] += 1
    print("Game-progress | winner-eval > loser-eval")
    for q in (25, 50, 75, 100):
        n, w = buckets[q]
        print(f"  {q:3d}% : {w}/{n} = {100*w/max(n,1):.1f}%")
    fm = sorted(final_margin)
    print(f"\nfinal-state margin: median={fm[len(fm)//2]:.0f} "
          f"p10={fm[len(fm)//10]:.0f} p90={fm[len(fm)*9//10]:.0f}  (n={len(fm)})")
    neg = sum(1 for x in fm if x <= 0)
    print(f"final states where winner NOT ahead: {neg}/{len(fm)} ({100*neg/len(fm):.1f}%)")
    print(f"\nterminal-override: decided states where evaluator returns +WIN for winner: "
          f"{term[1]}/{term[0]} ({100*term[1]/max(term[0],1):.1f}%)")


if __name__ == "__main__":
    main()
