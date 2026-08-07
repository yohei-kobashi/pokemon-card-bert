"""RL curriculum + hyper-parameters for the post-SFT LM agent.

TWO independent knobs:
  O = opponent distribution  (THE OBJECTIVE — who we must beat)
  P = pilot set              (which decks the LEARNING policy plays)

CURRENT CURRICULUM — two stages (user 2026-08-06, docs/rl_stages_v2.md):

  Stage 1  broad-11    P = STAGE_C_TARGETS (11);  O = the same 11, live-weighted
                       an 11x11 grid; the diagonal (MIRROR_FRAC) is same-shuffle mirror
  Stage 2  dragapult   P = {dragapult, dragapult_dusknoir};  O = the same 11

  Both stages learn from PLAYOUT-MEASURED Q labels (docs/rl_mirror_design.md), not from
  GRPO's per-game credit. That is the whole point of the revision: a uniform per-game
  reward is the DIAGNOSED cause of the two 12-round plateaus
  ([[rl-stage-a-plateau-diagnosis]], [[rl-plateau-five-refutations]]), so repeating it a
  third time is a refuted experiment, not a new one.

DEPRECATED — Stage A/B/C (2026-07-22 .. 2026-08-06). Kept so old runs can be reproduced;
do not start new work on them.

  Stage A  broad climb          P = all shippable decks;  O = heuristic-heavy -> self-play
  Stage B  meta realignment     P = all;                  O = reweighted to the LIVE meta
  Stage C  targeted special.    P = the TARGET deck(s) ONLY;  O = LIVE meta, NOT narrowed

  Why A is retired: it went 12 rounds flat, twice. And the breadth it was buying was
  illusory -- a training mix is ~77% base imitation pool covering all 65 decks, so the RL
  share is ~23%. Spread over 65 decks that is 0.35% per deck; concentrated on 11 it is
  2.05%, a 6x dose. Widening P never added breadth, it diluted the signal below resolution.
  Catastrophic forgetting is structurally unlikely because the base share does that job.
  Why B is retired: it only reweighted O to the live meta, which Stage 1 does from round 1.

Everything here is data, editable per run. The loop (tools/rl_loop.sh) reads a stage name
and asks this module for that stage's O, P, opponent-agent mix, and stop criterion.
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _all_decks():
    import library
    bad = ("_old", "_sample")
    return [d for d in library.list_decks() if not d.endswith(bad)]


# --- PROMPT FORMAT: one source of truth for the whole RL stack ---------------------
# RL initialises from an SFT checkpoint, so rollout and training MUST render states the
# way that checkpoint was trained. Every one of these knobs is silent when wrong -- no
# exception, no size change, just a policy scored on inputs it has never seen -- and the
# same class of bug shipped twice on 2026-07-27 (main.py not passing deck_mode/shuffle;
# lm/identify losing the ID segment in the bundle). RL is worse than inference here: a
# mismatched prompt does not merely score badly, it trains on the wrong thing.
#
# These values match rerank_gte_v37 == build_rerank --glossary none --deck-mode remaining
# --deck-shuffle. Change them ONLY together with the checkpoint they describe.
# v39 (2026-07-31). deck_mode="roles" groups DECK by prompt_roles with ids sorted inside a
# group, which removes the file-order fingerprint structurally, so deck_shuffle is off.
# board_facts adds `need:N` (type-aware energies short of any damaging attack) and `rt:N`
# on both sides: measured, the v37 prompt contained attack costs 0 times and retreat costs
# 0 times in 40,000 samples, and `attach` (20.3% of ceiling) and `retreat` (33.6%, chance
# 31.2%) are exactly the two kinds whose decisive input was therefore missing.
# identify="op" drops `ID ME`, redundant given DECK[].
# menu_dedup (v40, 2026-08-02) renders one menu entry per ACT. The cross-encoder is trained to
# rank the deduped candidate list, so the un-deduped menu was showing it 7.08 options for 5.36
# acts; on decisions that offer an attach the menu's attach share is inflated +4 to +6pt with a
# standard deviation of 9-11pt, and the size of that inflation is set by how many copies of the
# energy happen to be in hand -- noise with no bearing on whether attaching is right.
# CHECKPOINTS BEFORE v40 (rerank_gte_v39, rerank_loop2..4) WERE TRAINED WITHOUT IT: screen those
# with menu_dedup=False or they are scored on a format they never saw.
# hidden_facts (v41, 2026-08-05) adds the engine's hidden effect state: live damage with a KO
# marker per attack option, the post-attach `need` per attach option, the incoming threat on our
# Active, and the identity of Special Energies whose effect persists. +9.08 tokens/decision.
# Flipped here on 2026-08-05 when the v40 reranker loop was stopped -- rounds 4->8 measured
# -0.71pt +- 1.33 paired (flat), so there was nothing left to keep comparable. Everything from
# here on -- generation, screens, DAgger -- is v41.
PROMPT_FMT = dict(glossary="none", deck_mode="roles", deck_shuffle=False,
                  board_facts=True, identify="op", menu_dedup=True, hidden_facts=True)
PROMPT_FMT_V37 = dict(glossary="none", deck_mode="remaining", deck_shuffle=True)

# v41 (2026-08-05): everything v40 has, plus the engine's hidden effect state -- live damage
# with a KO marker on each attack option, the post-attach `need` on each attach option, the
# incoming threat on our Active, and the identity of Special Energies whose effect persists.
# See [[hidden-state-from-blob]]; +9.08 tokens/decision measured over 46,377 decisions.
#
PROMPT_FMT_V41 = dict(PROMPT_FMT)                       # alias; PROMPT_FMT IS v41 now
PROMPT_FMT_V40 = {k: v for k, v in PROMPT_FMT.items() if k != "hidden_facts"}

# Stamped on every pool row by build_rerank so the two can be told apart -- and so the old ones
# can be deleted -- without guessing from the text. Rows written before this existed have no
# `pfmt` key at all, which is exactly what identifies them.
PROMPT_VERSION = "v41"
PROMPT_VERSIONS = {"current": PROMPT_VERSION, "v41": "v41", "v40": "v40"}


# --- matched-sampling target (Stage A) --------------------------------------------
# rl_ratings.expected_wr is measured from engine_v2 vs engine_v2, so it says what the DECK
# does, not what the LM does with it. The learning policy is weaker than the engine on the
# same deck -- measured 2026-07-28 at ~12pt (crustle_stall -6.1, mega_lucario -15.8,
# alakazam -16.6) -- so a genuinely even game for the LM is a matchup the ENGINE wins ~62%.
# Aiming at 50 (the pre-2026-07-28 hard-coded value) hands the LM ~38% games; aiming below
# 50 makes it worse still. Revisit as RL closes the gap: the right long-run fix is to re-rate
# with the LM piloting, and this constant is the cheap stand-in until then.
LM_ENGINE_DEFICIT = 12.0
MATCH_TARGET_WR = 50.0 + LM_ENGINE_DEFICIT


# --- LIVE meta weights (opponent frequency) ---------------------------------------
# Update from the latest scrape; unlisted decks share the residual mass uniformly.
# REFRESHED 2026-08-06 from the 2026-08-05 top-500 scout
# (docs/meta/leaderboard_top500_2026-08-05.md; raw JSON scratchpad_replays/distribution.json).
# 494/500 teams classified, entries >= 0.004 cover 99.2% of them. `ogerpon_mono` and
# `dudunsparce_box` were scouted as `OTHER:` -- the classifier predates those decks -- and are
# folded back in by signature.
#
# Shifts vs the 2026-07-23 weights this replaces:
#   marnie_grimmsnarl  0.172 -> 0.358   a THIRD of the ladder, up 2.1x
#   alakazam_nz        0.212 -> 0.134   was #1, now #2; the Alakazam family roughly halved
#   archaludon         0.094 -> 0.024   collapsed
#   rockets_*          0.038 -> 0.020   and ZERO in the top 100 (see [[leaderboard-top100-meta-gap]])
#   ogerpon_mono          --  -> 0.057  newly expressible
#   dudunsparce_box       --  -> 0.024
#   staryu / iono_bellibolt / dragapult_dusknoir dropped out (below 0.004)
#
# Weight is POPULATION share and says nothing about how high an archetype reaches -- see
# STAGE_C_TARGETS below, where peak rank is the second criterion.
LIVE_META = {
    "marnie_grimmsnarl": 0.358, "alakazam_nz": 0.134, "crustle_geco": 0.057,
    "ogerpon_mono": 0.057, "crustle": 0.047, "alakazam": 0.040, "dragapult": 0.040,
    "cynthia_garchomp": 0.038, "mega_lucario_tr": 0.028, "mega_lopunny": 0.028,
    "mega_lucario": 0.028, "dudunsparce_box": 0.024, "archaludon": 0.024,
    "omatsuri": 0.018, "alakazam_nz_fez": 0.010, "raging_bolt": 0.008,
    "rockets_mewtwo": 0.008, "rockets_spidops": 0.008, "hydrapple": 0.006,
    "mega_starmie": 0.006, "ns_zoroark": 0.006, "mega_venusaur": 0.004,
    "crustle_stall": 0.004, "comfey_yveltal": 0.004, "rockets_honchkrow": 0.004,
}

# --- Stage-C target pilots (user-confirmed 2026-08-06; one RL run per target) ----------
# ELEVEN decks, chosen off the 2026-08-05 top-500 scout so that the SAME set is both the RL
# pilot roster (docs/rl_mirror_design.md) and the submission candidate list. Coverage:
#
#     393 / 500 of the top 500  = 78.6%      79 / 100 of the top 100      16 / 20 of the top 20
#
# Selected on TWO criteria, because usage count alone gets it wrong:
#   population  how many opponents run it  -> marnie_grimmsnarl 177, alakazam_nz 66, ...
#   frontier    how high the archetype gets -> mega_lucario_tr is only 14/500 but holds #2 AND
#               #3 (lb 1186.6 / 1160.7), and our decks/mega_lucario_tr.csv matches those two
#               builds at 1.00, so the whole gap there is piloting. It was nearly cut on
#               headcount; peak rank is the signal that saved it.
#
# dragapult / dragapult_dusknoir stay from the 2026-07-22 roster (human BDIF, high skill
# ceiling; the Dusknoir build patches the Grimmsnarl matchup for our Grimmsnarl-heavy field).
# dragapult_dusknoir is the one member with ZERO leaderboard presence and is kept deliberately
# as a submission candidate.
#
# DROPPED 2026-08-06: rockets_mewtwo, rockets_honchkrow. Team Rocket is 10/500 = 2.0% with
# ZERO teams in the top 100 (best #123 at lb 985.0 vs a 1002.2 cutoff), down from #4 at 1185.1
# in July. [[leaderboard-top100-meta-gap]]'s "Team Rocket is the missing package" is reversed
# there; do not re-add on the strength of that paragraph.
#
# Opponents in Stage C are NOT narrowed — they stay the live-frequency field (see stage("C")).
STAGE_C_TARGETS = ["dragapult", "dragapult_dusknoir", "marnie_grimmsnarl",
                   "alakazam_nz", "alakazam", "crustle_geco", "crustle",
                   "ogerpon_mono", "dudunsparce_box", "cynthia_garchomp",
                   "mega_lucario_tr"]

# --- Two-stage curriculum knobs (docs/rl_stages_v2.md) --------------------------------
# Stage 1 plays an 11x11 grid. MIRROR_FRAC is the share spent on the DIAGONAL (same deck
# both seats, same shuffle order). It is not the objective -- the ladder is off-diagonal --
# but the gate is a mirror paired screen, and mirror is the only setting whose null is
# exactly 0 ([[mirror-shuffle-mode]]). Left to the weight product the diagonal would be
# ~1/11 of a row and too thin to match the gate's distribution.
MIRROR_FRAC = 0.25
PLAYOUTS_PER_BRANCH = 16        # matches attach_label.py; below this the permutation null
                                # rejects nearly everything and the round yields no labels

# Share of Phase-1 state collection played by the RERANKER (DeBERTa) rather than the 4B
# decoder. The 4B is the RL vehicle -- it is unshippable at the 197.66 MiB cap and is used
# (a) to measure the method's ceiling cheaply and (b) as a data factory for DeBERTa. The Q
# label does not care who collected the state, but two things do: the state distribution
# (4B 53.5% vs DeBERTa 30.6% fleet mean are different policies reaching different boards)
# and which decisions the low-margin filter selects. Round 1 should MEASURE the overlap
# between the two models' selected branch points and then set this: high overlap -> 0.0,
# low overlap -> raise it. 0.3 is the starting guess, not a measurement.
DEBERTA_COLLECT_FRAC = 0.3

# --- TIME BOX (user 2026-08-06: "4B の Stage 1+2 は3〜4日でどこまで行けるか") ----------
# Competition deadline: 2026-08-16 23:59 UTC = 2026-08-17 08:59 JST (a Monday MORNING, so
# Sunday is effectively the last working day).
#
# Round length is the MEASURED one, not an estimate. instance2, r6 -> r7, steady state
# (excluding the one-off baseline re-screen):
#
#     screen 2.13h + collect/mix 1.75h + train 7.22h = 11.1h per round
#
# Phase 2 (branch + 16 playouts, ~3h) runs on instance1's CPU and never touches instance2's
# critical path, so 11.1h is the whole budget unit. 2026-08-06 06:30 -> 2026-08-10 00:00 is
# 89.5h = 8 rounds; budgeted as 7 (4 + 3) with one round of slack for slippage.
#
# THE CALENDAR STOP IS THE BINDING ONE. A plateau test that never fires would eat the entire
# box and Stage 2 would never run. STAGE1_DEADLINE_UTC forces the handover whatever the
# metric says.
RL_DEADLINE_UTC = "2026-08-10T00:00:00Z"        # hard stop for ALL 4B RL
STAGE1_DEADLINE_UTC = "2026-08-08T12:00:00Z"    # hand over to Stage 2 regardless of plateau
STAGE1_ROUNDS = 4
STAGE2_ROUNDS = 3

# GATE SIZING -- the single easiest thing to get wrong in this revision.
# Measured: i2 r7 scored +0.0077 +- 0.0114 on 63 decks x 40 games, so the per-deck delta
# sd is 1.14*sqrt(63) = 9.05pt. Two independent 40-game screens would predict
# sqrt(2*0.25/40) = 11.2pt, and 9.05 < 11.2, so the delta is SAMPLING-dominated and the
# true between-deck component is ~0. sd therefore falls as 1/sqrt(G):
#
#     63 decks x  40 games = 2,520 games -> sd 9.05pt -> SE 1.14pt   (today)
#     11 decks x 229 games = 2,519 games -> sd 3.78pt -> SE 1.14pt   (same budget, same SE)
#     11 decks x  40 games =   440 games -> sd 9.05pt -> SE 2.73pt   (naive narrowing: 2.4x worse)
#
# So narrowing the gate to 11 decks is FREE if the games are redistributed, and useless if
# they are not -- at SE 2.73 the "<= +1pt for 2 rounds" stop rule cannot fire on evidence.
# Re-measure sd every 2 rounds: if Stage 1 lifts some decks and not others the true
# between-deck component grows, sd climbs past 11.2pt, and deck COUNT starts to bind.
GATE_GAMES_PER_DECK = 229


# Stage-A FOCUS: strengthen the WEAK members of the submittable/meta set FIRST (user
# 2026-07-24). RL trains the PILOT, not the deck list, so FOCUS is the decks whose remaining
# gap is PILOTING (the Stage-C targets we actually submit), NOT the fleet's deck-broken tail
# (bottom-12 are DECK problems, pilot already fires 91-100% -> RL-inert, see
# [[fleet-roundrobin-and-weak-decks]]). Those tail decks stay only in the breadth floor.
STAGE_A_FOCUS_MASS = 0.75      # share of pilot draws spent on the FOCUS set
STAGE_A_TARGET_WR = 60.0       # a target at/above this is "strong enough"; below => headroom
STAGE_A_HEADROOM_FLOOR = 5.0   # min per-target mass so even a strong target keeps some RL


def _stageA_pilot_weights(decks):
    """Stage-A pilot distribution: concentrate FOCUS_MASS on STAGE_C_TARGETS, weighted by
    HEADROOM (weakest target gets the most RL), and spread the remainder uniformly over the
    rest of the fleet as a breadth floor (general competence + anti-forgetting). Headroom uses
    the latest round-robin strength (same table the matched sampler reads); with no table,
    FOCUS is uniform. The matched sampler still difficulty-matches each pilot to ~50% games,
    so a heavily-weighted weak target learns from CONTRASTIVE games, not blowouts."""
    import rl_ratings
    strength, _ = rl_ratings.load()
    focus = [d for d in decks if d in set(STAGE_C_TARGETS)]
    rest = [d for d in decks if d not in set(STAGE_C_TARGETS)]
    pw = {}
    if focus:
        head = {d: max(STAGE_A_HEADROOM_FLOOR,
                       STAGE_A_TARGET_WR - strength.get(d, STAGE_A_TARGET_WR))
                for d in focus}                       # missing strength => neutral (floor tilt)
        tot = sum(head.values())
        for d in focus:
            pw[d] = STAGE_A_FOCUS_MASS * head[d] / tot
    floor = (1.0 - STAGE_A_FOCUS_MASS) if focus else 1.0
    for d in rest:
        pw[d] = floor / len(rest) if rest else 0.0
    if not rest:                                      # degenerate: everything is focus
        s = sum(pw.values()) or 1.0
        pw = {d: w / s for d, w in pw.items()}
    return pw


# Where the two-stage curriculum reads its headroom from. This is the LM's own mirror
# screen (`mirror_match --a engine --b hf:<ckpt> --mirror`, merged), i.e. how well the
# POLICY pilots each deck. Written by the loop after every gate.
LM_SCREEN = os.path.join(ROOT, "evaluations", "lm_mirror_screen.json")
# Below this many games per deck the file is a screen, not a gate, and must not drive weights.
# 150 is the established per-matchup floor ([[deck-status-and-live-scores]]); the Stage-1 gate
# runs 229 ([[rl-curriculum-two-stage]]) and tools/stage1_loop.sh copies each round's gate here.
MIN_SCREEN_GAMES = 150

# Which (deck, kind) DECISIONS the Q-label budget buys, as opposed to which decks the games are
# played with. Produced by tools/retarget_cells.py from the observed gaps
# (diag_lm_losses.py --targets) re-weighted by the counterfactual (price_targets.py, both sides).
#
# The re-weighting is not cosmetic. Priced 2026-08-06 on the six largest observed gaps:
#
#   end       the largest observed effect in the fleet (z -10.90 pooled over 11 decks) and NOT
#             actionable on the side it was first priced -- where the LM declines to end its
#             turn, declining is right (dQ -0.10 to -0.38). It IS actionable on the other side:
#             where the LM ends, it should not have (pooled -0.0696, z -4.28 over 4 decks).
#             The policy ends its turn too early; it does not decline too often.
#   evolve    mega_lucario_tr, the one cell that is a mistake on the DECLINE side (+0.079,
#             z +2.88; take side agrees at +0.082). Its observed gap says winners evolve 29.9pp
#             LESS, so here the correlation and the counterfactual point opposite ways and the
#             counterfactual is the one that survives a controlled test.
#
# Regenerate whenever a new pricing run lands; the loop reads the file, not this constant.
QLABEL_TARGETS = os.path.join(ROOT, "evaluations", "lm_targets_priced.json")

# No single (deck, kind) cell may exceed this share of a round's branch budget. Uncapped, the
# one validated cell took 43.9%, which is the shape [[narrow-dagger-overfits]] measured: one-deck
# concentration moved the target +11.9pt and the fleet -2.75pt.
QLABEL_MAX_CELL_SHARE = 0.25
# Cap on the valued share of a training round's mix. The valued stream's size is set by how many
# batches happened to arrive from the other machine, so without this a slow round silently
# raises the fraction. 0.10 is where the attach labels sat when they paid off (9.2% of v40).
VALUED_MAX_FRAC = 0.10


def _stage12_pilot_weights(decks, screen_path=None):
    """Headroom weights for Stage 1, measured as the LM's OWN winrate per deck.

    NOT rl_ratings. That table is engine_v2's deck-vs-deck round-robin -- how strong the
    DECK is -- and using it here inverts the allocation. Measured on the real numbers:

        deck                rl_ratings  LM mirror   old w    correct
        dragapult_dusknoir     34.8       57.5%     0.303    least (LM's 2nd best)
        dragapult              46.9       30.0%     0.157    MOST  (LM's worst)
        dudunsparce_box         --        37.5%     0.060    2nd   (LM's 2nd worst)

    rl_ratings would have spent 30% of the RL on the deck the LM already plays second best
    and 6% on the second worst, because a weak DECK and a badly-PILOTED deck are different
    things ([[fleet-roundrobin-and-weak-decks]], [[weak-decks-pilot-vs-structural]]) and RL
    only moves the second. rl_ratings stays in use for Stage A's difficulty MATCHMAKING,
    which is a question about the deck.

    Falls back to uniform when no screen exists, and says so -- a silent fallback here looks
    exactly like a working run.
    """
    p = screen_path or LM_SCREEN
    wr, games = {}, []
    try:
        with open(p) as f:
            for d, v in (json.load(f).get("decks") or {}).items():
                if isinstance(v, dict) and v.get("p") is not None:
                    wr[d] = 100.0 * float(v["p"])
                    games.append(int(v.get("w", 0)) + int(v.get("l", 0)) + int(v.get("d", 0)))
    except Exception:
        pass
    # A THIN SCREEN CANNOT CARRY THIS ALLOCATION. At 40 games/deck the per-deck SE is ~8pt, and
    # the weights below are a linear function of the reading, so the noise IS the allocation.
    # Measured on the same checkpoint: the 40-game screen put dudunsparce_box at 37.5% (2nd
    # worst -> weight 0.200) and the 229-game gate put it at 55.3% (3rd best -> 0.043), while
    # crustle at 43.4% sat on the 0.044 floor and should have had 0.141. Same failure mode as
    # reading rl_ratings: an allocation driven by a measurement too coarse to support it.
    if games:
        med = sorted(games)[len(games) // 2]
        if med < MIN_SCREEN_GAMES:
            import sys
            print("[rl_config] %s has only ~%d games/deck (floor %d). Per-deck SE is ~%.0fpt and "
                  "these weights are linear in it -- point this at a gate, not a screen."
                  % (p, med, MIN_SCREEN_GAMES, 50.0 / max(1.0, med ** 0.5)), file=sys.stderr)
    if not wr:
        import sys
        print("[rl_config] no LM screen at %s -- Stage-1 pilot weights fall back to UNIFORM. "
              "Headroom targeting is OFF until the loop writes one." % p, file=sys.stderr)
        return {d: 1.0 / len(decks) for d in decks}
    missing = [d for d in decks if d not in wr]
    if missing:
        import sys
        print("[rl_config] LM screen has no entry for %s -- treated as neutral (%.0f%%)."
              % (", ".join(missing), STAGE_A_TARGET_WR), file=sys.stderr)
    head = {d: max(STAGE_A_HEADROOM_FLOOR, STAGE_A_TARGET_WR - wr.get(d, STAGE_A_TARGET_WR))
            for d in decks}
    tot = sum(head.values()) or 1.0
    return {d: h / tot for d, h in head.items()}


def _stageB_pilot_weights(decks):
    """Stage-B pilot distribution: 70% mass on the FOCUS set (Stage-C targets U the live-
    frequent decks, LIVE_META >= 0.05), 30% spread uniformly over the rest as a breadth
    floor (keeps general competence -> avoids the RL's-Razor forgetting when Stage C forks).
    See rl-design-value-free §7 / the curriculum notes."""
    focus = set(STAGE_C_TARGETS) | {d for d, w in LIVE_META.items() if w >= 0.05}
    F = [d for d in decks if d in focus]
    O = [d for d in decks if d not in focus]
    pw = {}
    for d in F:
        pw[d] = 0.70 / len(F) if F else 0.0
    for d in O:
        pw[d] = 0.30 / len(O) if O else 0.0
    if not O:                                   # degenerate: everything is focus
        for d in F:
            pw[d] = 1.0 / len(F)
    return pw


def stage(name, target=None):
    """Return the config dict for a stage: pilots P, opponent decks O (+weights),
    opponent-AGENT mix (heuristic vs LM-checkpoint), sampling temperature, and the
    plateau stop rule. `target` overrides the Stage-C pilot.

    `sampling` drives tools/rl_ratings.sample_pairs:
      "matched"  (Stage A) -- bias (pilot,opp) toward an expected winrate ~50% so weak
                              decks get even, contrastive games and can learn to play.
      "weighted" (Stage B/C) -- sample pairs by pilot_weights x opponent weights (no
                              difficulty kernel): focus the LEARNING on target/live decks
                              while facing the live-frequency field.
    `pilot_weights` (dict deck->w, missing=>drop from pilots for weighted mode)."""
    decks = _all_decks()
    common = dict(
        # GRPO / RAE / MARS knobs (docs/rl_design.md §3-5)
        # games_per_matchup is set PER STAGE below (a RAE group = games within one matchup).
        # Sizing note (2026-07-23): difficulty-matched Stage A produces CLOSE (~50%) games,
        # which run LONGER (more decisions) than uniform blowouts, and rollout is GPU-scoring
        # -bound -- so games/round = matchups x gpm is the wall-clock knob. Target ~1000-1500
        # games/round (~30-45 min) so the curriculum iterates: A gpm=10, B=16, C=24, with
        # RL_MATCHUPS ~128 (Stage C = 61 pairs, uncapped). 6144/round (256x24) was ~2h+.
        games_per_matchup=16,      # default (Stage B); A/C override below
        clip=0.2,                  # GRPO ratio clip
        kl_coef=0.02,              # KL to the frozen SFT reference (0 = pure REINFORCE+RAE)
        lr=5e-6,
        # MARS: terminal reward only (+1/-1) -> per-turn cumulative == terminal; the
        # turn-level machinery still normalizes advantage across a game's decisions.
        reward_win=1.0, reward_loss=-1.0,
        format_penalty=0.1,        # illegal/unparseable action token (should be ~0: we score legal cands)
    )

    # ---- CURRENT: the two-stage curriculum (docs/rl_stages_v2.md) --------------------
    if name in ("1", "2"):
        tgts = [d for d in STAGE_C_TARGETS if d in set(decks)]
        # O = the 11, by live frequency, renormalised. dragapult_dusknoir has ZERO
        # leaderboard presence so it never appears as an opponent -- correct, it is a pilot
        # we may submit, not a deck the field runs.
        opp = {d: LIVE_META.get(d, 0.0) for d in tgts}
        s = sum(opp.values())
        opp = {d: w / s for d, w in opp.items()} if s > 0 else {d: 1.0 / len(tgts) for d in tgts}

        if name == "2":
            # OPTIONAL TAIL (docs/rl_stages_v2.md, "recommended deviation"). The 11 are 78.6%
            # of the top 500; the missing 21.4% is real ladder. Off by default because the
            # user specified "the 11". Opponents outside the 11 have no competent LM pilot
            # after Stage 1, so they are played by engine_v2 -- which is why they are a
            # separate, reportable slice rather than mixed in silently.
            tail = float(os.environ.get("RL_STAGE2_TAIL", "0.0"))
            if tail > 0:
                rest = {d: LIVE_META.get(d, 0.0) for d in decks if d not in set(tgts)}
                rs = sum(rest.values())
                if rs > 0:
                    opp = {d: w * (1.0 - tail) for d, w in opp.items()}
                    for d, w in rest.items():
                        opp[d] = opp.get(d, 0.0) + tail * w / rs

        if name == "1":
            # headroom = the LM's OWN mirror screen, not rl_ratings (see _stage12_pilot_weights)
            pilots, pw = tgts, _stage12_pilot_weights(tgts)
            rounds, gpm, temp = STAGE1_ROUNDS, 24, 1.0
            # THREE stop conditions, and the calendar one is not a safety net -- it is the
            # binding one. Without it a plateau test that never fires eats the whole box and
            # Stage 2 never runs at all.
            stop = dict(kind="plateau", metric="mirror_paired_11",
                        window=2, min_slope=0.01,           # <=+1pt for 2 rounds (SE 1.14pt)
                        also_stop_if="dragapult>=0.50",
                        hard_deadline_utc=STAGE1_DEADLINE_UTC)
        else:
            pilots = [d for d in ("dragapult", "dragapult_dusknoir") if d in set(decks)]
            pw = {d: 1.0 / len(pilots) for d in pilots}
            rounds, gpm, temp = STAGE2_ROUNDS, 64, 0.9  # 20 cells x 64 = 1,280 games/round
            stop = dict(kind="live_or_plateau", metric="live_kaggle_or_field",
                        window=3, min_slope=0.005,
                        hard_deadline_utc=RL_DEADLINE_UTC)
        return dict(common,
            stage=name, precision="bf16",
            pilots=pilots, opponents=opp, pilot_weights=pw,
            sampling="weighted",
            games_per_matchup=gpm,
            # Labels come from playouts, so the opponent only has to be COMPETENT, not
            # learning. The 11 all have LM pilots by construction.
            opp_agent_mix=dict(heuristic=0.10, lm_selfplay=0.90),
            temperature=temp,
            max_rounds=rounds,
            # Data recipe: this curriculum is Q-labelled, not GRPO. rl_rollout reads these.
            label="playout_q",
            mirror_frac=MIRROR_FRAC if name == "1" else 0.0,
            playouts_per_branch=PLAYOUTS_PER_BRANCH,
            # WHICH DECISIONS the branch budget buys, priced rather than merely observed.
            qlabel_targets=QLABEL_TARGETS,
            qlabel_max_cell_share=QLABEL_MAX_CELL_SHARE,
            valued_max_frac=VALUED_MAX_FRAC,
            # Phase-1 state collection is split between the two models. The Q LABEL is
            # pilot-independent (16 engine_v2 playouts), which is why the 4B may generate
            # data the DeBERTa trains on and why this is NOT the killed distillation branch
            # ([[teacher-9b-adds-nothing]] killed using a big model's OPINION as the label).
            # What DOES stay 4B-shaped is the STATE distribution and which decisions the
            # margin filter picks, so a slice is collected with the reranker itself.
            collect_mix=dict(decoder=1.0 - DEBERTA_COLLECT_FRAC,
                             reranker=DEBERTA_COLLECT_FRAC),
            # Gate: 11 decks needs ~229 games EACH to hold the paired SE that 63x40 gave.
            # See docs/rl_stages_v2.md -- narrowing the gate without rescaling games silently
            # triples the SE and makes the +-1pt stop rule undecidable.
            eval_decks=tgts, eval_games=GATE_GAMES_PER_DECK,
            stop=stop,
        )

    if name == "A":
        # Broad climb. Heuristic-heavy opponents give a CLIMBABLE gradient (fresh-SFT
        # self-play is garbage-vs-garbage); anneal heuristic 0.70 -> 0.20 over the stage.
        return dict(common,
            stage="A", precision="bf16",   # all stages bf16 (2026-07-22 decision; see Stage C)
            games_per_matchup=8,           # broad climb: many matchups, fewer games each
                                           # (rollout is GPU-scoring-bound + matched games run
                                           #  long, so ~768 games/round @ RL_MATCHUPS=96)
            pilots=decks,
            opponents={d: 1.0 for d in decks},          # uniform O over the fleet
            pilot_weights=_stageA_pilot_weights(decks),  # <-- FOCUS on weak submittable/meta
                                                         #     targets (headroom-weighted);
                                                         #     breadth floor keeps the rest
            sampling="matched",                          # <-- difficulty-match (rl_ratings)
            match_target_wr=MATCH_TARGET_WR,             # <-- engine-units target, see above
                                                         #     STILL applies: FOCUS decks get
                                                         #     contrastive ~50% games, not blowouts
            # 2026-07-28 (user): START AT 100% engine_v2. The stated goal is "bring the LM up
            # to engine_v2's level using games against engine_v2", and every one of the 63
            # agents/*.py runs engine_v2 -- so training opponent, eval opponent and the target
            # baseline are the SAME agent. Reward (beat engine_v2) then measures the goal
            # directly, with no proxy in between; the loop can still anneal self-play in later.
            # It also removes the second GPU-resident model entirely (the LoRA set_adapter
            # trick died with the decoder; a cross-encoder opponent would be a full 294 MB copy).
            opp_agent_mix=dict(heuristic=1.00, lm_selfplay=0.00),
            temperature=1.0,       # tuned by the smoke test; 1.0 placeholder
            stop=dict(kind="plateau", metric="winrate_vs_heuristic",
                      window=3, min_slope=0.01),          # <1pt/round improvement over 3 rounds
        )

    if name == "B":
        # Meta realignment: SAME broad P, but reweight O to the LIVE meta BEFORE
        # specializing (self-play is 48pt wrong on alakazam; plateauing on the proxy
        # then specializing invests in the wrong objective). Opponents must be COMPETENTLY
        # piloted -> use LM checkpoints for the meta opponents, not raw heuristic.
        opp = {d: LIVE_META.get(d, 0.0) for d in decks}
        resid = max(0.0, 1.0 - sum(opp.values()))
        for d in decks:                                   # spread residual so tail decks still appear
            opp[d] += resid / len(decks)
        return dict(common,
            stage="B", precision="bf16",
            pilots=decks,
            opponents=opp,
            pilot_weights=_stageB_pilot_weights(decks),  # <-- focus LEARNING on target/live
            sampling="weighted",                         # pilot_w x opp_w, no difficulty kernel
            opp_agent_mix=dict(heuristic=0.20, lm_selfplay=0.80),  # competent (LM) meta opponents
            temperature=1.0,
            stop=dict(kind="plateau", metric="winrate_vs_livemeta",
                      window=3, min_slope=0.01,
                      focus=["alakazam", "marnie_grimmsnarl", "rockets_mewtwo", "crustle"]),
        )

    if name == "C":
        # Targeted specialization: narrow the PILOT to the target; the OPPONENT stays the
        # LIVE-frequency field (NOT narrowed). You must beat the whole live meta WITH the
        # target deck, with competently-piloted meta opponents.
        tgt = target or STAGE_C_TARGETS[0]
        opp = {d: LIVE_META.get(d, 0.0) for d in decks}
        resid = max(0.0, 1.0 - sum(opp.values()))
        for d in decks:
            opp[d] += resid / len(decks)
        return dict(common,
            stage="C", precision="bf16",   # bf16 like A/B: training precision doesn't set
                                           # CPU speed (GGUF ship-quant does), and bnb-NF4 !=
                                           # GGUF k-quant so QLoRA gave no real quant-awareness
            pilots=[tgt],                                 # <-- ONLY the target deck
            opponents=opp,                                # <-- live-weighted, NOT narrowed
            pilot_weights={tgt: 1.0},
            sampling="weighted",                          # target vs live field (no matchmaking)
            opp_agent_mix=dict(heuristic=0.10, lm_selfplay=0.90),
            temperature=0.9,       # slightly lower: refining, not exploring broadly
            games_per_matchup=24,  # ~61 live matchups x 24 = ~1.5k games/round (tight-ish RAE)
            stop=dict(kind="live_or_plateau", metric="live_kaggle_or_field",
                      window=4, min_slope=0.005),
        )

    raise SystemExit(f"unknown stage {name!r} (want A|B|C)")


if __name__ == "__main__":
    import sys
    s = sys.argv[1] if len(sys.argv) > 1 else "A"
    cfg = stage(s, sys.argv[2] if len(sys.argv) > 2 else None)
    view = dict(cfg)
    view["pilots"] = f"[{len(cfg['pilots'])} decks]"
    view["opponents"] = f"[{sum(1 for v in cfg['opponents'].values() if v>0)} nonzero]"
    print(json.dumps(view, indent=2, ensure_ascii=False))
