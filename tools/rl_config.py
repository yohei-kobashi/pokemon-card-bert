"""RL curriculum + hyper-parameters for the post-SFT LM agent (docs/rl_design.md).

TWO independent knobs (§8 of the design):
  O = opponent distribution  (THE OBJECTIVE — who we must beat)
  P = pilot set              (which decks the LEARNING policy plays)

Stage A  broad climb          P = all shippable decks;  O = heuristic-heavy -> self-play
Stage B  meta realignment     P = all;                  O = reweighted to the LIVE meta
Stage C  targeted special.    P = the TARGET deck(s) ONLY;  O = LIVE meta, NOT narrowed
         ^ IMPORTANT: Stage C narrows only the PILOT. The OPPONENT distribution stays the
           live-frequency mix (you must still beat the whole live field with the target deck).

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
# REFRESHED 2026-07-23 from a top-500 leaderboard scout (tools/leaderboard_distribution.py,
# 520 teams classified, ~96% covered). Big shifts vs the stale 2026-07 scrape: alakazam_nz
# is now the #1 deck (0.21, was 0.08); base alakazam collapsed (0.30 -> 0.09); archaludon
# (0.09) + cynthia_garchomp (0.05) newly present; mega_starmie (Mega Starmie ex/Froslass,
# 0.02) added. See [[leaderboard-top100-meta-gap]] / [[prompt-reopt-resft-plan]].
LIVE_META = {
    "alakazam_nz": 0.212, "marnie_grimmsnarl": 0.172, "archaludon": 0.094,
    "alakazam": 0.086, "cynthia_garchomp": 0.048, "crustle": 0.048,
    "alakazam_nz_fez": 0.046, "dragapult": 0.044, "crustle_geco": 0.038,
    "mega_lucario": 0.036, "mega_starmie": 0.022, "rockets_mewtwo": 0.018,
    "staryu": 0.018, "rockets_spidops": 0.016, "omatsuri": 0.01,
    "comfey_yveltal": 0.01, "ns_zoroark": 0.008, "mega_lucario_tr": 0.008,
    "crustle_stall": 0.006, "iono_bellibolt": 0.006, "dragapult_dusknoir": 0.004,
    "mega_lopunny": 0.004, "hydrapple": 0.004, "rockets_honchkrow": 0.004,
}

# --- Stage-C target pilots (user-confirmed 2026-07-22; one RL run per target) ----------
# Dragapult leads (human BDIF, high skill ceiling): BOTH the human-#1 "straight" build AND
# the Dusknoir build — the latter patches the Grimmsnarl matchup (Dusclops/Dusknoir remove
# Froslass -> 63% vs Grimmsnarl) for our Grimmsnarl-heavy LIVE field, while base dragapult
# is the human-Tier-1 list that beats Alakazam (the 44% live giant). Both decks used AS-IS
# (dragapult_dusknoir already ~95% matches the competitive Roman-G list). rockets_mewtwo +
# rockets_honchkrow (competitive-rebuilt, [[rockets-honchkrow-competitive-rebuild]]) are the
# two Team Rocket types; crustle/alakazam/marnie_grimmsnarl complete the live-field roster.
# Opponents in Stage C are NOT narrowed — they stay the live-frequency field (see stage("C")).
STAGE_C_TARGETS = ["dragapult", "dragapult_dusknoir", "rockets_mewtwo",
                   "rockets_honchkrow", "crustle", "alakazam", "marnie_grimmsnarl"]


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
