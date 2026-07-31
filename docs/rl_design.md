# RL Design — post-SFT reinforcement learning for the LM agent

Status: **design locked, not implemented.** Companion: `docs/ml_agent_plan.md` (SFT),
`docs/lm_input_format_v2.md` (prompt), memory `opponent-identification-in-prompt`,
`archetype-prediction`, `value-net-v1`, `value-scorer-turn-boundary`.

This memo supersedes the earlier "DPO, then maybe GRPO, reward = win/loss + evaluator"
sketch. Two things changed after research + analysis (2026-07-20):

1. **Base algorithm = GRPO with two targeted refinements (RAE + turn-level advantage),
   not DPO.**
2. **The run is value-free by default.** The learned value net is demoted to an
   optional, ablatable, *unbiased* baseline — never a reward term, never a bootstrap
   critic. Rationale in §4; this is the load-bearing decision.

---

## 1. Why not DPO

DPO is offline preference learning. It cannot use (a) the dense per-state evaluator
signal, (b) matchup conditioning, or (c) a self-play curriculum. GRPO + RAE is a strict
superset of what we'd get from DPO here, at similar implementation cost since we already
generate self-play rollouts (`tools/gen_selfplay.py`). DPO is dropped.

## 2. The setting, and why it selects the method

Our problem is unusual for LLM-RL and the peculiarities drive every choice:

- **Zero-sum, turn-based, self-play data already exists.**
- **Reward is observable:** we play the game out, so win/loss is *ground truth*, not a
  quantity to be predicted. (This is the crux for §4.)
- **Matchup confound:** 26.6% of outcome variance is the deck-vs-deck pairing; 32% of
  matchup cells are near-decided regardless of play (measured when GRPO was first scoped).
- **Long horizon, tiny actions:** a game is dozens of decisions (slowking runs 47–67
  turns), and each decision emits ~1 action token, not a long chain-of-thought.

Consequences:
- Length-normalization fixes (**Dr. GRPO**) are irrelevant — actions are one token.
- Sequence-level / MoE-stability (**GSPO**) is irrelevant to a dense-target policy head.
- The real problems are **matchup confounding** and **long-horizon credit assignment**.
  These are exactly what SPIRAL's RAE and MARS's turn-level advantage address.

## 3. Base algorithm: GRPO + RAE + turn-level advantage

**GRPO** (critic-free, group-relative baseline) is the base. For a state we sample a
group of rollouts and use the group's mean return as the baseline — no learned critic
required. This alone gives the variance reduction a critic would.

**RAE — Role-Conditioned Advantage Estimation (SPIRAL, arXiv:2506.24119).** Condition the
baseline on the *role*; in our game **role = the matchup (our deck × opponent deck, or at
least opponent archetype)**. Normalize returns *within the same matchup group* rather than
across the whole batch:

```
A(s,a) = R(trajectory) − mean_{same matchup}(R)      # matchup-conditioned baseline
```

This absorbs the 26.6% matchup confound directly: a win in a 30/70 matchup and a win in a
70/30 matchup no longer carry the same advantage. **RAE needs no value function** — it is
pure win/loss normalized within a role bucket. This resolves our single largest RL
obstacle with zero critic.

**Turn-level advantage (MARS, arXiv:2510.15414).** Group-relative advantage otherwise
smears one trajectory-level number across every decision in a 50-turn game. Use the
**sum-then-normalize** turn return: `R_k = Σ_{k'≥k} r_{k'}`, credit each turn's tokens with
the cumulative return from that turn onward, normalized within the (matchup) group.
Fine-grained credit assignment over the long horizon, still Monte-Carlo, still no critic.

**Opponent sampling / curriculum (SPIRAL).** Keep a population of past checkpoints and
sample opponents from it instead of always playing the latest policy. Cheap upgrade on top
of our existing self-play; prevents overfitting to a single exploitable opponent.

## 4. Value-free by default — the load-bearing decision

**Decision: the reward is terminal win/loss only. The value net is NOT in the reward and
NOT a bootstrap critic. It survives only as an optional, ablatable REINFORCE baseline.**

### 4.1 Why the value net's role collapses in RL

In **SFT** (`build_sft`) the value net was a *surrogate for the outcome* — it scored
trajectories we hadn't finished, so its error fed straight into which exemplars we imitated
(see `value-scorer-turn-boundary`). In **on-policy RL we observe the true win/loss**, so the
value net is no longer needed to guess the outcome. Its only remaining jobs are variance
reduction and per-step credit assignment — and **GRPO's group baseline already provides the
variance reduction for free** (that is the whole point of critic-free GRPO). The value net's
marginal value is therefore far lower in RL than it was in SFT.

### 4.2 The strong form of "value net degrades as the policy strengthens"

The degradation is two distinct failures, and only one is fixable:

1. **Distribution shift / staleness** — trained on v31 heuristic self-play; a stronger
   policy visits off-distribution states. *Partially* fixable by periodic refit on RL
   rollouts (but that refit always lags, and refitting is the PPO-style critic maintenance
   GRPO was designed to avoid).
2. **Feature-capacity ceiling** — the value net is **68 hand-designed features**. Expert
   lines (multi-turn combo reachability, reading the opponent's remaining hand) are not
   representable in 68 dimensions *no matter how fresh the data*. **Unfixable.** This is the
   real reason to distrust it at expert strength, and it grows precisely as the policy
   improves.

### 4.3 Usage mode decides whether the ceiling can hurt

The concern is entirely mode-dependent:

| Usage | Bias from a wrong V | Ceiling bites? | Verdict |
|---|---|---|---|
| Additive reward `r = win/loss + λ·V` | **Biases the objective** | Directly | **Forbidden** — teaches the policy to chase V's mistakes |
| Bootstrap critic, GAE(λ<1) | **Biased if V wrong** | At expert states | Avoid |
| REINFORCE baseline `A = R_MC − V(s)` | **Unbiased for ANY V** | **No** — only variance changes | Allowed, optional |

The REINFORCE-baseline row is the key fact: a state-dependent baseline subtracted from the
Monte-Carlo (turn-level) return is **unbiased for any V whatsoever**. A stale or
capacity-limited V there only inflates variance; it can never pull the policy the wrong way.
The ceiling cannot bite in this mode. The originally-planned "add the evaluator to the
reward" is exactly the forbidden top row — it is dropped.

### 4.4 Standing decision

- **Default: value-free.** GRPO + RAE + turn-level advantage, terminal reward only. This is
  what SPIRAL and MARS both do, and it is the natural consequence of §4.1 (outcome is
  observable in on-policy RL).
- **RAE already resolves the 26.6% matchup confound with no value function** — the biggest
  reason we reached for the evaluator is handled by matchup-conditioned normalization.
- **Value net retained only as an ablatable REINFORCE baseline** (`A = R_MC − V`),
  gated by a config flag, to be A/B'd against the plain group baseline. Drop it once the
  group is large enough that the group mean is a good baseline on its own.
- **Never** additive-reward, **never** GAE(λ<1) bootstrap.
- The one place a critic genuinely helps is very long games (per-step signal); but length is
  also where the capacity ceiling hurts most, so we lean value-free and lengthen credit
  reach via **turn-level MC + larger groups** rather than a critic.

## 5. Reward specification

```
terminal:  +1 win / −1 loss                       (ground truth, zero-sum)
format:    small penalty for an illegal/unparseable action token   (à la MARS)
shaping:   NONE by default (no evaluator term). If ever added, only as a
           potential-based term F = γΦ(s') − Φ(s) with Φ = value net and λ annealed
           → 0, relying on policy-invariance; but default is OFF.
baseline:  matchup-conditioned group mean (RAE), optionally minus V(s) as an unbiased
           control variate (ablation).
```

## 6. Open items / sequencing

1. Land SFT (v33) first — RL starts from the SFT checkpoint.
2. Implement GRPO rollout loop reusing `tools/gen_selfplay.py`; group by matchup for RAE.
3. Ship RAE + turn-level advantage together; measure whether matchup-conditioning removes
   the confound (compare per-matchup win-rate variance before/after).
4. Only then ablate the REINFORCE `−V(s)` baseline; keep only if it reduces gradient
   variance without cost. Expect it to fade as group size grows.
5. Opponent-population sampling once single-opponent overfitting is observed.
6. **DAPO** (dynamic sampling + decoupled clip) held in reserve purely for instability.

## 7. SFT interleaving during RL (prior-work-grounded)

Interleaving SFT *inside* the RL loop is standard and beneficial — but only for the right
reasons. Evidence: **DeepSeek-R1** (arXiv:2501.12948) alternates SFT↔RL over 4 stages, with
**Stage-3 = rejection-sampling SFT** (self-generate from the RL checkpoint, keep only the
winning/correct trajectories, mix with broad data, SFT to *consolidate* the skill and
*restore* general capability) before a final RL stage. The **ReST / ReST-EM / RAFT / RFT /
STaR** family is the same idea iterated: generate → filter by reward → SFT on the survivors.
Our value-free *listwise reward-weighted* update is already on this continuum, so a
hard-filter "SFT on winning games" step slots in cheaply.

**2026 caveat — do NOT insert SFT merely for anti-forgetting.** "RL's Razor" (arXiv:2509.04259)
shows on-policy RL forgets *less* than SFT (it stays near the base policy), and an SFT insert
can itself cause drift. So insert SFT for **(a) new-content injection** and **(b) policy
consolidation**, not as a reflexive forgetting patch.

**Policy for this pipeline:**
- **MANDATORY — SFT insert whenever new content is added** (a new deck, or a fleet-unique new
  card like Roto-Stick): RL rollouts of the current policy *cannot explore a card/line the
  policy has never seen*, so RL alone will never learn it. Generate self-play with the new
  build, `build_sft`, and resume-SFT (mixed with a broad fleet sample — see the honch_aug
  mixed top-up, memory `rockets-honchkrow-competitive-rebuild`).
- **OPTIONAL — rejection-sampling SFT consolidation between RL blocks** (R1 Stage-3 pattern):
  after a Stage A/B/C block, take the checkpoint, self-play, keep the **winning** games, and
  SFT on those decisions + a broad fleet replay. Stabilizes/distills the spiky RL policy;
  cheap because the rollouts already exist. Gate it on a real plateau/instability, not every
  block.
- **CONDITIONAL — broad SFT replay around Stage C narrowing**: narrowing P to one pilot *can*
  degrade the rest of the fleet, but per RL's Razor don't assume it does — **measure** the
  fleet win-rate before/after (the eval spine, §9) and only replay if degradation is real.
- **Drift hygiene for every insert** (ReST-EM): resume from a *stable* base (not a drifted
  checkpoint), **mix broad fleet data**, **low LR**, short. Consider model averaging / SWA
  over inserting SFT if the only goal is stability.

`rl_loop.sh` carries an opt-in `consolidate_sft` hook between stages; the new-content SFT is
run manually when content changes (it is not part of the automatic loop).

## 8. Training curriculum (opponent-distribution schedule)

**Framing: the opponent distribution O IS the objective; the pilot distribution P is a
separate knob.** "Train all decks → concentrate on target decks" mixes two axes:
- **P (pilot)** — which deck the learning policy plays. Broad P keeps the shared policy
  general; narrow P specializes it.
- **O (opponent)** — which decks it plays against. **O defines what win-rate we optimize.**
Schedule them separately.

**The non-negotiable pivot: self-play win-rate ≠ live win-rate.** Self-play overrates
stall/mill and is 48pt wrong on alakazam (memory `live-alakazam-beats-us`). Plateauing on
the self-play proxy and then specializing pours compute into the wrong objective. A
**meta-realignment stage** sits between "broad" and "targeted" — that is the main
refinement over the naive two-phase plan.

### Stage A — Broad climb to competence (anti-collapse)
- **O = heuristic-heavy.** Start ~70% fixed heuristic engine + ~30% self-play (past
  checkpoints, RAE population), anneal heuristic share 70%→20% as the policy strengthens.
  *Why heuristic early:* a fresh SFT policy self-playing produces garbage-vs-garbage with no
  climbable gradient; a fixed competent opponent gives a gradient to climb. Heuristic games
  are single-sided learning (our side only) — cleaner and more stable early.
- **P = all 62 decks, ~uniform** (drop the structurally-dead ones RL can't fix:
  iono_bellibolt, mega_dragonite 2/2/2 line — memory `iono-bellibolt-cannot-fuel-its-attacker`,
  `weak-deck-bottleneck-fixes`).
- Reward value-free terminal; RAE matchup-conditioned; turn-level advantage.
- **Goal:** LM ≥ heuristic across the held-out field. Purpose is *not to collapse* and to
  generalize, not to win the meta yet.
- **Plateau rule (concrete):** broad held-out win-rate improves < **1.5pt across 2
  consecutive eval checkpoints** → advance. (Eval sized so aggregate SE < 1pt, see §9.)

### Stage B — Meta realignment (reweight O to the live meta)
- **O shifts to match the LIVE leaderboard**, not the self-play field: up-weight alakazam,
  marnie_grimmsnarl, the scouted Team-Rocket / geco decks, and the top-100 distribution
  (memory `leaderboard-top100-meta-gap`, `scouted-decks-tr-geco`). Optionally add
  **live-exported games** as fixed strong opponents (memory `live-history-to-local-logs`).
- **P stays broad but meta-weighted.** This is the first stage optimizing the *real*
  objective. Expect self-play-favored decks (stall/mill) to lose apparent value here — that
  is the correction working, not a regression.
- **Gate to advance:** direction confirmed against a live-score spot check (§9 tier 3).

### Stage C — Targeted specialization (narrow P), live-calibrated
This is the user's "target decks" phase, and it must be calibrated to **what wins LIVE**,
not to an abstract notion of "meta". Three requirements, all confirmed with the user:

**(1) O = LIVE-FREQUENCY-weighted, not flat-meta.** Weight the opponent distribution to the
**observed live top-100 play-rate** (`tools/scout_decks.py`, memory
`leaderboard-deck-scouting`), so we optimize against the field we will actually face —
alakazam and crustle at their real live share, Team Rocket, etc. Still O ≠ the target decks
themselves (beat the field, don't mirror it).

**(2) The meta decks in O must be piloted COMPETENTLY — this bounds transfer.** Beating a
badly-played alakazam does not transfer to live. So opponents for the live-top decks are
drawn from: the best heuristic configs **plus the Stage A/B checkpoints that learned to
pilot alakazam / crustle / rockets well**, sampled from the population pool. This closes the
loop: **the reason Stage A/B train a broad P is precisely to mint the strong meta-pilot
opponents Stage C needs.** ("live で強いデッキは LLM もある程度プレイできる" is both a roster
property and an opponent-quality requirement.)

**(3) Target roster = a two-gate, data-driven selection** (not a fixed list). A deck enters
the roster only if BOTH hold, measured:
- **Pilotable** — the LLM reaches a competent held-out level piloting it.
- **Anti-meta capable** — it has a winning or even matchup vs the live top (alakazam,
  crustle, rockets), e.g. `deck` beats both current top-2 (memory
  `anti-meta-matchups-jul-2026`).
The roster may include the live-strong decks themselves (we should be able to pilot the meta)
**plus** dedicated anti-meta decks. Exclude the fuel-bound dead decks RL can't fix.

**Deckbuilding precedes RL — the structural gate.** RL cannot teach a deck to win a matchup
it structurally loses. mega_lucario loses to walls+spread *by construction* (memory
`mega-lucario-live-matchup-profile`); no amount of piloting fixes that. If a target deck
lacks the anti-meta matchup structurally, **fix the LIST first** (deckbuilding — e.g. the
buildable answer in `mega-lucario-real-list-tech`), then spend RL on it. Do not burn RL
compute on a deck that structurally cannot beat the live top.

- **Anti-forgetting floor:** keep ~15–20% of games uniform over all decks + RAE population
  sampling, so specialization does not tank the broad field (whack-a-mole guard).
- **Per-deck escalation:** default one shared deck-conditioned policy. Only if a target deck
  plateaus below its live rival, fork a per-deck LoRA head for it — do not fork by default
  (loses transfer).

### Cross-cutting mechanisms
- **Opponent-population sampling** throughout (SPIRAL) — sample opponents from a checkpoint
  pool, never only the latest, to avoid single-opponent exploitation cycles.
- **Note on the stall bias as *opponent*:** playing *against* stall is fine (learn to beat
  it); the bias only distorts when a stall deck is our *pilot* and self-play hands it cheap
  wins. Live-score gating (§9) + Stage B reweighting contain it.

## 9. Evaluation spine (the anti-delusion protocol)

Three tiers, increasing trust. **Never promote a checkpoint or advance a stage on tier 1
alone.**
1. **Self-play win-rate** — steering signal only, known biased (48pt off on alakazam). Fast.
2. **Held-out field** — a FIXED set of (pilot, opponent) matchups, seeds fixed, 150 games
   each, sized so the aggregate SE < 1pt (≈ 30+ matchups → ~4500 games). **Run this through
   an engine_v2 harness, NOT `tools/evaluate.py`** — evaluate.py drives the LEGACY engine
   (memory `two-engines-eval-tests-wrong-one`). Held-out means these matchups are never in
   the training opponent stream, else we overfit the eval. **For Stage C the held-out field
   is live-frequency-weighted and its meta decks are competently piloted** (§8 Stage C 1–2),
   so tier-2 tracks the live objective rather than a uniform self-play average.
3. **LIVE Kaggle score** — ground truth, the only real signal, slow, 150-game trust floor
   (memory `deck-status-and-live-scores`). Spot-check stage transitions against it.
- **Stage transitions gate on tier 2; sanity-check direction on tier 3.** Tier-1 is for
  intra-stage steering only.

## 10. Rejected

- **DPO** — offline, can't use matchup conditioning or curriculum (§1).
- **Evaluator in the reward** — forbidden mode (§4.3 top row); the value net poisons the
  objective at expert strength.
- **Bootstrap critic / GAE(λ<1)** — biased under a capacity-limited V (§4.3).
- **Dr. GRPO length fixes, GSPO** — premised on long CoT / MoE; our actions are one token.
