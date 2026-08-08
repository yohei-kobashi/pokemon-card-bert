# Mirror-paired shaped-return RL for dragapult_dusknoir (instance1)

No engine_v2 anywhere: not as the opponent, not as a playout policy, not as a label source.
Both seats are the policy itself, on the same decklist and the **same shuffle order**, so the
only difference between two games of a pair is the moves that were chosen.

This is deliberately a different method from instance2's. Instance2 learns a **preference
between two candidate moves**, measured by playing the rest of the game with engine_v2. This
one learns from **whole trajectories**, scored by a reward written from the deck's own game
plan plus the prize counts, with no external policy in the measurement at all.

---

## 1. Why the credit assignment has to change

`rl-stage-a-plateau-diagnosis` measured the failure mode we must not repeat: a per-game
outcome spread uniformly over every decision produced a flat curve for twelve rounds, and more
games did not help. One win/loss bit divided by ~80 decisions is not enough signal per
decision, and the noise between games swamps it.

The fix here is a **dense potential** Φ(s): a number computed from the board that goes up when
the deck is executing its plan. The per-step reward is the *change* in that potential.

`shaping-potential-refuted` is the warning attached to this idea: `eval_state`'s per-move delta
picked the better move at chance, and playout Q carried 17x more information. That result was
measured on a **generic** evaluator used as the signal itself. Two things differ here:

1. Φ is **deck-specific** — it names Phantom Dive's counters, Cursed Blast's prize cost, and
   Boss's Orders, not "material and board presence".
2. Φ is only a **shaping term**. The ground truth stays the terminal win/loss. Under
   potential-based shaping (Ng, Harada & Russell 1999) the reward

   ```
   F(s, a, s') = γ·Φ(s') − Φ(s)
   ```

   leaves the optimal policy **unchanged** for any Φ. A wrong Φ costs sample efficiency; it
   cannot make the agent prefer a losing line. That is the whole reason to write it as a
   difference of potentials rather than as points for doing things.

---

## 2. The potential Φ(s), from our perspective

Every term is read off the observation. `HP_i` is remaining HP, `dmg_i` damage counters ×10.

### 2.1 Prizes — the spine (weight 1.0, the unit of everything else)

```
prize_lead = (6 − my_prizes_remaining) − (6 − opp_prizes_remaining)
           = opp_prizes_remaining − my_prizes_remaining        ∈ [−6, +6]
Φ_prize = 1.00 × prize_lead
```

Both sides' counts, as required. **One prize is the unit**: every other term below is
expressed as a fraction of a prize, so the weights are not arbitrary numbers but answers to
"how much of a prize is this worth?".

This term alone makes Cursed Blast honest. It places 13 counters **and knocks itself out**, so
firing it moves Φ_prize by −1 immediately. It is only worth it if the 130 damage buys back
more than a prize through the terms below — which is exactly the human judgement.

### 2.2 Banked bench damage — what Phantom Dive's 6 counters are for (weight ≤ 0.45/body)

Raw "damage dealt" is the wrong measure: it rewards spraying counters where they never convert.
The plan says the counters exist to bring a body **into range of a future Phantom Dive**, so
the potential is *proximity to a killable state*, saturating at the threshold:

```
for each opponent Pokemon i (bench and active):
    need_i = max(0, HP_i − 200)             # damage still required for Phantom Dive to KO it
    ready_i = 1 if need_i == 0 else 0
    prog_i  = clip(dmg_i / max(1, HP_i − 200), 0, 1)   # 0..1 progress toward that threshold

Φ_spread = Σ_i [ 0.45·ready_i + 0.15·prog_i·(1 − ready_i) ]     capped at 3 bodies
```

`ready_i` is a body that a single Phantom Dive now removes: worth ~half a prize (it is one
attack away, and the attack is one we will make anyway). `prog_i` gives a gradient so the
first counters on a fresh 320 HP body are not worth zero. The 3-body cap stops the policy
from farming a spread it can never cash.

### 2.3 Attack readiness and the Crispin split (weight ≤ 0.30)

Phantom Dive costs {R}{P}. The guide's line — split the energy across two Drakloaks — is
insurance against losing the attacker, so the potential must reward **two bodies that can pay**,
not the total energy attached:

```
payers = #{ our Pokemon that can pay {R}{P} now (counting attached types) }
Φ_energy = 0.20·min(payers, 1) + 0.10·min(max(payers − 1, 0), 1)
```

The second payer is worth half the first: real, and much less than the first. Stacking a third
energy on one body scores nothing, which is the behaviour we want to discourage.

### 2.4 The engine: Drakloak count and Dragapult in play (weight ≤ 0.25)

```
Φ_setup = 0.10·min(#Drakloak_in_play, 2) / 2 · 2      # Recon Directive, one per Drakloak
        + 0.15·1{Dragapult ex in play}
```

Recon Directive is the deck's draw. Two Drakloaks is the practical target; more is not better
enough to chase.

### 2.5 Budew's item lock (weight 0.10, early only)

```
Φ_lock = 0.10 · 1{opponent is item-locked next turn} · 1{turn ≤ 6}
```

Small and time-limited: the lock buys setup turns, and setup only has value while we are still
setting up.

### 2.6 The full potential

```
Φ(s) = Φ_prize + Φ_spread + Φ_energy + Φ_setup + Φ_lock
```

Bounded roughly in [−6, +8]. **The terminal reward is separate and dominant:**

```
R_terminal = +1 win / −1 loss / 0 draw     (scaled ×3 so a win outweighs any Φ swing)
```

---

## 3. The learning signal: mirror pairing instead of a value net

`rl-design-value-free` records the constraint: no value net. The baseline comes from the deal
instead.

For a seed `s`, the shuffle is fixed and shared by both seats. Play the game **G times from the
same seed**, sampling actions from the policy at temperature τ. Every one of the G games faces
an *identical deal*; they differ only in the policy's own choices. So the group mean is a
baseline that has the draw luck already removed — which is the property mirror mode was built
for ([[mirror-shuffle-mode]]: null is exactly 0).

```
for each trajectory k:  R_k = 3·outcome_k + Σ_t [ γ·Φ(s_{t+1}) − Φ(s_t) ]
A_k = R_k − mean_k(R)                      # GRPO-style group baseline, no value net
```

Per-decision credit uses the **return-to-go** from that decision, not the whole-game number:

```
G_t^k = 3·outcome_k·γ^(T−t) + Σ_{u ≥ t} γ^(u−t)·[ γ·Φ(s_{u+1}) − Φ(s_u) ]
A_t^k = G_t^k − mean_k(G_t^k)      (matched by decision INDEX across the group)
```

This is the part the plateau diagnosis demands: two decisions in the same game now get
different credit.

### Loss

The reranker scores candidates; softmax over those scores **is** a policy.

```
π(a|s) = softmax(scores(s, candidates)/τ)
loss  = −Σ_t A_t·log π(a_t|s_t)  +  β_KL·KL(π ‖ π_ref)  −  β_H·H(π)
```

* `π_ref` is the **starting checkpoint**, and it stays there across rounds — the anchor mistake
  we made on instance2, where re-anchoring each round removed the only thing holding the SFT in
  place, must not be repeated here.
* The entropy bonus is not optional: a reranker trained by cross-entropy is very peaked, and
  with no exploration all G trajectories in a group are identical and every advantage is 0.
  **Measure the group's action-disagreement rate; if it is near zero the round is a no-op.**

---

## 4. What gets built

| file | role |
|---|---|
| `tools/dusk_potential.py` | Φ(s) from an observation, plus a `--selftest` that prints the term breakdown on real states |
| `tools/rl_rollout.py` | mirror self-play at temperature τ, G trajectories per seed, records (prompt, candidates, chosen, logits, Φ) per decision |
| `tools/rl_pg_train.py` | the policy-gradient update above, from a rollout file |
| `tools/rl_dusk_loop.sh` | rollout → train → gate, gated the same way as everything else |

The gate does **not** change: `dragapult_dusknoir` vs engine_v2, 400 games, mirror. engine_v2
is barred from training, not from measurement — an independent yardstick is the only reason we
can tell that three consecutive SFT rounds lost.

---

## 5. Validation ladder (each step must pass before the next)

Every silent failure this week came from skipping one of these.

1. **Φ is computable and sane.** Print the term breakdown on 20 real mid-game states. A hand
   check must agree that a state with two ready bodies scores above one with none.
2. **Φ correlates with winning.** On existing mirror logs, bucket decisions by Φ and check the
   eventual win rate rises monotonically. If Φ does not separate, the shaping is decoration —
   stop here rather than train on it. (This is the test `shaping-potential-refuted` would have
   failed.)
3. **The group actually diverges.** With τ set, ≥30% of decisions must differ somewhere within
   a group of G, or there is no gradient.
4. **Paired variance beats unpaired.** Compare sd(R) within-seed against across-seed. The whole
   argument for same-shuffle grouping is that the first is much smaller.
5. **Overfit probe.** 200 trajectories, no KL, high lr: the loss must collapse. A trainer that
   cannot do this must not spend a round (the instance2 probe rule).
6. **One gated round** at 400 games before any loop is started.

---

## 6. Honest risks

* **Φ may not separate** (step 2 kills the design if so). The prize term almost certainly does;
  the deck-specific terms are the bet.
* **Mirror self-play has no external pressure.** Both seats improve together and can settle
  into a mutually convenient equilibrium that a different opponent punishes. The engine_v2 gate
  is what would catch it — which is why the gate keeps engine_v2 even though training does not.
* **Reward hacking on the shaping terms.** Potential-based shaping bounds this in theory
  (Σ of differences telescopes to Φ(end) − Φ(start)); the practical check is that the gate
  moves, not that Φ moves.
* **This is a new trainer for a model that is already flat.** Three continuation rounds lost
  on this deck (45.5 → 41.5 → 30.7). A new objective is a reasonable response to that, but the
  base rate for "new training loop helps" in this project is not high, and the gate should be
  believed over the loss curve.
