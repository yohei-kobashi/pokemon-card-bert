"""night5: an ordinary RL round on the 4B, sized so its gate can actually resolve a real effect.

WHAT night4b SETTLED AND WHAT IT DID NOT.  The pair-confidence filter lifted held-out accuracy
51.9% -> 64.5% and moved the win rate +0.50 +- 2.24.  So the filter makes the labels easier to
fit and does not, on that sample, make the model play better -- the same shape as instance1's
seven straight non-adoptions and as [[plan-conformance-is-not-winning]].  It is not harmful and
it is cheap, so it stays ON as the method; what it is not is a result worth re-testing.

SO THIS ROUND ASKS THE ORDINARY QUESTION: does another round of DPO beat the policy we have?
Two arms differing only in learning rate, both filtered, gated against dpo_r8 itself as the
baseline arm -- so the number produced is "what this round is worth", and a round that helps at
one learning rate and hurts at the other is still a usable round rather than a wasted night.

SIZED FOR THE ANSWER.  night4b's +-2.24 could not see a 2pt effect, which is the size worth
adopting.  Collection doubles (100 games per deck per shard) and the gate runs 80 games per
(arm, opponent) across 8 opponents -- 1,920 gate games against 800, taking the SE to roughly
1.2pt.  The whole round fits in about four hours.
"""
import os

src = open("/root/night4b.sh").read()
s = src

s = s.replace("TAG=${TAG:-night4b}", "TAG=${TAG:-night5}", 1)
s = s.replace("LOG=${LOG:-/root/night4b.log}", "LOG=${LOG:-/root/night5.log}", 1)
s = s.replace("GAMES=${GAMES:-50}", "GAMES=${GAMES:-100}", 1)
s = s.replace("GATE_GAMES=${GATE_GAMES:-50}", "GATE_GAMES=${GATE_GAMES:-80}", 1)

# ---- section 4: two learning rates, both filtered -------------------------------------------
old = """for ARM in base filt; do
    Q=0.0; [ "$ARM" = "filt" ] && Q=$QMIN"""
new = """for ARM in a b; do
    # The filter is the METHOD now, not the variable; the variable is the learning rate, which
    # is what instance1's ladder moves when a round comes back flat.
    LR=5e-5; [ "$ARM" = "b" ] && LR=2e-5
    Q=$QMIN"""
assert s.count(old) == 1, "arm loop anchor"
s = s.replace(old, new)

old = """    say "train $ARM (qmin $Q)\""""
new = """    say "train $ARM (lr $LR, qmin $Q)\""""
assert s.count(old) == 1, "say anchor"
s = s.replace(old, new)

old = """--epochs "${EPOCHS:-3}" --beta 0.1 --lr 5e-5 --cdpo-calibrated \\"""
new = """--epochs "${EPOCHS:-3}" --beta 0.1 --lr "$LR" --cdpo-calibrated \\"""
assert s.count(old) == 1, "lr anchor"
s = s.replace(old, new)

# ---- section 5: gate the two arms against the policy they started from -----------------------
old = """say "gate: base (baseline) vs filt, 8 opponents x $GATE_GAMES games\""""
new = """say "gate: prev (baseline) vs a vs b, ${OPPS//,/ } x $GATE_GAMES games\""""
assert s.count(old) == 1, "gate say anchor"
s = s.replace(old, new)

old = """    --games "$GATE_GAMES" --seed 99000 --baseline base --opp-spec engine \\
    --arm "base=qwen:/root/out/lora_${TAG}_base" \\
    --arm "filt=qwen:/root/out/lora_${TAG}_filt" \\"""
new = """    --games "$GATE_GAMES" --seed 99000 --baseline prev --opp-spec engine \\
    --arm "prev=qwen:$PREV" \\
    --arm "a=qwen:/root/out/lora_${TAG}_a" \\
    --arm "b=qwen:/root/out/lora_${TAG}_b" \\"""
assert s.count(old) == 1, "gate arms anchor"
s = s.replace(old, new)

old = """grep -aE "vs |^base|^filt|arm " /root/gate_$TAG.log | tail -14 >> "$LOG\""""
new = """grep -aE "vs |^prev|^a |^b |arm |setup speed|t1 dreepy" /root/gate_$TAG.log | tail -20 >> "$LOG\""""
assert s.count(old) == 1, "grep anchor"
s = s.replace(old, new)

# ---- the summary block ------------------------------------------------------------------------
i = s.index("python3 - /root/gate_$TAG.json")
j = s.index("say \"NIGHT4B_DONE")
s = s[:i] + '''python3 - /root/gate_$TAG.json "$QMIN" <<'PY' >> "$LOG" 2>&1
import json, sys
j = json.load(open(sys.argv[1]))
a = j["arms"]
print("\\n=============== did this round beat the policy it started from? ===============")
print("qmin %s   (filter kept as the method, not re-tested)" % sys.argv[2])
for k in ("prev", "a", "b"):
    if k not in a:
        continue
    x = a[k]
    t = (x["delta_vs_baseline"] / x["se"]) if x.get("se") else 0.0
    print("  %-5s %5.1f%%   delta %+5.2f +- %4.2f (t %+.2f)%s"
          % (k, x["win_rate"], x["delta_vs_baseline"], x["se"], t,
             "   (baseline, lr 5e-5=a / 2e-5=b)" if k == "prev" else ""))
print("\\nper opponent:")
for o in sorted({k.split("|", 1)[1] for k in j["cells"]}):
    row = []
    for k in ("prev", "a", "b"):
        c = j["cells"].get("%s|%s" % (k, o))
        row.append("%5.1f" % (100.0 * c["win"] / c["games"]) if c else "    -")
    print("  %-24s prev %s   a %s   b %s" % (o, row[0], row[1], row[2]))
print("\\nADOPT only if an arm clears +2.0pt AND its own SE, and does not lose a cell by more")
print("than it gains elsewhere. night4b's +-2.24 could not see 2pt; this run's SE is ~1.2.")
PY
''' + s[j:]

s = s.replace('say "NIGHT4B_DONE', 'say "NIGHT5_DONE', 1)

open("/root/night5.sh", "w").write(s)
os.chmod("/root/night5.sh", 0o755)
print("wrote /root/night5.sh")
