import os
p = "/root/field_chain.sh"
s = open(p).read()

old = '''                --beta "$BETA" --temp "$TEMP" --qmin "$Q" | tee -a "$STATE/convert$R.log"'''
new = '''                --beta "$BETA" --temp "$TEMP" --qmin "$Q" \\
                --phi-min "${PHI_MIN:-0.10}" --phi-wc "${PHI_WC:-0.65}" \\
                | tee -a "$STATE/convert$R.log"'''
assert s.count(old) == 1, "convert anchor"
s = s.replace(old, new)

old2 = "export MIN_GAIN=${MIN_GAIN:-0.0}"
new2 = '''export MIN_GAIN=${MIN_GAIN:-0.0}
# Setup potential, 08-14. qmin drops ~60% of pairs as coin flips; those are labelled instead by
# which candidate left us further along the human opening (line bodies / Drakloak / energy that
# pays {R}{P}), measured at OUR turn boundary inside the playout, capped at the template so it
# cannot be farmed. Measured reach: 16.5% of pairs separate, +22% usable rows. The label is
# deliberately weak (0.65) because FORCING the same preference cost -2.25pt.
# PHI_MIN=0 turns it off and reproduces the previous converter exactly.
PHI_MIN=${PHI_MIN:-0.10}
PHI_WC=${PHI_WC:-0.65}'''
assert s.count(old2) == 1, "knob anchor"
s = s.replace(old2, new2)

open(p + ".new", "w").write(s)
os.replace(p + ".new", p)
os.chmod(p, 0o755)
print("field_chain: converter now rescues below-qmin pairs with the setup potential")
