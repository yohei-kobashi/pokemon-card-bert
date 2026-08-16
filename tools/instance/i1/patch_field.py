import os

p = "/root/field_chain.sh"
s = open(p).read()

# 1. The opponents. alakazam_nz + marnie are the top-500's top two and appeared in ZERO of our 19
#    live games. Gating on them produced fld_r1b, which beat the champion by +5.20pt on that panel
#    and then measured -1.87 to -4.00pt against the five decks we actually meet. That is a gate
#    overfit, and the fix is the panel, not the training.
old = 'OPPS=alakazam_nz,marnie_grimmsnarl'
new = 'OPPS=${OPPS:-mega_abomasnow_sample,dudunsparce_box,dragapult,archaludon,ogerpon_mono}'
assert s.count(old) == 1, ("opps", s.count(old))
s = s.replace(old, new)

# 2. The wrapper. Prohibitions-only beat the shipped R5 by +4.00 +- 1.65 over 750 games, and the
#    gain came from DROPPING the forcing rules, not from adding prohibitions (r5+prohibitions was
#    -0.27). Forcing a rule the model already obeys 84-100% of the time per turn can only take
#    away its exceptions.
old2 = 'R5=lethal_now,spread_aim,clops_hold,energy_line,energy_focus'
new2 = ('R5=lethal_now,spread_aim,clops_hold,energy_line,energy_focus\n'
        'PROH=clops_hold,judge_timing,spare_ex_bench,retreat_energy,stadium_replace\n'
        'WRAP_RULES=${WRAP_RULES:-$PROH}')
assert s.count(old2) == 1, ("r5", s.count(old2))
s = s.replace(old2, new2)

old3 = 'PFX="planfilter:$R5:"                          # the shipped configuration; gated 08-11'
new3 = ('PFX="planfilter:$WRAP_RULES:"                  # prohibitions-only; gated 08-12 at +4.00pt')
assert s.count(old3) == 1, ("pfx", s.count(old3))
s = s.replace(old3, new3)

# The reward exclusion must follow the wrapper: a rule the model no longer decides is one whose
# gradient is spent on nothing, and a rule it DOES decide must stay in the reward.
old4 = '--rule-weights --rule-exclude "$R5"'
new4 = '--rule-weights --rule-exclude "$WRAP_RULES"'
assert s.count(old4) == 1, ("excl", s.count(old4))
s = s.replace(old4, new4)

old5 = '''            --wrap "planfilter:$R5" \\'''
new5 = '''            --wrap "planfilter:$WRAP_RULES" \\'''
assert s.count(old5) == 1, ("wrapset", s.count(old5))
s = s.replace(old5, new5)

# 3. gpu_wait must not KILL the run. It gave up after 30 minutes and took round 2 with it when a
#    second gate shared the card. Wait far longer, and say so, rather than discarding an hour of
#    collection because something else was mid-gate.
old6 = '''        [ "$u" -le 2000 ] && return 0
        sleep 30
    done
    say "STOP: GPU held ${u} MiB for 30 min"; exit 1'''
new6 = '''        [ "$u" -le 2000 ] && return 0
        [ $((_i % 20)) -eq 0 ] && say "waiting for the GPU (${u} MiB in use)"
        _i=$((_i+1))
        sleep 30
    done
    say "STOP: GPU held ${u} MiB for 3 h -- something is wedged"; exit 1'''
assert s.count(old6) == 1, ("gpuwait", s.count(old6))
s = s.replace(old6, new6)
s = s.replace('    local u\n    for _ in $(seq 1 60); do',
              '    local u _i=0\n    for _ in $(seq 1 360); do', 1)

t = p + ".new"
open(t, "w").write(s)
os.chmod(t, 0o755)
os.replace(t, p)
print("field_chain: live panel, prohibitions wrapper, patient gpu_wait")
