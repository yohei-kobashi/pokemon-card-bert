"""night6: the 4B's RL round, run on top of the plan filter instead of beside it.

WHAT CHANGED TONIGHT.  Wrapping the existing dpo_r8 weights in the plan rules -- no training at
all -- is worth +10.62 +- 1.89 (t 5.63) across 800 paired games, with all eight opponents up and
the setup metrics moving with it (Dreepy on our turn 1: 1.66 -> 2.11, Drakloak on turn 2: 0.53 ->
1.07, a first Phantom Dive in 46% -> 60% of games).  Ten RL rounds across two machines produced
nothing that size.

So the filter is not an arm any more, it is the pilot.  That changes what a round MEANS:

  * collection runs through the filter, so the traces are of the policy we would actually ship,
    and the branch points are the decisions that policy really faces
  * the gate's baseline is the FILTERED previous policy, so a round has to beat the thing we
    already have rather than the thing we abandoned

Without this the round would train on states its own pilot never visits, and would be scored
against a baseline 10 points below what we ship -- which is how a flat round could look like a
win.
"""
import os

s = open("/root/night5.sh").read()

s = s.replace("TAG=${TAG:-night5}", "TAG=${TAG:-night6}", 1)
s = s.replace("LOG=${LOG:-/root/night5.log}", "LOG=${LOG:-/root/night6.log}", 1)
s = s.replace('say "NIGHT5_DONE', 'say "NIGHT6_DONE', 1)

# the rule set, and the wrapper spec built from it
old = "OPPS=${OPPS:-marnie_grimmsnarl,"
new = ("# The rules, as merged on instance1 at the round-20 boundary. PLAN_UPTO1 must be on or the\n"
       "# filter hands every \"choose up to 1\" menu -- i.e. every deck search -- back to the model.\n"
       "WRAP=${WRAP:-lethal_now,clops_hold,judge_timing,spare_ex_bench,retreat_energy,stadium_replace,search_bottom,setup_search}\n"
       "PF=\"planfilter:$WRAP:\"\n"
       "export PLAN_UPTO1=1\n"
       "OPPS=${OPPS:-marnie_grimmsnarl,")
assert s.count(old) == 1, "opps anchor"
s = s.replace(old, new, 1)

# collection through the filter
old = '--model "qwen:$PREV"'
new = '--model "${PF}qwen:$PREV"'
assert s.count(old) == 1, "collect anchor"
s = s.replace(old, new)

# and every gate arm through it too
for a, b in (('--arm "prev=qwen:$PREV"', '--arm "prev=${PF}qwen:$PREV"'),
             ('--arm "a=qwen:/root/out/lora_${TAG}_a"', '--arm "a=${PF}qwen:/root/out/lora_${TAG}_a"'),
             ('--arm "b=qwen:/root/out/lora_${TAG}_b"', '--arm "b=${PF}qwen:/root/out/lora_${TAG}_b"')):
    assert s.count(a) == 1, "arm anchor %r" % a
    s = s.replace(a, b)

s = s.replace("did this round beat the policy it started from?",
              "did this round beat the FILTERED policy it started from?")

open("/root/night6.sh", "w").write(s)
os.chmod("/root/night6.sh", 0o755)
print("wrote /root/night6.sh")
