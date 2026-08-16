import os
p = "/root/gend2.sh"
s = open(p).read()
old = 'CKPT=${CKPT:-/root/out/fld_r11a}'
new = ('# The champion moves now (MIN_GAIN=0.0 adopts on any positive delta), so the checkpoint is\n'
       '# read fresh each round from the pointer instance1 writes AFTER the weights have landed.\n'
       'CKPT_PTR=${CKPT_PTR:-/root/out/champion.txt}')
assert s.count(old) == 1, "ckpt anchor"
s = s.replace(old, new)

old2 = '[ -s "$CKPT/model.safetensors" ] || { say "STOP: $CKPT has no weights"; exit 1; }\nsay "start.'
new2 = 'say "start.'
assert s.count(old2) == 1, "guard anchor"
s = s.replace(old2, new2)

old3 = 'say "start. protagonist = planfilter + $CKPT (DeBERTa), opponents = reg (per-deck 4B)"'
new3 = 'say "start. protagonist = planfilter + the champion from $CKPT_PTR, opponents = reg (per-deck 4B)"'
assert s.count(old3) == 1, "say anchor"
s = s.replace(old3, new3)

old4 = '    R=$((R + 1)); echo "$R" > /root/gend_round'
new4 = '''    R=$((R + 1)); echo "$R" > /root/gend_round
    CKPT=$(cat "$CKPT_PTR" 2>/dev/null)
    if [ -z "$CKPT" ] || [ ! -s "$CKPT/model.safetensors" ]; then
        say "no usable champion at $CKPT_PTR yet -- waiting 120s"; sleep 120; R=$((R - 1)); continue
    fi
    [ "$CKPT" = "${SEEN:-}" ] || { say "champion for this round: $CKPT"; SEEN=$CKPT; }'''
assert s.count(old4) == 1, "loop anchor"
s = s.replace(old4, new4)
s = s.replace("R=$(cat /root/gend_round 2>/dev/null || echo 0)",
              "R=$(cat /root/gend_round 2>/dev/null || echo 0)\nSEEN=", 1)
open(p + ".n", "w").write(s)
os.replace(p + ".n", p)
os.chmod(p, 0o755)
print("gend2 reads the champion pointer each round")
