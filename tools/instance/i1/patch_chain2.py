import os
import shutil

p = "/root/mirror_chain2.sh"
s = open(p).read()

old = '        EPOCHS=$(python3 -c "print(round(min(2.0, 8000.0/$NR), 2))")'
new = (old + "\n"
       "        # The sweep pins these when it finds a setting whose two row orders agree;\n"
       "        # unset, the round keeps the original recipe.\n"
       '        [ -n "${EPOCHS_FIX:-}" ] && EPOCHS=$EPOCHS_FIX')
assert old in s, "epochs line not found"
s = s.replace(old, new, 1)

old2 = ('            --out /root/out/mrl2_r$R$V --lr 1e-5 --epochs "$EPOCHS" '
        '--accum 4 --l2sp 1e-3 \\')
new2 = ('            --out /root/out/mrl2_r$R$V --lr "${LR:-1e-5}" --epochs "$EPOCHS" '
        '--accum 4 \\\n            --l2sp "${L2SP:-1e-3}" \\')
assert old2 in s, "train line not found"
s = s.replace(old2, new2, 1)

old3 = 'say "train $V: beta $BETA temp $TEMP, $NR rows, epochs $EPOCHS"'
new3 = ('say "train $V: beta $BETA temp $TEMP, $NR rows, epochs $EPOCHS, '
        'lr ${LR:-1e-5}, l2sp ${L2SP:-1e-3}"')
assert old3 in s, "say line not found"
s = s.replace(old3, new3, 1)

open(p + ".new", "w").write(s)
shutil.copymode(p, p + ".new")
os.replace(p + ".new", p)
print("mirror_chain2.sh parameterised")
