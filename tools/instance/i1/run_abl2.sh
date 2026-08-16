set -u
for A in full empty no:card_roles \
         roles:463=win+473=win \
         roles:463=win+473=win+414=win \
         roles:891=line+474=line \
         roles:463=win+473=win+414=win+891=engine+474=engine ; do
  PROBE_ROOT=/root/ptcg/repo PROBE_ARM="$A" CUDA_VISIBLE_DEVICES="" python3 /root/probe3.py rockets_honchkrow 1200 40
done
echo ABL2_DONE
