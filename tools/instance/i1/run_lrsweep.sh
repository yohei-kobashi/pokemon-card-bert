set -u
cd /root/ptcg/repo
for LR in 1e-5 1e-4 5e-4; do
  echo "########## lr=$LR ##########"
  python tools/rank_probe.py --rollouts /root/out/branch8.jsonl.gz \
    --model /root/out/rlDL/A_r6_policy --epochs 1 --lr $LR --limit 12000 --trace-every 1500 2>&1 \
    | grep -E "benchmarks|limited|BEFORE|\[ *[0-9]+\]|^epoch"
done
echo "########## shuffle control at the best lr comes after ##########"
echo LRSWEEP_DONE
