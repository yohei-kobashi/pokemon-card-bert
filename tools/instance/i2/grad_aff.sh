set -u
cd /root/ptcg/repo
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
python3 tools/instance/dpo_teacher.py \
  --data /root/dpo_r5.jsonl.gz --init-from /root/out/dpo_r5 --ref-from /root/out/i2_r7 \
  --card-first /root/ptcg/repo/data/cardfirst_b_v39.json --out /root/out/discard_aff \
  --beta 0.1 --cdpo-calibrated --grad-affinity /root/loop_dpo/grad_affinity.json
echo GRAD_AFFINITY_DONE
