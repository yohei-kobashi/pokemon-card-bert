set -u
cd /root/ptcg/repo
export PYTHONPATH=cg-lib HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
G=/root/loop_dusk/gate_plan; rm -rf $G; mkdir -p $G
i=0
for OPPS in "marnie_grimmsnarl,alakazam_nz,alakazam" "crustle_geco,crustle,ogerpon_mono" "dudunsparce_box,cynthia_garchomp,dragapult" "mega_lucario_tr,slowking"; do
  nohup python3 -u tools/gate_protagonist.py --deck dragapult_dusknoir --opp "$OPPS" \
    --games 150 --seed $((1000 + i*100)) --baseline s1 \
    --arm "s1=hf:/root/out/dusk_s1@dusk" --arm "plan=hf:/root/out/plan_dusk@dusk" \
    --out $G/shard$i.json > $G/shard$i.log 2>&1 &
  i=$((i+1)); sleep 45
done
wait
python3 -u /root/dusk_gate_pool.py $G || echo "pool failed"
echo GATE_PLAN_DONE
