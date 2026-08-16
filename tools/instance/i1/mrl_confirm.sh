set -u
cd /root/ptcg/repo
export PYTHONPATH=cg-lib HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
O=/root/loop_dusk/mrl_confirm; mkdir -p $O
# Which mirror checkpoint is actually best? Round 3 was ADOPTED while measuring -5.00pt, because
# the verdict demanded d<=-2 AND t<=-2 and at n=320 the SE is 3.6pt -- t=-2 needs -7.2pt, so
# every drop between -2 and -7pt passed. Settle r2 vs r3 vs s1 head to head at 600 games.
python3 -u tools/gate_protagonist.py --deck dragapult_dusknoir --opp dragapult_dusknoir \
  --games 600 --seed 33000 --baseline s1 \
  --arm "s1=hf:/root/out/dusk_s1@dusk" \
  --arm "r1=hf:/root/out/mrl_r1@dusk" \
  --arm "r2=hf:/root/out/mrl_r2@dusk" \
  --arm "r3=hf:/root/out/mrl_r3@dusk" \
  --out $O/confirm.json 2>&1 | tail -14
echo MRL_CONFIRM_DONE
