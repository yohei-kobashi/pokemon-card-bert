#!/usr/bin/env bash
# At loop end (23:05Z): ALWAYS pre-build dusk_v4 with the final champion. Even if the champion
# never moved past fld_r49b, the BUNDLE has moved since dusk_v3: wrap 18 -> 21 rules
# (hammer_spare, lethal_line, draw_cap), the engine_v2 thin-board search fix, and the
# lethal_line/phantom_dive_chip dusk_plan. Build only -- the morning submit decision belongs
# to the user.
until [ "$(date -u +%s)" -ge "$(date -u -d 2026-08-16T23:05:00Z +%s)" ]; do sleep 120; done
CUR=$(cd /root/ptcg/repo && python3 -c "from lm import registry as r; print(r.resolve(\"dragapult_dusknoir\")[\"target\"])" 2>/dev/null)
echo "[prep $(date -u +%H:%M)] final champion: $CUR -- building dusk_v4" >> /root/sub4_run.log
bash /root/submit_dusk_v4.sh >> /root/sub4_run.log 2>&1
