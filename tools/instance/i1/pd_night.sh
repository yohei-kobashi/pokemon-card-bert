#!/usr/bin/env bash
# Overnight: where does each game stop on the Phantom Dive chain, across the field?
#
# Four opponents chosen to span the failure modes we know about: ogerpon_mono strips energy with
# four Crushing Hammers (25% of those games never reach a payable attacker), marnie is the
# opposite end where we already manage 1.3 dives a game, dragapult is the mirror-ish matchup we
# do best in, and ethan_hooh is a fast deck that should punish a slow setup. If the chain breaks
# in the same place against all four, it is ours; if it moves, it is the matchup.
#
# 300 games each so the per-game percentages carry ~3pt of noise rather than ~14.
set -u
cd /root/ptcg/repo_sb
W=lethal_now,clops_hold,judge_timing,spare_ex_bench,retreat_energy,stadium_replace,search_bottom,setup_search
CKPT=${CKPT:-/root/out/fld_r23b}
for D in ogerpon_mono marnie_grimmsnarl dragapult ethan_hooh; do
    PYTHONPATH=cg-lib:tools SB_UPTO1=1 nice -n 12 python3 -u tools/dusk_ogerpon_audit.py \
        --games 300 --opp "$D" --fmt dusk --seed 90000 \
        --mirror-so /root/ptcg/repo_sb/data/kaggle_engine_ext/libcg_mirror.so \
        --spec "planfilter:$W:hf:$CKPT" > "/root/pdn_$D.log" 2>&1 &
done
wait

{
    echo "=============================================================="
    echo " PHANTOM DIVE FORENSICS -- $(date -u +%Y-%m-%d\ %H:%M) UTC, champion $(basename $CKPT)"
    echo " human model for the bomb variant: Drakloak on turn 2, Dragapult ex on 3-4,"
    echo " then three Phantom Dives and one Cursed Bomb ends the game"
    echo "=============================================================="
    for D in ogerpon_mono marnie_grimmsnarl dragapult ethan_hooh; do
        echo
        echo "######## $D ########"
        grep -a "=== dragapult" "/root/pdn_$D.log"
        sed -n '/PHANTOM DIVE FORENSICS/,/when could we/p' "/root/pdn_$D.log" | head -22
        sed -n '/the opening, by OUR turn/,/first Phantom Dive/p' "/root/pdn_$D.log" | grep -v "off-line"
        sed -n '/first or second/,/other contexts/p' "/root/pdn_$D.log"
    done
    echo
    echo "== how to read the chain =="
    echo "Each row needs the row above it. A big drop between two rows is where the games are"
    echo "lost, and each drop has a different remedy:"
    echo "  in play -> can pay      : energy. Crispin and Night Stretcher are the levers we own."
    echo "  can pay -> is ACTIVE    : promotion. retreat_energy forbids retreating a body that"
    echo "                            carries {R}/{P}, which is exactly the body we want in front."
    echo "  is ACTIVE -> used       : the choice. Forcing it measured +0.33 +- 0.87, so this step"
    echo "                            is already fine and is NOT where to spend effort."
} > /root/pd_report.txt 2>&1
echo "PD_NIGHT_DONE" >> /root/pd_report.txt
