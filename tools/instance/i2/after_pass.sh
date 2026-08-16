#!/usr/bin/env bash
# After the five-deck pass: swap the sparring dusknoir, then add slowking.
#
# The swap waits for the pass to finish rather than landing mid-deck (user's call): an adapter
# trained half against mrl_r2 and half against a stronger champion would be measured against a
# moving target, and its gate -- new vs cur, both facing the same opponent -- would silently
# stop being a paired comparison.
#
# SLOWKING IS NOT LIKE THE OTHERS. Measured over 150 games in the fleet loop: 1,016 decisions
# per game against 120-152 for every other deck, and 1.13 pairs per 1,000 decisions against
# 8.0-8.7. Left on the standard settings it would collect for hours and still come back under
# the 500-pair floor, because dpo_branch samples at most --per-game points from each game and 15
# of 1,016 is 1.5%. So: fewer games, far more branch points per game.
set -u
export GAMES PER_GAME BUDGET
say() { echo "[after_pass $(date -u +%m-%d_%H:%M:%S)] $*"; }

# WAIT ON THE PROCESS, NOT ON A GREP OF THE WHOLE LOG. The first attempt greped for
# ALL_DECK_LORAS_DONE anywhere in deck_loras.log -- and matched the line left by the run that
# failed all five decks at 03:20 on a missing rl_config.DUSK_FMT. It fired instantly: slowking
# started while alakazam_nz was still mid-round, and the decklist and sparring champion both
# moved underneath a pass that was supposed to see one opponent throughout.
say "waiting for the five-deck pass"
while pgrep -f "[d]eck_loras.sh" >/dev/null; do sleep 120; done
if ! tail -3 /root/deck_loras.log 2>/dev/null | grep -aq "ALL_DECK_LORAS_DONE"; then
    say "deck_loras exited without ALL_DECK_LORAS_DONE at the end of its log -- stopping"
    exit 1
fi
say "pass complete"

# ---------------------------------------------------------------- 0. the decklist change
# Dawn -> Judge. Dawn searched a Basic/Stage1/Stage2 into HAND at SUPPORTER cost, and the deck
# already fetches the whole Dusknoir line with items -- Poke Pad (x4) takes any Pokemon without
# a Rule Box, which all three of Duskull/Dusclops/Dusknoir are, and Dusknoir reaches the hand in
# 90.2% of games without help. Meanwhile the supporter slot is the deck's scarcest resource:
# Boss's Orders is played on 3.8% of the menus that offer it because Crispin and Lillie's win the
# slot. Judge answers something real instead: Alakazam's Powerful Hand places 2 damage counters
# PER CARD IN ITS OWN HAND, so shuffling both hands to 4 halves its damage, and alakazam_nz is
# 15.6% of the ladder. Millar's NAIC list runs the Judge and no Dawn; this restores that.
#
# Applied HERE, at the pass boundary, so all five opponent adapters trained against one list.
# The new list is STAGED at /root/dusknoir_new.csv and moved into place here, not rsynced when
# it was authored: a deck file copied mid-pass would have given dragapult and dudunsparce_box a
# different opponent from marnie and ogerpon_mono, and their gates are "new vs cur against the
# same dusknoir" -- which stops being true the moment the list moves.
say "installing the updated dragapult_dusknoir decklist"
cd /root/ptcg/repo
if [ -s /root/dusknoir_new.csv ]; then
    cp decks/dragapult_dusknoir.csv /root/dusknoir_old.csv
    cp /root/dusknoir_new.csv decks/dragapult_dusknoir.csv
    say "decklist installed (previous kept at /root/dusknoir_old.csv)"
else
    say "no staged decklist at /root/dusknoir_new.csv -- keeping the current one"
fi
PYTHONPATH=cg-lib python3 -c "
import sys, collections; sys.path.insert(0,'.')
ids=[int(x) for x in open('decks/dragapult_dusknoir.csv') if x.strip()]
assert len(ids)==60, len(ids)
assert 1213 in ids and 1231 not in ids, 'Judge/Dawn swap did not land'
print('[deck] 60 cards, Judge in, Dawn out')
" || { say "DECK CHECK FAILED -- not proceeding"; exit 1; }

# ---------------------------------------------------------------- 1. swap the sparring dusknoir
cd /root/ptcg/repo
NEW=""
for C in mrl2_r7b mrl2_r7a mrl2_r6b mrl2_r6a mrl2_r5b; do
    [ -f "/root/out/$C/model.safetensors" ] && { NEW=$C; break; }
done
if [ -n "$NEW" ]; then
    WRAP_ARG=""
    if [ -s /root/DUSK_WRAP ]; then
        WRAP_ARG="--wrap $(cat /root/DUSK_WRAP)"
        say "rule wrapper: $(cat /root/DUSK_WRAP)"
    fi
    PYTHONPATH=cg-lib python3 tools/adapters.py set dragapult_dusknoir \
        --target "hf:$NEW" --fmt dusk $WRAP_ARG \
        --note "sparring champion, swapped after the five-deck pass" \
        && say "sparring dusknoir -> $NEW"
    PYTHONPATH=cg-lib python3 tools/adapters.py check || true
else
    say "no newer champion present on this machine -- keeping mrl_r2"
fi

# ---------------------------------------------------------------- 2. slowking
say "slowking round 1: 3 x 100 games, per-game 60 (7x the decisions, 1/7 the pair yield)"
GAMES=100 PER_GAME=60 BUDGET=12000 bash /root/deck_lora2.sh slowking 1 \
    && say "slowking done" || say "slowking FAILED"
say "AFTER_PASS_DONE"
