#!/bin/bash
# Let the loop finish the one thing still worth having -- the screen of rerank_loop4, the first
# model trained on the board-slot-deduped pool -- then stop it before it spends another 7 hours
# on a recipe that has already been judged.
#
# WHY STOP. Paired on the same 63 decks: base -> loop1 was +3.06pt with 0 decks becoming WORSE
# (16 left WORSE, exact p 0.000); loop1 -> loop2 was -4.25pt with 16 becoming WORSE (p 0.004);
# base -> loop2 is indistinguishable from the base (p 0.581). Two rounds netted nothing, and the
# loop's own policy is to retrain from BASE on an ever-growing DAgger pile, so a third round of
# the same recipe tests nothing new. rerank_loop2 stays the reference checkpoint.
#
# The trigger is history.tsv gaining its 4th round row, which the loop writes only AFTER the
# screen has merged and been summarised -- so the number is on disk before anything is killed.
set -u
LOG=/root/stop_after_screen.log
exec >> "$LOG" 2>&1
say() { echo "[stop $(date -u +%m-%d_%H:%M:%S)] $*"; }

HIST=/root/loop_rerank/history.tsv
have4() { awk '$1==4 {n++} END {exit !(n>=1)}' "$HIST" 2>/dev/null; }

say "watching $HIST for a round-4 row (currently $(wc -l < $HIST) rows)"
while ! have4; do
  if ! pgrep -f "[d]agger_loop2.sh" > /dev/null; then
    say "the loop exited on its own before writing a round-4 row -- nothing to stop"
    exit 0
  fi
  sleep 60
done
say "round-4 row present:"
tail -2 "$HIST"

# kill the loop first so it cannot start the collection, then whatever stage it had running
pkill -f "[d]agger_loop2.sh" && say "loop script stopped"
sleep 2
pkill -f "[c]ollect_dagger.py" && say "collection stopped"
pkill -f "[m]irror_match.py"   && say "leftover screen stopped"
say "GPU is free; rerank_loop2 remains the reference checkpoint"
