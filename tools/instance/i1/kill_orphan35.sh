#!/usr/bin/env bash
# Kill the round-35 collect python that the old field_chain spawned in the 30s race
# between its round-34 winner line and restart3's poll. It carries the OLD wrap
# (no front_dive) -- identified by that absence in its planfilter argument.
for P in $(pgrep -f 'setup_search:hf:/root/out/fld_r29a'); do
    if ! tr '\0' ' ' < /proc/$P/cmdline | grep -q front_dive; then
        kill "$P" 2>/dev/null && echo "killed $P"
    fi
done
sleep 5
for P in $(pgrep -f 'setup_search:hf:/root/out/fld_r29a'); do
    if ! tr '\0' ' ' < /proc/$P/cmdline | grep -q front_dive; then
        kill -9 "$P" 2>/dev/null && echo "killed -9 $P"
    fi
done
echo done
