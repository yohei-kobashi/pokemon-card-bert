#!/usr/bin/env bash
# vast periodically rewrites this instance's authorized_keys and drops instance1's key -- the
# documented silent killer of every i1->i2 transfer (see the two-instance-link memory). Keys
# listed in /root/keys_extra (a file vast does not touch) are re-appended within a minute of
# any rewrite.
while :; do
    while IFS= read -r k; do
        [ -n "$k" ] || continue
        grep -qF "$k" /root/.ssh/authorized_keys 2>/dev/null \
            || echo "$k" >> /root/.ssh/authorized_keys
    done < /root/keys_extra
    sleep 60
done
