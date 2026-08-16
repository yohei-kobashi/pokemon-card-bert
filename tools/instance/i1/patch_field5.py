import os

p = "/root/field_chain.sh"
s = open(p).read()

# Each round writes TWO 746 MB DeBERTa checkpoints and keeps at most one. Running to 08-15 on a
# 60 G disk with 17 G free, that is ~10 rounds to exhaustion -- and a full disk here does not fail
# politely: it corrupts the checkpoint being written. instance1 already hit 100% once, on 08-11.
helpers = '''prune_ckpts() {  # keep the champion and ONE fallback; a rejected arm is 746 MB of nothing
    local keep1="$1" keep2="$2" d
    for d in /root/out/fld_r*[ab]; do
        [ -d "$d" ] || continue
        [ "$d" = "$keep1" ] && continue
        [ "$d" = "$keep2" ] && continue
        rm -rf "$d"
    done
}

disk_ok() {      # refuse to START a train we cannot finish, rather than truncate one mid-write
    local free
    free=$(df -BG /root | awk 'NR==2{gsub("G","",$4); print $4}')
    [ "${free:-0}" -ge "${1:-5}" ]
}

'''
anchor = 'ok_gz() {'
assert s.count(anchor) == 1
s = s.replace(anchor, helpers + anchor, 1)

# Prune right after the verdict, once CUR is final for the round.
old = '    say "round $R winner: $WIN"'
new = '''    say "round $R winner: $WIN"
    PREV_CUR=${PREV_CUR:-}'''
assert s.count(old) == 1, ("verdict", s.count(old))
s = s.replace(old, new)

old2 = '''        CUR=/root/out/fld_r$R$WIN
        echo "$CUR" > "$STATE/current.txt"
        say "new champion: $CUR"'''
new2 = '''        PREV_CUR=$CUR
        CUR=/root/out/fld_r$R$WIN
        echo "$CUR" > "$STATE/current.txt"
        say "new champion: $CUR"'''
assert s.count(old2) == 1, ("adopt", s.count(old2))
s = s.replace(old2, new2)

# after the if/else that sets CUR
old3 = '''            --note "FIELD-gated champion, round $R arm $WIN (vs $OPPS, not the mirror)" || true
    fi
done'''
new3 = '''            --note "FIELD-gated champion, round $R arm $WIN (vs $OPPS, not the mirror)" || true
    fi
    prune_ckpts "$CUR" "$PREV_CUR"
    rm -f /root/fld_log$R.jsonl.gz "$STATE"/rows_r$R[ab].jsonl.gz
    say "disk: $(df -BG /root | awk 'NR==2{print $4}') free after pruning round $R"
done'''
assert s.count(old3) == 1, ("prune", s.count(old3))
s = s.replace(old3, new3)

# Guard before each training, with the same message shape as the other STOPs.
old4 = '''        gpu_wait
        say "train $V: beta $BETA'''
new4 = '''        gpu_wait
        disk_ok 5 || { say "STOP: only $(df -BG /root | awk 'NR==2{print $4}') free -- refusing to start a train that cannot finish"; exit 1; }
        say "train $V: beta $BETA'''
assert s.count(old4) == 1, ("diskguard", s.count(old4))
s = s.replace(old4, new4)

t = p + ".new"
open(t, "w").write(s)
os.chmod(t, 0o755)
os.replace(t, p)
print("field_chain: prunes rejected checkpoints each round, refuses to train under 5 G free")
