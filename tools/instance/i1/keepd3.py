import os
p = "/root/keepd.sh"
s = open(p).read()

# instance2's job changed from running its own RL rounds to generating traces, and the supervisor
# was still watching night6. The generator then died -- I killed it to patch it and the restart
# did not land -- and nobody noticed for half an hour. Watch what is actually running now.
old = '''    R=$(ssh $I2 -p $P $I2HOST '
        pgrep -f "[n]ight_chain.sh" >/dev/null || grep -aq CHAIN_DONE /root/night_chain.log 2>/dev/null \\
            || setsid --fork nohup bash /root/night_chain.sh >/dev/null 2>&1 </dev/null
        if pgrep -f "[n]ight6.sh|[n]ight_run.sh" >/dev/null; then echo RUNNING
        elif grep -aq NIGHT6_DONE /root/night6.log 2>/dev/null; then echo DONE
        else echo GONE; fi' 2>/dev/null)'''
new = '''    R=$(ssh $I2 -p $P $I2HOST '
        if pgrep -f "[g]end2.sh" >/dev/null; then echo RUNNING
        elif [ ! -s /root/out/champion.txt ]; then echo WAITING
        else echo GONE; fi' 2>/dev/null)'''
assert s.count(old) == 1, "i2 check anchor"
s = s.replace(old, new)

old2 = '''    case "${R:-UNREACHABLE}" in
        RUNNING|DONE) ;;
        GONE)
            say "night6 is not running and has not finished -- relaunching through night_run"
            ssh $I2 -p $P $I2HOST 'bash /root/go6.sh' 2>/dev/null \\
                && say "relaunched" || say "relaunch FAILED"
            ;;
        *) say "instance2 unreachable" ;;
    esac'''
new2 = '''    case "${R:-UNREACHABLE}" in
        RUNNING) ;;
        WAITING) say "instance2 has no champion yet -- ckptd has not delivered one" ;;
        GONE)
            say "the trace generator is down -- restarting it"
            ssh $I2 -p $P $I2HOST 'bash /root/go_gen.sh' 2>/dev/null \\
                && say "restarted" || say "restart FAILED"
            ;;
        *) say "instance2 unreachable" ;;
    esac
    # and the checkpoint sync, which is what keeps the generator on the current champion
    pgrep -f "[c]kptd.sh" >/dev/null || {
        say "ckptd down -- restarting"
        setsid --fork nohup bash /root/ckptd.sh >/dev/null 2>&1 </dev/null
    }
    pgrep -f "[g]enpull.sh" >/dev/null || {
        say "genpull down -- restarting"
        setsid --fork nohup bash /root/genpull.sh >/dev/null 2>&1 </dev/null
    }'''
assert s.count(old2) == 1, "case anchor"
s = s.replace(old2, new2)
open(p + ".n", "w").write(s)
os.replace(p + ".n", p)
os.chmod(p, 0o755)
print("keepd now supervises gend2 (i2), ckptd and genpull (i1)")
