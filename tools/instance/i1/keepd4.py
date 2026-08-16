import os, re
p = "/root/keepd.sh"
lines = open(p).read().splitlines(True)

# instance2's job changed from running its own RL rounds to generating traces, and the supervisor
# was still watching night6. The generator then died -- I killed it to patch it, the restart did
# not land, and nobody noticed for half an hour. Watch what actually runs now, and add the two
# instance1 daemons that keep the pipe alive.
i = next(k for k, l in enumerate(lines) if l.lstrip().startswith("R=$(ssh $I2"))
j = next(k for k, l in enumerate(lines[i:], i) if l.strip() == "esac")
new = '''    R=$(ssh $I2 -p $P $I2HOST '
        if pgrep -f "[g]end2.sh" >/dev/null; then echo RUNNING
        elif [ ! -s /root/out/champion.txt ]; then echo WAITING
        else echo GONE; fi' 2>/dev/null)
    case "${R:-UNREACHABLE}" in
        RUNNING) ;;
        WAITING) say "instance2 has no champion yet -- ckptd has not delivered one" ;;
        GONE)
            say "the trace generator is down -- restarting it"
            ssh $I2 -p $P $I2HOST 'bash /root/go_gen.sh' 2>/dev/null \\
                && say "restarted" || say "restart FAILED"
            ;;
        *) say "instance2 unreachable" ;;
    esac
    # the two daemons that keep the pipe alive: without ckptd the generator runs a stale
    # champion, without genpull the traces never reach the loop that needs them
    for W in ckptd genpull; do
        pgrep -f "[${W:0:1}]${W:1}.sh" >/dev/null || {
            say "$W down -- restarting"
            setsid --fork nohup bash /root/$W.sh >/dev/null 2>&1 </dev/null
        }
    done
'''
open(p + ".n", "w").writelines(lines[:i] + [new] + lines[j + 1:])
os.replace(p + ".n", p)
os.chmod(p, 0o755)
print("rewrote the instance2 block (lines %d-%d)" % (i + 1, j + 1))
