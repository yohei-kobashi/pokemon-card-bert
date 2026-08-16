import os
p = "/root/keepd.sh"
s = open(p).read()
old = '''    if ! pgrep -f "[r]estart_at_boundary.sh" >/dev/null; then'''
new = '''    # the two decision watchers added after this supervisor was written
    for W in hole_adopt restart_at_boundary; do
        pgrep -f "[${W:0:1}]${W:1}.sh" >/dev/null && continue
        case "$W" in
            hole_adopt)  grep -aq HOLE_ADOPT_DONE /root/hole_adopt.log 2>/dev/null && continue ;;
            restart_at_boundary) grep -aq RESTART_DONE /root/restart.log 2>/dev/null && continue ;;
        esac
        say "$W down and not yet fired -- restarting"
        setsid --fork nohup bash /root/$W.sh >/dev/null 2>&1 </dev/null
    done
    if false; then'''
assert s.count(old) == 1, "anchor"
s = s.replace(old, new)

# instance2 also supervises its chain watcher now
old2 = '''        if pgrep -f "[n]ight6.sh|[n]ight_run.sh" >/dev/null; then echo RUNNING'''
new2 = '''        pgrep -f "[n]ight_chain.sh" >/dev/null || grep -aq CHAIN_DONE /root/night_chain.log 2>/dev/null \\
            || setsid --fork nohup bash /root/night_chain.sh >/dev/null 2>&1 </dev/null
        if pgrep -f "[n]ight6.sh|[n]ight_run.sh" >/dev/null; then echo RUNNING'''
assert s.count(old2) == 1, "i2 anchor"
s = s.replace(old2, new2)

open(p + ".n", "w").write(s)
os.replace(p + ".n", p)
os.chmod(p, 0o755)
print("keepd now supervises hole_adopt and night_chain too")
