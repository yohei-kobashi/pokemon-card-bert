import os
p = "/root/restart_at_boundary.sh"
s = open(p).read()
old = 'say "waiting for round $WANT\'s verdict (then MIN_GAIN=0.0 takes effect)"'
new = ('# Idempotent: a supervisor (or a reboot) may start this again after it has already fired,\n'
       '# and a second firing would kill a round that is mid-collection for no reason.\n'
       'grep -aq RESTART_DONE "$LOG" 2>/dev/null && { say "already done -- nothing to do"; exit 0; }\n'
       + old)
assert s.count(old) == 1, "anchor"
open(p + ".n", "w").write(s.replace(old, new))
os.replace(p + ".n", p)
os.chmod(p, 0o755)
print("restart_at_boundary is now idempotent")
