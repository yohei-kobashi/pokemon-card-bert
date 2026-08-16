import os
p = "/root/statusd.sh"
s = open(p).read()
# A finished job stops writing, which is indistinguishable from a hung one by log age alone.
# Ask whether the launcher is still alive; only then does a quiet log mean trouble.
old = '''        [ -f /root/hole_$i.log ] && {
            C=$(cells /root/hole_$i.log)
            say "    hole gate s$i $C"
            case "$C" in *STALE*) ALERTS="$ALERTS hole-s$i-stale";; esac
        }'''
new = '''        [ -f /root/hole_$i.log ] && {
            C=$(cells /root/hole_$i.log)
            if pgrep -f "[h]ole_launch.sh" >/dev/null; then
                say "    hole gate s$i $C"
                case "$C" in *STALE*) ALERTS="$ALERTS hole-s$i-stale";; esac
            else
                say "    hole gate s$i $C (launcher exited -- finished)"
            fi
        }'''
assert s.count(old) == 1, "anchor"
open(p + ".n", "w").write(s.replace(old, new))
os.replace(p + ".n", p)
os.chmod(p, 0o755)
print("statusd: a quiet log only alerts while its launcher is still alive")
