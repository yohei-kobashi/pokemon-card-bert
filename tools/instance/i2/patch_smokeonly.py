import os

p = "/root/night_run.sh"
s = open(p).read()
old = 'echo "== launching the real run =="'
new = ('# SMOKE_ONLY exercises the harness itself without committing the night to a real run.\n'
       '[ "${SMOKE_ONLY:-0}" = 1 ] && { echo "== SMOKE_ONLY: harness validated, real run not started =="; exit 0; }\n\n'
       + old)
assert s.count(old) == 1
open(p + ".new", "w").write(s.replace(old, new))
os.replace(p + ".new", p)
os.chmod(p, 0o755)
print("SMOKE_ONLY added")
