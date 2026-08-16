import os
src = open('/root/gend2.sh').read()
import re
new, n = re.subn(r'2026-08-1\dT12:00:00Z', '2026-08-16T22:00:00Z', src)
open('/root/gend2.sh.new', 'w').write(new)
os.replace('/root/gend2.sh.new', '/root/gend2.sh')
print('replaced', n)
print([l for l in new.splitlines() if 'STOP_AFTER' in l][:2])
