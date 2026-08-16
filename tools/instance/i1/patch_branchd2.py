import os

p = "/root/branchd2.sh"
s = open(p).read()

# (1) Heartbeat.  instance2 cannot ssh to instance1, so a dead link is invisible from the side
#     that suffers from it -- the branch step just hangs for its full timeout.  The poll already
#     ssh's to instance2 every 120 s; make it leave a mtime behind, and instance2's preflight can
#     detect a dead link in one second instead of a hundred minutes.
old = """    REQ=$(ssh $I2 -p $I2PORT $I2HOST 'cat /root/branch_request2 2>/dev/null' 2>/dev/null \\"""
new = """    REQ=$(ssh $I2 -p $I2PORT $I2HOST 'touch /root/.branchd2_alive; cat /root/branch_request2 2>/dev/null' 2>/dev/null \\"""
assert s.count(old) == 1, "poll anchor"
s = s.replace(old, new)

# (2) Atomic delivery.  scp writes the reply under its final name, so `[ -s ]` on instance2 goes
#     true on the first byte.  night4b read it mid-transfer, got 1287 of 1626 rows and a gzip
#     that died on its own CRC.  Ship to .part and rename -- a rename is atomic, so the file
#     either is not there or is complete.
old2 = """    if scp $I2 -P $I2PORT /root/pairs_$TAG.jsonl.gz "$I2HOST:/root/pairs_$TAG.jsonl.gz"; then"""
new2 = """    if scp $I2 -P $I2PORT /root/pairs_$TAG.jsonl.gz "$I2HOST:/root/pairs_$TAG.jsonl.gz.part" \\
       && ssh $I2 -p $I2PORT $I2HOST "gzip -t /root/pairs_$TAG.jsonl.gz.part \\
              && mv -f /root/pairs_$TAG.jsonl.gz.part /root/pairs_$TAG.jsonl.gz"; then"""
assert s.count(old2) == 1, "ship anchor"
s = s.replace(old2, new2)

t = p + ".new"
open(t, "w").write(s)
os.replace(t, p)
os.chmod(p, 0o755)
print("patched")
