#!/usr/bin/env bash
for P in $(pgrep -f 'gen_r41s'); do kill "$P" 2>/dev/null; done
rm -f /root/gen_out/gtr_r42s0_fld_r29a.jsonl.gz /root/gen_out/gtr_r42s1_fld_r29a.jsonl.gz
rm -f /root/gen_out/gen_r42s0.jsonl.gz /root/gen_out/gen_r42s1.jsonl.gz
echo cleaned
