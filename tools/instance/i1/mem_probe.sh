set -u
CUDA_VISIBLE_DEVICES= taskset -c 0-3 python3 /root/time_bundle.py /root/ptcg/repo/submissions/rr_v37_crustle.tar.gz crustle 1 > /root/mem_run.log 2>&1 &
P=$!
PEAK=0
while kill -0 $P 2>/dev/null; do
  R=$(ps -o rss= -p $P 2>/dev/null | tr -d " ")
  [ -n "$R" ] && [ "$R" -gt "$PEAK" ] && PEAK=$R
  sleep 1
done
echo "PEAK RSS: $((PEAK/1024)) MiB"
cat /root/mem_run.log | grep -E "import|game 0"
echo "uncompressed bundle: $(tar tzvf /root/ptcg/repo/submissions/rr_v37_crustle.tar.gz | awk "{s+=\$3} END {print int(s/1048576)}") MiB"
echo MEM_DONE
