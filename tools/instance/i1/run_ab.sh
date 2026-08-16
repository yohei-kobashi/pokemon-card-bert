set -u
for D in rockets_honchkrow rockets_mewtwo; do
  echo "########## $D  BASELINE (repo) ##########"
  cd /root/ptcg/repo && CUDA_VISIBLE_DEVICES="" python3 /root/probe_honchkrow.py "$D" 900 48 2>&1 | tail -26
  echo "########## $D  FIXED (repo_fix, basic_search) ##########"
  cd /root/ptcg/repo_fix && CUDA_VISIBLE_DEVICES="" python3 /root/probe_fix.py "$D" 900 48 2>&1 | tail -26
done
echo AB_DONE
