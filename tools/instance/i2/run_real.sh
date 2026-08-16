cd /root
python3 sft_teacher.py --limit 150000 --epochs 1 --bsz 8 --accum 4 --eval-n 4000 --save-steps 400 --out /root/out/teacher9b
