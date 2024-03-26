
import os
import subprocess
import json
import sys

total = []

for K in range(1, 21):
    
    result = subprocess.run([f"python3", "-m torch.distributed.launch --master_port 47491 --nproc_per_node=8 --use_env main.py --ravitt_t 1.0 --ravitt_mode full --model deit_tiny_patch16_224 --batch-size 128 --data-path ../datasets/imagenet/data --eval --resume models/ravitt1.0-nocut-nomix/best_checkpoint.pth  --ravitt_k {K}"], stdout=subprocess.PIPE, cwd='/home/fquezada/Documents/deit-ravitt2').stdout.decode('utf-8')
    result = result.split("* Acc@1 ")
    print(K, result)
    total.append(result)

print(total)
