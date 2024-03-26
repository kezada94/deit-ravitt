#!/bin/bash

n=$1

if [ $n == 0 ]
then
	srun -p gpu --container-workdir=${PWD} --gres=gpu:A100:4 --container-name=cuda-12 --pty python3 -m torch.distributed.launch --master_port 42492 --nproc_per_node=4 --use_env main.py --ravitt_t 0.0 --ravitt_mode none --model deit_tiny_patch16_224 --finetune models/base-lr1e-3-nocut-nomix/best_checkpoint.pth --batch-size 256 --data-path ../../datasets/HER2-4  --output_dir models/base-lr1e-3-nocut-nomix-ft-her2_4-base --lr 0.001 --cutmix 0.0 --mixup 0.0 --data-set HER2-4 --classic yes
elif [ $n == 1 ]
then
	srun -p gpu --container-workdir=${PWD} --gres=gpu:A100:4 --container-name=cuda-12 --pty python3 -m torch.distributed.launch --master_port 42492 --nproc_per_node=4 --use_env main.py --ravitt_t 1.0 --ravitt_mode full --model deit_tiny_patch16_224 --finetune models/ravitt1.0-nocut-nomix/best_checkpoint.pth  --batch-size 256 --data-path ../../datasets/HER2-4  --output_dir models/ravitt1.0-lr1e-3-nocut-nomix-ft-her2_4-raviit1.0 --lr 0.001 --cutmix 0.0 --mixup 0.0 --data-set HER2-4 --classic yes
elif [ $n == 2 ]
then
	srun -p gpu --container-workdir=${PWD} --gres=gpu:A100:4 --container-name=cuda-12 --pty python3 -m torch.distributed.launch --master_port 42492 --nproc_per_node=4 --use_env main.py --ravitt_t 2.0 --ravitt_mode full --model deit_tiny_patch16_224 --finetune models/ravitt2.0-lr1e-3-nocut-nomix/best_checkpoint.pth  --batch-size 256 --data-path ../../datasets/HER2-4  --output_dir models/ravitt2.0-lr1e-3-nocut-nomix-ft-her2_4-raviit2.0 --lr 0.001 --cutmix 0.0 --mixup 0.0 --data-set HER2-4 --classic yes
elif [ $n == 3 ]
then
	srun -p gpu --container-workdir=${PWD} --gres=gpu:A100:4 --container-name=cuda-12 --pty python3 -m torch.distributed.launch --master_port 42492 --nproc_per_node=4 --use_env main.py --ravitt_t 3.0 --ravitt_mode full --model deit_tiny_patch16_224 --finetune models/ravitt3.0-lr1e-3-nocut-nomix/best_checkpoint.pth  --batch-size 256 --data-path ../../datasets/HER2-4  --output_dir models/ravitt3.0-lr1e-3-nocut-nomix-ft-her2_4-raviit3.0 --lr 0.001 --cutmix 0.0 --mixup 0.0 --data-set HER2-4 --classic yes
elif [ $n == 4 ]
then
	srun -p gpu --container-workdir=${PWD} --gres=gpu:A100:4 --container-name=cuda-12 --pty python3 -m torch.distributed.launch --master_port 42492 --nproc_per_node=4 --use_env main.py --ravitt_t 4.0 --ravitt_mode full --model deit_tiny_patch16_224 --finetune models/ravitt4.0-lr1e-3-nocut-nomix/best_checkpoint.pth  --batch-size 256 --data-path ../../datasets/HER2-4  --output_dir models/ravitt4.0-lr1e-3-nocut-nomix-ft-her2_4-raviit4.0 --lr 0.001 --cutmix 0.0 --mixup 0.0 --data-set HER2-4 --classic yes
elif [ $n == 5 ]
then
	srun -p gpu --container-workdir=${PWD} --gres=gpu:A100:4 --container-name=cuda-12 --pty python3 -m torch.distributed.launch --master_port 42492 --nproc_per_node=4 --use_env main.py --ravitt_t 5.0 --ravitt_mode full --model deit_tiny_patch16_224 --finetune models/ravitt5.0-lr1e-3-nocut-nomix/best_checkpoint.pth  --batch-size 256 --data-path ../../datasets/HER2-4  --output_dir models/ravitt5.0-lr1e-3-nocut-nomix-ft-her2_4-raviit5.0 --lr 0.001 --cutmix 0.0 --mixup 0.0 --data-set HER2-4 --classic yes
elif [ $n == 6 ]
then
	srun -p gpu --container-workdir=${PWD} --gres=gpu:A100:4 --container-name=cuda-12 --pty python3 -m torch.distributed.launch --master_port 42492 --nproc_per_node=4 --use_env main.py --ravitt_t 6.0 --ravitt_mode full --model deit_tiny_patch16_224 --finetune models/ravitt6.0-lr1e-3-nocut-nomix/best_checkpoint.pth  --batch-size 256 --data-path ../../datasets/HER2-4  --output_dir models/ravitt6.0-lr1e-3-nocut-nomix-ft-her2_4-raviit6.0 --lr 0.001 --cutmix 0.0 --mixup 0.0 --data-set HER2-4 --classic yes
elif [ $n == 7 ]
then
	srun -p gpu --container-workdir=${PWD} --gres=gpu:A100:4 --container-name=cuda-12 --pty python3 -m torch.distributed.launch --master_port 42492 --nproc_per_node=4 --use_env main.py --ravitt_t 7.0 --ravitt_mode full --model deit_tiny_patch16_224 --finetune models/ravitt7.0-lr1e-3-nocut-nomix/best_checkpoint.pth  --batch-size 256 --data-path ../../datasets/HER2-4  --output_dir models/ravitt7.0-lr1e-3-nocut-nomix-ft-her2_4-raviit7.0 --lr 0.001 --cutmix 0.0 --mixup 0.0 --data-set HER2-4 --classic yes
elif [ $n == 9 ]
then
	srun -p gpu --container-workdir=${PWD} --gres=gpu:A100:8 --container-name=cuda-12 --pty python3 -m torch.distributed.launch --master_port 42492 --nproc_per_node=8 --use_env main.py --ravitt_t 9.0 --ravitt_mode full --model deit_tiny_patch16_224 --finetune models/ravitt9.0-lr1e-3-nocut-nomix/best_checkpoint.pth  --batch-size 128 --data-path ../../datasets/HER2-4  --output_dir models/ravitt9.0-lr1e-3-nocut-nomix-ft-her2_4-raviit9.0 --lr 0.001 --cutmix 0.0 --mixup 0.0 --data-set HER2-4 --classic yes
fi
