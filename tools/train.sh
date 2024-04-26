#!/bin/bash

python tools/train.py \
    --save_dir ./results/babel/FlowMDM_retrained \
    --dataset babel \
    --batch_size 64 \
    --num_steps 1300000 \
    --rpe_horizon 100 \
    --train_platform_type WandbPlatform \
    --wandb_project mgpt \
    --wandb_entity lavaalex