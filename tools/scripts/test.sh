#!/usr/bin/env bash
TAG='5_percent_bev_base'
EXP='my_1st_exp'
EPOCH="30"

python ./tools/test.py \
    --cfg_file ./tools/cfgs/$TAG.yaml \
    --batch_size 2 \
    --workers 4 \
    --extra_tag my_1st_exp \
    --ckpt ./output/$TAG/$EXP/ckpt/checkpoint_epoch_$EPOCH.pth 
    # --start_epoch \
    # --eval_tag \
    # --eval_all \
