#!/usr/bin/env bash
# ln -s /gemini/output ./output
TAG='5_percent_bev_base'
EXP='my_2nd_exp'
python ./tools/train.py \
    --cfg_file ./tools/cfgs/$TAG.yaml \
    --batch_size 1 --epochs 30 \
    --extra_tag $EXP \
    --fix_random_seed \
    --logger_iter_interval 100 \
    --add_worker_init_fn \
    --workers 2 \
    --start_epoch 0 \
    --pretrained_model ./output/$TAG/my_2nd_exp/ckpt/latest_model.pth
