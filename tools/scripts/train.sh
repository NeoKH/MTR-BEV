#!/usr/bin/env bash
# ln -s /gemini/output ./output
python ./tools/train.py \
    --cfg_file './tools/cfgs/5_percent_bev_base.yaml' \
    --batch_size 1 --epochs 30 \
    --extra_tag my_1st_exp \
    --fix_random_seed \
    --logger_iter_interval 100 \
    --add_worker_init_fn \
    --workers 2 \
    --start_epoch 0
