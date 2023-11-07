#!/usr/bin/env bash
# ln -s /gemini/output ./output
TAG='100_percent_no_bev'
EXP='my_1st_exp'

python ./tools/train.py \
    --cfg_file ./tools/cfgs/$TAG.yaml \
    --batch_size 4 --epochs 30 \
    --extra_tag $EXP \
    --fix_random_seed \
    --logger_iter_interval 100 \
    --ckpt_save_interval 1 \
    --add_worker_init_fn \
    --workers 4 \
    --max_ckpt_save_num 20 \
    --start_epoch 9
    

# python ./tools/demo.py \
#     --eval_all --batch_size 4 --workers 2 \
#     --extra_tag $EXP \
#     --cfg_file "./tools/cfgs/$TAG.yaml"
