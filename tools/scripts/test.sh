#!/usr/bin/env bash
TAG='5_percent_bev_base'
EXP='my_1st_exp'
python ./tools/demo.py \
    --eval_all --batch_size 2 --workers 4 \
    --extra_tag $EXP \
    --cfg_file "./tools/cfgs/$TAG.yaml"
    # --ckpt_id 30
TAG='5_percent_no_bev'
python ./tools/demo.py \
    --eval_all --batch_size 2 --workers 4 \
    --extra_tag $EXP \
    --cfg_file "./tools/cfgs/$TAG.yaml"
