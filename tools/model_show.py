import _init_path
import argparse
import datetime
import glob
import os
import re
import time
from pathlib import Path

import numpy as np
import torch
from tensorboardX import SummaryWriter

from eval_utils import eval_utils
from mtr.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from mtr.datasets import build_dataloader
from mtr.models import model as model_utils
from mtr.utils import common_utils


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    
    # ckpt parameters
    parser.add_argument(
        '--cfg_file', type=str, 
        default='/tools/cfgs/5_percent_bev_base.yaml', 
        help='specify the config for training'
    )
    parser.add_argument(
        '--ckpt_dir', type=str, 
        default='./output/5_percent_bev_base', 
        help='specify a ckpt directory to be evaluated if needed'
    )
    parser.add_argument('--extra_tag', type=str, default='my_1st_exp', help='extra tag for this experiment')
    parser.add_argument('--ckpt_num', type=str, default="30", help='checkpoint to start from')
    parser.add_argument('--save_to_file', action='store_true', default=False, help='')
    
    # DDP parameters
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
    parser.add_argument('--tcp_port', type=int, default=18888, help='tcp port for distrbuted training')
    parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')
    parser.add_argument('--fix_random_seed', action='store_true', default=True, help='')
    parser.add_argument('--max_waiting_mins', type=int, default=30, help='max waiting minutes')
    
    parser.add_argument('--eval_tag', type=str, default='default', help='eval tag for this experiment')
    parser.add_argument('--start_epoch', type=int, default=0, help='')
    
    parser.add_argument('--eval_all', action='store_true', default=False, help='whether to evaluate all checkpoints')
    
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER, help='set extra config keys if needed')
    
    
    args = parser.parse_args() # 解析命令行参数
    if args.set_cfgs is not None: # 读取其他参数
        cfg_from_list(args.set_cfgs, cfg)
    cfg_from_yaml_file(args.cfg_file, cfg) # 读取配置文件参数
    
    return args, cfg # 命令行参数 配置文件参数
    

def main():
    args, cfg = parse_config()
    np.random.seed(1024)

    if args.launcher == 'none': # 
        dist_test = False
        total_gpus = 1
    else:
        total_gpus, cfg.LOCAL_RANK = getattr(common_utils, 'init_dist_%s' % args.launcher)(
            args.tcp_port, args.local_rank, backend='nccl'
        )
        dist_test = True

    if args.batch_size is None:
        # 如果没有在命令行指定BatchSize,
        # 则读取配置文件里的 BATCH_SIZE_PER_GPU
        args.batch_size = cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU
    else:
        assert args.batch_size % total_gpus == 0, 'Batch size should match the number of gpus'
        args.batch_size = args.batch_size // total_gpus
    
    #! TODO
    cfg.TAG = Path(args.cfg_file).stem # i.e. 5_percent_bev_base
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'
    
    # output_dir = cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG / args.extra_tag
    output_dir = cfg.ROOT_DIR / 'output' / cfg.TAG / args.extra_tag
    # output_dir.mkdir(parents=True, exist_ok=True)
    eval_output_dir = output_dir / 'eval'
    
    if args.eval_all: # 测试所有已保存的模型
        epoch_id = None
        eval_output_dir = eval_output_dir / "eval_all_default"
    else: # 测试单个模型
        
        
    
    
    
if __name__=="__main__":
    model_path = "./output/"
    cfg_path = "./tools/cfgs/5_percent_bev_base.yaml"
    cfg_from_yaml_file(cfg_path, cfg)
    cfg.TAG = Path(cfg_path).stem
    cfg.EXP_GROUP_PATH = '/'.join(cfg_path.split('/')[1:-1])

    model = model_utils.MotionTransformer(config=cfg.MODEL)
    log_file = Path("./") / ('log_eval_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)
    it, epoch = model.load_params_from_file(filename=model_path, logger=logger, to_cpu=False)

