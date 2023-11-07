import _init_path
import argparse
import datetime
import glob
import os
from natsort import ns, natsorted
import re
import datetime
from pathlib import Path
from tqdm import tqdm
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
        default='./tools/cfgs/5_percent_bev_base.yaml', 
        help='specify the config for training'
    )
    parser.add_argument('--group_name', type=str, default='', help='group name')
    parser.add_argument('--extra_tag', type=str, default='my_1st_exp', help='extra tag for this experiment')
    parser.add_argument('--ckpt_id', type=str, default="30", help='checkpoint to start from')
    # parser.add_argument('--save_to_file', action='store_true', default=False, help='')
    parser.add_argument('--mkdir', action='store_true', default=False, help='mkdir if needed')
    
    # DDP parameters
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
    parser.add_argument('--tcp_port', type=int, default=18888, help='tcp port for distrbuted training')
    parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')
    parser.add_argument('--fix_random_seed', action='store_true', default=True, help='')
    parser.add_argument('--max_waiting_mins', type=int, default=30, help='max waiting minutes')
    
    parser.add_argument('--start_epoch', type=int, default=0, help='')
    parser.add_argument('--batch_size', type=int, default=None, required=False, help='batch size for training')
    parser.add_argument('--workers', type=int, default=2, help='number of workers for dataloader')
    
    parser.add_argument('--eval_all', action='store_true', default=False, help='whether to evaluate all checkpoints')
    
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER, help='set extra config keys if needed')
    
    
    args = parser.parse_args() # 解析命令行参数
    if args.set_cfgs is not None: # 读取其他参数
        cfg_from_list(args.set_cfgs, cfg)
    cfg_from_yaml_file(args.cfg_file, cfg) # 读取配置文件参数
    
    return args, cfg # 命令行参数 配置文件参数


def eval_one(model,ckpt_path,loader,logger,ckpt_name,dist_test):
    if ckpt_path!=None:
        it, epoch = model.load_params_from_file(filename=ckpt_path, logger=logger, to_cpu=dist_test)
    logger.info(f'====================== LOAD MODEL {ckpt_name} for EVALUATION ======================')
    model.cuda()
    model.eval()
    # start evaluation
    if dist_test:
        if not isinstance(model, torch.nn.parallel.DistributedDataParallel):
            num_gpus = torch.cuda.device_count()
            local_rank = cfg.LOCAL_RANK % num_gpus
            model = torch.nn.parallel.DistributedDataParallel(
                    model,
                    device_ids=[local_rank],
                    broadcast_buffers=False
            )
    model.eval()
    
    start_time = datetime.datetime.now()
    
    # for循环
    pred_lists = []
    pred_score_lists = []
    gt_lists = []
    types_list = []
    for i in tqdm(range(len(loader))):
        batch_dict = next(iter(loader))
        batch_pred_dicts = model(batch_dict)
        
        pred_scores = batch_pred_dicts['pred_scores']
        pred_trajs = batch_pred_dicts['pred_trajs']
        num_center_objects, num_modes, num_timestamps, num_feat = pred_trajs.shape
        
        input_dict = batch_pred_dicts['input_dict']
        center_objects_world = input_dict['center_objects_world'].type_as(pred_trajs)
        center_gt_trajs = input_dict['center_gt_trajs_src'].type_as(pred_trajs)
        center_objects_type = input_dict['center_objects_type']
        
        pred_trajs_world = common_utils.rotate_points_along_z(
            points=pred_trajs.view(num_center_objects, num_modes * num_timestamps, num_feat),
            angle=center_objects_world[:, -5].view(num_center_objects)
        ).view(num_center_objects, num_modes, num_timestamps, num_feat)
        pred_trajs_world[:, :, :, 0:2] += center_objects_world[:, None, None, 0:2]
        
        
        pred_lists.append(pred_trajs_world[:, :, :, 0:2]) #（k,6,12,2）
        gt_lists.append(center_gt_trajs[:, 5:, 0:2]) #（k,12,2）
        
        pred_score_lists.append(pred_scores) # (k,6)
        types_list.extend(center_objects_type) # (k)
    
    pred_trajs_world = torch.cat(pred_lists,dim=0) #（M,6,12,2）
    gt_trajs_world = torch.cat(gt_lists,dim=0) #（M,12,2)
    N,M,T,_ = pred_trajs_world.shape
    
    distances = pred_trajs_world - gt_trajs_world[:,None]
    distances = distances.squeeze()
    
    l2 = torch.norm(distances.view(-1,2),dim=1).view(N,M,T)
    
    fde = l2[:,:,-1].squeeze() # (N,M)
    min_fde = np.min(fde.cpu().numpy(),axis=1) # (N)
    
    vehicle_fde = np.mean(min_fde[np.argwhere(np.array(types_list)=="TYPE_VEHICLE")])
    human_fde = np.mean(min_fde[np.argwhere(np.array(types_list)=="TYPE_PEDESTRIAN")])
    
    ade = np.mean(l2.cpu().numpy(),axis=-1) #(N,M)
    min_ade = np.min(ade,axis=1)
    vehicle_ade = np.mean(min_ade[np.argwhere(np.array(types_list)=="TYPE_VEHICLE")])
    human_ade = np.mean(min_ade[np.argwhere(np.array(types_list)=="TYPE_PEDESTRIAN")])
    
    # if cfg.LOCAL_RANK == 0: # 只在rank==0的机器上打印信息
    logger.info('====================== Performance of %s ======================' % ckpt_name)
    
    logger.info(f"vehicle ade: {vehicle_ade:.3f}")
    logger.info(f"vehicle fde: {vehicle_fde:.3f}")
    logger.info(f"pedestrian ade: {human_ade:.3f}")
    logger.info(f"pedestrian fde: {human_fde:.3f}")
    
    logger.info(f"Total Spend Time: {datetime.datetime.now() - start_time}")
    logger.info('======================Evaluation done.======================')

def main():
    args, cfg = parse_config()

    if args.fix_random_seed:
        common_utils.set_random_seed(666)
    
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
    
    cfg.TAG = Path(args.cfg_file).stem # i.e. 5_percent_bev_base
    output_dir = cfg.ROOT_DIR / 'output' / args.group_name /cfg.TAG / args.extra_tag
    if args.mkdir and not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
    eval_output_dir = output_dir / 'eval'
    
    if args.eval_all: # 测试所有已保存的模型
        ckpt_dir = output_dir / "ckpt"
        eval_output_dir = eval_output_dir / "eval_all_default"
    else: # 测试单个模型
        if not "latest" in args.ckpt_id:
            ckpt_file_path = output_dir / "ckpt" / f"checkpoint_epoch_{args.ckpt_id}.pth"
        else:
            ckpt_file_path = output_dir / "ckpt" / "latest_model.pth"
        eval_output_dir = eval_output_dir / f"epoch_{args.ckpt_id}"
    if not eval_output_dir.exists():
        eval_output_dir.mkdir(parents=True, exist_ok=True)
    
    # log
    log_file = eval_output_dir / f"log_eval_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}.txt"
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)
    logger.info('======================Start logging======================')
    gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
    logger.info(f"CUDA_VISIBLE_DEVICES={gpu_list}")
    if dist_test:
        logger.info(f"total_batch_size: {total_gpus * args.batch_size}")
    for key, val in vars(args).items():
        logger.info(f"{key:16} {val}")
    log_config_to_file(cfg, logger=logger)
    
    test_set, test_loader, sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        batch_size=args.batch_size,
        dist=dist_test, workers=args.workers, logger=logger, training=False
    )
    
    model = model_utils.MotionTransformer(config=cfg.MODEL)
    
    if args.eval_all:
        # 把 ckpt目录下所有的.pth文件列出来
        ckpt_files_path = natsorted([x for x in ckpt_dir.iterdir() if x.suffix == ".pth"],alg=ns.PATH)
        for ckpt_file_path in ckpt_files_path:
            ckpt_name = ckpt_file_path.stem
            with torch.no_grad():
                eval_one(model,ckpt_file_path,test_loader,logger,ckpt_name,dist_test)
    else:
        with torch.no_grad():
            ckpt_name = ckpt_file_path.stem()
            eval_one(model,ckpt_file_path,test_loader,logger,ckpt_name,dist_test)
    
  
if __name__=="__main__":
    main()

