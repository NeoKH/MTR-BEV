# Motion Transformer (MTR): https://arxiv.org/abs/2209.13508
# Published at NeurIPS 2022
# Written by Shaoshuai Shi 
# All Rights Reserved


import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from .context_encoder import build_context_encoder
from .motion_decoder import build_motion_decoder
import torchvision.transforms.functional as TF

class MotionTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model_cfg = config
        self.use_bev = self.model_cfg.USE_BEV
        
        self.context_encoder = build_context_encoder(self.model_cfg.CONTEXT_ENCODER)
        self.motion_decoder = build_motion_decoder(
            in_channels=self.context_encoder.num_out_channels,
            config=self.model_cfg.MOTION_DECODER
        )
        
        if self.use_bev:
            from mmcv import Config
            self.bev_cfg = Config.fromfile(self.model_cfg.BEV_CONFIG_PATH)
            self.checkpoint_path = self.model_cfg.BEV_CHECKPOINT_PATH
            self.bev_h = self.bev_cfg.bev_h_
            self.bev_w = self.bev_cfg.bev_w_
            self.resolution = 102.4 / self.bev_h
            self.fuse_conv_bn = self.bev_cfg.fuse_conv_bn
            self.img_size = torch.tensor([self.bev_h,self.bev_w]).float().cuda()
            # self.init_register()
            self.build_bevformer()
        
    def build_bevformer(self):
        from bev.mmdet3d_plugin.bevformer.detectors.bevformer import BEVFormer
        from collections import OrderedDict
        from mmcv.runner import _load_checkpoint,load_state_dict
        from mmcv.cnn import fuse_conv_bn
        self.bev_cfg.model.train_cfg = None
        self.bev_former = BEVFormer(
            use_grid_mask=self.bev_cfg.model.use_grid_mask,
            video_test_mode=self.bev_cfg.model.video_test_mode,
            img_backbone = self.bev_cfg.model.img_backbone,
            img_neck = self.bev_cfg.model.img_neck,
            bev_head = self.bev_cfg.model.bev_head,
        )
        # 加载checkpoint
        checkpoint = _load_checkpoint(self.checkpoint_path, map_location='cpu')
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        new_state_dict = OrderedDict()
        for key in state_dict.keys():
            if 'cls_branches' in key or 'reg_branches' in key:
                    continue
            if 'pts_bbox_head' in key:
                new_key = key.replace("pts_bbox_head","bev_head")
                new_state_dict[new_key] = state_dict[key]
                continue
            new_state_dict[key] = state_dict[key]
        
        load_state_dict(self.bev_former, new_state_dict, strict=False)
        
        if self.fuse_conv_bn:
            self.bev_former = fuse_conv_bn(self.bev_former)
        
        # self.bev_former = MMDataParallel(self.bev_former, device_ids=[0]).eval()
    
    def get_bev_feature(self,batch_dict):
        input_dict = batch_dict['input_dict']
        
        bev_data = input_dict['bev_data']
        features = self.bev_former(rescale=True, **bev_data).detach() #(B,H*W,256)
        
        center_objects = input_dict['center_objects_world']
        distance_to_ego = input_dict["distance_to_ego"].cuda()
        angle_to_ego = input_dict["angle_to_ego"].cuda()
        
        num_channels = features.shape[-1]
        visibilities = center_objects[:,7]
        
        bev_list = []
        for batch_idx in range(len(batch_dict['batch_sample_count'])):
            N = batch_dict['batch_sample_count'][batch_idx]
            bev_feature = features[batch_idx][None].repeat(N,1,1).view(N,self.bev_h,self.bev_w,-1).permute(0,3,1,2)
            bev_feature = bev_feature.view(-1,self.bev_h,self.bev_w)
            # 创建一个标准化的网格坐标
            grid_y, grid_x = torch.meshgrid([torch.linspace(-1, 1, 200), torch.linspace(-1, 1, 200)])
            grid = torch.stack((grid_x, grid_y), dim=2).unsqueeze(0)  # 添加一个维度以匹配img_feature
            
            
            angle_radians = angle_to_ego * (3.1415926 / 180)
            # 旋转 bev feature
            new_feature = TF.rotate(bev_feature[id],float(angle_to_ego[id]),expand=True).view(N,-1,self.bev_h,self.bev_w)
            # 此时的 bev feature 仍然以ego为中心,但是方向变成了 agent的前进方向 为正方向.
            return_feature = torch.zeros((N,num_channels,10,10)).float().cuda()
            for id in range(N):
                if visibilities[id] <= 0.3:
                    # 被遮挡的 Agent, 直接返回0向量
                    # bev_list.append(torch.zeros(256,10,10).cuda())
                    continue
                # 旋转 bev feature
                xy = distance_to_ego[id]
                if torch.sum(torch.abs(xy)/self.resolution < self.img_size/2+5) < 2: # 超出范围的 Agent, 直接返回0向量
                    bev_list.append(torch.zeros(256,10,10).cuda())
                    continue
                # Crop出需要的区域
                left,top = xy/self.resolution + self.img_size/2 - 5
                crop_feature = TF.crop(new_feature,int(left),int(top),10,10)
                bev_list.append(crop_feature)
        
        batch_dict['input_dict']["bev_features"] = torch.stack(bev_list,dim=0)
                
        return batch_dict

    
    def forward(self, batch_dict):
        if self.use_bev:
            batch_dict = self.get_bev_feature(batch_dict)
        
        batch_dict = self.context_encoder(batch_dict)
        batch_dict = self.motion_decoder(batch_dict)

        if self.training:
            loss, tb_dict, disp_dict = self.get_loss()

            tb_dict.update({'loss': loss.item()})
            disp_dict.update({'loss': loss.item()})
            return loss, tb_dict, disp_dict

        return batch_dict

    def get_loss(self):
        loss, tb_dict, disp_dict = self.motion_decoder.get_loss()

        return loss, tb_dict, disp_dict

    def load_params_with_optimizer(self, filename, to_cpu=False, optimizer=None, logger=None):
        if not os.path.isfile(filename):
            raise FileNotFoundError

        logger.info('==> Loading parameters from checkpoint %s to %s' % (filename, 'CPU' if to_cpu else 'GPU'))
        loc_type = torch.device('cpu') if to_cpu else None
        checkpoint = torch.load(filename, map_location=loc_type)
        epoch = checkpoint.get('epoch', -1)
        it = checkpoint.get('it', 0.0)

        self.load_state_dict(checkpoint['model_state'], strict=True)

        if optimizer is not None:
            logger.info('==> Loading optimizer parameters from checkpoint %s to %s'
                        % (filename, 'CPU' if to_cpu else 'GPU'))
            optimizer.load_state_dict(checkpoint['optimizer_state'])

        if 'version' in checkpoint:
            print('==> Checkpoint trained from version: %s' % checkpoint['version'])
        # logger.info('==> Done')
        logger.info('==> Done (loaded %d/%d)' % (len(checkpoint['model_state']), len(checkpoint['model_state'])))

        return it, epoch

    def load_params_from_file(self, filename, logger, to_cpu=False):
        if not os.path.isfile(filename):
            raise FileNotFoundError

        logger.info('==> Loading parameters from checkpoint %s to %s' % (filename, 'CPU' if to_cpu else 'GPU'))
        loc_type = torch.device('cpu') if to_cpu else None
        checkpoint = torch.load(filename, map_location=loc_type)
        model_state_disk = checkpoint['model_state']
        # print(model_state_disk.keys())
        # print(model_state_disk['bev_former.bev_head.transformer.can_bus_mlp.2.weight'].shape) # torch.Size([256, 128])
        # print(model_state_disk['bev_former.bev_head.transformer.can_bus_mlp.2.weight'][0,:10]) # torch.Size([256, 128])
        version = checkpoint.get("version", None)
        if version is not None:
            logger.info('==> Checkpoint trained from version: %s' % version)

        logger.info(f'The number of disk ckpt keys: {len(model_state_disk)}')
        model_state = self.state_dict()
        model_state_disk_filter = {}
        for key, val in model_state_disk.items():
            if key in model_state and model_state_disk[key].shape == model_state[key].shape:
                model_state_disk_filter[key] = val
            else:
                if key not in model_state:
                    print(f'Ignore key in disk (not found in model): {key}, shape={val.shape}')
                else:
                    print(f'Ignore key in disk (shape does not match): {key}, load_shape={val.shape}, model_shape={model_state[key].shape}')

        model_state_disk = model_state_disk_filter

        missing_keys, unexpected_keys = self.load_state_dict(model_state_disk, strict=False)
        new_missing_keys = []
        for missing_key in missing_keys:
            if "bev_former" in missing_key:
                new_missing_keys.append(missing_key)
        logger.info(f'Missing keys: {new_missing_keys}')
        logger.info(f'The number of missing keys: {len(new_missing_keys)}')
        logger.info(f'The number of unexpected keys: {len(unexpected_keys)}')
        logger.info('==> Done (total keys %d)' % (len(model_state)))

        epoch = checkpoint.get('epoch', -1)
        it = checkpoint.get('it', 0.0)

        return it, epoch


