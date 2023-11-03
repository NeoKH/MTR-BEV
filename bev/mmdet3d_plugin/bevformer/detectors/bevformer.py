import torch
from mmcv.runner import force_fp32, auto_fp16
from mmdet.models import DETECTORS,HEADS,BACKBONES
from mmdet3d.core import bbox3d2result
from mmdet3d.models import builder
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from bev.mmdet3d_plugin.models.utils.grid_mask import GridMask
import time
import copy
import numpy as np
import mmdet3d
# from projects.mmdet3d_plugin.models.utils.bricks import run_time
from mmcv.runner.base_module import BaseModule

# @BACKBONES.register_module()
class BEVFormer(BaseModule):
    """BEVFormer.
    Args:
        video_test_mode (bool): Decide whether to use temporal information during inference.
    """
    
    def __init__(
        self,
        use_grid_mask=False,
        img_backbone=None,
        img_neck=None,
        bev_head = None,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
        video_test_mode=False
    ):
        super(BEVFormer,self).__init__()
        
        if img_backbone:
            self.img_backbone = builder.build_backbone(img_backbone)
        if img_neck is not None:
            self.img_neck = builder.build_neck(img_neck)
        if bev_head is not None:
            from ..dense_heads.bevformer_head import BEVFormerHead
            # self.bev_head = builder.build_head(bev_head)
            self.bev_head = BEVFormerHead(
                transformer = bev_head.transformer,
                positional_encoding=bev_head.positional_encoding,
                bev_h=bev_head.bev_h,
                bev_w=bev_head.bev_w,
                num_query=bev_head.num_query,
                # num_classes=bev_head.num_classes,
                # in_channels=bev_head.in_channels
            )
        
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        
        if pretrained is None:
            img_pretrained = None
            pts_pretrained = None
        elif isinstance(pretrained, dict):
            img_pretrained = pretrained.get('img', None)
            pts_pretrained = pretrained.get('pts', None)
        else:
            raise ValueError(
                f'pretrained should be a dict, got {type(pretrained)}')
        
        self.grid_mask = GridMask(True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.use_grid_mask = use_grid_mask

        # temporal
        self.video_test_mode = video_test_mode
        self.prev_frame_info = {
            'prev_bev': None,
            'scene_token': None,
            'prev_pos': 0,
            'prev_angle': 0,
        }
    
    def with_img_neck(self):
        """bool: Whether the detector has a neck in image branch."""
        return hasattr(self, 'img_neck') and self.img_neck is not None
    
    def extract_img_feat(self, img, img_metas, len_queue=None):
        """Extract features of images."""
        B = img.size(0)
        if img is not None:
            
            # input_shape = img.shape[-2:]
            # # update real input shape of each single img
            # for img_meta in img_metas:
            #     img_meta.update(input_shape=input_shape)

            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_()
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.reshape(B * N, C, H, W)
            if self.use_grid_mask:
                img = self.grid_mask(img)

            img_feats = self.img_backbone(img)
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)

        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            if len_queue is not None:
                img_feats_reshaped.append(img_feat.view(int(B/len_queue), len_queue, int(BN / B), C, H, W))
            else:
                img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        return img_feats_reshaped
    
    @auto_fp16(apply_to=('img'))
    def extract_feat(self, img, img_metas=None, len_queue=None):
        """Extract features from images and points."""

        img_feats = self.extract_img_feat(img, img_metas, len_queue=len_queue)
        
        return img_feats
   
    def obtain_history_bev(self, imgs_queue, img_metas_list):
        """Obtain history BEV features iteratively. To save GPU memory, gradients are not calculated.
        """
        prev_bev = None
        bs, len_queue, num_cams, C, H, W = imgs_queue.shape
        imgs_queue = imgs_queue.reshape(bs*len_queue, num_cams, C, H, W)
        img_feats_list = self.extract_feat(img=imgs_queue, len_queue=len_queue)
        for i in range(len_queue):
            img_metas = [each[i] for each in img_metas_list]
            if not img_metas[0]['prev_bev_exists']:
                prev_bev = None
            # img_feats = self.extract_feat(img=img, img_metas=img_metas)
            img_feats = [each_scale[:, i] for each_scale in img_feats_list]
            prev_bev = self.bev_head(
                img_feats, img_metas, prev_bev)
        return prev_bev
     
    def forward(self, img_metas, img=None, **kwargs):
        self.eval()
        with torch.no_grad():
            img = img.cuda()
            len_queue = img.size(1)
            prev_img = img[:, :-1, ...]
            img = img[:, -1, ...]
            prev_img_metas = copy.deepcopy(img_metas)
            prev_bev = self.obtain_history_bev(prev_img, prev_img_metas)
            
            img_metas = [each[len_queue-1] for each in img_metas]
            if not img_metas[0]['prev_bev_exists']:
                prev_bev = None
            img_feats = self.extract_feat(img=img, img_metas=img_metas)
            new_prev_bev = self.bev_head(img_feats, img_metas, prev_bev)

        # for var, name in [(img_metas, 'img_metas')]:
        #     if not isinstance(var, list):
        #         raise TypeError('{} must be a list, but got {}'.format(
        #             name, type(var)))
        # img = [img] if img is None else img
        
        # if img_metas[0][0]['scene_token'] != self.prev_frame_info['scene_token']:
        #     # the first sample of each scene is truncated
        #     self.prev_frame_info['prev_bev'] = None
        # # update idx
        # self.prev_frame_info['scene_token'] = img_metas[0][0]['scene_token']
        
        # # do not use temporal information
        # if not self.video_test_mode:
        #     self.prev_frame_info['prev_bev'] = None
            
        # # Get the delta of ego position and angle between two timestamps.
        # tmp_pos = copy.deepcopy(img_metas[0][0]['can_bus'][:3])
        # tmp_angle = copy.deepcopy(img_metas[0][0]['can_bus'][-1])
        # if self.prev_frame_info['prev_bev'] is not None:
        #     img_metas[0][0]['can_bus'][:3] -= self.prev_frame_info['prev_pos']
        #     img_metas[0][0]['can_bus'][-1] -= self.prev_frame_info['prev_angle']
        # else:
        #     img_metas[0][0]['can_bus'][-1] = 0
        #     img_metas[0][0]['can_bus'][:3] = 0
            
        # new_prev_bev = self.simple_test(img_metas[0], img[0], prev_bev=self.prev_frame_info['prev_bev'], **kwargs) #
        
        # # During inference, we save the BEV features and ego motion of each timestamp.
        # self.prev_frame_info['prev_pos'] = tmp_pos
        # self.prev_frame_info['prev_angle'] = tmp_angle
        # self.prev_frame_info['prev_bev'] = new_prev_bev
        
        return new_prev_bev
    
    def simple_test(self, img_metas, img=None, prev_bev=None, rescale=False): #! rescale is not used
        img_feats = self.extract_feat(img=img, img_metas=img_metas)
        
        new_prev_bev = self.bev_head(img_feats, img_metas, prev_bev)
        

        return new_prev_bev