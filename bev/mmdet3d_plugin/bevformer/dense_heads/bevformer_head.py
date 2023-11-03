import copy
import torch
import torch.nn as nn

from mmcv.cnn import Linear, bias_init_with_prob
from mmcv.utils import TORCH_VERSION, digit_version
from mmdet.core import (multi_apply, multi_apply, reduce_mean)
from mmdet.models.utils.transformer import inverse_sigmoid
from mmdet.models import HEADS
from mmdet.models.dense_heads import DETRHead
from mmdet.models.utils import build_transformer
from mmcv.cnn.bricks.transformer import FFN, build_positional_encoding
from mmdet3d.core.bbox.coders import build_bbox_coder
# from projects.mmdet3d_plugin.core.bbox.util import normalize_bbox
from mmcv.runner import force_fp32, auto_fp16

from mmcv.runner.base_module import BaseModule

# @HEADS.register_module()
class BEVFormerHead(BaseModule):
    def __init__(
        self,
        # *args,
        transformer=None,
        code_weights=None,
        positional_encoding=None,
        bev_h=30,
        bev_w=30,
        num_query=900,
        # num_classes = 10,
        # in_channels = 256,
        # **kwargs
    ):
        super(BEVFormerHead, self).__init__() # num_classes,in_channels
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.num_query = num_query
        self.embed_dims = transformer.embed_dims
        self.fp16_enabled = False
        
        if code_weights is not None:
            self.code_weights = code_weights
        else:
            self.code_weights = [1.0, 1.0, 1.0,
                                 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2]
        
        self.code_weights = nn.Parameter(torch.tensor(
            self.code_weights, requires_grad=False), requires_grad=False) #! what's this
        
        self.transformer = build_transformer(transformer)
        self.positional_encoding = build_positional_encoding(positional_encoding)
        
        self.pc_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
        self.real_w = self.pc_range[3] - self.pc_range[0]
        self.real_h = self.pc_range[4] - self.pc_range[1]
        
        self.bev_embedding = nn.Embedding(self.bev_h * self.bev_w, self.embed_dims)
        self.query_embedding = nn.Embedding(self.num_query, self.embed_dims * 2) #! not used
        
    def _init_layers(self):
        # self.bev_embedding = nn.Embedding(self.bev_h * self.bev_w, self.embed_dims)
        # self.query_embedding = nn.Embedding(self.num_query, self.embed_dims * 2) #! not used
        pass
        
    def init_weights(self):
        """Initialize weights of the DeformDETR head."""
        self.transformer.init_weights()
    
    @auto_fp16(apply_to=('mlvl_feats'))
    def forward(self, mlvl_feats, img_metas, prev_bev=None,  only_bev=True):
        bs, num_cam, _, _, _ = mlvl_feats[0].shape
        dtype = mlvl_feats[0].dtype
        # object_query_embeds = self.query_embedding.weight.to(dtype)
        bev_queries = self.bev_embedding.weight.to(dtype)
        bev_mask = torch.zeros((bs, self.bev_h, self.bev_w),device=bev_queries.device).to(dtype)
        bev_pos = self.positional_encoding(bev_mask).to(dtype)
        
        return self.transformer.get_bev_features(
                mlvl_feats,
                bev_queries,
                self.bev_h,
                self.bev_w,
                grid_length=(self.real_h / self.bev_h, self.real_w / self.bev_w),
                bev_pos=bev_pos,
                img_metas=img_metas,
                prev_bev=prev_bev,
        )