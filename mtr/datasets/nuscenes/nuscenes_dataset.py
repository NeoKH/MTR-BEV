import os
import zlib
import argparse
import numpy as np
from pathlib import Path
import pickle
import torch
import warnings
warnings.filterwarnings("ignore")

import mmcv
import torch
import torchvision.transforms.functional as TF
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, collate
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint, _load_checkpoint, load_state_dict,
                         wrap_fp16_model)

from mtr.datasets.dataset import DatasetTemplate
from mtr.utils import common_utils
from mtr.config import cfg

class NuscenesDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, training=True, logger=None,):
        super().__init__(dataset_cfg, training, logger)
        self.data_root = cfg.ROOT_DIR / self.dataset_cfg.DATA_ROOT
        self.data_path = self.data_root / self.dataset_cfg.SPLIT_DIR[self.mode]
        if not self.data_path.exists():
            assert False, f"{self.data_path} does not exist"
        
        self.file_path_list = [self.data_path / f for f in self.data_path.iterdir() if f.is_file() and f.suffix == ".pkl"]
        self.file_path_list = self.file_path_list[::self.dataset_cfg.SAMPLE_INTERVAL[self.mode]]
        
        self.type_dict = {0:"TYPE_PEDESTRIAN", 1:"TYPE_VEHICLE", 2:"TYPE_EGO"}
        
        if self.dataset_cfg.USE_BEV:
            from mmcv import Config
            self.bev_cfg = Config.fromfile(self.dataset_cfg.BEV_CONFIG_PATH)
            self.init_register()
            self.init_bev_dataset()
    
    def init_register(self,):
        if hasattr(self.bev_cfg, 'plugin'):
            if self.bev_cfg.plugin:
                import importlib
                if hasattr(self.bev_cfg, 'plugin_dir'):
                    plugin_dir = self.bev_cfg.plugin_dir
                    _module_dir = os.path.dirname(plugin_dir)
                    _module_dir = _module_dir.split('/')
                    _module_path = _module_dir[0]

                    for m in _module_dir[1:]:
                        _module_path = _module_path + '.' + m
                    print(_module_path)
                    plg_lib = importlib.import_module(_module_path)
                else:
                    # import dir is the dirpath for the config file
                    _module_dir = os.path.dirname(self.bev_config_path)
                    _module_dir = _module_dir.split('/')
                    _module_path = _module_dir[0]
                    for m in _module_dir[1:]:
                        _module_path = _module_path + '.' + m
                    print(_module_path)
                    plg_lib = importlib.import_module(_module_path)
    
    def __len__(self):
        return len(self.file_path_list)
    
    def __getitem__(self, idx):
        data_path = self.file_path_list[idx]
        with open(data_path,"rb") as f:
            data_dict = pickle.load(f)

        ret_infos = self.get_data(data_dict)

        if self.dataset_cfg.USE_BEV:
            _, sample_token = data_dict['scenario_id'].split('_')
            ego_idx = np.argwhere(ret_infos["obj_types"] == "TYPE_EGO")[0][0]
            distance_to_ego = -ret_infos['obj_trajs'][:,ego_idx,-1,:2].view(-1,2)
            angle_to_ego = torch.from_numpy(np.rad2deg(ret_infos['obj_trajs'][:,ego_idx,-1,-5].view(-1).numpy()))
            ret_infos["distance_to_ego"] = distance_to_ego
            ret_infos["angle_to_ego"] = angle_to_ego
            
            bev_idx = np.argwhere(self.bev_tokens==sample_token)[0][0]
            bev_data = self.bev_dataset[bev_idx]
            ret_infos["bev_data"] = bev_data
            
        return ret_infos
    
    def get_data(self,data_dict):
        center_idxs  = data_dict['center_idxs']  # (N)
        observe_data = data_dict['observe_data'] # (N,5,21)
        predict_data = data_dict['predict_data'] # (N,12,7)
        map_polyline = data_dict['map_polyline'][0] # [(m1,3),(m2,3),...]

        # center_objects, center_ids= self.get_center_objects(observe_data,center_idxs) # (k, 21) (k)
        center_objects = torch.from_numpy(observe_data[center_idxs,-1,:]).float() # (k, 21)
        center_ids = torch.from_numpy(np.nonzero(center_idxs)[0]) # (k)
        
        center_objects_type = torch.argmax(center_objects[:,8:11], dim=1)
        
        # (N,5,21) --> (k,N,5,21) , (N,12,7) --> (k,N,12,7)
        obs_trajs, pred_trajs = \
            self.create_agent_data_for_center_objects(center_objects,observe_data,predict_data)
        
        obj_trajs_pos = obs_trajs[:,:,-1,:3].clone()
        
        past_mask = obs_trajs[:,:,:,6].clone()
        obs_trajs[past_mask==0]=0.
        
        future_trajs = pred_trajs[:,:,:,[0,1,3,4]].clone() # (k,N,12,4)
        future_mask = pred_trajs[:,:,:,5].clone()  # (k,N,12)
        # print("future_trajs: ",future_trajs.shape)
        # print("future_mask: ",future_mask.shape)
        # print(center_ids)
        
        future_trajs[future_mask==0]=0.
        
        center_gt_trajs = future_trajs[[range(len(center_ids))],center_ids][0].clone() # (k,12,4)
        center_gt_trajs_mask = future_mask[[range(len(center_ids))],center_ids][0].clone() # (k,12)
        center_gt_final_valid_idx = torch.sum(center_gt_trajs_mask,dim=1).clone() -1 # (k,)
        
        # print(center_gt_trajs.shape)
        # print(center_gt_trajs_mask.shape)
        # print(center_gt_final_valid_idx)
        
        observe_data = torch.from_numpy(observe_data)
        obj_types = torch.argmax(observe_data[:,-1,8:11], dim=1).numpy()
        
        predict_data = torch.from_numpy(predict_data)
        center_past_trajs = torch.zeros((len(center_ids),observe_data.shape[1]+predict_data.shape[1],10)) 
        # (k,5+12,10): [cx, cy, cz, dx, dy, dz, heading, vel_x, vel_y, valid]
        center_past_trajs[:,:5,:6] = observe_data[center_ids,:,:6]
        center_past_trajs[:,:5,6:9] = observe_data[center_ids,:,-5:-2]
        center_past_trajs[:,:5,9] = observe_data[center_ids,:,6]
        center_past_trajs[:,5:,:3] = predict_data[center_ids,:,:3]
        center_past_trajs[:,5:,3:6] = center_past_trajs[:,0,3:6][:,None,:].repeat(1,predict_data.shape[1],1)
        center_past_trajs[:,5:,6] = predict_data[center_ids,:,-1]
        center_past_trajs[:,5:,7:9] = predict_data[center_ids,:,3:5]
        center_past_trajs[:,5:,9] = predict_data[center_ids,:,5]
        
        map_polylines, map_polylines_mask, map_polylines_center = \
            self.get_map_data(map_polyline,center_objects,self.dataset_cfg.get('CENTER_OFFSET_OF_MAP', (10.0, 0)))
        
        
        return_data = {
            # "scenario_id": [data_dict['scenario_id']], #! 存疑 .split('_')[-1]
            
            "obj_trajs": obs_trajs,
            "obj_trajs_mask": past_mask > 0,
            "obj_trajs_last_pos": obj_trajs_pos,
            
            "track_index_to_predict": center_ids,
            
            "obj_types": np.array([self.type_dict[int(x)] for x in obj_types]),
            
            "center_objects_world": center_objects,
            "center_objects_id": center_ids, #! 与 track_index_to_predict 重复
            "center_objects_type": np.array([self.type_dict[int(x)] for x in center_objects_type]), 
            
            "obj_trajs_future_state": future_trajs,
            "obj_trajs_future_mask": future_mask,
            
            "center_gt_trajs": center_gt_trajs,
            "center_gt_trajs_mask": center_gt_trajs_mask,
            "center_gt_final_valid_idx": center_gt_final_valid_idx,
            "center_gt_trajs_src": center_past_trajs, #!
            
            "map_polylines": map_polylines,
            "map_polylines_mask": map_polylines_mask > 0,
            "map_polylines_center": map_polylines_center,
        }
        
        return return_data
        
    def create_agent_data_for_center_objects(self,center_objects,observe_data,predict_data):
        k = center_objects.shape[0]
        N,T,C = observe_data.shape
        
        center_xyz=center_objects[:, 0:3] # (k,3)
        heading = center_objects[:,-5] # (k)
        
        obs_trajs = torch.from_numpy(observe_data).float()[None].repeat(k,1,1,1) # (k,N,5,21)
        obs_trajs[:, :, :, 0:3] -= center_xyz[:, None, None, :] # 平移
        obs_trajs[:, :, :, 0:2] = self.rotate_points_along_z(
            obs_trajs[:, :, :, 0:2].view(k,-1,2),
            angle = -heading
        ).view(k,N,T,2)

        pred_trajs = torch.from_numpy(predict_data).float()[None].repeat(k,1,1,1)
        pred_trajs[:, :, :, 0:3] -= center_xyz[:, None, None, :] # 平移
        pred_trajs[:, :, :, 0:2] = self.rotate_points_along_z(
            pred_trajs[:, :, :, 0:2].view(k,-1,2),
            angle = -heading
        ).view(k,N,pred_trajs.shape[2],2)
        
        # 旋转速度
        obs_trajs[:, :, :, 17:19] = self.rotate_points_along_z(
            obs_trajs[:, :, :, 17:19].view(k,-1,2),
            angle = -heading
        ).view(k,N,T,2)
        pred_trajs[:, :, :, 3:5] = self.rotate_points_along_z(
            pred_trajs[:, :, :, 3:5].view(k,-1,2),
            angle = -heading
        ).view(k,N,pred_trajs.shape[2],2)
        
        # 偏移角做相应调整
        obs_trajs[:,:,:,-5] -= heading[:,None,None]
        pred_trajs[:,:,:,-5] -= heading[:,None,None]
        
        return obs_trajs,pred_trajs
        
    def get_map_data(self,map_polyline,center_objects,center_offset):
        """
            map_polyline: list of (num_points, 7): [x, y, z, dir_x, dir_y, dir_z, global_type]
            center_objects: (k,21)
            center_offset (2):, [offset_x, offset_y]
        """
        k = center_objects.shape[0]
        
        # transform object coordinates by center objects
        def transform_to_center_coordinates(neighboring_polylines, neighboring_polyline_valid_mask):
            neighboring_polylines[:, :, :, 0:3] -= center_objects[:, None, None, 0:3]
            neighboring_polylines[:, :, :, 0:2] = common_utils.rotate_points_along_z(
                points=neighboring_polylines[:, :, :, 0:2].view(k, -1, 2),
                angle=-center_objects[:, -5]
            ).view(k, -1, batch_polylines.shape[1], 2)
            neighboring_polylines[:, :, :, 3:5] = common_utils.rotate_points_along_z(
                points=neighboring_polylines[:, :, :, 3:5].view(k, -1, 2),
                angle=-center_objects[:, -5]
            ).view(k, -1, batch_polylines.shape[1], 2)

            # use pre points to map
            # (num_center_objects, num_polylines, num_points_each_polyline, num_feat)
            xy_pos_pre = neighboring_polylines[:, :, :, 0:2]
            xy_pos_pre = torch.roll(xy_pos_pre, shifts=1, dims=-2)
            xy_pos_pre[:, :, 0, :] = xy_pos_pre[:, :, 1, :]
            neighboring_polylines = torch.cat((neighboring_polylines, xy_pos_pre), dim=-1)

            neighboring_polylines[neighboring_polyline_valid_mask == 0] = 0
            return neighboring_polylines, neighboring_polyline_valid_mask
        
        # 拆分 polyline
        num_points = self.dataset_cfg.NUM_POINTS_EACH_POLYLINE
        new_polylines = []
        new_polylines_mask = []
        
        for polyline in map_polyline:
            polyline = torch.tensor(polyline)
            if polyline.shape[0] <= num_points:
                new_polyline = torch.zeros((num_points,polyline.shape[1])).float()
                new_polyline_mask = torch.zeros((num_points)).float()
                new_polyline[:len(polyline)] = polyline
                new_polyline_mask[:len(polyline)] = 1
                new_polylines.append(new_polyline)
                new_polylines_mask.append(new_polyline_mask)
            else:
                for i in range(0,polyline.shape[0],num_points):
                    if i + num_points > polyline.shape[0]:
                        tmp_polyline = polyline[i:]
                    else:
                        tmp_polyline = polyline[i:i+num_points]
                    
                    new_polyline = torch.zeros((num_points,polyline.shape[1])).float()
                    new_polyline_mask = torch.zeros((num_points)).float()
                    
                    new_polyline[:len(tmp_polyline)] = tmp_polyline
                    new_polyline_mask[:len(tmp_polyline)] = 1
                    
                    new_polylines.append(new_polyline)
                    new_polylines_mask.append(new_polyline_mask)
        
        batch_polylines = torch.stack(new_polylines,dim=0)
        batch_polylines_mask = torch.stack(new_polylines_mask,dim=0)
        
        if batch_polylines.shape[0] > self.dataset_cfg.NUM_OF_SRC_POLYLINES:
            # 如果 polyline 过多,筛选出距离近的前k个polyline
            polyline_center = batch_polylines[:, :, 0:2].sum(dim=1) / torch.clamp_min(batch_polylines_mask.sum(dim=1).float()[:, None], min=1.0)
            center_offset_rot = torch.from_numpy(np.array(center_offset, dtype=np.float32))[None, :].repeat(k, 1)
            center_offset_rot = common_utils.rotate_points_along_z(
                points=center_offset_rot.view(k, 1, 2),
                angle=center_objects[:, -5]
            ).view(k, 2)
            pos_of_map_centers = center_objects[:, 0:2] + center_offset_rot
            dist = (pos_of_map_centers[:, None, :] - polyline_center[None, :, :]).norm(dim=-1)  # (k, num_polylines)
            topk_dist, topk_idxs = dist.topk(k=self.dataset_cfg.NUM_OF_SRC_POLYLINES, dim=-1, largest=False)
            map_polylines = batch_polylines[topk_idxs]  # (k, num_topk_polylines, num_points_each_polyline, 7)
            map_polylines_mask = batch_polylines_mask[topk_idxs]  # (k, num_topk_polylines, num_points_each_polyline)
        else:
            map_polylines = batch_polylines[None, :, :, :].repeat(k, 1, 1, 1)
            map_polylines_mask = batch_polylines_mask[None, :, :].repeat(k, 1, 1)
        
        map_polylines, map_polylines_mask = transform_to_center_coordinates(
            neighboring_polylines=map_polylines,
            neighboring_polyline_valid_mask=map_polylines_mask
        )
        
        temp_sum = (map_polylines[:, :, :, 0:3] * map_polylines_mask[:, :, :, None].float()).sum(dim=-2)  # (k, num_polylines, 3)
        map_polylines_center = temp_sum / torch.clamp_min(map_polylines_mask.sum(dim=-1).float()[:, :, None], min=1.0)  # (k, num_polylines, 3)
        
        return map_polylines, map_polylines_mask, map_polylines_center
        
    def rotate_points_along_z(self,points,angle):
        """
        Args:
            points: (B, N, 3 + C)
            angle: (B), angle along z-axis, angle increases x ==> y
        Returns:

        """
        cosa = torch.cos(angle)
        sina = torch.sin(angle)
        zeros = angle.new_zeros(points.shape[0])
        if points.shape[-1] == 2:
            rot_matrix = torch.stack((
                cosa,  sina,
                -sina, cosa
            ), dim=1).view(-1, 2, 2).float()
            points_rot = torch.matmul(points, rot_matrix)
        else:
            ones = angle.new_ones(points.shape[0])
            rot_matrix = torch.stack((
                cosa,  sina, zeros,
                -sina, cosa, zeros,
                zeros, zeros, ones
            ), dim=1).view(-1, 3, 3).float()
            points_rot = torch.matmul(points[:, :, 0:3], rot_matrix)
            points_rot = torch.cat((points_rot, points[:, :, 3:]), dim=-1)
        return points_rot
    
    def get_center_objects(self,observe_data,center_idxs):
    #     """根据超参数MAX_AGENTS_EACH_SCENE限制要预测的数目,目前限制为行人和车辆各最多10个"""
    #     center_objects = torch.from_numpy(observe_data[center_idxs,-1,:]).float() # (k, 21)
    #     center_ids = torch.from_numpy(np.nonzero(center_idxs)[0]) # (k)
    #     MAX_AGENTS = self.dataset_cfg.MAX_AGENTS_EACH_SCENE
    #     if np.sum(center_idxs) < MAX_AGENTS*2 :
    #         center_objects = torch.from_numpy(observe_data[center_idxs,-1,:]).float() # (k, 21)
    #         center_ids = torch.from_numpy(np.nonzero(center_idxs)[0]) # (k)
    #         return center_objects,center_ids
    #     else:
    #         vehicle_idxs = np.bitwise_and(center_idxs,observe_data[:,4,9]==1)
    #         pedestrian_idxs = np.bitwise_and(center_idxs,observe_data[:,4,8]==1)
            
    #         vehicle_ids = torch.from_numpy(np.nonzero(vehicle_idxs)[0])
    #         pedestrian_ids = torch.from_numpy(np.nonzero(pedestrian_idxs)[0])
            
    #         vehicle_nums = np.sum(vehicle_idxs)
    #         pedestrian_nums = np.sum(pedestrian_idxs)
            
    #         if vehicle_nums > MAX_AGENTS and pedestrian_nums > MAX_AGENTS:
    #             vehicle_idxs[vehicle_ids[MAX_AGENTS:]] = False
    #             center_vehicle = torch.from_numpy(observe_data[vehicle_idxs,-1,:]).float()

    #             pedestrian_idxs[pedestrian_ids[MAX_AGENTS:]] = False
    #             center_pedestrian = torch.from_numpy(observe_data[pedestrian_idxs,-1,:]).float()
                
    #             center_objects = torch.cat((center_vehicle,center_pedestrian),dim=0)
    #             center_ids = torch.cat((vehicle_ids[:MAX_AGENTS],pedestrian_ids[:MAX_AGENTS]))
    #             return center_objects,center_ids
    #         elif vehicle_nums > MAX_AGENTS and pedestrian_nums <= MAX_AGENTS:
    #             vehicle_idxs[vehicle_ids[MAX_AGENTS:]] = False
    #             center_vehicle = torch.from_numpy(observe_data[vehicle_idxs,-1,:]).float()
    #             center_pedestrian = torch.from_numpy(observe_data[pedestrian_idxs,-1,:]).float()
    #             center_objects = torch.cat((center_vehicle,center_pedestrian),dim=0)
    #             center_ids = torch.cat((vehicle_ids[:MAX_AGENTS],pedestrian_ids))
    #             return center_objects,center_ids
    #         elif vehicle_nums <= MAX_AGENTS and pedestrian_nums > MAX_AGENTS:
    #             pedestrian_idxs[pedestrian_ids[MAX_AGENTS:]] = False
    #             center_pedestrian = torch.from_numpy(observe_data[pedestrian_idxs,-1,:]).float()
    #             center_vehicle = torch.from_numpy(observe_data[vehicle_idxs,-1,:]).float()
    #             center_objects = torch.cat((center_vehicle,center_pedestrian),dim=0)
    #             center_ids = torch.cat((vehicle_ids,pedestrian_ids[:MAX_AGENTS]))
    #             return center_objects,center_ids
    #         else:
    #             center_vehicle = torch.from_numpy(observe_data[vehicle_idxs,-1,:]).float()
    #             center_pedestrian = torch.from_numpy(observe_data[pedestrian_idxs,-1,:]).float()
    #             center_objects = torch.cat((center_vehicle,center_pedestrian),dim=0)
    #             center_ids = torch.cat((vehicle_ids,pedestrian_ids))
    #             return center_objects,center_ids
        pass
    
    def init_register(self,):
        if hasattr(self.bev_cfg, 'plugin'):
            if self.bev_cfg.plugin:
                import importlib
                if hasattr(self.bev_cfg, 'plugin_dir'):
                    plugin_dir = self.bev_cfg.plugin_dir
                    _module_dir = os.path.dirname(plugin_dir)
                    _module_dir = _module_dir.split('/')
                    _module_path = _module_dir[0]

                    for m in _module_dir[1:]:
                        _module_path = _module_path + '.' + m
                    print(_module_path)
                    plg_lib = importlib.import_module(_module_path)
                else:
                    # import dir is the dirpath for the config file
                    _module_dir = os.path.dirname(self.bev_config_path)
                    _module_dir = _module_dir.split('/')
                    _module_path = _module_dir[0]
                    for m in _module_dir[1:]:
                        _module_path = _module_path + '.' + m
                    print(_module_path)
                    plg_lib = importlib.import_module(_module_path)
    
    def init_bev_dataset(self):
        from mmdet3d.datasets import build_dataset
        
        if self.training:
            self.bev_cfg.data.test['ann_file'] = self.bev_cfg.data.test['ann_file'].replace('mode','train')
        else:
            self.bev_cfg.data.test['ann_file'] = self.bev_cfg.data.test['ann_file'].replace('mode','val')
        
        self.bev_dataset = build_dataset(self.bev_cfg.data.test)
        bev_data_infos = self.bev_dataset.data_infos
        self.bev_tokens = np.array([data_info['token'] for data_info in bev_data_infos])

    def init_bevformer(self):
        from bev.mmdet3d_plugin.bevformer.detectors.bevformer import BEVFormer
        from collections import OrderedDict
        self.bev_cfg.model.train_cfg = None
        model = BEVFormer(
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
        
        load_state_dict(model, new_state_dict, strict=False)
        
        if self.fuse_conv_bn:
            model = fuse_conv_bn(model)
        
        self.model = MMDataParallel(model, device_ids=[0]).eval()
    
    def generate_prediction_dicts(self, batch_dict, output_path=None):
        """
        Args:
            batch_dict:
                pred_scores: (num_center_objects, num_modes)
                pred_trajs: (num_center_objects, num_modes, num_timestamps, 7)

              input_dict:
                center_objects_world: (num_center_objects, 21)
                center_objects_type: (num_center_objects)
                center_objects_id: (num_center_objects)
                center_gt_trajs_src: (num_center_objects, num_timestamps, 10)
        """
        input_dict = batch_dict['input_dict']

        pred_scores = batch_dict['pred_scores']
        pred_trajs = batch_dict['pred_trajs']
        center_objects_world = input_dict['center_objects_world'].type_as(pred_trajs)

        num_center_objects, num_modes, num_timestamps, num_feat = pred_trajs.shape
        assert num_feat == 7
        
        pred_trajs_world = common_utils.rotate_points_along_z(
            points=pred_trajs.view(num_center_objects, num_modes * num_timestamps, num_feat),
            angle=center_objects_world[:, -5].view(num_center_objects)
        ).view(num_center_objects, num_modes, num_timestamps, num_feat) # 旋转
        pred_trajs_world[:, :, :, 0:2] += center_objects_world[:, None, None, 0:2] # 平移
        
        pred_dict_list = []
        batch_sample_count = batch_dict['batch_sample_count']
        start_obj_idx = 0
        for bs_idx in range(batch_dict['batch_size']):
            cur_scene_pred_list = []
            for obj_idx in range(start_obj_idx, start_obj_idx + batch_sample_count[bs_idx]):
                single_pred_dict = {
                    'scenario_id': input_dict['scenario_id'][obj_idx], #!!!
                    'pred_trajs': pred_trajs_world[obj_idx, :, :, 0:2].cpu().numpy(),  #!!!
                    'pred_scores': pred_scores[obj_idx, :].cpu().numpy(),
                    'object_id': input_dict['center_objects_id'][obj_idx],  #!!!
                    'object_type': input_dict['center_objects_type'][obj_idx],
                    'gt_trajs': input_dict['center_gt_trajs_src'][obj_idx].cpu().numpy(),  #!!!
                    'track_index_to_predict': input_dict['track_index_to_predict'][obj_idx].cpu().numpy()  #!!!
                }
                cur_scene_pred_list.append(single_pred_dict)
            pred_dict_list.append(cur_scene_pred_list)
            start_obj_idx += batch_sample_count[bs_idx]
        assert start_obj_idx == num_center_objects
        assert len(pred_dict_list) == batch_dict['batch_size']
        
        return pred_dict_list
    
    def evaluation(self, pred_dicts, output_path=None, eval_method='nuscenes', **kwargs):
        pass 
        

if __name__=='__main__':
    
    nusc_data = NuscenesDataset()

    