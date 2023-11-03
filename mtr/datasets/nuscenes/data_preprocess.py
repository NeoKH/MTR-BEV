import math
import zlib
import random
import torch
import pickle
import argparse
import multiprocessing
from multiprocessing import Process
from tqdm import tqdm
from pathlib import Path
import torch.utils.data as torch_data

from nuscenes import NuScenes
from nuscenes.eval.prediction.splits import get_prediction_challenge_split
from nuscenes.map_expansion.map_api import NuScenesMap, discretize_lane
from nuscenes.prediction.input_representation.static_layers import StaticLayerRasterizer, get_lanes_in_radius, correct_yaw, quaternion_yaw
from nuscenes.prediction.input_representation.agents import *
from shapely.geometry import Point

class NuScenesProcessor(torch_data.Dataset):
    def __init__(self, src_dir = "" ,dst_dir = "",version = 'v1.0-trainval',split = "train",cores=8):
        super().__init__()
        self.nuscenes = NuScenes(version, dataroot=src_dir)
        self.helper = PredictHelper(self.nuscenes)
        self.cores = cores
        
        self.split = split
        self.dst_dir = Path(dst_dir) / self.split
        if not self.dst_dir.exists():
            self.dst_dir.mkdir(parents=True)
        
        # Observe and prediction horizons
        self.obs_time = 2 # seconds
        self.pred_time = 6 # seconds
        self.freq = 2 # Hz
        
        self.obs_len  = self.freq * self.obs_time
        self.pred_len = self.freq * self.pred_time
        
        self.subdivide_len = 5
        self.lane_feature_dim = 11
        
        self.token_list = get_prediction_challenge_split(self.split, dataroot=self.helper.data.dataroot)
        
        self.data_num = len(self.token_list)
        self.data_per_core = int(self.data_num / self.cores) + 1
        
        self.visibility_trans = [0.,0.3,0.6,1.]
    
    def __len__(self):
        return len(self.token_list)
    
    def __getitem__(self, index) -> dict:
        token = self.token_list[index]

        self.ego_instance_token, self.sample_token = token.split('_')
        self.sample = self.nuscenes.get('sample', self.sample_token)
        self.scene  = self.nuscenes.get('scene', self.sample['scene_token'])
        self.log    = self.nuscenes.get('log', self.scene['log_token'])
        self.map = NuScenesMap(dataroot= DATAROOT, map_name=self.log['location'])
        
        
        self.batch_observe,self.batch_predict, self.center_idxs = self.get_origin_trajs()
        
        data_dict = {
            "scenario_id": token, # str
            "observe_data":self.batch_observe, # (N,5,21)
            "predict_data":self.batch_predict, # (N,12,7)
            "center_idxs":self.center_idxs, # [True,False,...]
        }
        if np.sum(self.center_idxs)>0:
            data_dict["map_polyline"] = self.get_origin_map(), # [(m1,7),(m2,7),...]
            save_path = self.dst_dir / f"{token}.pkl"
            with open(save_path,"wb") as f:
                pickle.dump(data_dict, f)
        else:
            return
        
    def get_origin_trajs(self):
        obs_seq = []
        obs_masks = []
        obs_yaws = []
        obj_vis = []

        pred_seq = []
        pred_masks = []
        pred_yaws = []
        
        obj_types = []
        obj_sizes = []
        
        for j,anno_token in enumerate(self.sample['anns']): # 遍历场景中的Agent
            sample_ann = self.nuscenes.get('sample_annotation', anno_token)
            category = sample_ann['category_name']
            ins_token = sample_ann['instance_token']
            
            past_infos = self.helper.get_past_for_agent(ins_token,self.sample_token,seconds=self.obs_time,in_agent_frame=False,just_xy=False)
            future_infos = self.helper.get_future_for_agent(ins_token,self.sample_token,seconds=self.pred_time,in_agent_frame=False,just_xy=False)
            obj_info = self.helper.get_sample_annotation(instance_token=ins_token,sample_token=self.sample_token)
            
            pose_at_now = obj_info['translation']
            past_traj_global = [x["translation"] for x in past_infos]
            future_traj_global = [x["translation"] for x in future_infos]
            
            
            # 处理历史轨迹
            # 有的历史轨迹不足4帧，有两种处理思路，一是用mask标记有效位置，二是用一些临近帧去填补
            # 这里既用了临近帧填补，也用了mask标记
            if len(past_traj_global) == 0: # 没有历史轨迹
                if len(future_traj_global)==0: # 也没有未来轨迹,丢弃
                    continue
                elif len(future_traj_global)!=0: # 有未来轨迹
                    continue
                    past_traj_global = list(np.array([pose_at_now] * self.obs_len)) # 用当前帧填补整个历史
                    obs_mask = [False]*self.obs_len
                    visibilities = [-1]*self.obs_len
                    obs_yaw = [quaternion_yaw(Quaternion(obj_info['rotation']))]*self.obs_len
                    obj_size = [obj_info["size"]]*self.obs_len
                    # print(f"第{i:04d}片段第{j:03d}目标的历史轨迹为0,未来轨迹偏移为{future_traj_global[0,:]-future_traj_global[-1,:]}")
            elif 0< len(past_traj_global) < self.obs_len: # 历史帧数量不够
                obs_mask = [False]*(self.obs_len - len(past_traj_global)) + [True]*len(past_traj_global)
                visibilities = [0.]*(self.obs_len - len(past_traj_global)) + [self.visibility_trans[int(x["visibility_token"])-1] for x in past_infos]
                obs_yaw = [quaternion_yaw(Quaternion(d['rotation'])) for d in past_infos]
                obs_yaw = [obs_yaw[0]]*(self.obs_len - len(past_traj_global)) + obs_yaw
                obj_size = [obj_info["size"]]*(self.obs_len - len(past_traj_global)) + [x["size"]for x in past_infos]
                past_traj_global = [past_traj_global[0]] * (self.obs_len - len(past_traj_global)) + list(past_traj_global)
            else:
                past_traj_global = list(past_traj_global)
                obs_mask = [True]*self.obs_len
                obs_yaw = [quaternion_yaw(Quaternion(d['rotation'])) for d in past_infos]
                visibilities = [self.visibility_trans[int(x["visibility_token"])-1]for x in past_infos]
                obj_size = [x["size"]for x in past_infos]
            
            # 添加now
            past_traj_global.append(pose_at_now)
            obs_mask.append(True)
            visibilities.append(self.visibility_trans[int(obj_info["visibility_token"])-1])
            obj_size.append(obj_info["size"])
            obs_yaw.extend([quaternion_yaw(Quaternion(obj_info['rotation']))])
            
            # 处理未来轨迹
            if len(future_traj_global)==0:
                continue
                future_traj_global = list(np.array([pose_at_now] * self.pred_len)) # 用当前帧填补整个未来
                pred_mask = [False]*self.pred_len
                pred_yaw = [quaternion_yaw(Quaternion(obj_info['rotation']))]*self.pred_len
            elif 0< len(future_traj_global) < self.pred_len: # 未来帧数量不够
                pred_mask = [True]*len(future_traj_global) + [False]*(self.pred_len-len(future_traj_global))
                pred_yaw = [quaternion_yaw(Quaternion(d['rotation'])) for d in future_infos]
                pred_yaw.extend([pred_yaw[-1]]*(self.pred_len-len(future_traj_global)))
                future_traj_global = list(future_traj_global) + [future_traj_global[-1]]*(self.pred_len-len(future_traj_global))
            else:
                future_traj_global = list(future_traj_global)
                pred_mask = [True]*len(future_traj_global)
                pred_yaw = [quaternion_yaw(Quaternion(d['rotation'])) for d in future_infos]
            
            
            # agent 类型信息
            if category.startswith('human'):
                obj_types.append(0)
            elif category.startswith('vehicle') and ins_token != self.ego_instance_token:
                obj_types.append(1)
            elif category.startswith('vehicle') and ins_token == self.ego_instance_token:
                obj_types.append(2) # ego vehicle
            else:
                continue
                obj_types.append(3) # Unknown object
            
            obj_sizes.append(np.array(obj_size))
            obj_vis.append(np.array(visibilities))
            
            obs_seq.append(past_traj_global)
            obs_masks.append(obs_mask)
            obs_yaws.append(obs_yaw)
            
            pred_seq.append(future_traj_global)
            pred_masks.append(pred_mask)
            pred_yaws.append(pred_yaw)
            
        obs_seq = np.stack(obs_seq) # (N,5,3)
        obs_masks = np.stack(obs_masks).astype(np.int) # (N,5)
        
        obs_vels = np.zeros((obs_seq.shape[0],obs_seq.shape[1],obs_seq.shape[2]-1)) # (N,5,2)
        obs_vels[:,1:,:] = obs_seq[:,1:,:2] - obs_seq[:,:-1,:2]
        obs_vels[:,0,:] = obs_vels[:,1,:]
        
        obs_accs = np.zeros_like(obs_vels) # (N,5,2)
        obs_accs[:,1:,:] = obs_vels[:,1:,:] - obs_vels[:,:-1,:]
        
        obs_yaws = np.array(obs_yaws) # (N,5)
        pred_yaws = np.array(pred_yaws) # (N,12)
        
        obj_sizes = np.stack(obj_sizes) # (N,5,3)

        _obj_types = np.array(obj_types,dtype=np.int) # (N)
        obj_types = np.eye(3)[_obj_types] # (N,3)
        obj_types = np.repeat(obj_types[:,None,:],self.obs_len+1,axis=1) # (N,5,3)
        
        obj_vis = np.stack(obj_vis) # (N,5)
        
        time_embed = np.repeat(np.eye(self.obs_len+1)[None,:,:],obj_vis.shape[0],axis=0) # (N,5,5)
        
        batch_observe = np.concatenate((
            obs_seq, # 3
            obj_sizes, # 3
            obs_masks[:,:,None], # 1
            obj_vis[:,:,None], # 1
            obj_types, # 3
            time_embed, # 5
            obs_yaws[:,:,None], # 1
            obs_vels, # 2
            obs_accs, # 2
        ), axis=-1)
        
        pred_seq = np.stack(pred_seq) # (N,12,3)
        pred_masks = np.stack(pred_masks) # (N,12)
        pred_vels = np.zeros_like(pred_seq[:,:,:2]) # (N,12,2)
        pred_vels[:,1:,:] = pred_seq[:,1:,:2] - pred_seq[:,:-1,:2]
        pred_vels[:,0,:]  = pred_seq[:,0,:2]  - obs_seq[:,-1,:2]
        
        batch_predict = np.concatenate((
            pred_seq, # 3
            pred_vels, # 2
            pred_masks[:,:,None], # 1
            pred_yaws[:,:,None] # 1
        ),axis=-1) # (N,12,7)
        
        # 挑选出可以作为预测目标的 agent
        ego_idx = np.argwhere(batch_observe[:,-1,10]==1)[0][0]
        obs_valid_cnt = np.sum(batch_observe[:,:,6],axis=1) # valid mask
        pre_valid_cnt = np.sum(batch_predict[:,:,-2],axis=1) # valid mask
        center_idxs = np.bitwise_and(obs_valid_cnt>=5, pre_valid_cnt>=11)
        center_idxs[ego_idx] = False # 不预测ego
        
        #  排除长时间静止不动的轨迹
        for i,center_idx in enumerate(center_idxs):
            if center_idx:
                pred_trajs = batch_predict[i,:,:2]
                pose_now = batch_observe[i,-1,:2]
                _trajs = pred_trajs - pose_now
                if np.sum(_trajs) == 0.0:
                    center_idxs[i] = False
                end_point = batch_predict[i,-1,:2]
                start_point = batch_observe[i,0,:2]
                dis = np.abs(end_point-start_point)
                if dis[0] < 0.1 and dis[1] < 0.1:
                    center_idxs[i] = False
        
        # vehicle_cnt = np.sum(np.bitwise_and(select_idxs,batch_observe[:,4,9]==1))
        # pedestrian_cnt = np.sum(np.bitwise_and(select_idxs,batch_observe[:,4,8]==1))
        return batch_observe, batch_predict, center_idxs
       
    def get_origin_map(self):
        """
            
        """
        def get_polyline(token_list,resolution_meters=1,lane_type=0):
            lane_data = self.map.discretize_lanes(token_list,resolution_meters=resolution_meters)
            if len(lane_data)>0:
                poses = [np.array(x) for x in lane_data.values()]
                directions = [get_polyline_direction(x) for x in poses]
                lane_type = [np.ones_like(x[:,0])[:,None]*lane_type for x in poses]
                polylines = [np.concatenate((poses[i],directions[i],lane_type[i]),axis=1)for i in range(len(poses))]
                return polylines
            else:
                return [np.zeros((0, 7), dtype=np.float32)]
        
        # print(self.batch_observe[:,:,:2].shape)
        # print(self.batch_predict[:,:,:2].shape)
        # 计算场景中所有Agent坐标的平均值作为查询Map的中心点
        # center = batch_pos.reshape(-1,2).mean(axis=0)
        # batch_dis = np.linalg.norm(batch_pos-center[None,:], axis=-1)
        # max_dis = np.max(batch_dis)
        
        batch_pos = np.concatenate((
            self.batch_observe[self.center_idxs,:,:2],
            self.batch_predict[self.center_idxs,:,:2]),
            axis=1
        ).reshape(-1,2)
        
        min_pos = np.min(batch_pos,axis=0) -5
        max_pos = np.max(batch_pos,axis=0) +5
        box_coords = (*min_pos,*max_pos)
        layer_names=["lane","lane_connector"]
        lane_types = { "lane":1,  "lane_connector":0}
        lane_dict = self.map.get_records_in_patch(box_coords,layer_names)
        
        polylines = get_polyline(lane_dict['lane'],lane_type=lane_types['lane'])
        polylines += get_polyline(lane_dict['lane_connector'],lane_type=lane_types['lane_connector'])
        
        return polylines
    
    def extract_multiprocess(self):
        """
        the parallel process of extracting data
        """
        def run(queue: multiprocessing.Queue):
            process_id = queue.get()
            if process_id == self.cores - 1:
                li = list(range(process_id * self.data_per_core, self.data_num))
            elif process_id is not None:
                li = list(range(process_id * self.data_per_core, (process_id + 1) * self.data_per_core))

            for idx in tqdm(li):
                self.__getitem__(idx)

        queue = multiprocessing.Queue(self.cores)
        processes = [Process(target=run, args=(queue,)) for _ in range(self.cores)]
        
        for each in processes:
            each.start()
        for i in range(self.cores):
            queue.put(i)
        while not queue.empty():
            pass
        for each in processes:
            each.join()
    
def get_polyline_direction(polyline):
    polyline_pre = np.roll(polyline, shift=1, axis=0)
    polyline_pre[0] = polyline[0]
    diff = polyline - polyline_pre
    polyline_dir = diff / np.clip(np.linalg.norm(diff, axis=-1)[:, np.newaxis], a_min=1e-6, a_max=1000000000)
    return polyline_dir

    
if __name__=="__main__":
    # VERSION = 'v1.0-mini'
    # SPLIT = 'mini_train'
    # DATAROOT = '../../../data/nuscenes/v1.0-mini'
    VERSION = 'v1.0-trainval'
    SPLIT = 'train'
    DATAROOT = '../../../data/nuscenes'
    SAVEROOT = '../../../data/processed_data'
    CORES = 10
    
    processor = NuScenesProcessor(DATAROOT,SAVEROOT,VERSION,SPLIT,CORES)
    # processor.__getitem__(4717)
    processor.extract_multiprocess()
    # processor = NuScenesProcessor(DATAROOT,SAVEROOT,VERSION,"val",CORES)
    # processor.extract_multiprocess()
    
    