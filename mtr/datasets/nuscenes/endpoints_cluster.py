



from mtr.config import cfg
import yaml
import pickle
import numpy as np
from sklearn.cluster import KMeans
from mtr.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from nuscenes_dataset  import NuscenesDataset
import tqdm

def get_endpoints(dataset_cfg,training=True):
    dataset = NuscenesDataset(dataset_cfg, training=training)
    
    vehicle_list = []
    pedestrian_list = []
    for i,data in enumerate(dataset):
        idx_to_predict = data["track_index_to_predict"].numpy() # (k)
        gt_infos = data["obj_trajs_future_state"].numpy() # (k,N,12,4)
        
        
        center_objects_type = data["center_objects_type"]
        
        pedestrian_idx = np.argwhere(center_objects_type=="TYPE_PEDESTRIAN").reshape(-1)
        vehicle_idx = np.argwhere(center_objects_type=="TYPE_VEHICLE").reshape(-1)
        
        pedestrian_index = idx_to_predict[pedestrian_idx]
        vehicle_index = idx_to_predict[vehicle_idx]
        
        pedestrian_endpoints = gt_infos[pedestrian_idx,pedestrian_index,-1,:2]
        vehicle_endpoints = gt_infos[vehicle_idx,vehicle_index,-1,:2]

        pedestrian_list.extend(list(pedestrian_endpoints))
        vehicle_list.extend(list(vehicle_endpoints))
        
        # print(pedestrian_endpoints.shape)
        # if i >=1:
        #     break
    p_endpoints_np = np.stack(pedestrian_list,axis=0)
    v_endpoints_np = np.stack(vehicle_list,axis=0)
    
    return v_endpoints_np, p_endpoints_np


def get_cluster_centers(endpoints):
    endpoints = np.stack(endpoints)
    kmeans = KMeans(n_clusters=64, random_state=0).fit(endpoints)
    # print(endpoints.shape)
    # print(kmeans.cluster_centers_.shape)
    return kmeans.cluster_centers_

if __name__ == "__main__":
    cfg_p = cfg.ROOT_DIR / "tools/cfgs/bev_mtr_100_data.yaml"
    cfg_from_yaml_file(cfg_p, cfg)
    
    train_v_endpoints_np,train_p_endpoints_np = get_endpoints(cfg.DATA_CONFIG, training=True)
    # print(train_v_endpoints_np.shape)
    # print(train_p_endpoints_np.shape)
    val_v_endpoints_np,val_p_endpoints_np = get_endpoints(cfg.DATA_CONFIG, training=False) #
    # print(val_v_endpoints_np.shape)
    # print(val_p_endpoints_np.shape)
    v_endpoints_np = np.concatenate([train_v_endpoints_np, val_v_endpoints_np],axis=0)
    p_endpoints_np = np.concatenate([train_p_endpoints_np, val_p_endpoints_np],axis=0)
    
    # print(v_endpoints_np.shape)
    # print(p_endpoints_np.shape)
    
    data_dict = {
        "TYPE_VEHICLE": get_cluster_centers(v_endpoints_np),
        "TYPE_PEDESTRIAN": get_cluster_centers(p_endpoints_np)
    }
    
    with open(cfg.DATA_CONFIG.ENDPOINTS_FILE,"wb") as f:
        pickle.dump(data_dict, f)

