# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import torch
from mmcv.image import tensor2imgs
from os import path as osp
import pickle
import gzip
from tqdm import tqdm

def save_zipped_pickle(obj, filename, protocol=-1):
    with gzip.open(filename, 'wb') as f:
        pickle.dump(obj, f, protocol)

def custom_single_gpu_test(
    model,
    data_loader,
    out_dir=None
):
    model.eval()
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in tqdm(enumerate(data_loader)):
        with torch.no_grad():
            result = model(rescale=True, **data)
        # print(result.shape)
        # torch.save(result.clone(),"./result.pt")
        # result.view(result.shape[0],200,200,result.shape[-1])
        # img_metas = data['img_metas'][0].data[0][0]
        # print(img_metas.keys())
        # img_metas = dict()
        # img_metas['bev'] = result.clone()
        # # print(img_metas)
        # save_zipped_pickle(img_metas, './img_metas.pklz')
        
        # break
        
        batch_size = len(result)
        for _ in range(batch_size):
            prog_bar.update()



def custom_multi_gpu_test(
    
):
    
    pass