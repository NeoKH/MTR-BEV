_base_ = [
    # '../datasets/custom_nus-3d.py', #! 疑似没有用到
    # '../_base_/default_runtime.py' #! 疑似没有用到
]
#
plugin = True
plugin_dir = 'bev/mmdet3d_plugin/'

fuse_conv_bn = False

# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
voxel_size = [0.2, 0.2, 8]

img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False
)

# For nuScenes we usually do 10-class detection
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
] #! 疑似没有用到

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=True
)

_dim_ = 256
_pos_dim_ = _dim_//2
_ffn_dim_ = _dim_*2
_num_levels_ = 4
bev_h_ = 200
bev_w_ = 200
queue_length = 4 # each sequence contains `queue_length` frames.

img_backbone = dict(
    type='ResNet', depth=101,
    num_stages=4, out_indices=(1, 2, 3),
    frozen_stages=1,
    norm_cfg=dict(type='BN2d', requires_grad=False),
    norm_eval=True,
    style='caffe',
    dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False), # original DCNv2 will print log when perform load_state_dict
    stage_with_dcn=(False, False, True, True)
)
img_neck=dict(
    type='FPN',
    in_channels=[512, 1024, 2048],
    out_channels=_dim_,
    start_level=0,
    add_extra_convs='on_output',
    num_outs=4,
    relu_before_extra_convs=True
)
encoder = dict(
    type='BEVFormerEncoder',
    num_layers=6,
    pc_range=point_cloud_range,
    num_points_in_pillar=4,
    return_intermediate=False,
    transformerlayers = dict(
        type='BEVFormerLayer',
        attn_cfgs=[
            dict(
                type='TemporalSelfAttention',
                embed_dims=_dim_,
                num_levels=1),
            dict(
                type='SpatialCrossAttention',
                pc_range=point_cloud_range,
                deformable_attention=dict(
                    type='MSDeformableAttention3D',
                    embed_dims=_dim_,
                    num_points=8,
                    num_levels=_num_levels_),
                embed_dims=_dim_,
            )
        ],
        feedforward_channels=_ffn_dim_,
        ffn_dropout=0.1,
        operation_order=('self_attn', 'norm', 'cross_attn', 'norm','ffn', 'norm')
    )
)
decoder=dict(
    type='DetectionTransformerDecoder',
    num_layers=6,
    return_intermediate=True,
    transformerlayers=dict(
        type='DetrTransformerDecoderLayer',
        attn_cfgs=[
            dict(
                type='MultiheadAttention',
                embed_dims=_dim_,
                num_heads=8,
                dropout=0.1),
                dict(
                type='CustomMSDeformableAttention',
                embed_dims=_dim_,
                num_levels=1),
        ],

        feedforward_channels=_ffn_dim_,
        ffn_dropout=0.1,
        operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                            'ffn', 'norm')
    )
)
transformer = dict(
    type='PerceptionTransformer',
    rotate_prev_bev=True,
    use_shift=True,
    use_can_bus=True,
    embed_dims=_dim_,
    encoder = encoder,
    decoder = decoder
)
bev_head = dict(
    type='BEVFormerHead',
    bev_h=bev_h_,
    bev_w=bev_w_,
    num_query=900, #! 有用到
    num_classes=10, #! 有用到
    in_channels=_dim_,
    # sync_cls_avg_factor=True, #! 没有用到
    # with_box_refine=True, #! 没有用到
    # as_two_stage=False, #! 没有用到
    transformer = transformer,
    positional_encoding=dict(
        type='LearnedPositionalEncoding',
        num_feats=_pos_dim_,
        row_num_embed=bev_h_,
        col_num_embed=bev_w_,
    )
)

model = dict(
    type='BEVFormer',
    use_grid_mask=True, #! extract_img_feat 用到了
    video_test_mode=True,
    img_backbone = img_backbone,
    img_neck = img_neck,
    bev_head = bev_head,
)

test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1600, 900),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='CustomCollect3D', keys=['img'])
        ])
]
train_pipeline = test_pipeline

dataset_type = 'CustomNuScenesDataset'
data_root = 'data/nuscenes/'
file_client_args = dict(backend='disk')
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=0,
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + f"nuscenes_infos_temporal_mode.pkl",
        pipeline=test_pipeline,
        bev_size=(bev_h_, bev_w_),
        classes=class_names, #! 疑似没有用到
        modality=input_modality,
        test_mode=True, #! 疑似没有用到
        use_valid_flag=True, #! 疑似没有用到
    ),
    shuffler_sampler=dict(type='DistributedGroupSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler')
)

optimizer = dict( #! 疑似没有用到
    type='AdamW',
    lr=2e-4,
    paramwise_cfg=dict( custom_keys={'img_backbone': dict(lr_mult=0.1)}),
    weight_decay=0.01
)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2)) #! 疑似没有用到
lr_config = dict( #! 疑似没有用到
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3
)

total_epochs = 24
evaluation = dict(interval=1, pipeline=test_pipeline) #! 这是什么?

runner = dict(type='EpochBasedRunner', max_epochs=total_epochs) #! 这是什么?
load_from = 'ckpts/r101_dcn_fcos3d_pretrain.pth' #! 疑似没有用到

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ]
)

checkpoint_config = dict(interval=1) #! 疑似没有用到