_base_ = '../yolov5/yolov5_s-v61_syncbn_fast_8xb16-300e_coco.py'

data_root = 'data/rock/'
class_name = ('tuff', 'granite', 'feldsparphyric rhyolite', 'granodiorite', 'jar', 'foam')
num_classes = len(class_name)
metainfo = dict(classes=class_name, palette=[(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228),
                                             (0, 60, 100)])
channels = [192, 384, 768]
# pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth'
load_from = 'D:/PythonProjects/mmyolo/work_dirs/rock_swyolo_coco/best_coco_bbox_mAP_epoch_310.pth'

# Adaptive anchor based on tools/analysis_tools/optimize_anchors.py
anchors = [
    [(9, 7), (13, 14), (26, 19)],  # P3/8
    [(32, 33), (55, 32), (48, 46)],  # P4/16
    [(85, 36), (133, 37), (284, 36)]  # P5/32
]

# Max training epoch
max_epochs = 400
# Set batch size to 12
train_batch_size_per_gpu = 16
# dataloader num workers
train_num_workers = 4
widen_factor = 1.0

model = dict(
    backbone=dict(
        _delete_=True,
        type='mmdet.SwinTransformer',
        # Swin Transformer-tiny
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.6,
        patch_norm=True,
        out_indices=(1, 2, 3),
        with_cp=False,
        convert_weights=True,
        # freeze the parameters in and before the first stage
        frozen_stages=1
    ),
    neck=dict(
        in_channels=channels,
        out_channels=channels,
        widen_factor=widen_factor,
        type='YOLOv5PAFPN'
    ),
    bbox_head=dict(
        head_module=dict(
            in_channels=channels,
            num_classes=num_classes,
            widen_factor=widen_factor,
            type='YOLOv5HeadModule',
        ),
        prior_generator=dict(base_sizes=anchors)
    ),
)

train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        # Dataset annotation file of json path
        # ann_file='train/annotation_rock.json',
        ann_file='train/annotation_rock.json',
        # Dataset prefix
        data_prefix=dict(img='train/'))
)

val_dataloader = dict(
    dataset=dict(
        metainfo=metainfo,
        data_root=data_root,
        ann_file='val/annotation_rock.json',
        data_prefix=dict(img='val/'))
)

test_dataloader = val_dataloader

_base_.optim_wrapper.optimizer.batch_size_per_gpu = train_batch_size_per_gpu

val_evaluator = dict(ann_file=data_root + 'val/annotation_rock.json')
test_evaluator = dict(ann_file=data_root + 'val/annotation_rock.json')

default_hooks = dict(
    # Save weights every 10 epochs and a maximum of two weights can be saved.
    # The best model is saved automatically during model evaluation
    checkpoint=dict(interval=10, max_keep_ckpts=2, save_best='auto'),
    # The warmup_mim_iter parameter is critical.
    # The default value is 1000 which is not suitable.
    param_scheduler=dict(max_epochs=max_epochs, warmup_mim_iter=10),
    # The log printing interval is 5
    logger=dict(type='LoggerHook', interval=5)
)

# The evaluation interval is 10
train_cfg = dict(max_epochs=max_epochs, val_interval=10)

visualizer = dict(vis_backends=[dict(type='LocalVisBackend'), dict(type='WandbVisBackend')])

# avoid multiple prediction
model_test_cfg = dict(
    multi_label=False
)
# ---------------------------------------------------------------------------
albu_train_transforms = [
    dict(type='Blur', p=0.01),
    dict(type='MedianBlur', p=0.01),
    dict(type='ToGray', p=0.01),
    dict(type='CLAHE', p=0.01)
]

pre_transform = [
    dict(type='LoadImageFromFile', backend_args=_base_.backend_args),
    dict(type='LoadAnnotations', with_bbox=True)
]

last_transform = [
    dict(
        type='mmdet.Albu',
        transforms=albu_train_transforms,
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_bboxes_labels', 'gt_ignore_flags']),
        keymap={
            'img': 'image',
            'gt_bboxes': 'bboxes'
        }),
    dict(type='YOLOv5HSVRandomAug'),
    dict(type='mmdet.RandomFlip', prob=0.5),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'flip',
                   'flip_direction'))
]

train_pipeline = [
    *pre_transform,
    dict(
        type='Mosaic',
        img_scale=(640, 640),
        pad_val=114.0,
        # max_cached_images=20,
        pre_transform=pre_transform),
    dict(
        type='YOLOv5RandomAffine',
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        scaling_ratio_range=(1 - 0.9, 1 + 0.9),
        max_aspect_ratio=100,
        # img_scale is (width, height)
        border=(-320, -320),
        border_val=(114, 114, 114)
    ),
    *last_transform
]

train_pipeline_stage2 = [
    *pre_transform,
    dict(type='YOLOv5KeepRatioResize', scale=(640, 640)),
    dict(
        type='LetterResize',
        scale=(640, 640),
        allow_scale_up=True,
        pad_val=dict(img=114.0)),
    dict(
        type='YOLOv5RandomAffine',
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        scaling_ratio_range=(1 - 0.9, 1 + 0.9),
        max_aspect_ratio=100,
        border_val=(114, 114, 114)),
    *last_transform
]

train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        # Dataset annotation file of json path
        # ann_file='train/annotation_rock.json',
        ann_file='train/annotation_rock.json',
        # Dataset prefix
        data_prefix=dict(img='train/'),
        pipeline=train_pipeline
    )
)

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='SGD',
        lr=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        nesterov=True,
        batch_size_per_gpu=train_batch_size_per_gpu),
    constructor='YOLOv5OptimizerConstructor')

default_hooks = dict(
    param_scheduler=dict(
        type='YOLOv5ParamSchedulerHook',
        scheduler_type='cosine',
        lr_factor=0.01,
        max_epochs=max_epochs,
        warmup_mim_iter=10),
    checkpoint=dict(
        type='CheckpointHook',
        interval=10,
        save_best='auto',
        max_keep_ckpts=2))

custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0001,
        update_buffers=True,
        strict_load=False,
        priority=49),
    dict(
        type='mmdet.PipelineSwitchHook',
        switch_epoch=max_epochs - 10,
        switch_pipeline=train_pipeline_stage2)
]

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_interval=10,
    dynamic_intervals=[((max_epochs - 10),
                        1)])
