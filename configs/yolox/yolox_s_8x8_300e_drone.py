_base_ = './yolox_s_8x8_300e_coco.py'

dataset_type = 'CocoDataset'
data_root = '../data/drone/data/'

img_scale = (640, 640)

train_pipeline = [
    dict(type='Mosaic', img_scale=img_scale, pad_val=114.0),
    dict(
        type='RandomAffine',
        scaling_ratio_range=(0.1, 2),
        border=(-img_scale[0] // 2, -img_scale[1] // 2)),
    dict(
        type='MixUp',
        img_scale=img_scale,
        ratio_range=(0.8, 1.6),
        pad_val=114.0),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', flip_ratio=0.5),
    # According to the official implementation, multi-scale
    # training is not considered here but in the
    # 'mmdet/models/detectors/yolox.py'.
    dict(type='Resize', img_scale=img_scale, keep_ratio=True),
    dict(
        type='Pad',
        pad_to_square=True,
        # If the image is three-channel, the pad value needs
        # to be set separately for each channel.
        pad_val=dict(img=(114.0, 114.0, 114.0))),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1), keep_empty=False),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]


train_dataset = dict(
    type='MultiImageMixDataset',
    dataset=dict(
        type=dataset_type,
        ann_file=data_root + 'train/coco/train.json',
        img_prefix=data_root + 'train/images/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True)
        ],
        filter_empty_gt=False,
    ),
    pipeline=train_pipeline)

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=1,
    train=train_dataset,
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'val/coco/val.json',
        img_prefix=data_root + 'val/images/',),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'test/coco/test.json',
        img_prefix=data_root + 'test/images/',))

evaluation = dict(interval=10)

model = dict(
    bbox_head=dict(
        num_classes=4,))

checkpoint_config = dict(interval=10)
runner = dict(max_epochs=200)
load_from = 'checkpoints/yolox_s_8x8_300e_coco_20211121_095711-4592a793.pth'