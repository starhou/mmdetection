dataset_type = 'CocoDataset'
data_root = '../data/drone/data/'


data = dict(
    samples_per_gpu=32,
    workers_per_gpu=16,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'train/coco/train.json',
        img_prefix=data_root + 'train/images/',),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'val/coco/val.json',
        img_prefix=data_root + 'val/images/',),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'test/coco/test.json',
        img_prefix=data_root + 'test/images/',))

evaluation = dict(interval=10, metric=['bbox', 'segm'])
