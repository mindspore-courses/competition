# Copyright (c) Tencent Inc. All rights reserved.
from typing import Tuple, Union

from .datasets import MultiModalDataset, YOLOv5LVISV1Dataset
from .transforms import YOLOResize, LoadAnnotations, LoadImageFromFile, LoadText, PackDetInputs

import mindspore.dataset as ds
__all__ = ('build_lvis_testloader', )


def build_lvis_testloader(img_scale: Union[int, Tuple[int, int]] = (640, 640),
                          anno_file: str = 'lvis/lvis_v1_minival_inserted_image_name.json',):

    # build transform
    test_pipeline = [
        LoadImageFromFile(),
        YOLOResize(scale=img_scale),
        LoadAnnotations(with_bbox=True),
        LoadText(),
        PackDetInputs(meta_keys=('img_id', 'img_path', 'ori_shape',
                                 'img_shape', 'scale_factor', 'pad_param', 'texts')),
    ]

    # build dataset
    lvis_dataset = YOLOv5LVISV1Dataset(
        data_root='data/coco/',
        test_mode=True,
        ann_file=anno_file,
        data_prefix=dict(img=''),
        pipeline=test_pipeline,
    )
    
    test_dataset = MultiModalDataset(
        dataset=lvis_dataset, class_text_path='data/texts/lvis_v1_class_texts.json', pipeline=test_pipeline)
    # if hasattr(test_dataset, 'full_init'):
    #     test_dataset.full_init()

    # # build sampler
    # test_sampler = DefaultSampler(dataset=test_dataset, shuffle=False, seed=None if diff_rank_seed else seed)

    # # build dataloader
    # test_dataloader = DataLoader(
    #     dataset=test_dataset,
    #     sampler=test_sampler,
    #     batch_size=val_batch_size_per_gpu,
    #     num_workers=val_num_workers,
    #     persistent_workers=persistent_workers,
    #     pin_memory=True,
    #     drop_last=False,
    #     collate_fn=pseudo_collate  # `pseudo_collate`
    # )

    # test_dataloader = ds.GeneratorDataset(source=test_dataset, column_names=["data_info"])

    return test_dataset
    # return test_dataloader
