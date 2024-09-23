# Copyright (c) Tencent Inc. All rights reserved.
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Union

import mindspore.nn as nn
from mindspore import Tensor
import time

__all__ = ('YOLOWorldDetector', )

class YOLOWorldDetector(nn.Cell):
    """YOLO-World arch

    train_step(): forward() -> loss() -> extract_feat()
    val_step(): forward() -> predict() -> extract_feat()
    """

    def __init__(self,
                 backbone: nn.Cell,
                 neck: nn.Cell,
                 bbox_head: nn.Cell,
                 mm_neck: bool = False,
                 num_train_classes: int = 80,
                 num_test_classes: int = 80,
                 data_preprocessor: Optional[nn.Cell] = None,) -> None:
        super().__init__()

        self.mm_neck = mm_neck
        self.num_train_classes = num_train_classes
        self.num_test_classes = num_test_classes

        self.backbone = backbone
        self.neck = neck
        self.bbox_head = bbox_head
        self.data_preprocessor = data_preprocessor


    @property
    def with_neck(self) -> bool:
        return hasattr(self, 'neck') and self.neck is not None

    def val_step(self, data: Union[tuple, dict, list]) -> list:
        data = self.data_preprocessor(data, False)
        
        return self(**data, mode='predict')  # type: ignore

    def test_step(self, data: Union[dict, tuple, list]) -> list:
        return self.val_step(data)

    def construct(self, data: Union[dict, tuple, list]) -> Union[dict, list, tuple, Tensor]:
        data_info = self.data_preprocessor(data, False)
        res, time_dict = self.predict(data_info["inputs"], data_info["data_samples"])
        return res, time_dict


    def parse_losses(self, losses: Dict[str, Tensor]) -> Tuple[Tensor, Dict[str, Tensor]]:
        log_vars = []
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, Tensor):
                log_vars.append([loss_name, loss_value.mean()])
            elif isinstance(loss_value, Union[List[Tensor], Tuple[Tensor]]):
                log_vars.append([loss_name, sum(_loss.mean() for _loss in loss_value)])
            else:
                raise TypeError(f'{loss_name} is not a tensor or list of tensors')

        loss = sum(value for key, value in log_vars if 'loss' in key)
        log_vars.insert(0, ['loss', loss])
        log_vars = OrderedDict(log_vars)  # type: ignore

        return loss, log_vars  # type: ignore


    def predict(self, batch_inputs: Tensor, batch_data_samples: Union[List, dict], rescale: bool = True) -> list:
        
        start = time.perf_counter()
        img_feats, txt_feats, time_dict = self.extract_feat(batch_inputs, batch_data_samples)
        enc_time = time.perf_counter()
        self.bbox_head.num_classes = txt_feats[0].shape[0]
        # results_list = self.bbox_head.predict(img_feats, txt_feats, batch_data_samples, rescale=rescale)
        results_list = self.bbox_head(img_feats, txt_feats, batch_data_samples, rescale=rescale)
        pred_time = time.perf_counter()
        
        # print(f'enc_time: {(enc_time - start):.4f}\n',
        #       f'pred_time: {(pred_time - enc_time):.4f}\n',
        #       '_'*20)

        batch_data_samples = self.add_pred_to_datasample(batch_data_samples, results_list)
        time_dict.update({"pred_time":pred_time-enc_time, "all_time":pred_time-start, })
        return batch_data_samples, time_dict

    def _forward(self,
                 batch_inputs: Tensor,
                 batch_data_samples: Optional[Union[List, dict]] = None) -> Tuple[List[Tensor]]:
        img_feats, txt_feats = self.extract_feat(batch_inputs, batch_data_samples)
        results = self.bbox_head.forward(img_feats, txt_feats)
        return results

    def extract_feat(self, batch_inputs: Tensor, batch_data_samples: Union[List, dict]) -> Tuple[Tuple[Tensor], Tensor]:
        txt_feats = None
        if batch_data_samples is None:
            texts = self.texts
            txt_feats = self.text_feats
        elif isinstance(batch_data_samples, dict) and 'texts' in batch_data_samples['img_metas']:
            texts = batch_data_samples['img_metas']['texts']
        elif isinstance(batch_data_samples, list) and ('texts' in batch_data_samples[0]['img_metas']):
            texts = [data_sample['img_metas']['texts'] for data_sample in batch_data_samples]
        elif hasattr(self, 'text_feats'):
            texts = self.texts
            txt_feats = self.text_feats
        else:
            raise TypeError('batch_data_samples should be dict or list.')
        if txt_feats is not None:
            # forward image only
            img_feats = self.backbone.forward_image(batch_inputs)
        else:
            img_feats, txt_feats, time_dict = self.backbone(batch_inputs, texts)
        
        if self.with_neck:
            if self.mm_neck:
                img_feats = self.neck(img_feats, txt_feats)
            else:
                img_feats = self.neck(img_feats)
        return img_feats, txt_feats, time_dict

    def add_pred_to_datasample(self, data_samples: List, results_list: List) -> List:
        for data_sample, pred_instances in zip(data_samples, results_list):
            data_sample['pred_instances'] = pred_instances
        # samplelist_boxtype2tensor(data_samples)
        return data_samples

    def reparameterize(self, texts: List[List[str]]) -> None:
        # encode text embeddings into the detector
        self.texts = texts
        self.text_feats = self.backbone.forward_text(texts)
