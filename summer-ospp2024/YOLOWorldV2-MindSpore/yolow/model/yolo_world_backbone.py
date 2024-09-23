# Copyright (c) Tencent Inc. All rights reserved.
from typing import List, Tuple

import  mindspore.nn as nn
from mindspore import Tensor
import time
__all__ = ("MultiModalYOLOBackbone")

class MultiModalYOLOBackbone(nn.Cell):

    def __init__(self,
                 image_model: nn.Cell,
                 text_model: nn.Cell,
                 frozen_stages: int = -1,
                 with_text_model: bool = True) -> None:
        super().__init__()
        self.with_text_model = with_text_model
        self.image_model = image_model
        if self.with_text_model:
            self.text_model = text_model
        else:
            self.text_model = None
        self.frozen_stages = frozen_stages
        self._freeze_stages()

    def _freeze_stages(self):
        """Freeze the parameters of the specified stage so that they are no
        longer updated."""
        if self.frozen_stages >= 0:
            for i in range(self.frozen_stages + 1):
                m = getattr(self.image_model, self.image_model.layers[i])
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        self._freeze_stages()

    def construct(self, image: Tensor, text: List[List[str]]) -> Tuple[Tuple[Tensor], Tensor]:
        start = time.perf_counter()
        img_feat = self.image_model(image)
        img_enc_time = time.perf_counter()
        txt_feat = self.text_model(text)
        txt_enc_time = time.perf_counter()

        # print('_'*20+'\n',
        #       f'img_enc_time: {(img_enc_time - start):.4f}\n'
        #       f'txt_enc_time: {(txt_enc_time - img_enc_time):.4f}')

        return img_feat, txt_feat, {'img_enc_time': img_enc_time - start, 'txt_enc_time': txt_enc_time - img_enc_time}

    def forward_text(self, text: List[List[str]]) -> Tensor:
        if self.with_text_model:
            return self.text_model(text)
        return None  # no text model

    def forward_image(self, image: Tensor) -> Tuple[Tensor]:
        return self.image_model(image)
