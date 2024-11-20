# Copyright (c) Tencent Inc. All rights reserved.
import itertools
import warnings
from typing import List, Sequence

import os


# To avoid warnings from huggingface transformers (seems a bug)
# FutureWarning: `resume_download` is deprecated and will be removed in version 1.0.0.
# Downloads always resume when possible. If you want to force a new download, use `force_download=True`.
warnings.simplefilter(action='ignore', category=FutureWarning)

__all__ = ('HuggingCLIPLanguageBackbone', )

from mindnlp.transformers import CLIPTextModelWithProjection as CLIPTP
from mindnlp.transformers import CLIPTextConfig as CLIPTextConfig
from mindnlp.transformers import AutoTokenizer as AutoTokenizer

from mindspore import Tensor
import mindspore.nn as nn


class HuggingCLIPLanguageBackbone(nn.Cell):

    def __init__(self, model_name: str, frozen_modules: Sequence[str] = (), dropout: float = 0.0) -> None:
        super().__init__()

        self.frozen_modules = frozen_modules

        #mindspore tokenizer
        self.ms_tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=".mindnlp", local_files_only=True)
        #mindspore config
        ms_configuration = CLIPTextConfig.from_pretrained(model_name, attention_dropout=dropout, cache_dir=".mindnlp", local_files_only=True)
        
        #mindspore model
        self.ms_model = CLIPTP.from_pretrained(model_name, config=ms_configuration, cache_dir=".mindnlp", local_files_only=True)

        # self._freeze_modules()

    def construct(self, text: List[List[str]]) -> Tensor:
        num_per_batch = [len(t) for t in text]
        
        assert max(num_per_batch) == min(num_per_batch), ('number of sequences not equal in batch')
        ms_text = list(itertools.chain(*text))

        ms_text = self.ms_tokenizer(text=ms_text, return_tensors='ms', padding=True)

        ms_txt_outputs = self.ms_model(**ms_text)


        # ms_txt_features 和 txt_features 对齐
        ms_txt_features = ms_txt_outputs.text_embeds
        ms_txt_features = ms_txt_features / ms_txt_features.norm(dim=-1, keepdim=True)  # mindspore.Tensor.norm(ord=None, dim=-1, keepdim=True) ord为None时默认为2-norm
        ms_txt_features = ms_txt_features.reshape(-1, num_per_batch[0],
                                      ms_txt_features.shape[-1])
        return ms_txt_features

    def _freeze_modules(self):
        if len(self.frozen_modules) == 0:
            # not freeze
            return
        if self.frozen_modules[0] == "all":
            self.model.eval()
            for _, module in self.model.named_modules():
                module.eval()
                for param in module.parameters():
                    param.requires_grad = False
            return
        for name, module in self.model.named_modules():
            for frozen_name in self.frozen_modules:
                if name.startswith(frozen_name):
                    module.eval()
                    for param in module.parameters():
                        param.requires_grad = False
                    break
