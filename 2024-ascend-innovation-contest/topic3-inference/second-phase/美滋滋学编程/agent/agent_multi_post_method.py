"""agent"""
import queue
import threading
import copy
import signal
import socket
import logging
import time
import psutil
import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed, wait
from multiprocessing import Process, shared_memory
try:
    import mindspore_lite as mslite
    IMPORT_LITE_FAILED = False
except ImportError:
    IMPORT_LITE_FAILED = True
from mindspore.common.tensor import Tensor
from mindspore_serving.serving_utils.err_code import AgentStatus
from mindspore_serving.models.post_sampling.topk import post_sampling, softmax_np
from mindspore_serving.sub_process.sub_process import listen_agents_after_startup
from mindspore_serving.config.config import ServingConfig
from mindspore_serving.models.build_inputs import build_inputs

import mindspore
from mindformers import AutoConfig, AutoModel
from tools.post_sampling_model import temperature_TopK, ArgmaxPost

pool = ThreadPoolExecutor(max_workers=20, thread_name_prefix='test_thread')

# 加载模型函数，并设置MindSpore的上下文。
def load_model_for_kbk(cfg: ServingConfig, rank_id: int, device_id: int):
    # 加载模型
    model_config = cfg.model_config
    # logging.info(f"load model {model_config.model_name} in rank {rank_id} start.")

    # 0: mindspore.GRAPH_MODE, 1: mindspore.PYNATIVE_MODE
    mindspore.set_context(mode=mindspore.GRAPH_MODE, device_id=device_id, device_target="Ascend")
    model = AutoModel.from_config(model_config.model_cfg_path)
    model.set_train(False)

    # logging.info(f"load model {model_config.model_name} in rank {rank_id} end.")
    return model

# 加载GE（Graph Engine）模型，执行预热操作并准备解码模型。
def load_model_for_ge(cfg: ServingConfig, rank_id: int, device_id: int):
    # 加载模型
    model_path = cfg.model_path
    model_config = cfg.model_config
    context = mslite.Context()

    warmup_func = build_inputs(cfg.warmup_inputs, module_type='warmup_inputs')
    context.ascend.device_id = device_id
    context.ascend.rank_id = rank_id
    context.ascend.provider = "ge"
    context.target = ["Ascend"]
    # 单模型
    if len(model_path.decode_model) == 0:
        model0 = mslite.Model()
        model0.build_from_file(model_path.prefill_model[0], mslite.ModelType.MINDIR, context, model_path.prefill_ini[0])
        model1 = None
        return model0, model1
    # rank_table_file放在config_file中
    all_models = [mslite.Model()]  # prefill
    # decode
    for _ in model_path.decode_ini:
        all_models.append(mslite.Model())
    model_group = mslite.ModelGroup(mslite.ModelGroupFlag.SHARE_WEIGHT)
    model_group.add_model(all_models)
    # logging.info('load prefill model ...')
    all_models[0].build_from_file(model_path.prefill_model[rank_id], mslite.ModelType.MINDIR, context,
                                  model_path.prefill_ini[0])
    # warm up prefill model
    # logging.info('warmup prefill model ...')
    prefill_batch_size = model_config.prefill_batch_size[0] if len(model_config.prefill_batch_size) > 0 else 1
    # 加入PA判断
    if model_config.page_attention:
        prefill_seq_length = model_config.seq_length[-1]
        inc_seq_len = cfg.pa_config.decode_seq_length
        prefill_inputs_list = warmup_func.get_warmup_inputs(seq_length=prefill_seq_length,
                                                            batch_size=prefill_batch_size,
                                                            full_model=True,
                                                            use_current_index=model_config.current_index,
                                                            # 这里需要考虑加入use_current_index测试
                                                            page_attention=model_config.page_attention,
                                                            zactivate_len=model_config.zactivate_len,
                                                            decode_seq_length=inc_seq_len,
                                                            block_size=cfg.pa_config.block_size)

    else:
        prefill_seq_length = model_config.seq_length[0] if len(model_config.seq_length) > 0 else 2048
        prefill_inputs_list = warmup_func.get_warmup_inputs(seq_length=prefill_seq_length,
                                                            batch_size=prefill_batch_size,
                                                            full_model=True,
                                                            use_current_index=model_config.current_index,
                                                            page_attention=model_config.page_attention,
                                                            zactivate_len=model_config.zactivate_len,
                                                            model_type=model_config.model_type)
    prefill_lite_inputs = [mslite.Tensor(np.ascontiguousarray(item)) for item in prefill_inputs_list]
    if rank_id == 0:
        for item in prefill_lite_inputs:
            pass
            # print("prefill item ", item.shape, item.dtype)
            # logging.debug(f"prefill item shape is {item.shape}, type is {item.dtype}")

    all_models[0].predict(prefill_lite_inputs)
    # logging.info('warmup prefill model finish')
    # logging.info('warmup decode model ...')
    if len(model_path.decode_ini) != len(model_config.zactivate_len):
        # logging.error('length of zactivate is not consistent with that of decode_ini')
        # padding invalid act_len list
        model_config.zactivate_len = [2 for _ in range(len(model_path.decode_ini))]
    for i in range(len(model_path.decode_ini)):
        act_len = model_config.zactivate_len[i]

        # print(f"starting warm up {act_len} decode model")
        # logging.info(f"starting warm up {act_len} decode model")
        if len(model_config.decode_batch_size) == 0:
            raise ValueError("length of model_config.decode_batch_size should at least be 1, but got 0")
        warm_batch_size = model_config.decode_batch_size[0] if len(model_config.decode_batch_size) > 0 else 1
        warm_seq_length = 1
        if model_config.page_attention:
            # print(f"decode warmup page attention")
            # warm_seq_length = model_config.seq_length[-1]
            inc_seq_len = cfg.pa_config.decode_seq_length
            decode_inputs_list = warmup_func.get_warmup_inputs(seq_length=warm_seq_length,
                                                               batch_size=warm_batch_size,
                                                               full_model=False,
                                                               use_current_index=model_config.current_index,
                                                               page_attention=model_config.page_attention,
                                                               zactivate_len=model_config.zactivate_len,
                                                               decode_seq_length=inc_seq_len,
                                                               block_size=cfg.pa_config.block_size)  # zactivate_len这里是否要加上zactivate_len
        else:
            # print(f"not decode warmup page attention")
            decode_inputs_list = warmup_func.get_warmup_inputs(seq_length=warm_seq_length,
                                                               batch_size=warm_batch_size,
                                                               full_model=False,
                                                               use_current_index=model_config.current_index,
                                                               valid_length=[act_len - 1],
                                                               page_attention=model_config.page_attention,
                                                               zactivate_len=model_config.zactivate_len,
                                                               model_type=model_config.model_type)

        decode_lite_inputs = [mslite.Tensor(np.ascontiguousarray(item)) for item in decode_inputs_list]
        # if rank_id == 0:
        #     for item in decode_lite_inputs:
        #         print("decode item ", item.shape, item.dtype, flush=True)
        #         logging.debug(f"decode item shape is {item.shape}, type is {item.dtype}")
        #         print(item, flush=True)
        # logging.debug(f"==================before build_from_file")
        all_models[i + 1].build_from_file(model_path.decode_model[rank_id], mslite.ModelType.MINDIR, context,
                                          model_path.decode_ini[i])
        # logging.debug(f"=====================after build_from_file")
        # if rank_id == 0:
        #     model_in = all_models[i + 1].get_inputs()
        #     for m_in in model_in:
        #         print(m_in.shape, m_in.dtype, flush=True)
        all_models[i + 1].predict(decode_lite_inputs)
        # logging.debug(f"=====================finish warm up")
        # print(f"finish warm up {act_len} decode model")
    # logging.debug(f"====================warmup all decode models finish'")
    # print('warmup all decode models finish', flush=True)
    # print(f"load model {model_path.prefill_model[rank_id]} and {model_path.decode_model[rank_id]} in rank {rank_id} "
        #   f"successful", flush=True)
    return all_models[0], all_models[1:]

# 加载后处理模型函数
def load_post_model(model_path, config_file, rank_id, device_id):
    context = mslite.Context()
    # print('device_id: ', device_id)
    # print('rank_id: ', rank_id)
    context.ascend.device_id = device_id
    context.ascend.rank_id = rank_id
    context.ascend.provider = "ge"
    context.target = ["Ascend"]
    # print('post sampling config: ', config_file)
    model = mslite.Model()
    # print('post model path:', model_path)
    if not os.path.exists(model_path):
        # logging.error(f"load post-sampling model_path {model_path} not exists.")
        raise ValueError(f"load post-sampling model_path  {model_path} not exists.")

    if not os.path.exists(config_file):
        # logging.error(f"load post-sampling post_model_ini {config_file} not exists.")
        raise ValueError(f"load post-sampling post_model_ini {config_file} not exists.")

    model.build_from_file(model_path, mslite.ModelType.MINDIR, context, config_file)
    # logging.debug(f"load post-sampling model  {model_path} successful")
    return model

# 解码参数类
class DecodeParams:
    def __init__(self,
                 do_sample: bool = True,
                 top_k: int = 1,
                 top_p: float = 1.0,
                 temperature: float = 1.0,
                 repetition_penalty: float = 1.0,
                 decode_index: int = -1,
                 current_index: int = 0,
                 valid_length: int = 0,
                 init_reset: bool = False,
                 ge_token: int = 0
                 ):
        self.do_sample = do_sample
        self.top_k = top_k
        self.top_p = top_p
        self.temperature = temperature
        self.repetition_penalty = repetition_penalty
        self.decode_index = decode_index
        self.current_index = current_index
        self.valid_length = valid_length
        self.init_reset = init_reset
        self.ge_token = ge_token


"""
work_agent.proto实现, 供worker调用
"""

# 用于加载和运行模型的逻辑，包括预处理、推理和后处理步骤。
class WorkAgent:
    def __init__(self, rank_id, cfg: ServingConfig):
        self.rank_id = rank_id
        model_path = cfg.model_path
        serving_config = cfg.serving_config
        device_id = rank_id + serving_config.start_device_id

        if cfg.model_config.backend == "ge":
            self.prefill, self.decode = load_model_for_ge(cfg, rank_id, device_id)
            self.argmax_model = load_post_model(model_path.argmax_model,
                                                model_path.post_model_ini,
                                                rank_id,
                                                device_id)
            self.topk_model = load_post_model(model_path.topk_model,
                                              model_path.post_model_ini,
                                              rank_id,
                                              device_id)
        else:
            self.mindspore_model = load_model_for_kbk(cfg, rank_id, device_id)
            self.argmax_model = ArgmaxPost()
            self.topk_model = temperature_TopK()

        self.shm_names = []
        self.init_reset = None
        self.current_index = None
        self.valid_length = None
        self.tensor_shape = None
        self.pre_input_ids = None
        self.is_prefill = True
        self.target = None
        self.post_mode_list = None
        self.input_length = None
        self.targets = []
        self.kbk_targets = None
        self.decode_params_map = {}
        self.status = AgentStatus.unconnected
        self.current_batch_size = None
        self.config = cfg
        self.basic_input_func = build_inputs(cfg.basic_inputs, module_type="basic_inputs")
        self.extra_input_func = build_inputs(cfg.extra_inputs, module_type="extra_inputs")

    def warmup_and_multiPrefill(self,cfg:ServingConfig):
        batchsize = cfg['model_config']['prefill_batch_size'][0]
        seq_len = cfg['model_config']['seq_length'][0]
        input_ids = np.array([
        [1, 2, 3, 4, 5, 6, 7, 8] + [0] * (seq_len - 8)
        ] * batchsize, dtype=np.int64)
        valid_length_each_example = np.array([8] * batchsize, dtype=np.int64)
        current_index = np.array([7] * batchsize, dtype=np.int64)
        block_tables = np.array([
        [0] + [-1] * (256 - 1)
        for i in range(batchsize)
        ], dtype=np.int32)
        slot_mapping = np.array(list(range(10)) + [-1] * (seq_len*batchsize - 10), dtype=np.int32)
        model_kwargs = {"current_index": current_index}
        self.mindspore_model.is_first_iteration = True
        # self.mindspore_model.config.batch_size=4
        res, current_index = self.mindspore_model.forward(input_ids=input_ids,
                                                valid_length_each_example=valid_length_each_example,
                                                generation_config=self.mindspore_model.config,
                                                block_tables=block_tables,
                                                slot_mapping=slot_mapping,
                                                prefill=True,
                                                **model_kwargs)
        # print("over")

    def _post_sampling_argmax_npu(self, outputs_np) -> np.ndarray:
        """
        Args:
           outputs_np: np.ndarray or ms.Tensor, (bs, 1, vocab_size)
        """
        post_inputs = self.argmax_model.get_inputs()
        if isinstance(outputs_np, np.ndarray):
            post_inputs[0].shape = outputs_np.shape
            post_inputs[0].set_data_from_numpy(outputs_np)
        else:
            post_inputs[0].shape = outputs_np.shape
            post_inputs[0] = outputs_np
        post_sampling_out = self.argmax_model.predict(post_inputs)
        return post_sampling_out[0].get_data_to_numpy().astype(np.int32)

    # 用于执行后处理（取最大概率的预测）操作。
    @staticmethod
    def _post_sampling_argmax_host(outputs) -> np.ndarray:
        if isinstance(outputs, mslite.Tensor):
            outputs = outputs.get_data_to_numpy()
        outputs.reshape((outputs.shape[0], outputs.shape[-1]))
        argmax_out = np.argmax(outputs, axis=-1)
        return np.array([argmax_out]).astype(np.int32)[0]

    @staticmethod
    def do_sample(decode_params, p_args, outs, targets, index, candidate_token_num: int = 1):
        """
        Args:
           decode_params: decode parameters for current client request
           p_args: numpy.ndarray, index
           outs: numpy.ndarray, probs
           targets: batch targets after sampling
           index: the batch index of current request
           candidate_token_num: default top_p_num
        """
        topp = decode_params.top_p
        topk = decode_params.top_k
        if topk > 100:
            topk = 100
            # logging.error('top k out of range [1,100]')
        outs = outs[:topk]
        if topp < 1.0:
            outs_ = np.cumsum(softmax_np(outs), axis=-1)
            top_p_num = sum(outs_ < topp)
            if top_p_num == 0:
                top_p_num = candidate_token_num
            outs = outs[:top_p_num]
            p_args = p_args[:top_p_num]

        p = softmax_np(outs)
        target_index = np.random.choice(len(p), p=p)
        targets[index] = p_args[target_index]

    def _post_sampling_topk_npu(self, outputs_np, decode_index, prefill=True) -> np.ndarray:
        """
        Args:
           outputs_np: np.ndarray or ms.Tensor, (bs, 1, vocab_size)
        """
        decode_params = self.decode_params_map[int(decode_index[0])]
        self.targets.clear()
        tempreture_ = np.array([decode_params.temperature], dtype=np.float32)

        post_inputs = self.topk_model.get_inputs()

        if isinstance(outputs_np, np.ndarray):
            post_inputs[0].shape = outputs_np.shape
            post_inputs[0].set_data_from_numpy(outputs_np)

        else:
            post_inputs[0].shape = outputs_np.shape
            post_inputs[0] = outputs_np

        post_inputs[1].shape = tempreture_.shape
        post_inputs[1].set_data_from_numpy(tempreture_)

        post_sampling_out = self.topk_model.predict(post_inputs)
        outs = post_sampling_out[0].get_data_to_numpy().astype(np.float16)
        p_args = post_sampling_out[1].get_data_to_numpy()
        thread_num = self.current_batch_size
        targets = np.zeros((thread_num,), np.int32)
        all_task = [pool.submit(self.do_sample, self.decode_params_map[decode_index[i]], p_args[i], outs[i], targets, i)
                    for i in range(thread_num)]
        wait(all_task)
        return targets

    def _post_sampling_topk_kbk(self, outputs_np, decode_index) -> np.ndarray:
        """
        Args:
           outputs_np: np.ndarray or ms.Tensor, (bs, 1, vocab_size)
        """
        decode_params = self.decode_params_map[int(decode_index[0])]
        self.targets.clear()
        tempreture_ = np.array([decode_params.temperature], dtype=np.float32)
        tempreture_t = Tensor(tempreture_, mindspore.float32)
        post_sampling_out = self.topk_model(outputs_np, tempreture_t)
        outs = post_sampling_out[0].asnumpy().astype(np.float16)
        p_args = post_sampling_out[1].asnumpy()
        thread_num = self.current_batch_size
        targets = np.zeros((thread_num,), np.int32)
        all_task = [pool.submit(self.do_sample, self.decode_params_map[decode_index[i]], p_args[i], outs[i], targets, i)
                    for i in range(thread_num)]
        wait(all_task)
        return targets

    def _get_seq_length(self, input_ids, is_prefill):
        max_length = 0
        if not is_prefill:
            if self.config.model_config.page_attention:
                return self.config.pa_config.decode_seq_length
        for item in input_ids:
            if isinstance(item, list):
                max_length = max(max_length, len(item))
            else:
                max_length = max(max_length, 1)
        if self.config.model_config.seq_type == 'dyn':
            seq_length = max_length
        elif len(self.config.model_config.seq_length) > 1:
            seq_length = self._get_seq_length_dynmic_dinning(self.config.model_config.seq_length, max_length)
        else:
            if len(self.config.model_config.seq_length) == 0 and self.config.model_config.seq_type != 'dyn':
                # logging.error('seq length is None ! using default 2048')
                seq_length = 2048
            else:
                seq_length = self.config.model_config.seq_length[0]
        return seq_length

    @staticmethod
    def _get_seq_length_dynmic_dinning(seq_list, seq_length):
        for data in seq_list:
            if seq_length < data:
                return data
        return seq_list[-1]

    @staticmethod
    def _padding(origin_inputs, seq_length, default_padding_values):
        pad_ids = list()
        for item in origin_inputs:
            pad_length = seq_length - len(item)
            # if pad_length < 0:
            #     logging.error('input sequence length is over max in serving system!')
            pad_item = np.pad(item, (0, pad_length), 'constant', constant_values=default_padding_values)
            pad_ids.append(pad_item)
        return np.array(pad_ids)

    def _post_sampling_topk_host(self, outputs, decode_index, prefill):
        """
         topk top-p in cpu, time-cost function,
        """
        if isinstance(outputs, mslite.Tensor):
            outputs = outputs.get_data_to_numpy()
        outputs = np.reshape(outputs, (outputs.shape[0], outputs.shape[-1]))
        thread_num = self.current_batch_size
        targets = np.zeros((thread_num,), np.int32)
        all_task = [pool.submit(post_sampling, np.array(item), self.decode_params_map[decode_index[i]], targets, i)
                    for i, item in enumerate(outputs)]
        wait(all_task)
        return targets

    def multi_thread_post_sampling(self, outputs_np, outputs_shm, decode_index_np, bs=1):

        self.targets.clear()
        all_task = [pool.submit(self.do_post_sampling, outputs_np[i], outputs_shm,
                                decode_index_np[i], i) for i in range(bs)]

        for x in as_completed(all_task):
            res = x.result()
            self.targets.append(res)
        return self.targets

    def get_consistent_batch(self, decode_index):
        not_do_sample_list = []
        do_sample_list = []
        for index in decode_index:
            do_sample_index = self.decode_params_map[index].do_sample
            if do_sample_index is True:
                do_sample_list.append(index)
            else:
                not_do_sample_list.append(index)
        if len(do_sample_list) >= 1 and len(not_do_sample_list) >= 1:
            for item in not_do_sample_list:
                self.decode_params_map[item].top_k = 1
            do_sample = True
        else:
            do_sample = self.decode_params_map[decode_index[0]].do_sample
        return do_sample

    def do_post_sampling(self, outputs_np, outputs_shm, output_logprob_shm, decode_index, prefill=True):
        #logging.info("do_post_sampling outputs_np type is {}, value is {}".format(outputs_np.dtype, outputs_np))
        do_sample = self.get_consistent_batch(decode_index)
        if self.config.model_config.backend == "ge":
            if self.config.serving_config.enable_host_post_sampling:
                if not do_sample:
                    target = self._post_sampling_argmax_host(outputs_np)
                    target.reshape((self.current_batch_size,))
                    target = np.squeeze(target, axis=1)
                else:
                    target = self._post_sampling_topk_host(outputs_np, decode_index, prefill)
            else:
                if not do_sample:
                    target = self._post_sampling_argmax_npu(outputs_np)
                else:
                    target = self._post_sampling_topk_npu(outputs_np, decode_index, prefill)
            output_info = outputs_np.get_data_to_numpy()
        else:
            if not do_sample:
                self.targets.clear()
                target = self.argmax_model(outputs_np)
            else:
                target = self._post_sampling_topk_kbk(outputs_np, decode_index)
            if isinstance(target, Tensor):
                target = target.asnumpy()
            output_info = outputs_np.asnumpy()

        #logging.info("do_post_sampling target type is {}, value is {}".format(target.dtype, target))
        #logging.info("do_post_sampling output_info type is {}, value is {}".format(output_info.dtype, output_info))
        if self.rank_id == 0:
            if prefill:
                for index in decode_index:
                    tmp = np.ndarray((index + self.current_batch_size,), dtype=target.dtype, buffer=outputs_shm.buf)
                    tmp[index: index + self.current_batch_size] = target[:]

                    logprob_list = []
                    for idx, tag in enumerate(target):
                        logprob_list.append(output_info[idx][int(tag)])
                    tmp_logprob = np.ndarray((index + self.current_batch_size,), dtype=np.float64,
                                             buffer=output_logprob_shm.buf)
                    tmp_logprob[index: index + self.current_batch_size] = logprob_list[:]
                    self.targets[index: index + self.current_batch_size] = target[:]
            else:
                tmp = np.ndarray((self.current_batch_size,), dtype=target.dtype, buffer=outputs_shm.buf)
                tmp[:] = target[:]

                logprob_list = []
                for idx, tag in enumerate(target):
                    if len(output_info.shape) == 2:
                        logprob_list.append(output_info[idx][int(tag)])
                    else:
                        logprob_list.append(output_info[idx][0][int(tag)])
                tmp_logprob = np.ndarray((self.current_batch_size,), dtype=np.float64, buffer=output_logprob_shm.buf)
                tmp_logprob[:] = logprob_list[:]
                self.targets[:] = target[:]


    def do_post_sampling1(self, outputs, outputs_shm, output_logprob_shm, decode_index, prefill=True):
        outputs_np=outputs[0]
        indices=outputs[3][0]
        max_values=outputs[3][1]
        indices=indices.asnumpy().reshape(-1)
        max_values=max_values.asnumpy().reshape(-1)
        # logging.info("do_post_sampling outputs_np type is {}, value is {}".format(outputs_np.dtype, outputs_np))
        do_sample = self.get_consistent_batch(decode_index)
        if self.config.model_config.backend == "ge":
            if self.config.serving_config.enable_host_post_sampling:
                if not do_sample:
                    target = self._post_sampling_argmax_host(outputs_np)
                    target.reshape((self.current_batch_size,))
                    target = np.squeeze(target, axis=1)
                else:
                    target = self._post_sampling_topk_host(outputs_np, decode_index, prefill)
            else:
                if not do_sample:
                    target = self._post_sampling_argmax_npu(outputs_np)
                else:
                    target = self._post_sampling_topk_npu(outputs_np, decode_index, prefill)
            output_info = outputs_np.get_data_to_numpy()
        else:
            if not do_sample:
                self.targets.clear()
                target = self.argmax_model(outputs_np)
            else:
                target = self._post_sampling_topk_kbk(outputs_np, decode_index)
            if isinstance(target, Tensor):
                target = target.asnumpy()
            output_info = outputs_np.asnumpy()

        # logging.info("do_post_sampling target type is {}, value is {}".format(target.dtype, target))
        # logging.info("do_post_sampling output_info type is {}, value is {}".format(output_info.dtype, output_info))
        if self.rank_id == 0:
            if prefill:
                for index in decode_index:
                    tmp = np.ndarray((index + self.current_batch_size,), dtype=indices.dtype, buffer=outputs_shm.buf)
                    tmp[index: index + self.current_batch_size] = indices[:]

                    logprob_list = []
                    for idx, tag in enumerate(indices):
                        logprob_list.append(max_values[idx])

                    logprob_list123 = []
                    for idx, tag in enumerate(target):
                        logprob_list123.append(output_info[idx][int(tag)])
                    
                    # 比较 logprob_list 和 logprob_list123 是否逐个元素相等
                    differences = []
                    for i in range(len(logprob_list)):
                        if logprob_list[i] != logprob_list123[i]:
                            differences.append((i, logprob_list[i], logprob_list123[i]))

                    # 输出比较结果
                    if not differences:
                        print("logprob_list and logprob_list123 are element-wise equal.")
                    else:
                        print("logprob_list and logprob_list123 have differences:")
                        for diff in differences:
                            print("Difference at index {}: logprob_list = {}, logprob_list123 = {}".format(diff[0], diff[1], diff[2]))
                    
                    tmp_logprob = np.ndarray((index + self.current_batch_size,), dtype=np.float64,
                                             buffer=output_logprob_shm.buf)
                    tmp_logprob[index: index + self.current_batch_size] = logprob_list[:]
                    self.targets[index: index + self.current_batch_size] = indices[:]
            else:
                tmp = np.ndarray((self.current_batch_size,), dtype=indices.dtype, buffer=outputs_shm.buf)
                tmp[:] = indices[:]

                logprob_list = []
                for idx, tag in enumerate(indices):
                    logprob_list.append(max_values[idx])

                logprob_list123 = []
                for idx, tag in enumerate(target):
                    if len(output_info.shape) == 2:
                        logprob_list123.append(output_info[idx][int(tag)])
                    else:
                        logprob_list123.append(output_info[idx][0][int(tag)])

                # 比较 logprob_list 和 logprob_list123 是否逐个元素相等
                differences = []
                for i in range(len(logprob_list)):
                    if logprob_list[i] != logprob_list123[i]:
                        differences.append((i, logprob_list[i], logprob_list123[i]))

                # 输出比较结果
                if not differences:
                    print("111logprob_list and logprob_list123 are element-wise equal.")
                else:
                    print("logprob_list and logprob_list123 have differences:")
                    for diff in differences:
                        print("Difference at index {}: logprob_list = {}, logprob_list123 = {}".format(diff[0], diff[1], diff[2]))
                
                tmp_logprob = np.ndarray((self.current_batch_size,), dtype=np.float64, buffer=output_logprob_shm.buf)
                tmp_logprob[:] = logprob_list[:]
                self.targets[:] = indices[:]

    def handle_inference_results(self,outputs, outputs_shm, output_logprob_shm, decode_index, prefill=True):
        # self.targets.clear()
        indices=outputs[0]
        max_values=outputs[1]


        # 记录索引和对应的最大值日志
        # logging.info("handle_inference_results indices type is {}, value is {}".format(indices.dtype, indices))
        # logging.info("handle_inference_results max_values type is {}, value is {}".format(max_values.dtype, max_values))
        indices=indices.asnumpy().reshape(-1)
        max_values=max_values.asnumpy().reshape(-1)
        # 当 rank_id 为 0 时，将数据写入共享内存
        if self.rank_id == 0:
            if prefill:
                for index in decode_index:
                    tmp = np.ndarray((index + self.current_batch_size,), dtype=indices.dtype, buffer=outputs_shm.buf)
                    tmp[index: index + self.current_batch_size] = indices[:]


                    tmp_logprob = np.ndarray((index + self.current_batch_size,), dtype=np.float64,
                                             buffer=output_logprob_shm.buf)
                    tmp_logprob[index: index + self.current_batch_size] = max_values[:]

                # 预填充模式
                # for idx,index in enumerate(decode_index):
                #     # 将最大值索引写入共享内存
                #     tmp = np.ndarray((index + 1,), dtype=indices.dtype, buffer=outputs_shm.buf)
                #     tmp[index: index + 1] = indices[idx]

                #     # logprob_list = []
                #     # for idx, tag in enumerate(indices):
                #     #     logprob_list.append(max_values[idx])


                #     # 将对应的最大值写入共享内存
                #     tmp_logprob = np.ndarray((index + 1,), dtype=np.float64, buffer=output_logprob_shm.buf)
                #     tmp_logprob[index: index + 1] = max_values[idx]
                    
                #     # 更新 targets
                #     # self.targets[index: index + 1] = [indices[idx]]
            else:
                # 非预填充模式
                tmp = np.ndarray((self.current_batch_size,), dtype=indices.dtype, buffer=outputs_shm.buf)
                tmp[:] = indices[:]

                # logprob_list = []
                # for idx, tag in enumerate(indices):
                #     logprob_list.append(max_values[idx])


                # 将对应的最大值写入共享内存
                tmp_logprob = np.ndarray((self.current_batch_size,), dtype=np.float64, buffer=output_logprob_shm.buf)
                tmp_logprob[:] = max_values[:]
                
                # 更新 targets
                # self.targets[:] = indices[:]

    def model_choice_seq(self, act_len, decode_model_map):
        if len(decode_model_map) == 1:
            return decode_model_map[0]
        act_len_list = self.config.model_config.zactivate_len
        # if len(act_len_list) != len(decode_model_map):
        #     logging.error('act_len config is inconsistent with decode'
        #                   'model ini,please check them')
        model_index = act_len_list.index(act_len)
        # logging.debug('current act_len model is: {}'.format(act_len))
        return decode_model_map[model_index]

    def predict(self, shape_list=None, current_batch=None, batch_valid_flag=None):
        # 进入predict函数，首先将agent的状态设置为忙碌状态。
        self.status = AgentStatus.busy
        # 初始化一个空列表tmp_shms，用于存储共享内存对象。
        tmp_shms = []
        start_time = time.time()
        existing_shm0 = shared_memory.SharedMemory(name=self.shm_names[0])#创建一个与现有共享内存相关联的 SharedMemory 对象，允许对该共享内存进行读写操作。
        tmp_shms.append(existing_shm0)

        output_shm = shared_memory.SharedMemory(name=self.shm_names[5])
        tmp_shms.append(output_shm)

        output_logprob_shm = shared_memory.SharedMemory(name=self.shm_names[6])
        tmp_shms.append(output_logprob_shm)

        gen_params_id = 4
        gen_params_shm = shared_memory.SharedMemory(name=self.shm_names[gen_params_id])
        tmp_shms.append(gen_params_shm)
        # 如果当前状态为预填充
        if self.is_prefill:
            # 获取first_group数组中的当前索引current_index_，并将其压缩为一维数组current_index。
            first_group = np.ndarray((shape_list[0]), dtype=np.int32, buffer=existing_shm0.buf)
            current_index_ = first_group[:, shape_list[0][1] - 3: shape_list[0][1] - 2]#取出倒数第3个值，代表当前句子最后一个token位置（从0开始计数）
            current_index = np.squeeze(current_index_, axis=-1)

            # 获取first_group数组中的有效长度valid_length_（句子token长度），并根据配置将其转换为int64或int32类型的数组valid_length。
            valid_length_ = first_group[:, shape_list[0][1] - 1: shape_list[0][1]]
            if self.config.model_config.current_index or self.config.model_config.backend == "kbk":
                valid_length = np.squeeze(valid_length_, axis=-1).astype(np.int64)
            else:
                valid_length = np.squeeze(valid_length_, axis=-1).astype(np.int32)
            # 从first_group数组中获取输入IDinput_ids。
            input_ids = first_group[:, :shape_list[0][1] - 3]
            gen_params_id = 1  # 改为1，正向取值，原始shape_list只有两个值，现在多加了两个
            shape_params = shape_list[gen_params_id]
            # 将共享内存gen_params_shm中的数据转为gen_params数组。
            gen_params = np.ndarray(shape_params, dtype=np.float16, buffer=gen_params_shm.buf)
            # 从gen_params数组中提取生成参数
            do_sample_list = gen_params[:, 0].astype(np.bool_)
            top_p_list = gen_params[:, 1]
            top_k_list = gen_params[:, 2].astype(np.int32)
            temperature_list = gen_params[:, 3]
            repetition_penalty_list = gen_params[:, 4]
            decode_index_list = gen_params[:, 5].astype(np.int32)
            # 添加baichuanPA block_tables_shape slot_mapping_shape
            if self.config.model_config.page_attention:
                # logging.debug("prefill before page aatention get shape list")
                # print("model_name:", self.config.model_config.model_name)
                block_tables_shape = shape_list[2]  # 这里的shapeindex会不会变？？
                slot_mapping_shape = shape_list[3]
            # 遍历shape_list，获取额外输入的共享内存对象，将其添加到tmp_shms列表中，并将其数据转为extra_input数组。
            extra_input = []
            for i in range(1, len(shape_list) - 1):#i 1-2 两个
                existing_shm = shared_memory.SharedMemory(name=self.shm_names[i])
                tmp_shms.append(existing_shm)
                # To Do np.int64 ?
                extra_input.append(np.ndarray((shape_list[i]), dtype=np.int64, buffer=existing_shm.buf))

            if self.config.model_config.backend == "ge":# 跳过
                # pa or static model type don't need 'act_len' parameter
                if self.config.model_config.page_attention or (
                        self.config.model_config.model_name == 'wizard_coder' and self.config.model_config.model_type == "static"):
                    extra_input = []
                else:
                    extra_input = self.extra_input_func.get_extra_inputs(input_ids, current_index, None, True,
                                                                         valid_length,
                                                                         zactivate_len=self.config.model_config.zactivate_len)
            # 设置当前批次大小self.current_batch_size。
            self.current_batch_size = len(input_ids)
            init_reset = []
            decode_index = []
            for i in range(self.current_batch_size):
                # 遍历当前批次，创建DecodeParams实例，初始化解码参数，并更新decode_params_map。
                decode_params = DecodeParams(
                    do_sample=bool(do_sample_list[i]),
                    top_p=top_p_list[i],
                    top_k=int(top_k_list[i]),
                    temperature=temperature_list[i],
                    repetition_penalty=repetition_penalty_list[i],
                    decode_index=int(decode_index_list[i]),
                    current_index=int(current_index[i]),
                    valid_length=int(valid_length[i]),
                    init_reset=False
                )
                self.decode_params_map[decode_params.decode_index] = decode_params
                # 将init_reset和decode_index转换为NumPy数组。
                init_reset.append(decode_params.init_reset)
                decode_index.append(decode_params.decode_index)
            init_reset = np.array(init_reset, dtype=np.bool_)
            decode_index_np = np.array(decode_index, dtype=np.int64)
        else:
            # keep decode map size equal to current batch size
            # extend
            # 用于存储当前索引、有效长度、初始化重置标志和解码索引。
            current_index = []
            valid_length = []
            init_reset = []
            decode_index = []
            self.current_batch_size = current_batch
            current_batch_size = self.current_batch_size
            # 检查当前批次大小是否与有效标志列表batch_valid_flag的长度相等。如果不相等，记录错误日志，并清空batch_valid_flag，然后将其重置为与当前批次大小相同的列表，所有值为1。
            if self.current_batch_size != len(batch_valid_flag):
                logging.error("batch size is not equal to the length of batch_valid_flag: batch size is {}, "
                              "batch_valid_flag is {}".format(self.current_batch_size, batch_valid_flag))
                batch_valid_flag.clear()
                batch_valid_flag = [1 for _ in range(self.current_batch_size)]
            # 获取上一次推理的decode_params_map的大小，即之前的批次大小。这里不是指上一次推理而是指decode_params_map中存放的推理请求
            before_batch_size = len(self.decode_params_map.keys())
            # 如果之前的批次大小小于当前批次大小，需要扩展decode_params_map
            if before_batch_size < current_batch_size:
                # 从共享内存中读取上一次的推理结果tokenid，类型为int32。
                input_ids = np.ndarray((before_batch_size,), dtype=np.int32, buffer=output_shm.buf)
                # 获取填充ID，即模型配置中的结束标记。
                pad_input_id = self.config.model_config.end_token
                # 计算需要扩展的长度。
                add_length = self.current_batch_size - before_batch_size
                # 创建一个长度为扩展长度的数组addition_input_ids，填充值为结束标记。
                addition_input_ids = np.array(add_length * [pad_input_id], dtype=np.int32)
                # 将addition_input_ids附加到input_ids中。
                input_ids = np.append(input_ids, addition_input_ids)
                # 获取目标批次大小。
                target_batch = self.current_batch_size
                # 获取最后一个解码参数的键值（用于填充）。
                pad_key = list(self.decode_params_map.keys())[-1]
                # padding_obj = self.decode_params_map[pad_key]
                # 遍历目标批次大小，如果当前索引不在decode_params_map中，进行填充操作
                for j in range(target_batch):
                    if j not in self.decode_params_map:
                        # 复制最后一个解码参数对象padding_obj。
                        padding_obj = copy.deepcopy(self.decode_params_map[pad_key])
                        padding_obj.current_index = 0
                        padding_obj.valid_length = 1
                        padding_obj.decode_index = j
                        self.decode_params_map[j] = padding_obj
            else:#如果之前的批次大小大于等于当前批次大小，需要收缩decode_params_map
                # pop
                while len(self.decode_params_map.keys()) > current_batch_size:
                    # 逐项弹出decode_params_map中的元素，直到其大小等于当前批次大小
                    self.decode_params_map.popitem()
                # 从共享内存中读取上一次的推理输出token ID，类型为int32。
                input_ids = np.ndarray((current_batch_size,), dtype=np.int32, buffer=output_shm.buf)
            # 对decode_params_map按键值进行排序。
            self.decode_params_map = dict(sorted(self.decode_params_map.items(), key=lambda x: x[0]))
            # 遍历decode_params_map中的每个键值对
            for key in self.decode_params_map.keys():
                # 获取解码参数对象
                decode_params = self.decode_params_map[key]
                decode_params.current_index = decode_params.current_index + 1
                decode_params.valid_length = decode_params.valid_length + 1
                decode_params.init_reset = True  # 修改原始代码bug
                # 根据batch_valid_flag的值，更新current_index和valid_length
                if batch_valid_flag[key] == 1:
                    current_index.append(decode_params.current_index)
                    valid_length.append(decode_params.valid_length)
                else:
                    current_index.append(0)
                    valid_length.append(1)
                # 将初始化重置标志和解码索引分别添加到init_reset和decode_index列表（就是按0123排列）中。
                init_reset.append(decode_params.init_reset)
                decode_index.append(decode_params.decode_index)

            if self.config.model_config.backend == "ge":
                # pa or static model type don't need 'act_len' parameter
                if self.config.model_config.page_attention or (
                        self.config.model_config.model_name == 'wizard_coder' and self.config.model_config.model_type == "static"):
                    extra_input = []
                else:
                    extra_input = self.extra_input_func.get_extra_inputs(input_ids, current_index, None, False,
                                                                         valid_length,
                                                                         zactivate_len=self.config.model_config.zactivate_len)
            # 将current_index转换为NumPy int32数组
            current_index = np.array(current_index, dtype=np.int32)

            if self.config.model_config.current_index or self.config.model_config.backend == "kbk":
                valid_length = np.array(valid_length, dtype=np.int64)
            else:
                valid_length = np.array(valid_length, dtype=np.int32)
            init_reset = np.array(init_reset, dtype=np.bool_)
            decode_index_np = np.array(decode_index, dtype=np.int64)
            # 调整input_ids的形状为二维数组。
            input_ids = input_ids.reshape((-1, 1))
            # 加入PA特性，获取block_tables_shape（bs，256）和slot_mapping_shape（bs）。
            if self.config.model_config.page_attention:
                block_tables_shape = shape_list[0]
                slot_mapping_shape = shape_list[1]

        # 初始化block_tables_np和slot_mapping_np为None。
        block_tables_np = None
        slot_mapping_np = None
        # 如果启用了PA，获取共享内存对象block_tables_shm和slot_mapping_shm。
        # 将共享内存中的数据转为block_tables_np和slot_mapping_np数组，并记录调试日志。
        if self.config.model_config.page_attention:
            block_tables_shm = shared_memory.SharedMemory(name=self.shm_names[7])  # 这里的共享内存index要改
            slot_mapping_shm = shared_memory.SharedMemory(name=self.shm_names[8])
            block_tables_np = np.ndarray((block_tables_shape), dtype=np.int32, buffer=block_tables_shm.buf)
            slot_mapping_np = np.ndarray((slot_mapping_shape), dtype=np.int32, buffer=slot_mapping_shm.buf)


        if self.config.model_config.backend == "ge":
            if self.config.model_config.page_attention:
                # logging.debug("page attention beofore predict")
                if self.is_prefill:
                    tmp_in = [input_ids, valid_length, slot_mapping_np]
                else:
                    tmp_in = [input_ids, valid_length, block_tables_np, slot_mapping_np]
            else:
                tmp_in = self.basic_input_func.get_inputs(input_ids, current_index, init_reset, valid_length,
                                                          self.config.model_config.current_index, decode_index_np,
                                                          self.config.model_config.model_type)
                if len(extra_input) > 0:
                    tmp_in.extend(extra_input)

            # for tmp in tmp_in:
            #     pass
                # logging.debug("item shape is {}, dtype is {}".format(tmp.shape, tmp.dtype))
                # logging.debug("item is {}".format(tmp))

            outputs = self.predict_for_ge(extra_input, start_time, tmp_in)
        else:#kbk
            # 获取序列长度seq_length。其实直接返回4096就可以
            seq_length = self._get_seq_length(input_ids, False)
            # init kbk_targets, shape(current_batch, seq_length), default value: self.config.model_config.pad_token_id
            # 初始化kbk_targets，形状为(decode_batch_size, seq_length)，默认值为pad_token_id。
            if self.kbk_targets is None:
                decode_batch_size = self.config.model_config.decode_batch_size[0]
                # logging.debug(
                #     f"predict decode_batch_size value is {decode_batch_size}, seq_length value is {seq_length}")
                self.kbk_targets = np.full((decode_batch_size, seq_length), self.config.model_config.pad_token_id)

            # logging.debug(
            #     f"predict self.kbk_targets value is {self.kbk_targets}, shape is {self.kbk_targets.shape}, dtype is {self.kbk_targets.dtype}")

            # decode [[4]]
            # logging.debug(
            #     f"predict input_ids before update value is {input_ids}, shape is {input_ids.shape}, dtype is {input_ids.dtype}")

            # decode 时，将原始decoding的tokenid的batchsize，1的shape，变成了batchsize，hidden_size的shape，每行的第一个是原来的tokenid，后面全部填0
            if input_ids.shape[1] == 1:#这里将input_ids的type由int32变成float64了
                input_ids = np.concatenate((input_ids, np.zeros((input_ids.shape[0], seq_length - 1))), axis=1)
                # logging.debug(
                #     f"predict input_ids before update value is {input_ids}, shape is {input_ids.shape}, dtype is {input_ids.dtype}")

            # 遍历decode_index
            for idx, index in enumerate(decode_index):
                # index = int(decode_index[0])#index和idx
                # logging.debug(f"===============predict self.kbk_targets len:{len(self.kbk_targets)}")
                # logging.debug(f"===============predict decode index:{index}")
                if self.is_prefill:
                    # logging.debug("kbk predict prefill start.")
                    self.kbk_targets[index] = input_ids[idx]
                else:
                    # logging.debug("kbk predict decode start.")
                    current_index_value = int(current_index[idx])
                    # logging.debug(
                        # f"predict decode before curr_input_ids==============current_index_value：{current_index_value}")
                    self.kbk_targets[index][current_index_value:current_index_value + 1] = input_ids[idx][:1]
                    input_ids[idx] = self.kbk_targets[index]#从decoding input_ids的位置中取出那个单独的tokenid，加入到kbk_target中，再将这句目前生成的全部token移动到input_ids中
                    # logging.debug(
                        # f"predict input_ids after update value is {input_ids}, shape is {input_ids.shape}, dtype is {input_ids.dtype}")
            # 进行推理，获取输出
            # logging.debug(f"predict current_index value is {current_index}")
            # logging.debug(
                # f"predict valid_length value is {valid_length}, shape is {valid_length.shape}, dtype is {valid_length.dtype}")
            logging.info('agent pre-process time is {}'.format((time.time() - start_time) * 1000))
            outputs = self.predict_for_kbk(current_index, input_ids, valid_length, block_tables_np, slot_mapping_np)
        # 记录后处理开始时间
        post_time = time.time()
        if self.rank_id == 0:
            multi_thread_time = time.time()
            if self.is_prefill:
                # self.do_post_sampling(outputs, output_shm, output_logprob_shm, decode_index_np, prefill=True)
                self.handle_inference_results(outputs, output_shm, output_logprob_shm, decode_index_np, prefill=True)
            else:
                # self.do_post_sampling(outputs, output_shm, output_logprob_shm, decode_index_np, prefill=False)
                self.handle_inference_results(outputs, output_shm, output_logprob_shm, decode_index_np, prefill=False)
                # logging.info('multi_thread_post_sampling time is {}'.format((time.time() - multi_thread_time) * 1000))
            # logging.info('target is  {}'.format((self.targets)))
        logging.info('post_time is {}'.format((time.time() - post_time) * 1000))
        logging.info('npu_total_time is {}'.format((time.time() - start_time) * 1000))
        # 将agent状态设置为非忙碌状态。
        self.status &= ~AgentStatus.busy
        # 返回目标结果和共享内存对象列表。
        return self.targets, tmp_shms

    def predict_for_ge(self, extra_input, start_time, tmp_in):
        # 调用ms lite进行推理
        if len(extra_input) > 0:
            model = self.prefill if self.is_prefill else self.model_choice_seq(len(extra_input[0]), self.decode)
        else:
            model = self.prefill if self.is_prefill else self.decode[0]
        lite_inputs = [mslite.Tensor(np.ascontiguousarray(item)) for item in tmp_in]
        # logging.info('agent pre-process time is {}'.format((time.time() - start_time) * 1000))
        predict_time = time.time()
        if self.config.model_config.model_name == 'wizard_coder' and self.config.model_config.model_type == "static":
            if self.is_prefill:
                init_reset_ms_tensor = mslite.Tensor(np.array([False], np.bool_))
            else:
                init_reset_ms_tensor = mslite.Tensor(np.array([True], np.bool_))
            outputs_list = model.predict((lite_inputs[0], lite_inputs[1], init_reset_ms_tensor, lite_inputs[2]))
        else:
            outputs_list = model.predict(lite_inputs)
        # logging.debug(
        #     "outputs tensor after model predict type {}, value is {}".format(outputs_list[0].dtype, outputs_list[0]))
        # logging.info('predict time is {}'.format((time.time() - predict_time) * 1000))
        outputs = outputs_list[0]
        return outputs

    def predict_for_kbk(self, current_index, input_ids, valid_length, block_tables, slot_mapping):
        # 封装调用模型参数
        model_kwargs = {"current_index": current_index}
        # model_inputs = self.mindspore_model.prepare_inputs_for_generation(input_ids, **model_kwargs)
        # logging.debug(f"predict model_inputs value is {model_inputs}")
        # 调用mindformers进行推理
        predict_time = time.time()
        if self.mindspore_model.config.use_past:
            # logging.debug(f"predict before pa predict_for_kbk.")
            if self.is_prefill:
                self.mindspore_model.is_first_iteration = True
            res, current_index = self.mindspore_model.forward(input_ids=input_ids,
                                                    valid_length_each_example=valid_length,
                                                    generation_config=self.mindspore_model.config,
                                                    block_tables=block_tables,
                                                    slot_mapping=slot_mapping,
                                                    prefill=self.is_prefill,
                                                    **model_kwargs)
        # else:
            # res = self.mindspore_model(**model_inputs)
        
        if self.is_prefill:
            logging.info('prefillpredict time is {}'.format((time.time() - predict_time) * 1000))
            bs=len(input_ids)
            logging.info('prefill batch is {}'.format(bs))
            return res[0]
            # out_idx=res[0].asnumpy()[:bs]
            # out_log=res[1].asnumpy()[:bs]
            # res1=Tensor(out_idx)
            # res2=Tensor(out_log)
            # return (res1,res2)
        else:
            logging.info('decodepredict time is {}'.format((time.time() - predict_time) * 1000))
            # logging.info("mindspore_model res : %s;", res)
            # outputs = res[0] if isinstance(res, tuple) else res
            return res[0]





def start_agent_socket_server(i, cfg: ServingConfig, startup_queue):
    logging.basicConfig(level=logging.ERROR,
                        filename=f"./output/agent_{i}.log",
                        filemode='w',
                        format=
                        '%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s')
    """启动agent进程, 由_agent_process进行调用, 创建agent进程"""
    # if IMPORT_LITE_FAILED:
    #     logging.warning("import mindspore_lite failed, using kbk backend.")
    work_agent = WorkAgent(i, cfg)
    work_agent.warmup_and_multiPrefill(cfg)

    agent_ports = cfg.serving_config.agent_ports
    agent_ip = cfg.serving_config.agent_ip
    agent_address = (agent_ip, agent_ports[i])
    # print(agent_address)

    parent_process = psutil.Process(os.getppid())
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(agent_address)
    server.listen(5)

    startup_queue.put(i)

    # 绑定method
    # print("start agent socket server in rank{}".format(i), flush=True)

    while True:
        if not parent_process.is_running():
            # logging.warning(f"detect parent pid={parent_process.pid} has exited, child begin to exit")
            server.close()
            return

        conn, client_addr = server.accept()
        # todo workagent = WorkAgent(config)
        while True:
            if not parent_process.is_running():
                # logging.warning(
                #     f"detect parent pid={parent_process.pid} has exited, child begin to exit")
                server.close()
                return
            try:
                data = conn.recv(4096)
                if not data:
                    break
                data = data.decode()
                # logging.debug(f"data received is {data}")
                # worker 和 agent建联
                if data.startswith('#'):
                    if work_agent.status & AgentStatus.unconnected == AgentStatus.unconnected:
                        data = data[1:]
                        work_agent.shm_names = data.split(",")#第一次建立连接时，获取到共享内存的地址
                        work_agent.status = AgentStatus.connected
                        # print("connect succes")
                        conn.sendall("succes".encode())
                    else:
                        # print("connect failed")
                        conn.sendall("failed".encode())
                elif data.startswith('*'):
                    # 全量推理
                    work_agent.is_prefill = True
                    data = data[1:]
                    shape_strs = data.split(",")#将收到的数据按照逗号分开data '1 4099,1 6,1 256,4096'
                    input_shapes = []
                    for shape_str in shape_strs:
                        shape = list(map(int, shape_str.split(" ")))
                        input_shapes.append(shape)
                    _, _ = work_agent.predict(shape_list=input_shapes)# input_shapes[[1, 4099], [1, 6], [1, 256], [4096]]
                    if i == 0:
                        conn.sendall("1".encode())
                elif data.startswith('a'):
                    # 增量推理 data'a_4_1 0 0 0_4 256_4' 按照下划线进行分割的a为增量
                    decode_data = data.split('_')
                    # 取出当前decode的batchsize大小，如果启动PA，就从倒数第四个位置取，如果不开启则从倒数第二个位置取
                    current_batch_dyn = int(decode_data[-4]) if cfg.model_config.page_attention else int(
                        decode_data[-2])
                    batch_valid_flag = []
                    batch_valid = decode_data[-3] if cfg.model_config.page_attention else decode_data[-1]
                    for ele in batch_valid.split(" "):
                        batch_valid_flag.append(int(ele))# 获得有效的batch flag列表
                    # 增加 block_tables和slot_mapping 的 shape
                    input_shapes = []
                    if cfg.model_config.page_attention:
                        for shape_str in [decode_data[-2], decode_data[-1]]:
                            shape = list(map(int, shape_str.split(" ")))
                            input_shapes.append(shape)
                    work_agent.is_prefill = False # input_shapes[[batchsize, 256], [batchsize]]
                    _, _ = work_agent.predict(current_batch=current_batch_dyn, batch_valid_flag=batch_valid_flag,
                                              shape_list=input_shapes)
                    if i == 0:
                        conn.sendall("1".encode())
                elif data.startswith('e'):
                    # worker退出获取agent状态，free状态下才允许退出
                    if work_agent.status & AgentStatus.busy == AgentStatus.busy:
                        print("busy")
                        conn.sendall("busy".encode())
                    else:
                        work_agent.status = AgentStatus.unconnected
                        print("free")
                        conn.sendall("free".encode())
                elif data.startswith('r'):
                    # reset agents status
                    work_agent.status = AgentStatus.unconnected
                    print("reset succes")
                    conn.sendall("succes".encode())
            except ConnectionResetError:
                break
            except RuntimeError:
                logging.error("predict failed, abandon prompts")
                conn.sendall("2".encode())
        conn.close()


def handler(sig_num, addition):
    os.killpg(os.getpgid(os.getpid()), signal.SIGKILL)


def startup_agents(config, startup_queue):
    signal.signal(signal.SIGTERM, handler)
    signal.signal(signal.SIGINT, handler)
    agent_ports = config.serving_config.agent_ports
    subprocess_list = []
    log_dir = os.path.join(os.getcwd(), "output")
    # print("------------", log_dir)
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    for i in range(len(agent_ports)):
        p = Process(target=start_agent_socket_server, args=(i, config, startup_queue))
        p.start()
        subprocess_list.append(p)
    listen_agents_after_startup(subprocess_list)
# def startup_agents(config, startup_queue):
#     # 设置信号处理器
#     signal.signal(signal.SIGTERM, handler)
#     signal.signal(signal.SIGINT, handler)

#     # 获取单一的端口
#     agent_port = config.serving_config.agent_ports[0]

#     # 创建日志目录
#     log_dir = os.path.join(os.getcwd(), "output")
#     if not os.path.exists(log_dir):
#         os.mkdir(log_dir)

#     # 直接启动代理，不使用多进程
#     start_agent_socket_server(0, config, startup_queue)