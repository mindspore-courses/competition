import time
from typing import List, Tuple, Deque
import logging

from queue import Queue
import copy
from mindspore_serving.serving_utils.constant import *
from mindspore_serving.serving_utils.entry import EntryMetaData, EntryStatus, EntryData
from mindspore_serving.config.config import ServingConfig
from mindspore_serving.schedule.cache_engine import ServingBlockMemPool, ServingCacheEngine
import heapq

class Schedule:
    """static batch strategy"""

    def __init__(self, config: ServingConfig):
        self.waiting_request_queue: List[EntryMetaData] = Deque([])#存储等待处理的推理请求队列。
        self.running_request_list: List[EntryMetaData] = []# 存储当前正在执行的请求列表。
        self.count_of_invalid_sample = 0# 记录当前批次中的无效请求数量。
        self.config = config
        self.batch_size = config.model_config.decode_batch_size[0]#当前批次的大小。
        self.eos_token = config.model_config.end_token
        self.batch_waiting_time = config.serving_config.prefill_batch_waiting_time#在预填充或解码时，批次的最大等待时间。
        self.decode_batch_waiting_time = config.serving_config.decode_batch_waiting_time
        self.batching_strategy = config.model_config.batching_strategy
        self.max_input_len = config.model_config.seq_length[-1] if len(config.model_config.seq_length) > 0 else 4096
        # batch中有效token的最大index, 初始化为-1
        self.max_valid_index = -1
        self.dyn_batch = config.model_config.decode_batch_size#动态批次大小列表，用于动态调整批次的大小。

    def get_dyn_batch(self):
        return self.batch_size

    def get_queue_len(self):
        return len(self.waiting_request_queue)

    def add_entrys(self, entry_meta_data: EntryMetaData):
        # 将一个新请求 entry_meta_data 添加到等待队列 waiting_request_queue 中，并将其状态设置为 WAITING。
        entry_meta_data.get_entry_data().set_status(EntryStatus.WAITING)
        heapq.heappush(self.waiting_request_queue, entry_meta_data)

    def _padding_batch_size(self):
        while len(self.running_request_list) < self.batch_size:
            entry_meta_data = copy.deepcopy(self.running_request_list[-1])
            entry_meta_data.entry_data.set_status(EntryStatus.PADDING_INVAILED)
            self.running_request_list.append(entry_meta_data)

    def _over_all_complete_entry(self):
        for index, _ in enumerate(self.running_request_list):
            self.running_request_list[index].get_entry_data().set_status(EntryStatus.FINISHED_STOPPED)

    def _padding_request_into_batching_list(self, index):
        if not self.waiting_request_queue:
            time.sleep(self.batch_waiting_time / float(len(self.running_request_list)))
            if not self.waiting_request_queue:
                entry_meta_data = copy.deepcopy(self.running_request_list[-1])

                if entry_meta_data.entry_data.get_prompt_len() + entry_meta_data.entry_data.get_max_token_len() >= self.max_input_len:
                    entry_meta_data.get_entry_data().set_status(EntryStatus.INPUT_OUTOFRANGE)
                else:
                    entry_meta_data.get_entry_data().set_status(EntryStatus.PADDING_INVAILED)

                entry_meta_data.get_entry_data().set_decode_index(index)
                self.running_request_list.append(entry_meta_data)
                logging.debug(f'waiting and add invalid request in batch init, batch size index is {index}')
            else:
                data = heapq.heappop(self.waiting_request_queue)
                if data.entry_data.get_prompt_len() + data.entry_data.get_max_token_len() >= self.max_input_len:
                    data.get_entry_data().set_status(EntryStatus.INPUT_OUTOFRANGE)
                else:
                    data.get_entry_data().set_status(EntryStatus.RUNNING)

                data.get_entry_data().set_decode_index(index)
                self.running_request_list.append(data)
                logging.debug(f'add new valid request in batch, batch size index is {index}')
        else:
            data = heapq.heappop(self.waiting_request_queue)
            logging.debug('get_nowait2')

            if data.entry_data.get_prompt_len() + data.entry_data.get_max_token_len() >= self.max_input_len:
                data.get_entry_data().set_status(EntryStatus.INPUT_OUTOFRANGE)
            else:
                data.get_entry_data().set_status(EntryStatus.RUNNING)

            data.get_entry_data().set_decode_index(index)
            self.running_request_list.append(data)
            logging.debug(f'add new valid request in batch, batch size index is {index}')

    def _get_next_batch(self):
        # 从等待队列中获取一批新的请求，填充到 running_request_list 中。
        # 如果队列中没有足够的有效请求，则用无效请求进行填充。
        self.running_request_list.clear()
        count = 0
        # no request in schedule queue, return
        if not self.waiting_request_queue:
            return
        # add request into batching list
        while self.waiting_request_queue:
            if count >= self.batch_size:
                break
            data = heapq.heappop(self.waiting_request_queue)

            if data.entry_data.get_prompt_len() + data.entry_data.get_max_token_len() >= self.max_input_len:
                data.get_entry_data().set_status(EntryStatus.INPUT_OUTOFRANGE)
            else:
                data.get_entry_data().set_status(EntryStatus.RUNNING)

            data.get_entry_data().set_decode_index(count)
            self.running_request_list.append(data)
            logging.debug(f'add new valid request in batch, batch size index is {count}')
            count += 1
        # if batching list not full, add invalid padding request into batching list
        # 如果批次大小不足，使用无效请求填充（通过 _padding_request_into_batching_list 方法）。
        if len(self.running_request_list) < self.batch_size + 1:
            for index in range(len(self.running_request_list), self.batch_size):
                self._padding_request_into_batching_list(index)

    def _all_samples_in_batch_is_over(self) -> bool:
        res = True
        for _, data in enumerate(self.running_request_list):
            if data.get_entry_data().get_status() == EntryStatus.RUNNING:
                res = False
        return res

    def checkout_entry(self) -> List[bool]:
        """
            检查批次中哪些请求已经结束，可以被移除。
          request in FINISHED_LENGTH_CAPPED, FINISHED_STOPPED, PADDING_INVAILED status can be cut out
        """
        checkout_list = []
        for index, data in enumerate(self.running_request_list):
            check_ = False
            # max_length, cut out finished request in batch
            if data.get_entry_data().get_status() == EntryStatus.FINISHED_LENGTH_CAPPED:
                check_ = True
            # eos, cut out finished request in batch
            elif data.get_entry_data().get_status() == EntryStatus.FINISHED_STOPPED:
                check_ = True
            elif data.get_entry_data().get_status() == EntryStatus.PADDING_INVAILED:
                check_ = True
            checkout_list.append(check_)
        return checkout_list

    def _padding_new_prompt_to_batch(self, index):
        # 当批次大小不足时，将无效的请求（或等待队列中的请求）填充到批次中。
        # queue is empty, no new request in schedule queue
        # 如果等待队列为空，程序会等待一段时间。
        if not self.waiting_request_queue:
            # waiting
            time.sleep(self.batch_waiting_time / float(len(self.running_request_list)))
            # no new request, continue finished valid decode
            if not self.waiting_request_queue:
                # logging.debug('waiting and no new request, continue finished valid decode')
                return
            # new requestes in queue
            else:
                # logging.debug('add a new request into batching list')
                data = heapq.heappop(self.waiting_request_queue)
                # logging.debug('get_nowait3')
                if data.entry_data.get_prompt_len() + data.entry_data.get_max_token_len() >= self.max_input_len:
                    data.get_entry_data().set_status(EntryStatus.INPUT_OUTOFRANGE)

                else:
                    data.get_entry_data().set_status(EntryStatus.RUNNING)
                data.get_entry_data().set_decode_index(index)
                self.running_request_list[index] = data
                # logging.debug(f'add new valid request in batch, batch size index is {index}')
        else:
            # 否则，将等待队列中的下一个请求加入 running_request_list。
            # logging.debug('add a new request into batching list')
            data = heapq.heappop(self.waiting_request_queue)
            # logging.debug('get_nowait4')
            if data.entry_data.get_prompt_len() + data.entry_data.get_max_token_len() >= self.max_input_len:
                data.get_entry_data().set_status(EntryStatus.INPUT_OUTOFRANGE)
            else:
                data.get_entry_data().set_status(EntryStatus.RUNNING)
            data.get_entry_data().set_decode_index(index)
            self.running_request_list[index] = data
            # logging.debug(f'add new valid request in batch, batch size index is {index}')

    def _update_status_after_one_itreation(self):
        # 在每次推理迭代后，更新批次中无效请求的数量，并确定最后一个有效请求的索引。
        self.count_of_invalid_sample = 0
        """checkout and update number of invalid request in batching list"""
        self.max_valid_index = -1
        for index, data in enumerate(self.running_request_list):
            data_status = data.get_entry_data().get_status()
            if data_status == EntryStatus.FINISHED_STOPPED or data_status == EntryStatus.FINISHED_LENGTH_CAPPED:
                self.count_of_invalid_sample += 1
            elif data_status == EntryStatus.RUNNING:
                self.max_valid_index = index

    def _determine_batch_size(self):
        logging.debug("def _determine_batch_size(self):")
        # 根据当前请求队列的状态和已完成的请求数量动态调整批次大小。
        self._update_status_after_one_itreation()
        bf_batch = self.batch_size
        queue_len = len(self.waiting_request_queue)
        bs_list_len = len(self.dyn_batch)
        # 1. 请求队列长度大于当前batch_size，扩容
        if self.max_valid_index == -1 or queue_len > self.batch_size:
            # 获取最接近waiting list长度的batch档位
            dyn_index = queue_len
        # 2. 请求队列长度小于count_of_invalid_sample，根据max_valid_index动态到邻近档位
        elif queue_len < self.count_of_invalid_sample:
            # max_valid_index左侧有多少结束的token
            left_free_num = self.count_of_invalid_sample - (self.batch_size - self.max_valid_index - 1)
            if queue_len <= left_free_num:
                dyn_index = self.max_valid_index + 1
            else:
                # 请求队列中全部补齐会到哪个index
                dyn_index = queue_len - left_free_num + self.max_valid_index + 1
        else:
            dyn_index = self.max_valid_index + 1 + queue_len - self.count_of_invalid_sample
        bs_after_changing = self.batch_size
        if dyn_index <= 0:
            # 默认值
            bs_after_changing = self.dyn_batch[0]
        else:
            for i in range(1, bs_list_len):
                if dyn_index > self.dyn_batch[bs_list_len - i - 1]:
                    bs_after_changing = self.dyn_batch[bs_list_len - i]
                    break
        self.batch_size = bs_after_changing if bs_after_changing > 0 else self.dyn_batch[0]
        af_batch = self.batch_size
        if af_batch != bf_batch:
            logging.debug(('----bs changed from  {} '.format(bf_batch)))
            logging.debug(('----bs changed to  {} '.format(af_batch)))
        if bf_batch >= af_batch:
            self.running_request_list = self.running_request_list[:af_batch]
        else:
            bf_batch = 0 if self.max_valid_index == -1 else bf_batch
            block_size = 0
            if self.config.model_config.page_attention:
                block_size = self.config.pa_config.block_size
            for i in range(bf_batch, af_batch):
                entry_meta_data = EntryMetaData(page_attention=self.config.model_config.page_attention,
                                                request_id=PADDING_REQUEST_ID,
                                                is_prompt=True,
                                                entry_data=EntryData(prompt_tokens=[self.eos_token],
                                                                     max_token_len=5000),
                                                entry_id=-1,
                                                prompt=PADDING_PROMPT,
                                                block_size=block_size)
                entry_meta_data.get_entry_data().set_decode_index(i)
                entry_meta_data.get_entry_data().set_status(EntryStatus.PADDING_INVAILED)
                self.running_request_list.append(entry_meta_data)

    def _continuous_batch(self):
        logging.debug("def _continuous_batch(self):")
        # init batch size when running_request_list is empty.
        if len(self.running_request_list) == 0:
            self._get_next_batch()
        # update invalid request number in batching list.
        self._update_status_after_one_itreation()
        if self.count_of_invalid_sample == self.batch_size:
            self._get_next_batch()
        # update status after one inference step
        else:
            checkout_list = self.checkout_entry()
            for index, data in enumerate(checkout_list):
                if data and index < self.batch_size:
                    # logging.debug('----{}-th prefill request in batching padded to batch.'.format(index))
                    self._padding_new_prompt_to_batch(index)

    def _insert_new_prompt_to_batch_pa(self, index):
        # logging.debug('add a new request into batching list')
        # data = self.waiting_request_queue.get_nowait()
        # 从 waiting_request_queue 中取出一个请求，检查其输入长度
        data = heapq.heappop(self.waiting_request_queue)
        if data.entry_data.get_prompt_len() + data.entry_data.get_max_token_len() >= self.max_input_len:
            data.get_entry_data().set_status(EntryStatus.INPUT_OUTOFRANGE)
        else:
            data.get_entry_data().set_status(EntryStatus.RUNNING)
        # 将新的请求插入到当前批次的指定索引位置，并更新其解码索引。
        data.get_entry_data().set_decode_index(index)
        self.running_request_list[index] = data
        logging.debug(f'add new valid request in batch, batch size index is {index}')

    def try_substitute_entry(self):
        # 检查无效请求，获取当前批次中无效的请求列表。
        checkout_list = self.checkout_entry()
        # 找到无效请求的索引，存入 is_invalid_index_list。
        is_invalid_index_list = []
        for index, is_invalid in enumerate(checkout_list):
            if is_invalid:
                is_invalid_index_list.append(index)
        if not is_invalid_index_list:
            # logging.debug("no invalid entry to substitute")
            return False
        # 如果有空槽位，尝试替代一条新请求：
        # 选择第一个无效请求的索引 index_to_substitute
        index_to_substitute = is_invalid_index_list[0]
        logging.debug("trying to substitute old entry at index: %s", index_to_substitute)
        # 如果新entry需要的block数量，小于can_substitute entry的block数量 + mem pool全局剩余block数量
        new_entry = self.waiting_request_queue[0]
        # 检查等待队列中的第一个新请求，判断它是否可以使用缓存资源
        if new_entry.cache_engine.try_use_budget(new_entry.get_entry_data().get_len()):
            self._insert_new_prompt_to_batch_pa(index_to_substitute)
            return True
        logging.debug("failed inserting to existing entry")
        # 如果空间不足，那么连第一条waiting的请求就无法替换，直接退出
        return False

    def reset_all_budgets(self):
        # 重置当前批次中所有请求的缓存预算，释放它们占用的资源。
        # logging.debug("current running list")
        for entry in self.running_request_list:
            entry.cache_engine.release_budget()

    def can_predict_current_batch(self):
        checkout_list = self.checkout_entry()
        for index, is_invalid in enumerate(checkout_list):
            # 对于batch中running的请求
            if is_invalid:
                continue
            entry = self.running_request_list[index]
            entry_cache_engine = entry.cache_engine
            if entry.is_prompt:
                if not entry_cache_engine.try_use_budget(entry.get_entry_data().get_len()):
                    return False
            else:
                if not entry_cache_engine.try_use_budget(1):
                    return False
        # logging.debug("can decode current batch return true")
        return True

    def try_initialize_paddings_pa(self):
        # running list和batch size不匹配时，添加padding位补充
        # 场景：1.启动server后，runninglist为空；2.升档后
        # logging.debug("try initialize paddings...")
        # 如果 running_request_list 的长度已经等于 batch_size，则无需填充，直接返回。
        if len(self.running_request_list) == self.batch_size:
            return
        # 如果 running_request_list 的长度超过 batch_size，抛出错误，因为批次大小不应该超过配置的批次大小。
        elif len(self.running_request_list) > self.batch_size:
            raise RuntimeError("running list size: %s larger than batch size: %s!", len(self.running_request_list),
                               self.batch_size)
        block_size = 0
        if self.config.model_config.page_attention:
            block_size = self.config.pa_config.block_size
        # 对于批次中尚未填满的索引位置，从当前索引位置到批次大小之间，逐个创建无效请求（填充请求）。
        for index in range(len(self.running_request_list), self.batch_size):
            padding_entry = EntryMetaData(page_attention=self.config.model_config.page_attention,
                                          request_id=PADDING_REQUEST_ID,
                                          is_prompt=False,  # True
                                          entry_data=EntryData(prompt_tokens=[self.config.model_config.end_token],
                                                               max_token_len=5000),
                                          entry_id=-1,
                                          prompt=PADDING_PROMPT,
                                          block_size=block_size)
            padding_entry.get_entry_data().set_decode_index(index)
            padding_entry.get_entry_data().set_status(EntryStatus.PADDING_INVAILED)
            cache_engine = padding_entry.cache_engine
            cache_engine.assign_null_block()
            self.running_request_list.append(padding_entry)

    def insert_padding_entry(self, index):
        block_size = 0
        if self.config.model_config.page_attention:
            block_size = self.config.pa_config.block_size
        padding_entry = EntryMetaData(page_attention=self.config.model_config.page_attention,
                                      request_id=PADDING_REQUEST_ID,
                                      is_prompt=False,  # True
                                      entry_data=EntryData(prompt_tokens=[self.config.model_config.end_token],
                                                           max_token_len=5000),
                                      entry_id=-1,
                                      prompt=PADDING_PROMPT,
                                      block_size=block_size)
        padding_entry.get_entry_data().set_decode_index(index)
        padding_entry.get_entry_data().set_status(EntryStatus.PADDING_INVAILED)
        cache_engine = padding_entry.cache_engine
        cache_engine.assign_null_block()
        self.running_request_list[index] = padding_entry

    def try_swap_valid_entries(self):
        is_invalid_list = self.checkout_entry()
        num_tokens_index_list = []
        for index, is_invalid in enumerate(is_invalid_list):
            if is_invalid:
                continue
            num_tokens_index_list.append((self.running_request_list[index].get_entry_data().get_len(), index))
        if not num_tokens_index_list:
            raise RuntimeError("no valid entry to pop!")

        num_tokens_index_list.sort(key=lambda x: x[0])
        _, index_to_swap = num_tokens_index_list[0]

        # 释放一条长度最短的valid entries（认为是最后进来的，TODO：按照时间顺序pop掉最晚进来的entry）
        entry_to_swap = self.running_request_list[index_to_swap]
        entry_to_swap.get_entry_data().set_status(EntryStatus.WAITING)
        entry_to_swap.get_entry_data().set_decode_index(0)
        entry_to_swap.is_prompt = True
        entry_to_swap.cache_engine.release_cache()
        # append回waiting list
        logging.warning("swap entry out, index: %s", index_to_swap)
        heapq.heappush(self.waiting_request_queue, entry_to_swap)
        # 用padding替代
        # logging.debug("inserting padding to popped entry %s", index_to_swap)
        self.insert_padding_entry(index_to_swap)

    def _continuous_batch_pa(self):
        logging.debug("def _continuous_batch_pa(self):")
        # 重置整个缓存池的预算。
        ServingBlockMemPool.instance().reset_budget()
        # ServingBlockMemPool.instance().log_status()
        # 为当前批次中的空位填充无效请求。
        self.try_initialize_paddings_pa()
        # self.log_running_list("schedule start running status")
        # 判断batch内的running entry，能否进行本轮推理？
        num_entry_swapped_out = 0
        while not self.can_predict_current_batch():
            # 如果不能，swap出去已有请求
            self.reset_all_budgets()
            self.try_swap_valid_entries()
            num_entry_swapped_out += 1
        # 如果有请求被交换，重置缓存预算，并结束当前循环。
        if num_entry_swapped_out:
            self.reset_all_budgets()
            return
        # 3. 处理新请求
        # logging.debug("determine if can process new request...")
        # 当等待队列不为空时，继续尝试使用 try_substitute_entry() 来替换已结束的请求，填充新的有效请求。
        while self.waiting_request_queue:
            # 如果有空batch槽，尝试插入
            # logging.debug("has new entry, trying to enter current batch")
            if not self.try_substitute_entry():
                # 尝试失败，退出
                break
        # 最后，重置所有缓存资源，确保准备好进行推理。
        self.reset_all_budgets()
        # ServingBlockMemPool.instance().log_status()

    def _static_batch(self):
        if self._all_samples_in_batch_is_over() or len(self.running_request_list) == 0:
            self._get_next_batch()
        # updata status after one inference step
        self._update_status_after_one_itreation()
        # if all samples in batch is invalid status, a static batch is over
        if self.count_of_invalid_sample == self.batch_size:
            self._get_next_batch()

    def schedule(self) -> Tuple[List[EntryMetaData], int]:
        # 如果 dyn_batch 存在多个批次大小，调用 _determine_batch_size 动态调整批次大小
        if self.dyn_batch and len(self.dyn_batch) > 1:
            self._determine_batch_size()
        # 根据配置的 batching_strategy 和是否启用了 page_attention，决定使用哪种批次调度策略：
        if self.batching_strategy == 'static':
            self._static_batch()
        elif not self.config.model_config.page_attention and self.batching_strategy == 'continuous':
            self._continuous_batch()
        elif self.config.model_config.page_attention:  # 加入PA
            self._continuous_batch_pa()
        else:
            raise ValueError("Invalid batching strategy!, please setting static or continuous")
        return self.running_request_list, self.batch_size

    # 增加对 PA的_finished_request处理
    def _finished_pa_request(self, index, token, eos_id):
        # eos
        if token == eos_id:
            logging.debug("a request finished, token equal to {}".format(token))
            self.running_request_list[index].get_entry_data().set_status(EntryStatus.FINISHED_STOPPED)
            self.running_request_list[index].cache_engine.release_cache()
            self.running_request_list[index].cache_engine.assign_null_block()
            return

        # max len
        entry_data = self.running_request_list[index].get_entry_data()
        if entry_data.max_token_len <= entry_data.get_output_len():
            logging.debug("a request reached the max generate token length")
            self.running_request_list[index].get_entry_data().set_status(EntryStatus.FINISHED_LENGTH_CAPPED)
            self.running_request_list[index].cache_engine.release_cache()
            self.running_request_list[index].cache_engine.assign_null_block()
            return

        if entry_data.get_len() >= self.config.model_config.max_generate_length:
            logging.debug("a request reached seq len: %s, index: %s", self.config.max_generate_length, index)
            self.running_request_list[index].get_entry_data().set_status(EntryStatus.FINISHED_LENGTH_CAPPED)
            self.running_request_list[index].cache_engine.release_cache()
            self.running_request_list[index].cache_engine.assign_null_block()
            return

        # input outofrange
        if entry_data.status == EntryStatus.INPUT_OUTOFRANGE or entry_data.status == EntryStatus.EMPTY_PROMPT_TOKEN:
            self.running_request_list[index].get_entry_data().set_status(EntryStatus.FINISHED_STOPPED)
            self.running_request_list[index].cache_engine.release_cache()
            self.running_request_list[index].cache_engine.assign_null_block()
            return
        # predict failed
        if token == -1:
            logging.debug("a request predict failed, token equal to {}".format(token))
            self.running_request_list[index].get_entry_data().set_status(EntryStatus.FINISHED_STOPPED)
            self.running_request_list[index].cache_engine.release_cache()
            self.running_request_list[index].cache_engine.assign_null_block()
            return

    def _finished_request(self, index, token, eos_id):
        # eos
        # 如果生成的 token 等于结束符 eos_id，表示推理已经完成。将当前请求的状态设为 FINISHED_STOPPED，并记录日志说明请求已完成。
        if token == eos_id:
            logging.debug("a request finished, token equal to {}".format(token))
            self.running_request_list[index].get_entry_data().set_status(EntryStatus.FINISHED_STOPPED)
            return

        # max len
        # 检查当前请求的生成 token 数是否已达到或超过其最大 token 长度 max_token_len。
        # 如果是，将请求状态设置为 FINISHED_LENGTH_CAPPED，表示请求达到最大长度并终止生成。
        entry_data = self.running_request_list[index].get_entry_data()
        if entry_data.max_token_len <= entry_data.get_output_len():
            logging.debug("a request reached the max generate token length")
            self.running_request_list[index].get_entry_data().set_status(EntryStatus.FINISHED_LENGTH_CAPPED)
            return

        # input outofrange
        # 如果当前请求的状态为 INPUT_OUTOFRANGE（输入超出范围）或者 EMPTY_PROMPT_TOKEN（输入为空），
        # 则将请求状态设置为 FINISHED_STOPPED，表示请求不能再继续推理。
        if entry_data.status == EntryStatus.INPUT_OUTOFRANGE or entry_data.status == EntryStatus.EMPTY_PROMPT_TOKEN:
            self.running_request_list[index].get_entry_data().set_status(EntryStatus.FINISHED_STOPPED)
            return
        # predict failed
        # 如果生成的 token 为 -1，表示模型推理失败，终止请求，将其状态设为 FINISHED_STOPPED，并记录日志说明失败原因。
        if token == -1:
            logging.debug("a request predict failed, token equal to {}".format(token))
            self.running_request_list[index].get_entry_data().set_status(EntryStatus.FINISHED_STOPPED)
            return

    def upate_entries_after_one_step(self, outputs: List[int], eos_id: int, index_list: List[int] = None):
        # 在每次推理迭代之后更新每个请求的状态，包括输出 token 的更新、状态变更以及 PA 机制的处理。
        """update status after ever iteration"""
        # optimize prefill multi-batch later
        if index_list is not None:
            # idx: index_list and outputs data index, index: batch list index.
            for idx, index in enumerate(index_list):
                self.running_request_list[index].is_prompt = False
                # invalid prompt
                # 如果请求状态是 PADDING_INVAILED，表示该请求是一个无效的填充请求，直接跳过。
                if self.running_request_list[index].get_entry_data().get_status() == EntryStatus.PADDING_INVAILED:
                    continue

                # 如果请求状态是 INPUT_OUTOFRANGE，则使用 INPUT_OUT_OF_TOKEN 作为生成的 token。
                if self.running_request_list[index].get_entry_data().get_status() == EntryStatus.INPUT_OUTOFRANGE:
                    update_token = INPUT_OUT_OF_TOKEN[0]
                # 如果请求状态是 EMPTY_PROMPT_TOKEN，则使用 INPUT_EMPTY_TOKEN 作为生成的 token。
                elif self.running_request_list[index].get_entry_data().get_status() == EntryStatus.EMPTY_PROMPT_TOKEN:
                    update_token = INPUT_EMPTY_TOKEN[0]
                # 否则，使用 outputs 中的生成 token。
                else:
                    update_token = outputs[idx]

                self.running_request_list[index].get_entry_data().updata_output_tokens(update_token)
                # valid prompt 区分PA处理
                if self.config.model_config.page_attention:
                    self._finished_pa_request(index, update_token, eos_id)
                else:
                    self._finished_request(index, update_token, eos_id)
        # decode
        else:
            # 如果没有提供 index_list，则表示需要遍历所有请求并逐个更新。
            for index, token in enumerate(outputs):
                if self.running_request_list[index].get_entry_data().get_status() != EntryStatus.RUNNING:  # 改动
                    continue
                # update new token to result list
                self.running_request_list[index].get_entry_data().updata_output_tokens(token)
                # 区分PA处理
                if self.config.model_config.page_attention:
                    self._finished_pa_request(index, token, eos_id)
                else:
                    self._finished_request(index, token, eos_id)

    def abort_entry(self,
                    request_id: str):
        logging.debug(f"Abort entry called with request_id: {request_id} of type {type(request_id)}")
        for index, data in enumerate(self.running_request_list):
            # logging.debug(f"Abort entry data.status{self.running_request_list[index].get_entry_data().get_status()} == request_id:: {data.request_id}")
            if data.request_id == request_id:
                # logging.debug(f"Abort entry data.request_id:f{data.request_id} == request_id:: {request_id} of type {type(request_id)}")
                self.running_request_list[index].get_entry_data().set_status(EntryStatus.FINISHED_STOPPED)

