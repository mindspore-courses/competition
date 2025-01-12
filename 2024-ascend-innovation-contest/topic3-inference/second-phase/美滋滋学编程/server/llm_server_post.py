import asyncio
import logging
import subprocess

from mindspore_serving.master.master import AsyncMaster
from mindspore_serving.master.response_async_queue import AsyncResultsOfOneRequest
from mindspore_serving.master.utils import ResponseOutput, ModelInfo
from mindspore_serving.master.request_resister_engine import RequestEngine
from mindspore_serving.config.config import ServingConfig


# from mindspore_serving.serving_utils.register import import_all_modules_for_register

# import_all_modules_for_register()

# LLMServer 类用于管理请求队列、推理任务，并与模型进行交互处理推理请求。
class LLMServer:
    """
    使用一个异步的队列 (request_queue) 存储请求，并通过 master 对未完成的请求进行批处理推理。
       request_queue(FIFO): add request into a async queue, and monitor request status(is_finished),
                            mapping inference result of each iteration to corresponding request
                            result queue(used in stream return).
       master: Continuously getting unfinished request from request_queue, conducting batch strategy,
               and doing one step inference using ms-lite, after get result of one iteration, client
               get stream inference result from request_queue, and update to request_queue.
    """

    def __init__(self, config: ServingConfig):
        logging.debug("LLMServer init")
        self.request_engine = RequestEngine()
        self.background_loop = None#异步事件循环的标志，初始值为 None。
        self.master = AsyncMaster(config)#是一个 AsyncMaster 实例，用于处理实际的推理任务。
        self.status = 0#用于标记服务器的状态（0 表示停止，1 表示运行）。
        self.config = config#保存配置信息。

    @property
    def is_running(self) -> bool:
        # 是一个只读属性，用来判断后台循环是否正在运行。
        return self.background_loop is not None

    async def run_loop(self):
        # 一个异步方法，持续运行，只要 status 为 1。
        while self.status:
            # 每次调用 step() 处理一个推理步骤，并通过 asyncio.sleep(0) 让出控制权。
            # logging.debug("await self.step()")
            await self.step()
            await asyncio.sleep(0)

    def start_background_loop(self) -> None:
        # 用于启动后台循环。
        # todo
        self.status = 1
        """Start the background loop."""
        if self.is_running:
            raise RuntimeError("Background loop is already running.")
        # 创建并启动异步任务 run_loop()。
        self.background_loop = asyncio.get_event_loop().create_task(self.run_loop())

    async def register_request(self,
                               request_id: str,
                               **add_request_info) -> AsyncResultsOfOneRequest:
        #异步方法注册一个新的请求。
        logging.debug("background loop {}".format(self.background_loop))
        if self.background_loop is None:
            self.start_background_loop()

        # 注册请求，并返回结果流 res_stream。
        res_stream = self.request_engine.register_request(
            request_id,
            **add_request_info)
        # logging.debug("register_request over")
        return res_stream

    def _abort(self, request_id: str) -> None:
        """Abort a request.
        Args:
            request_id: The unique id of the request.
        """
        self.request_engine.abort_request(request_id)

    async def step(self):
        # loop consuming from request_engine
        # logging.debug("async def step(self):")
        if self.status == 0:
            return
        # 先从 request_engine 中获取新的请求（推理请求信息参数字典的一个列表）和完成的请求（request_id的set容器）。
        new_requests, finished_requests = self.request_engine.get_requests_from_register_pool()
        # logging.debug(f"Finished requests type: {type(finished_requests)}, value: {finished_requests}")
        # 将新请求添加到 master 的调度池中。
        for new_request in new_requests:
            self.master.add_requests_to_schedule_pool(**new_request)
        # 如果有完成的请求，则调用 _master_abort 方法进行清理。
        if finished_requests:
            await self._master_abort(finished_requests)
        # 调用 master.step_async() 执行推理，推理结果通过 request_engine.process_request_output() 处理。
        request_outputs = await self.master.step_async()
        # Put the outputs into the corresponding streams.
        if request_outputs is not None:
            for request_output in request_outputs:
                self.request_engine.process_request_output(request_output)

    def get_total_tokens(self):
        return self.master.get_number_of_total_tokens()

    def get_bs_current(self):
        return self.master.get_current_batch()

    def get_queue_current(self):
        return self.master.get_current_requestes_nums()

    async def generate_answer(
            self,
            request_id: str,
            **add_request_info
    ) -> ResponseOutput:

        # Preprocess the request.
        try:
            # logging.DEBUG(f"async def generate_answer:{request_id}")
            res_stream = await self.register_request(request_id, **add_request_info)
            # logging.debug(f"async def generate_answer:{request_id}")

            async for request_output in res_stream:
                yield request_output

        except Exception as e:
            # If there is an exception, abort the request.
            self._abort(request_id)
            raise e

    async def _master_abort(self, request_ids):
        logging.debug(f"Master abort called with request_ids type: {type(request_ids)}, value: {request_ids}")
        # self.master.abort_request(request_ids)
        for request_id in request_ids:
            self.master.abort_request(request_id)

    def stop(self):
        # 1. stop background
        self.status = 0
        self.master.stop()

    def get_dockerId(self):
        p = subprocess.Popen("cat /proc/self/cgroup | grep /docker | head -1 | cut -d/ -f3", shell=True,
                             stdout=subprocess.PIPE)
        out = p.stdout.read()
        id = str(out, 'utf-8')
        return id

    def get_serverd_model_info(
            self
    ) -> ModelInfo:
        max_seq_length = int(self.config.model_config.seq_length[-1])
        max_decode_batch_size = int(self.config.model_config.decode_batch_size[-1])
        docker_id = self.get_dockerId()
        serverd_model_info = ModelInfo(docker_label=docker_id,
                                       max_batch_total_tokens=max_seq_length * max_decode_batch_size,
                                       max_concurrent_requests=self.master.get_current_requestes_nums(),
                                       max_input_length=max_seq_length, max_total_tokens=max_decode_batch_size,
                                       model_dtype=self.config.model_config.model_dtype,
                                       model_id=self.config.model_config.model_name)
        return serverd_model_info
