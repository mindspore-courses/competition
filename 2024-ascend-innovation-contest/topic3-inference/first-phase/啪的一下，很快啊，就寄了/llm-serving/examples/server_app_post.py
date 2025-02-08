# Copyright 2023 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""MindSpore Serving server app"""

# This is the HTTP server that handles requests

import os
import json
import uuid
import time
import logging
import argparse
from contextlib import asynccontextmanager
from typing import List

import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from sse_starlette.sse import EventSourceResponse, ServerSentEvent

from mindspore_serving.client.client_utils import ClientRequest, Parameters, ValidatorUtil
from mindspore_serving.config.config import ServingConfig, check_valid_config
from mindspore_serving.server.llm_server_post import LLMServer
from mindspore_serving.serving_utils.constant import *

DEBUG_WIN = os.getenv('DEBUG_WIN')
work_path = os.path.dirname(os.path.abspath(__file__))
logging.basicConfig(level=logging.INFO,
                    filename='./output/server_app.log',
                    filemode='w',
                    format=
                    '%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s')

llm_server = None
config = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # init LLMServer
    if not check_valid_config(config):
        yield
    global llm_server
    llm_server = LLMServer(config)
    yield
    llm_server.stop()
    print('---------------server app is done---------------')


app = FastAPI(lifespan=lifespan)


async def get_full_res(request, results):
    #ts = time.time()
    all_texts = ''
    tokens_list = []
    finish_reason = ""
    output_tokens_len = 0
    async for result in results:
        finish_reason = result.finish_reason
        output_tokens_len = result.output_tokens_len
        for index, output in enumerate(result.outputs):
            answer_texts = output.text
            if answer_texts is None:
                continue

            res_list = {
                "id": output.index,
                "logprob": output.logprob,
                "special": output.special,
                "text": answer_texts
            }

            tokens_list.append(res_list)
            all_texts += answer_texts

    ret = {
        "generated_text": all_texts,
        "finish_reason": finish_reason,
        "generated_tokens": output_tokens_len,
        "prefill": [tokens_list[0]],
        "seed": 0,
        "tokens": tokens_list,
        "top_tokens": [
            [tokens_list[0]]
        ],
        "details": None
    }
    yield (json.dumps(ret, ensure_ascii=False) + '\n').encode("utf-8")
    #print('[get_full_res]', time.time() - ts)


async def get_full_res_sse(request, results):
    #ts = time.time()
    all_texts = ''
    tokens_list = []
    finish_reason = ""
    output_tokens_len = 0
    async for result in results:
        finish_reason = result.finish_reason
        output_tokens_len = result.output_tokens_len
        for index, output in enumerate(result.outputs):
            answer_texts = output.text
            if answer_texts is None:
                continue

            res_list = {
                "id": output.index,
                "logprob": output.logprob,
                "special": output.special,
                "text": answer_texts
            }
            tokens_list.append(res_list)
            all_texts += answer_texts

    ret = {
        "event": "message",
        "retry": 30000,
        "generated_text": all_texts,
        "finish_reason": finish_reason,
        "generated_tokens": output_tokens_len,
        "prefill": [tokens_list[0]],
        "seed": 0,
        "tokens": tokens_list,
        "top_tokens": [
            [tokens_list[0]]
        ],
        "details": None
    }
    yield (json.dumps(ret, ensure_ascii=False) + '\n').encode("utf-8")
    #print('[get_full_res_sse]', time.time() - ts)


async def get_stream_res(request, results):
    #ts = time.time()

    all_texts: List[str] = []
    tokens_list = []
    finish_reason: str = None
    output_tokens_len = 0
    token_index = 0
    async for result in results:
        finish_reason = result.finish_reason
        output_tokens_len = result.output_tokens_len
        for index, output in enumerate(result.outputs):
            answer_texts = output.text
            if answer_texts is None:
                continue
            else:
                token_index += 1

            res_list = {
                "id": output.index,
                "logprob": output.logprob,
                "special": output.special,
                "text": answer_texts
            }
            tokens_list.append(res_list)
            all_texts.append(answer_texts)

            ret = {
                "details": None,
                "generated_text": answer_texts,
                "tokens": res_list,
                "top_tokens": [
                    res_list
                ],
            }
            #logging.debug("get_stream_res one token_index is {}".format(token_index))
            yield ("data:" + json.dumps(ret, ensure_ascii=False) + '\n').encode("utf-8")

    return_full_text = request.parameters.return_full_text
    if return_full_text:
        ret = {
            "generated_text": ''.join(all_texts),
            "finish_reason": finish_reason,
            "generated_tokens": output_tokens_len,
            "prefill": [tokens_list[0]],
            "seed": 0,
            "tokens": tokens_list,
            "top_tokens": [
                [tokens_list[0]]
            ],
            "details": None
        }
        yield ("data:" + json.dumps(ret, ensure_ascii=False) + '\n').encode("utf-8")

    #print('[get_stream_res]', time.time() - ts)


async def get_stream_res_sse(request, results): # <- step 3
    #ts = time.time()

    all_texts: list[str] = []
    tokens_list = []
    finish_reason: str = None
    output_tokens_len = 0
    token_index = 0
    async for result in results:
        finish_reason = result.finish_reason
        output_tokens_len = result.output_tokens_len
        for index, output in enumerate(result.outputs):
            answer_texts = output.text
            if answer_texts is None:
                continue
            else:
                token_index += 1

            res_list = {
                "id": output.index,
                "logprob": output.logprob,
                "special": output.special,
                "text": answer_texts
            }
            tokens_list.append(res_list)
            all_texts.append(answer_texts)

            tmp_ret = {
                "details": None,
                "generated_text": answer_texts,
                "tokens": res_list,
                "top_tokens": [
                    res_list
                ]
            }
            ret = {
                "event": "message",
                "retry": 30000,
                "data": tmp_ret
            }
            #logging.debug("get_stream_res one token_index is {}".format(token_index))
            yield (json.dumps(ret, ensure_ascii=False) + '\n').encode("utf-8")

    return_full_text = request.parameters.return_full_text
    if return_full_text:
        full_tmp_ret = {
            "details": None,
            "generated_text": ''.join(all_texts),
            "finish_reason": finish_reason,
            "generated_tokens": output_tokens_len,
            "prefill": [tokens_list[0]],
            "seed": 0,
            "tokens": tokens_list,
            "top_tokens": [
                [tokens_list[0]]
            ]
        }
        ret = {
            "event": "message",
            "retry": 30000,
            "data": full_tmp_ret
        }
        yield (json.dumps(ret, ensure_ascii=False) + '\n').encode("utf-8")

    #print('[get_stream_res_sse]', time.time() - ts)


def send_request(request: ClientRequest):       # <- step 2
    #ts = time.time()
    request_id = str(uuid.uuid1())

    if request.parameters is None:
        request.parameters = Parameters()
    if request.parameters.do_sample is None:
        request.parameters.do_sample = False
    if request.parameters.top_k is None:
        request.parameters.top_k = 3
    if request.parameters.top_p is None:
        request.parameters.top_p = 1.0
    if request.parameters.temperature is None:
        request.parameters.temperature = 1.0
    if request.parameters.repetition_penalty is None:
        request.parameters.repetition_penalty = 1.0
    if request.parameters.max_new_tokens is None:
        request.parameters.max_new_tokens = 300
    if request.parameters.return_protocol is None:
        request.parameters.return_protocol = "sse"
    if request.parameters.decoder_input_details is None:
        request.parameters.decoder_input_details = False
    if request.parameters.details is None:
        request.parameters.details = False
    if request.parameters.return_full_text is None:
        request.parameters.return_full_text = False
    if request.parameters.seed is None:
        request.parameters.seed = 0
    if request.parameters.stop is None:
        request.parameters.stop = []
    if request.parameters.top_n_tokens is None:
        request.parameters.top_n_tokens = 0
    if request.parameters.truncate is None:
        request.parameters.truncate = False
    if request.parameters.typical_p is None:
        request.parameters.typical_p = 0
    if request.parameters.watermark is None:
        request.parameters.watermark = False

    if not ValidatorUtil.validate_top_k(request.parameters.top_k, config.model_config.vocab_size):
        request.parameters.top_k = 1
    if not ValidatorUtil.validate_top_p(request.parameters.top_p) and request.parameters.top_p < 0.01:
        request.parameters.top_p = 0.01
    if not ValidatorUtil.validate_top_p(request.parameters.top_p) and request.parameters.top_p > 1.0:
        request.parameters.top_p = 1.0
    if not ValidatorUtil.validate_temperature(request.parameters.temperature):
        request.parameters.temperature = 1.0
        request.parameters.do_sample = False

    params = {
        "prompt": request.inputs,
        "do_sample": request.parameters.do_sample,
        "top_k": request.parameters.top_k,
        "top_p": request.parameters.top_p,
        "temperature": request.parameters.temperature,
        "repetition_penalty": request.parameters.repetition_penalty,
        "max_token_len": request.parameters.max_new_tokens,
    }

    #print('generate_answer params: ', params)
    global llm_server
    results = llm_server.generate_answer(request_id, **params)
    #print('[send_request]', time.time() - ts)
    return results


@app.post("/models/llama2")
async def async_generator(request: ClientRequest):
    results = send_request(request)

    if request.stream:
        if request.parameters.return_protocol == "sse":
            print('get_stream_res_sse...')
            return EventSourceResponse(get_stream_res_sse(request, results),
                                       media_type="text/event-stream",
                                       ping_message_factory=lambda: ServerSentEvent(
                                           **{"comment": "You can't see this ping"}),
                                       ping=600)
        else:
            print('get_stream_res...')
            return StreamingResponse(get_stream_res(request, results))
    else:
        print('get_full_res...')
        return StreamingResponse(get_full_res(request, results))


@app.post("/models/llama2/generate")
async def async_full_generator(request: ClientRequest):
    results = send_request(request)
    #print('get_full_res...')
    return StreamingResponse(get_full_res(request, results))


@app.post("/models/llama2/generate_stream")     # <- step 1
async def async_stream_generator(request: ClientRequest):
    if int(os.getenv('RUN_LEVEL', 99)) <= 1:
        async def get_stream_res_sse_dummy():
            yield json.dumps({"data": {'generated_text':'fake'}}).encode("utf-8")
        return EventSourceResponse(
            get_stream_res_sse_dummy(),
            media_type="text/event-stream",
            ping_message_factory=lambda: ServerSentEvent(**{"comment": "You can't see this ping"}),
            ping=600
        )

    results = send_request(request)
    if request.parameters.return_protocol == "sse":
        #print('get_stream_res_sse...')
        return EventSourceResponse(
            get_stream_res_sse(request, results),
            media_type="text/event-stream",
            ping_message_factory=lambda: ServerSentEvent(**{"comment": "You can't see this ping"}),
            ping=600
        )
    else:
        p#rint('get_stream_res...')
        return StreamingResponse(get_stream_res(request, results))


def run_server_app(config):
    print('server port is: ', config.server_port)
    uvicorn.run(app, host="127.0.0.1", port=config.server_port)


async def _get_batch_size():
    global llm_server
    batch_size = llm_server.get_bs_current()
    ret = {'event': "message", "retry": 30000, "data": batch_size}
    yield json.dumps(ret, ensure_ascii=False)


async def _get_request_numbers():
    global llm_server
    queue_size = llm_server.get_queue_current()
    ret = {'event': "message", "retry": 30000, "data": queue_size}
    yield json.dumps(ret, ensure_ascii=False)


async def _get_serverd_model_info():
    global llm_server
    serverd_model_info = llm_server.get_serverd_model_info()
    ret = {
        "docker_label": serverd_model_info.docker_label,
        "max_batch_total_tokens": serverd_model_info.max_batch_total_tokens,
        "max_concurrent_requests": serverd_model_info.max_concurrent_requests,
        "max_input_length": serverd_model_info.max_input_length,
        "max_total_tokens": serverd_model_info.max_total_tokens,
        "model_device_type": "CANN",
        "model_dtype": serverd_model_info.model_dtype,
        "model_id": serverd_model_info.model_id,
        "model_pipeline_tag": "text-generation",
        "version": "2.3"
    }
    #print(ret)
    yield json.dumps(ret, ensure_ascii=False)


@app.get("/serving/get_bs")
async def get_batch_size():
    return EventSourceResponse(_get_batch_size(),
                               media_type="text/event-stream",
                               ping_message_factory=lambda: ServerSentEvent(**{"comment": "You can't see this ping"}),
                               ping=600)


@app.get("/serving/get_request_numbers")
async def get_request_numbers():
    return EventSourceResponse(_get_request_numbers(),
                               media_type="text/event-stream",
                               ping_message_factory=lambda: ServerSentEvent(**{"comment": "You can't see this ping"}),
                               ping=600)


@app.get("/serving/get_serverd_model_info")
async def get_serverd_model_info():
    return EventSourceResponse(_get_serverd_model_info(),
                               media_type="text/event-stream",
                               ping_message_factory=lambda: ServerSentEvent(**{"comment": "You can't see this ping"}),
                               ping=600)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='YAML config files')
    args = parser.parse_args()

    config = ServingConfig(args.config)
    if DEBUG_WIN:   # NOTE: this fucking path CANNOT contain white spaces
        config.tokenizer.vocab_file = '../performance_serving/tokenizer.model'
    run_server_app(config.serving_config)
