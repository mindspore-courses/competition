import time
import json
import requests
import argparse
import os
import numpy as np
import threading
from mindformers import LlamaTokenizer
import log

time_now = time.strftime("%Y-%m-%d-%H_%M", time.localtime())
LOGGER = log.logger_for_test("test_llama", f"./testLog/test_performance_{time_now}.log")
LLAMA2_tokenizer = "./tokenizer.model"  # 换模型不需要换tokenizer
RESULT = []

CompletedProgress = 0


def init_tokenizer(model_path=LLAMA2_tokenizer):
    tokenizer = LlamaTokenizer(model_path)
    return tokenizer


def get_text_token_num(tokenizer, text):
    tokens = tokenizer.tokenize(text)
    num_tokens = len(tokens)
    # print("token num in text is ", num_tokens)
    return num_tokens


def poisson_random_s(interval):
    poisson_random_ms = np.random.poisson(interval * 1000, 1000)[0]
    LOGGER.info(f"poisson random interval time is {poisson_random_ms / 1000}s")
    return poisson_random_ms / 1000


Tokenizer = init_tokenizer()


# 延迟tms定时器
def delayMsecond(t):
    t = t * 1000  # 传入s级别
    start, end = 0, 0
    start = time.time_ns()  # 精确至ns级别
    while end - start < t * 1000000:
        end = time.time_ns()


class MyThread(threading.Thread):
    def __init__(self, func, args=()):
        super(MyThread, self).__init__()
        self.func = func
        self.args = args

    def run(self):
        self.result = self.func(*self.args)

    def get_result(self):
        threading.Thread.join(self)
        try:
            return json.loads(self.result)
        except Exception:
            return None


class LargeModelClient:
    def __init__(self, port):
        self.url_generate_all = f'http://localhost:{port}/models/llama2/generate'
        self.url_generate_stream = f'http://localhost:{port}/models/llama2/generate_stream'

    def send_request(self, testcase, all_counts):
        global CompletedProgress
        # print("testcase is {}".format(testcase))
        inputs = testcase["input"]
        # inputs = "<s><|User|>:{}<eoh>\n<|Bot|>:".format(inputs)
        body = {"inputs": inputs}
        para = {}
        return_full_text = testcase["return_full_text"] if "return_full_text" in testcase else False
        do_sample = testcase["do_sample"]
        max_new_tokens = testcase["max_new_tokens"] if "max_new_tokens" in testcase else False
        topk_k = testcase["topk_k"] if "topk_k" in testcase else False
        top_p = testcase["top_p"] if "top_p" in testcase else False
        temperature = testcase["temperature"] if "temperature" in testcase else False
        stream = testcase["stream"]
        if max_new_tokens:
            para["max_new_tokens"] = max_new_tokens
        if temperature:
            para["temperature"] = temperature
        if topk_k:
            para["topk_k"] = topk_k
        if top_p:
            para["top_p"] = top_p
        para["do_sample"] = do_sample
        para["return_full_text"] = return_full_text
        # print(para)
        body["parameters"] = para
        if stream:
            res = self.return_stream(body, stream)
        else:
            res = self.return_all(body, stream)
        CompletedProgress += 1
        LOGGER.info(f"{res}\nTest Progress --> {CompletedProgress}/{all_counts}")
        RESULT.append(res)
        return res

    def return_all(self, request_body, stream):
        url = self.url_generate_stream if stream else self.url_generate_all
        headers = {"Content-Type": "application/json", "Connection": "close"}
        start_time = time.time()
        resp = requests.request("POST", url, data=json.dumps(request_body), headers=headers)
        resp_text = resp.text
        resp.close()
        res_time = time.time() - start_time
        # print(resp_text)
        return {"input": request_body["inputs"],
                "resp_text": json.loads(resp_text)["generated_text"],
                "res_time": res_time}

    def return_stream(self, request_body, stream):
        url = self.url_generate_stream if stream else self.url_generate_all
        headers = {"Content-Type": "application/json", "Connection": "close"}
        start_time = time.time()
        resp = requests.request("POST", url, data=json.dumps(request_body), headers=headers, stream=True)
        lis = []
        first_token_time = None
        for i, line in enumerate(resp.iter_lines(decode_unicode=True)):
            if line:
                if i == 0:
                    first_token_time = time.time() - start_time
                    LOGGER.info(f"first_token_time is {first_token_time}")
                # print(json.loads(line))
                # if
                print(json.loads(line)["data"][0]["generated_text"])
                lis.append(json.loads(line)["data"][0]["generated_text"])
                # data = json.loads(line)
                # print(data['id'])
        res_time = time.time() - start_time
        if request_body["parameters"]["return_full_text"]:
            # print(request_body["parameters"]["return_full_text"])
            resp_text = lis[-1]
            # print("******stream full text********")
            # print(resp_text)
        else:
            # print("******stream completeness result********")
            resp_text = "".join(lis)
            # print("".join(lis))
        return {"input": request_body["inputs"],
                "resp_text": resp_text,
                "res_time": res_time,
                "first_token_time": first_token_time}


def generate_thread_tasks(testcases, all_count, port):
    client = LargeModelClient(port)
    print(all_count)
    i = 0
    thread_tasks = []
    k = 0
    while True:
        print(k, ":", all_count)
        if i > len(testcases) - 1:
            i = 0
        thread_tasks.append(MyThread(client.send_request, (testcases[i], all_count)))
        i += 1
        k += 1
        if k == all_count:
            break
    LOGGER.info(f"thread_tasks length is {len(thread_tasks)}")
    return thread_tasks


def test_main(port, inputs, outputs, x, out_dir, test_all_time=3600):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    print('start Test...')
    testcases = []
    for i, input_string in enumerate(inputs):
        testcase = {"input": f"{input_string}", "do_sample": "False", "return_full_text": "True", "stream": True,
                    "max_new_tokens": get_text_token_num(Tokenizer, outputs[i])}
        testcases.append(testcase)
        # print(testcase)
    LOGGER.info(f"testcases length is {len(testcases)}")
    # 每次发送的间隔时间
    interval = round(1 / x, 2)
    all_counts = int(test_all_time * x)
    # 1h内一共需要发送多少次请求
    thread_tasks = generate_thread_tasks(testcases, all_counts, port)
    start_time = time.time()
    LOGGER.info(f"Start send request, avg interval is {interval}")
    for task in thread_tasks:
        task.start()
        delayMsecond(poisson_random_s(interval))

    for task in thread_tasks:
        task.join()

    end_time = time.time()
    LOGGER.info(f"All Tasks Done; Exec Time is {end_time - start_time}")
    gen_json = os.path.join(out_dir, f"result_{x}_x.json")
    with open(gen_json, "w+") as f:
        f.write(json.dumps(RESULT))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="test serving performance")
    parser.add_argument("-X", "--qps", help='x req/s', required=True, type=float)
    parser.add_argument("-P", "--port", help='port, default is 8000', required=True)
    parser.add_argument("-O", "--out_dir", help='dir for saving results', required=True)
    parser.add_argument("-T", "--test_time", help='test all time, default 1h', required=False, type=int, default=3600)
    args = parser.parse_args()
    with open("./alpaca_5010.json") as f:
        alpaca_data = json.loads(f.read())
    INPUTS_DATA = []
    OUTPUTS_DATA = []
    for data in alpaca_data:
        input_ = data["instruction"] + ":" + data["input"] if data["input"] else data["instruction"]
        INPUTS_DATA.append(input_)
        OUTPUTS_DATA.append(data["output"])
    test_main(args.port, INPUTS_DATA, OUTPUTS_DATA, args.qps, args.out_dir, args.test_time)
