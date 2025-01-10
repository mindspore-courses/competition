from mindspore_serving.agent.agent_multi_post_method_task1 import *
from mindspore_serving.agent.agent_multi_post_method_task1 import WorkAgent as WorkAgentBase


class WorkAgent(WorkAgentBase):

    # NOTE: only this part is different w.r.t. to two tasks
    def predict_for_kbk(self, current_index, input_ids, valid_length, block_tables, slot_mapping):
        # 封装调用模型参数
        model_kwargs = {"current_index": current_index}
        model_inputs = self.mindspore_model.prepare_inputs_for_generation(input_ids, **model_kwargs)
        logging.debug(f"predict model_inputs value is {model_inputs}")

        #2222222222222222222222222222222222222222222222
        predict_time = time.time()

        # FIXME: 很奇怪，我们产生的文件名不能直接匹配标准参考文件名，在这里修复一下 :(
        input_ids_str = '.'.join([str(id) if idx == 0 else f'0{id}' for idx, id in enumerate(input_ids[0].tolist())])
        current_index_str = str(current_index.item())
        filename = str(predict_time) + '_' + input_ids_str[:20] + '_' + current_index_str + '.npy'
        logging.info(f"***********************filename = {filename}")

        #2222222222222222222222222222222222222222222222

        # 调用mindformers进行推理 
        if self.mindspore_model.config.use_past:
            #logging.debug(f"predict before pa predict_for_kbk.")
            if self.is_prefill:
                self.mindspore_model.is_first_iteration = True
            res, current_index = self.mindspore_model.forward(
                input_ids=input_ids,
                valid_length_each_example=valid_length,
                generation_config=self.mindspore_model.config,
                block_tables=block_tables,
                slot_mapping=slot_mapping,
                prefill=self.is_prefill,
                **model_kwargs)
            #logging.info("use_past true mindspore_model res : %s;", res)
            #logging.info("use_past true mindspore_model current_index : %s;", current_index)
        else:
            res = self.mindspore_model(**model_inputs)
        logging.info('predict time is {}'.format((time.time() - predict_time) * 1000))
        #logging.info("mindspore_model res : %s;", res)
        outputs = res[0] if isinstance(res, tuple) else res

        # 222222222222222222222222222222222222222222222222222222222

        if DEBUG_WIN:
            save_path = '../file_npy/'
        else:
            save_path = '/home/ma-user/work/file_npy/'
        if not os.path.isdir(save_path):
            logging.info("Create %s to cache logits" % save_path)
            os.makedirs(save_path)

        fpath = os.path.join(save_path, filename)

        if current_index[0] % 100 == 0:
            if not os.path.exists(fpath):
                with open(fpath, 'wb') as f:
                    np.save(f, outputs.numpy())
            else:
                logging.info("Cache logits path: %s already exists, skipped" % fpath)

        # 222222222222222222222222222222222222222222222222222222222

        return outputs


def start_agent_socket_server(i, cfg: ServingConfig, startup_queue):
    logging.basicConfig(level=logging.INFO,
                        filename=f"./output/agent_{i}.log",
                        filemode='w',
                        format=
                        '%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s')
    """启动agent进程, 由_agent_process进行调用, 创建agent进程"""
    if IMPORT_LITE_FAILED:
        logging.warning("import mindspore_lite failed, using kbk backend.")
    work_agent = WorkAgent(i, cfg)

    agent_ports = cfg.serving_config.agent_ports
    agent_ip = cfg.serving_config.agent_ip
    agent_address = (agent_ip, agent_ports[i])
    print(agent_address)

    parent_process = psutil.Process(os.getppid())
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(agent_address)
    server.listen(5)

    startup_queue.put(i)

    # 绑定method
    print("start agent socket server in rank{}".format(i), flush=True)

    while True:
        if not parent_process.is_running():
            logging.warning(f"detect parent pid={parent_process.pid} has exited, child begin to exit")
            server.close()
            return

        conn, client_addr = server.accept()
        # todo workagent = WorkAgent(config)
        while True:
            if not parent_process.is_running():
                logging.warning(f"detect parent pid={parent_process.pid} has exited, child begin to exit")
                server.close()
                return
            try:
                data = conn.recv(4096)
                if not data: break
                data = data.decode()
                logging.debug(f"data received is {data}")

                # worker 和 agent建联
                if data.startswith('#'):
                    if work_agent.status & AgentStatus.unconnected == AgentStatus.unconnected:
                        data = data[1:]
                        work_agent.shm_names = data.split(",")
                        work_agent.status = AgentStatus.connected
                        print("connect succes")
                        conn.sendall("succes".encode())
                    else:
                        print("connect failed")
                        conn.sendall("failed".encode())
                elif data.startswith('*'):
                    # 全量推理
                    work_agent.is_prefill = True
                    data = data[1:]
                    shape_strs = data.split(",")
                    input_shapes = []
                    for shape_str in shape_strs:
                        shape = list(map(int, shape_str.split(" ")))
                        input_shapes.append(shape)
                    _, _ = work_agent.predict(shape_list=input_shapes)
                    if i == 0:
                        conn.sendall("1".encode())
                elif data.startswith('a'):
                    # 增量推理
                    decode_data = data.split('_')
                    # 增加PA的判断
                    current_batch_dyn = int(decode_data[-4]) if cfg.model_config.page_attention else int(decode_data[-2])
                    batch_valid_flag = []
                    batch_valid = decode_data[-3] if cfg.model_config.page_attention else decode_data[-1]
                    for ele in batch_valid.split(" "):
                        batch_valid_flag.append(int(ele))
                    # 增加 block_tables和slot_mapping 的 shape
                    input_shapes = []
                    if cfg.model_config.page_attention:
                        for shape_str in [decode_data[-2], decode_data[-1]]:
                            shape = list(map(int, shape_str.split(" ")))
                            input_shapes.append(shape)
                    work_agent.is_prefill = False
                    _, _ = work_agent.predict(current_batch=current_batch_dyn, batch_valid_flag=batch_valid_flag, shape_list=input_shapes)
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


def startup_agents_debug_win(config, startup_queue):
    signal.signal(signal.SIGTERM, handler)
    signal.signal(signal.SIGINT, handler)
    log_dir = os.path.join(os.getcwd(), "output")
    print("------------", log_dir)
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    start_agent_socket_server(0, config, startup_queue)


def startup_agents(config, startup_queue):
    if DEBUG_WIN: return startup_agents_debug_win(config, startup_queue)

    signal.signal(signal.SIGTERM, handler)
    signal.signal(signal.SIGINT, handler)
    agent_ports = config.serving_config.agent_ports
    subprocess_list = []
    log_dir = os.path.join(os.getcwd(), "output")
    print("------------", log_dir)
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    for i in range(len(agent_ports)):
        p = Process(target=start_agent_socket_server, args=(i, config, startup_queue))
        p.start()
        subprocess_list.append(p)
    listen_agents_after_startup(subprocess_list)
