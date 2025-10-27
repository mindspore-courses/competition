

# MindSpore 报错解决地图文档映射字典
# 使用方式: docs_mapping = {...}
# MindSpore 大模型报错解决地图文档映射字典
# 使用方式: llm_docs_mapping = {...}

import re


def normalize_text_for_matching(text):
    """
    标准化文本用于模糊匹配，只保留汉字、字母和数字
    """
    if not text:
        return ""
    # 保留汉字(4e00-9fff)、字母(a-zA-Z)、数字(0-9)
    normalized = re.sub(r"[^\u4e00-\u9fff\u3400-\u4dbfa-zA-Z0-9]", "", text)
    return normalized.lower()


llm_docs_mapping = {
    "MindSpore2.2.10使用Flash attention特性报错AttributeError: module 'mindspore.nn'has no attribute'FlashAttention'": "https://www.hiascend.com/developer/blog/details/0239142395981501014",
    "MindSpore保存模型提示：need to checkwhether you is batch size and so on in the 'net' and 'parameter dict' are same.": "https://www.hiascend.com/developer/blog/details/0239142396278591015",
    "MindSpore开启profiler功能报错IndexError:list index out of range": "https://www.hiascend.com/developer/blog/details/0262142396420663009",
    "MindSpore多机运行Profiler报错ValueError: not enough values to unpack (expected 4, got 0)": "https://www.hiascend.com/developer/blog/details/0281142396597298011",
    "MindSpore在yaml文件的callbacks中配置SummaryMonitor后，开启summary功能失效": "https://www.hiascend.com/developer/blog/details/0239142396896763016",
    "MindSpore模型权重功能无法保存更新后的权重": "https://www.hiascend.com/developer/blog/details/0281144493431225096",
    "MindSpore开启MS_DISABLE_REF_MODE导致报错The device address type is wrong:type name in address:CPU,type name in context:Ascend": "https://www.hiascend.com/developer/blog/details/0239144493589693100",
    "MindSpore和tbe版本不匹配问题及解决": "https://www.hiascend.com/developer/blog/details/0262144493684918096",
    "Ascend上构建MindSpore报has no member named 'update output desc dpse' ;did you mean 'update_output_desc_dq'?": "https://www.hiascend.com/developer/blog/details/0281144493807853097",
    "MindSpore开启profile，使用并行策略报错ValueError: When dıstrıbuted loads are slıced we1ghts, sınk mode must be set True.": "https://www.hiascend.com/developer/blog/details/0281144493879398098",
    "模型启动时报Malloc device memory failed": "https://www.hiascend.com/forum/thread-0272155360513957210-1-1.html",
    "MindSpore盘古模型报错Failed to allocate memory.Possible Cause: Available memory is insufficient.": "https://www.hiascend.com/developer/blog/details/02100159778812756004",
    "Ascend910环境分离部署时请求超时": "https://www.hiascend.com/developer/blog/details/0246170061405686062",
    "MindSpore报错ImportError cannot import name \"build dataset loader' from 'mindformers.dataset. dataloader'": "https://www.hiascend.com/developer/blog/details/0262145616108548170",
    "MindSpore大模型报错xxx is not a supported default model or a valid path to checkpoint.": "https://www.hiascend.com/developer/blog/details/0261155358622619198",
    "llama2模型转换报错ImportError: cannot import name 'swap_cache' from 'mindspore._c_expression'": "https://www.hiascend.com/developer/blog/details/0261155360751938201",
    "MindSpore报错BrokenPipeError: Errno 32] Broken pipe": "https://www.hiascend.com/developer/blog/details/0228160995615231019",
    "昇腾910上CodeLlama报错Session options is not equal in diff config infos when models' weights are shared, last session options": "https://www.hiascend.com/developer/blog/details/02114177922488903031",
    "集成通信库初始化init()报错要求使用mpirun启动多卡训练": "https://www.hiascend.com/developer/blog/details/02104167639988674043",
    "INFNAN模式溢出问题": "https://www.hiascend.com/developer/blog/details/0280167638654803068",
    "Mindrecoder 格式转换报错ValueError: For 'Mul'. x.shape and y.shape need to broadcast.": "https://www.hiascend.com/developer/blog/details/0239145616204418169",
    "数据处理成Mindrecord数据时出现ImportError: cannot import name 'tik' from 'te'": "https://www.hiascend.com/developer/blog/details/0261154237556241059",
    "MindSpore数据并行报错Call GE RunGraphWithStreamAsync Failed，EL0004: Failed to allocate memory.": "https://www.hiascend.com/developer/blog/details/0265156071599686298",
    "MindSpore训练报错TypeError: Invalid Kernel Build Info! Kernel type: AICPU_KERNEL, node: Default/Concat-op1": "https://www.hiascend.com/developer/blog/details/0295162714910585084",
    "MindSpore模型正向报错Sync stream failed:Ascend_0，plog显示Gather算子越界": "https://www.hiascend.com/developer/blog/details/02102160470077384050",
    "model.train报错Exception in training: The input value must be int and must > 0, but got '0' with type 'int'.": "https://www.hiascend.com/developer/blog/details/0241162719570943071",
    "数据集异常导致编译(model.build)或者训练(model.train)卡住": "https://www.hiascend.com/developer/blog/details/02105164623402250033",
    "baichaun2-13b 在Ascend910上持续溢出": "https://www.hiascend.com/developer/blog/details/0246170060326623061",
    "MTP数据集分布式读写锁死，Failed to execute the sql SELECT NAME from SHARD NAME;] while verifying meta file, database is locked]": "https://www.hiascend.com/developer/blog/details/02104167637806553041",
    "MTP任务卡死，平台报错信息'ROOT_CLUSTER'] job failed.": "https://www.hiascend.com/developer/blog/details/0296165722495413106",
    "大模型获取性能数据方法": "https://discuss.mindspore.cn/t/topic/190",
    "模型解析性能数据": "https://discuss.mindspore.cn/t/topic/192",
    "数据集处理结果精细对比": "https://discuss.mindspore.cn/t/topic/144",
    "通过优化数据来加速训练速度": "https://discuss.mindspore.cn/t/topic/867",
    "MindSpore分布式模型并行报错：operator Mul init failed或者CheckStrategy failed.": "https://www.hiascend.com/developer/blog/details/0267127625597174071",
    "MindSpore跑模型并行报错ValueError: array split does not result in an equal division": "https://www.hiascend.com/developer/blog/details/0235127625960108068",
    "MindSpore训练大模型报错：BrokenPipeError: Errno 32] Broken pipe, EOFError": "https://www.hiascend.com/developer/blog/details/0275135488550849042",
    "MindSpore大模型并行需要在对应的yaml里面做哪些配置": "https://www.hiascend.com/developer/blog/details/0231150793908483061",
    "模型并行显示内存溢出": "https://www.hiascend.com/forum/thread-0298150793958505076-1-1.html",
    "MindSpore模型Pipeline并行发现有些卡的log中loss为0": "https://www.hiascend.com/developer/blog/details/0222150793995453065",
    "MindSpore8卡报Socket times out问题": "https://www.hiascend.com/developer/blog/details/0222150794121931066",
    "MindSpore模型Pipeline并行训练报错RuntimeError: Stage 0 should has at least 1 parameter. but got none.": "https://www.hiascend.com/developer/blog/details/0237150794406782097",
    "并行策略为8:1:1时报错RuntimeError: May you need to check if the batch size etc. in your 'net' and 'parameter dict' are same.": "https://www.hiascend.com/developer/blog/details/0222151319271723099",
    "模型并行策略为 1:1:8 时报错RuntimeError: Stage num is 8 is not equal to stage used: 5": "https://www.hiascend.com/developer/blog/details/0298151319366478127",
    "单机4卡分布式推理失报错RuntimeError: Ascend kernel runtime initialization failed. The details refer to 'Ascend Error Message'.": "https://www.hiascend.com/forum/thread-0265154245383154065-1-1.html",
    "MindSpore跑分布式报错TypeError: The parameters number of the function is 636, but the number of provided arguments is 635.": "https://www.hiascend.com/developer/blog/details/0261156071523854301",
    "MindSpore分布式8节点报错Call GE RunGraphWithStreamAsync Failed, ret is: 4294967295": "https://www.hiascend.com/developer/blog/details/0272156071756966303",
    "MindFormers进行单机八卡调用时报错No parameter is entered. Notice that the program will run on default 8 cards.": "https://www.hiascend.com/developer/blog/details/0215159162798360070",
    "MindSpore分布式并行报错The strategy is XXX, shape XXX cannot be divisible by strategy value XXX": "https://www.hiascend.com/developer/blog/details/02100160469670839072",
    "MTP使用多进程生成mindrecord，报错RuntimeError: Unexpected error. Internal ERROR] Failed to write mindrecord meta files.": "https://www.hiascend.com/developer/blog/details/02104162718883467066",
    "流水线并行报错Reshape op can't be a border.": "https://www.hiascend.com/developer/blog/details/02105164624450893034",
    "增加数据并行数之后模型占用显存增加": "https://www.hiascend.com/developer/blog/details/02102180946269921004",
    "MindSpore分布式ckpt权重A转换为其他策略的分布式权重B": "https://www.hiascend.com/developer/blog/details/0215170999041958157",
    "流水线并行没开Cell共享导致编译时间很长": "https://www.hiascend.com/developer/blog/details/0255167638982837041",
    "多机训练报错：import torch_npu._C ImportError: libascend_hal.so: cannot open shared object file: No such file or directory": "https://www.hiascend.com/developer/blog/details/0290165512295945073",
    "docker执行报错：RuntimeError: Maybe you are trying to call 'mindspore.communication.init()' without using 'mpirun'": "https://www.hiascend.com/developer/blog/details/02115183802559475007",
    "MindSpore报错：TypeError: Multiply values for specific argument: query_embeds": "https://www.hiascend.com/developer/blog/details/0277135488364790036",
    "Tokenizer指向报错TypeError GPT2Tokenizer: __init__ () missing 2 required positional arguments: 'vocab_file' and \"merges_file": "https://www.hiascend.com/developer/blog/details/0281145616307116183",
    "Tokenizer文件缺失报错TypeError:__init_() missing 2 required positional arguments: 'vocab_file' and 'merge_file'": "https://www.hiascend.com/developer/blog/details/0281145616432621184",
    "模型推理报错RuntimeError A model class needs to define a `prepare inputs fordgeneration` method in order to use .generate()`": "https://www.hiascend.com/developer/blog/details/0239145616859321170",
    "MindSpore模型推理报错：memory isn't enough and alloc failed, kernel name: kernel_graph_@ HostDSActor, alloc size: 8192B": "https://www.hiascend.com/developer/blog/details/0244147496229267003",
    "MindSpore模型推理报错TypeError: cell_reuse() takes 0 positional arguments but 1 was given": "https://www.hiascend.com/developer/blog/details/0293147496373577002",
    "运行wizardcoder迁移代码报错broken pipe": "https://www.hiascend.com/developer/blog/details/0259147496649863001",
    "MindSpore Lite模型加载报错RuntimeError: build from file failed! Error is Common error code.": "https://www.hiascend.com/developer/blog/details/0294147496774074004",
    "Mindspore在自回归推理时的精度对齐设置": "https://www.hiascend.com/developer/blog/details/0244147496914744004",
    "MindSpore大模型打开pp并行或者梯度累积之后loss不溢出也不收敛": "https://www.hiascend.com/developer/blog/details/0297148719627648054",
    "Ascend上用ADGEN数据集评估时报错not support in PyNative RunOp!": "https://www.hiascend.com/developer/blog/details/0297148719847621055",
    "qwen1.5_1.8B推理出现回答混乱问题及解决": "https://www.hiascend.com/developer/blog/details/0272154238185135070",
    "使用单卡Ascend910进行LLaMA2-7B推理,速度缓慢": "https://www.hiascend.com/developer/blog/details/0265154246101034066",
    "MindSpore大模型微调时报溢出及解决": "https://www.hiascend.com/developer/blog/details/0265155357876933195",
    "MindSpore大模型在线推理速度慢及解决方案": "https://www.hiascend.com/developer/blog/details/0286155358239022182",
    "Llama推理报参数校验错误TypeError: The input value must be int. but got 'NoneType.": "https://www.hiascend.com/developer/blog/details/0265155360210516196",
    "MindSpore模型报错Reason: Memory resources are exhausted.": "https://www.hiascend.com/developer/blog/details/0272156071000957302",
    "MindSpore2.2.10 ge图模式报错: Current execute mode is KernelByKernel, the processes must be launched with OpenMPI or ...": "https://www.hiascend.com/developer/blog/details/0265157712779659499",
    "baichuan2-13b算子溢出 loss跑飞问题和定位": "https://www.hiascend.com/developer/blog/details/0205160997110803018",
    "MindSpore训练异常中止：Try to send request before Open()、Try to get response before Open()、Response is empty": "https://www.hiascend.com/developer/blog/details/0204161684348351016",
    "Ascend 910训练脚本刚运行就报错：RuntimeError: Initialize GE failed!": "https://www.hiascend.com/developer/blog/details/0204162714448935072",
    "昇腾910上算子溢出问题分析": "https://www.hiascend.com/developer/blog/details/0284180153149981043",
    "使用ops.nonzero算子报错TypeError": "https://www.hiascend.com/developer/blog/details/02114179635563096169",
    "训练过程中评测超时导致训练过程发生中断": "https://www.hiascend.com/developer/blog/details/0297178618490577115",
    "昇腾910上CodeLlama推理报错get fail deviceLogicId0]": "https://www.hiascend.com/developer/blog/details/02114177922152391030",
    "昇腾910上CodeLlama导出mindir模型报错rankTablePath is invalid": "https://www.hiascend.com/developer/blog/details/0297177921597914049",
    "权重文件被异常修改导致加载权重提示Failed to read the checkpoint file": "https://www.hiascend.com/developer/blog/details/0201177824363228027",
    "Mindformers模型启动时因为host侧OOM导致任务被kill": "https://www.hiascend.com/developer/blog/details/02112175946346217157",
    "昇腾910FlashAttention适配alibi问题": "https://www.hiascend.com/developer/blog/details/02113174898457169049",
    "llama3.1-8b的lora微调，不开启权重转换会导致维度不匹配，开启了之后会报错找不到rank1的ckpt，但是strategy目录里面是全的": "https://www.hiascend.com/developer/blog/details/02111172404490038237",
    "Mindspore+MIndformer训练plog报错halMemAlloc failed，drvRetCode=6": "https://www.hiascend.com/developer/blog/details/02108171017217647137",
    "MindSpore的离线权重转换接口说明及转换过程": "https://www.hiascend.com/developer/blog/details/02111171009711308147",
    "MindSpore的ckpt格式完整权重和分布式权重互转": "https://www.hiascend.com/developer/blog/details/02111171008716808146",
    "MindSpore+MindFormer-r.1.2.0微调qwen1.5 报错": "https://www.hiascend.com/developer/blog/details/0255167836651883049",
    "MIndformer训练plog中算子GatherV2_xxx_high_precision_xx报错": "https://www.hiascend.com/developer/blog/details/0290166325908215125",
    "mindformers进行Lora微调后的权重合并": "https://www.hiascend.com/developer/blog/details/0205183800543322007",
    "pangu-100b 2k集群线性度问题定位": "https://www.hiascend.com/developer/blog/details/0205183803443524009",
    "Mixtral 8*7B 大模型精度问题总结": "https://www.hiascend.com/developer/blog/details/0226183804207627012",
    "大模型内存占用调优": "https://discuss.mindspore.cn/t/topic/99",
    "大模型网络算法参数和输出对比": "https://discuss.mindspore.cn/t/topic/145",
    "使用jit编译加速时编译流程中的常见if控制流问题": "https://discuss.mindspore.cn/t/topic/276",
    "jit编译加速功能开启时，如何避免多次重新编译": "https://discuss.mindspore.cn/t/topic/286",
    "编译时报错ValueError: Please set a unique name for the parameter.": "https://discuss.mindspore.cn/t/topic/292",
    "大模型动态图训练内存优化": "https://discuss.mindspore.cn/t/topic/898",
    "大模型精度收敛分析和调优": "https://discuss.mindspore.cn/t/topic/189",
    "Dump工具应用—算子执行报错输入数据值越界": "https://discuss.mindspore.cn/t/topic/213",
    "Dump工具应用—网络训练溢出": "https://discuss.mindspore.cn/t/topic/214",
    "模型训练长稳性能抖动或劣化问题经验总结": "https://discuss.mindspore.cn/t/topic/250",
    "msprobe工具应用–网络训练溢出": "https://discuss.mindspore.cn/t/topic/297",
    "msprobe精度定位工具常见问题整理": "https://discuss.mindspore.cn/t/topic/211",
    "模型编译的性能优化总结": "https://discuss.mindspore.cn/t/topic/866",
    "大模型动态图训练性能调优指南": "https://discuss.mindspore.cn/t/topic/897",
    "MindSpore权重自动切分后报错ValueError: Failed to read the checkpoint. please check the correct of the file.": "https://www.hiascend.com/developer/blog/details/0255148720028067046",
    "MindSpore2机自动切分模型权重报错NotADirectoryError: ./output/transformed chec kpoint/rank_15 is not a real directory": "https://www.hiascend.com/developer/blog/details/0255148721012536047",
    "Mindspore并行策略下hccl_tools工具使用报错": "https://www.hiascend.com/developer/blog/details/0297148721151570056",
    "MindSpore大模型报错: Inner Error! EZ9999 InferShape] The k-axis of a(131072) and b(16384) tensors must be the same.": "https://www.hiascend.com/developer/blog/details/02104162718590038065",
    "Transformers报错google.protobuf.message.DecodeError: Wrong wire type in tag.": "https://www.hiascend.com/developer/blog/details/0238165721450201098",
    "报错日志不完整": "https://www.hiascend.com/developer/blog/details/0222151319437782100",
    "日志显示没有成功加载预训练模型：model built, but weights is unloaded, since the config has no attribute or is None.": "https://www.hiascend.com/developer/blog/details/0298151319494480128",
    "工作目录问题：'from mindformers import Trainer'报错ModuleNotFoundError:No module named ' mindformers'": "https://www.hiascend.com/developer/blog/details/0237151319576093130",
    "NFS上生成mindrecord报错Failed to write mindrecord meta files": "https://www.hiascend.com/developer/blog/details/0214160473378616064",
    "盘古-智子38B在昇腾910上，greedy模式下无法固定输出": "https://www.hiascend.com/developer/blog/details/0255180944901763004",
    "昇腾上moxing拷贝超时导致Notify超时": "https://www.hiascend.com/developer/blog/details/0275180154904672040",
    "MTP Ascend910切换不同型号设备报错：KeyError：‘group_list’": "https://www.hiascend.com/developer/blog/details/0296165722965762108",
}

docs_mapping = {
    "MindSpore数据集加载-调试小工具 py-spy": "https://www.hiascend.com/developer/blog/details/0229107677003446146",
    "【MindData】如何将自有数据高效的生成MindRecord格式数据集，并且防止爆内存": "https://www.hiascend.com/developer/blog/details/0230106819649046082",
    "【MindSpore Dataset】在使用Dataset处理数据过程中内存占用高，怎么优化？": "https://www.hiascend.com/developer/blog/details/0232106819326340079",
    "如何处理数据集加载多进程(multiprocessing)错误": "https://www.hiascend.com/developer/blog/details/0232107677482846133",
    "MindRecord-Windows下中文路径问题Unexpected error. Failed to open file": "https://www.hiascend.com/developer/blog/details/0232107677596017134",
    "MindRecord数据集格式-Windows下数据集报错Invalid file, DB file can not match": "https://www.hiascend.com/developer/blog/details/0231107677803288124",
    "MindSpore数据集加载-GeneratorDataset卡住、卡死": "https://www.hiascend.com/developer/blog/details/0229107678187952148",
    "MindSpore-GeneratorDataset报错误Unexpected error. Invalid data type.": "https://www.hiascend.com/developer/blog/details/0231107678315400125",
    "MindSpore数据集加载-GeneratorDataset数据处理报错：The pointer cnode is null": "https://www.hiascend.com/developer/blog/details/0230106992306834091",
    "MindSpore数据集报错【The data pipeline is not a tree】": "https://www.hiascend.com/developer/blog/details/0230107678474985121",
    "MindSpore数据增强报错：Use Decode() for encoded data or ToPIL()for decoded data.": "https://www.hiascend.com/developer/blog/details/0232107678668309135",
    "MindSpore自定义数据增强报错【args should be Numpy narray.Got <class 'tuple'>】": "https://www.hiascend.com/developer/blog/details/0230107678833189122",
    "MindSpore数据集加载-GeneratorDataset功能及常见问题": "https://www.hiascend.com/developer/blog/details/0229106992810800100",
    "MindSpore数据加载报错【too many open files】": "https://www.hiascend.com/developer/blog/details/0231107678973789126",
    "MindSpore数据增强报错【TypeError: Invalid with type】": "https://www.hiascend.com/developer/blog/details/0229107679078336149",
    "MindSpore数据集格式报错【MindRecord File could not open successfully】": "https://www.hiascend.com/developer/blog/details/0231107679243990127",
    "MindSpore数据集加载报错【'IdentitySampler' object has no attribute 'child_sampler'】": "https://www.hiascend.com/developer/blog/details/0229107679386960150",
    "MindSpore数据集加载baoc【'DictIterator' has no attribute 'get_next'】": "https://www.hiascend.com/developer/blog/details/0230107679565465123",
    "MindSpore数据集加载报错【IndexError: list index out of range】": "https://www.hiascend.com/developer/blog/details/0232107679694236136",
    "MindSpore数据增强后，内存不足，自动退出": "https://www.hiascend.com/developer/blog/details/0230107679768460124",
    "MindSpore报错RuntimeError: Syntax error. Invalid data, Page size: 1048576 is too small to save a blob row.": "https://www.hiascend.com/developer/blog/details/0231107680001698128",
    "MindSpore报错TypeError: parse() missing 1 required positional argument:'self'": "https://www.hiascend.com/developer/blog/details/0229107794478075160",
    "MindSpore报错RuntimeError: Exception thrown from PyFunc.": "https://www.hiascend.com/developer/blog/details/0232107680321371137",
    "MindSpore报错RuntimeError: Thread ID 140706176251712 Unexpected error.": "https://www.hiascend.com/developer/blog/details/0229107680452655151",
    "MindSpore报错RuntimeError:the size of column_names is:1 and number of returned NumPy array is:2": "https://www.hiascend.com/developer/blog/details/0232107680713769138",
    "MindSpore报错RuntimeError:Invalid data, the number of schema should be positive but got: 0. Please check the input schema.": "https://www.hiascend.com/developer/blog/details/0231107680972273129",
    "图像类型错误导致执行报错：TypeError: img should be PIL image or NumPy array. Got <class 'list'>.": "https://www.hiascend.com/developer/blog/details/0230107681084420126",
    '通道顺序错误引起matplotlib.image.imsave执行报错:raise ValueError"Third dimension must be 3 or 4"': "https://www.hiascend.com/developer/blog/details/0231107681198509130",
    "cv2.imwrite保存Tensor引起类型报错:cv2.error: OpenCV(4.6.0) :-1: error: (-5:Bad argument) in function 'imwrite'": "https://www.hiascend.com/developer/blog/details/0232107681300607140",
    "cv2保存图片类型错误执行报错，由于没有将tensor转换为numpy，导致cv2.imwrite运行失败": "https://www.hiascend.com/developer/blog/details/0232107681441536141",
    "MindSpore报错RuntimeError: Invalid data, the number of schema should be positive but got: 0. Please checkthe input schema.": "https://www.hiascend.com/developer/blog/details/0236108382186972003",
    "MindSpore报错MRMOpenError: MindRecord File could not open successfully.": "https://www.hiascend.com/developer/blog/details/0235108562491930004",
    "MindSpore报错RuntimeError: Invalid python function, the 'source' of 'GeneratorDataset' should return same number ...": "https://www.hiascend.com/developer/blog/details/0235108563799778005",
    "MindSpore报错ValueError: when loss_fn is not None, train_dataset should return two elements, but got 3": "https://www.hiascend.com/developer/blog/details/0236108566643627005",
    "MindSpore数据类型转换结果不符合预期": "https://www.hiascend.com/developer/blog/details/0231110445874171005",
    "MindSpore报错TypeError:parse() missing 1 required positional argument: ‘self’": "https://www.hiascend.com/developer/blog/details/0229107794478075160",
    '使用mindspore.ops.interpolate报错ValueError:For "scale_factor" option cannot currentiy be set with the mode = bilinear = 4D': "https://www.hiascend.com/developer/blog/details/0239126021196615012",
    "mindspore.dataset.Dataset.split切分数据集时randomize=True时分割出的数据不够随机问题": "https://www.hiascend.com/developer/blog/details/0216133277723500037",
    "MindSpore拆分dataset输入给多输入模型": "https://www.hiascend.com/developer/blog/details/0272133348193463049",
    "使用ImageFolderDataset读取图片在进行PIL转化的时候出现报错": "https://www.hiascend.com/developer/blog/details/0274134631995158065",
    "MindSpore如何对使用了自定义采样器的数据集进行分布式采样": "https://www.hiascend.com/developer/blog/details/0275134725805208081",
    "ImageFolderDataset读取图片在进行PIL转化的时候出现报错": "https://www.hiascend.com/developer/blog/details/0274134895783024089",
    "使用MindSpore读取数据报错RuntimeError:Exception thrown from dataset pipeline. Refer to 'Dataset Pipline Error Message'.": "https://www.hiascend.com/developer/blog/details/0284138875727260077",
    "使用MindSpore对vision.SlicePatches的数据集切分和合并": "https://www.hiascend.com/developer/blog/details/0265154928713548138",
    "MindSpore对vision.SlicePatches的数据集切分": "https://www.hiascend.com/developer/blog/details/0272155140603727176",
    "MindSpore调用Dataset.batch()中per_batch_map函数出错": "https://www.hiascend.com/developer/blog/details/0296165508787980087",
    "报错：mindspore._c_expression.typing.TensorType object is not callable": "https://www.hiascend.com/developer/blog/details/0229107244187957128",
    "Ascend print数据落盘使用": "https://www.hiascend.com/developer/blog/details/0231107602436329120",
    "MindSpore在construct中进行标量计算": "https://www.hiascend.com/developer/blog/details/0232111067444997004",
    "MindSpore报错：module 'mindspore.dataset.vision' has no attribute 'Normalize'": "https://www.hiascend.com/developer/blog/details/0263180173583161049",
    "使用Dataset处理数据过程中如何优化内存占用高的问题": "https://discuss.mindspore.cn/t/topic/310",
    "MindSpore报错“RuntimeError: Exceed function call depth limit 1000”": "https://www.hiascend.com/developer/blog/details/0231107350879722104",
    "MindSpore报错:AttributeError:Tensor has no attribute": "https://www.hiascend.com/developer/blog/details/0231107351782306108",
    "MindSpore报错AttributeError:NoneType has no attribute...": "https://www.hiascend.com/developer/blog/details/0231107351017810105",
    "Mindspore 报错:the dimension of logits must be equal to 2, but got 3": "https://www.hiascend.com/developer/blog/details/0230107682360889127",
    "MindSpore报错ValueError: For 'Pad', all elements of paddings must be gt;= 0.": "https://www.hiascend.com/developer/blog/details/0232107682548597142",
    "MindSpore报错ValueError:For xx,the x shape:xx must be equal to xxx": "https://www.hiascend.com/developer/blog/details/0229107682820707152",
    "MindSpore报类型错误TypeError: For 'Tensor', the type of `input_data` should be one of '['Tensor', 'ndarray',]'": "https://www.hiascend.com/developer/blog/details/0229107683010760153",
    "MindSpore报错 ScatterNdUpdate这个算子在Ascend硬件上不支持input0是int32的数据类型": "https://www.hiascend.com/developer/blog/details/0232107351416081120",
    "MindSpore报错TypeError: ScalarAdd不支持bool类型": "https://www.hiascend.com/developer/blog/details/0229107683282903154",
    "MindSpore报RuntimeError:ReduceSum算子不支持8维及以上的输入而报错": "https://www.hiascend.com/developer/blog/details/0229107683436161155",
    "MindSpore报错:Unsupported parameter type for python primitive, the parameter value is KeywordArg key : axis, value:(2,3)": "https://www.hiascend.com/developer/blog/details/0231107750766539138",
    "MindSpore报错RuntimeError:Primitive ScatterAddapos;s bprop not defined": "https://www.hiascend.com/developer/blog/details/0230106904350795086",
    "MindSpore报运行时错误: x shape的C_in除以group应等于weight的C_in": "https://www.hiascend.com/developer/blog/details/0229107751020214157",
    "昇思报错“input_x”形状的乘积应等于“input_shape”的乘积，但“input_x”形状的积为65209，“input_sshape”的积为65308": "https://www.hiascend.com/developer/blog/details/0230107751179845131",
    "MindSpore报错ValueError:x.shape和y.shape不能广播，得到i:-2，x.shapes:[2，5]，y.shape:[3，5]": "https://www.hiascend.com/developer/blog/details/0231107751415909139",
    "MindSpore报错ValueError: For 'MatMul', the input dimensions must be equal, but got 'x1_col': 12800 and 'x2_row': 10": "https://www.hiascend.com/developer/blog/details/0231107752390459140",
    "MindSpore报错：无法在AI CORE或AI CPU内核信息候选列表中为Default/Pow-op0选择有效的内核信息": "https://www.hiascend.com/developer/blog/details/0230107752596874132",
    "MindSpore报错ValueError: `x rank` in `NLLLoss` should be int and must in [1, 2], but got `4` with type `int": "https://www.hiascend.com/developer/blog/details/0229107752780642158",
    "MindSpore报错Select GPU kernel op BatchNorm fail! Incompatible data type!": "https://www.hiascend.com/developer/blog/details/0231107752908306141",
    "MindSpore报错ValueError: For 'MirrorPad', paddings must be a Tensor with type of int64, but got None.": "https://www.hiascend.com/developer/blog/details/0232107753018057147",
    "MindSpore报错 For primitive TensorSummary, the v rank 必须大于等于0": "https://www.hiascend.com/developer/blog/details/0232107764821552149",
    "MindSpore报错TypeError: For 'CellList', each cell should be subclass of Cell, but got NoneType.": "https://www.hiascend.com/developer/blog/details/0230107764935997134",
    "MindSpore报错RuntimeError: Call runtime rtStreamSynchronize failed. Op name: Default/CTCGreedyDecoder-op2": "https://www.hiascend.com/developer/blog/details/0232107765087084150",
    "MindSpore报错ValueError：padding_idx in Embedding超出范围的报错": "https://www.hiascend.com/developer/blog/details/0231107765203397143",
    "MindSpore报错ValueError: seed2 in StandardNormal should be int and must &gt;= 0, but got -3 with type int.": "https://www.hiascend.com/developer/blog/details/0230107765331977135",
    "MindSpore报错 Ascend 环境下ReduceMean不支持8维及其以上的输入": "https://www.hiascend.com/developer/blog/details/0229108037306667164",
    "MindSpore报错： Conv2D第三维输出数据类型必须是正整数或者SHP_ANY, but got -59": "https://www.hiascend.com/developer/blog/details/0229108037447145165",
    "MindSpore报错TypeError: 对于TopK的输入类型必须是int32， float16或者float32， 而实际得到的是float64.": "https://www.hiascend.com/developer/blog/details/0231108037663019150",
    "MindSpore报错 ValueError:Minimum inputs size 0 does not match the requires signature size 2": "https://www.hiascend.com/developer/blog/details/0231108037794118151",
    "MindSpore报错ValueError: Currently half_pixel_centers=True only support in Ascend device_target, but got CPU": "https://www.hiascend.com/developer/blog/details/0230108037901142145",
    "MindSpore报错ValueError:输出形状的每一个值都应该大于零， 实际出现了负数": "https://www.hiascend.com/developer/blog/details/0231108038179836152",
    '昇思报错"The function construct need xx positional argument ..."怎么办': "https://www.hiascend.com/developer/blog/details/0230106556970619074",
    "MindSpore报错算子AddN的输入类型(kNumberTypeBool,kNumberTypeBool)和输出类型kNumberTypeBool不支持": "https://www.hiascend.com/developer/blog/details/0231108038339131153",
    "MindSpore报错should be initialized as a 'Parameter' type in the 'init' function, but got '2.0' with type 'float.": "https://www.hiascend.com/developer/blog/details/0231108038445963154",
    "MindSpore报错RuntimeError: The 'add' operation does not support the type [kMetaTypeNone, Int64].": "https://www.hiascend.com/developer/blog/details/0229108038622012166",
    "MindSpore报错：When eval 'Tensor(self.data, mindspore.float32)' by using Fallback feature": "https://www.hiascend.com/developer/blog/details/0230108038785148146",
    "MindSpore报错RuntimeError: Net parameters weight shape xxx i": "https://www.hiascend.com/developer/blog/details/0229106884746176082",
    "GPU设备算力不足导致计算结果错误cublasGemmEx failed": "https://www.hiascend.com/developer/blog/details/0229106469255191065",
    "GPU环境运行MindSpore报错：设卡失败 SetDevice failed": "https://www.hiascend.com/developer/blog/details/0232107350495289115",
    "Ascend环境使用mindspore报Total stream number xxx exceeds the limit of 1024, secrch details information in mindspore's FAQ.": "https://www.hiascend.com/developer/blog/details/0232108039029264156",
    "LoadTask Distribute Task Failed 报错解决": "https://www.hiascend.com/developer/blog/details/0231108039303484155",
    "GPU训练提示分配流失败cudaStreamCreate failed": "https://www.hiascend.com/developer/blog/details/0229106468010505064",
    "如何处理GPU训练过程中出现内存申请大小为0的错误【The memory alloc size is 0】": "https://www.hiascend.com/developer/blog/details/0231106450074928065",
    "MindSpore报错ValueError: Please input the correct checkpoint": "https://www.hiascend.com/developer/blog/details/0231106884934629079",
    "加载checkpoint的时候报warning日志 quot;xxx parameters in the net are not": "https://www.hiascend.com/developer/blog/details/0230106885093411084",
    "执行时遇到 For context.set_context, package type xxx support devic": "https://www.hiascend.com/developer/blog/details/0229106885219029083",
    "MindSpore PyNative模式下The pointer top_cell_ is null错误": "https://www.hiascend.com/developer/blog/details/0232108039729499157",
    "【MindSpore】Ascend环境运行mindspore脚本报：网络脚本的设备被占用，当前MindSpore框架在Ascend环境只支持每张卡运行一个网络脚本": "https://www.hiascend.com/forum/thread-0231108039894174156-1-1.html",
    "ms报错ValueError: Please input the correct checkpoint": "https://www.hiascend.com/developer/blog/details/0231106884934629079",
    "MindSpore报错ValueError: For 'MatMul', the input dimensions must be equal, but got 'x1_col': 2 and 'x2_row': 1.": "https://www.hiascend.com/developer/blog/details/0231107752390459140",
    "没有ckpt文件导致模型加载执行报错：ckpt does not exist, please check whether the 'ckpt_file_name' is correct.": "https://www.hiascend.com/developer/blog/details/0231108040812361157",
    "Tensor张量shape不匹配导致执行报错：ValueError:x.shape和y.shape不能广播": "https://www.hiascend.com/developer/blog/details/0231108040967734158",
    "自定义loss没有继承nn.Cell导致执行报错：ParseStatement Unsupported statement 'Try'.": "https://www.hiascend.com/developer/blog/details/0231108041178218159",
    "MindSpore:For 'Optimizer',the argument parameters must be Iterable type,but got<class'mindspore.common.tensor.Tensor'>.": "https://www.hiascend.com/developer/blog/details/0230108041455649147",
    "形参与实参的不对应导致ops.GradOperation执行报错：The parameters number of the function is 2, but the number of provided arguments is 1.": "https://www.hiascend.com/developer/blog/details/0231108041570644160",
    "return回来的参数承接问题导致执行报错：AttributeError: 'tuple' object has no attribute 'asnumpy'": "https://www.hiascend.com/developer/blog/details/0231108041767369161",
    "维度数错误引起模型输入错误：For primitive Conv2D, the x shape size must be equal to 4, but got 3.": "https://www.hiascend.com/developer/blog/details/0232108041888051159",
    "MindSpore报错：The value parameter,it's name 'xxxx' already exsts. please set a unique name for the parameter .": "https://www.hiascend.com/developer/blog/details/0231108042042394162",
    "construct方法名称错误引起损失函数执行报错:The 'sub' operation does not support the type TensorFloat32, None.": "https://www.hiascend.com/developer/blog/details/0231107244019807098",
    "MindSpore报错RuntimeError:The sub operation does not support the type TensorFloat32, None.": "https://www.hiascend.com/developer/blog/details/0232108042402327160",
    "注释不当报错：There are incorrect indentations in definition or comment of function: 'Net.construct'.": "https://www.hiascend.com/developer/blog/details/0230108042613165148",
    "静态图执行卡死问题：For MakeTuple, the inputs should not be empty..node:xxx": "https://www.hiascend.com/developer/blog/details/0229106990749902096",
    "MindSpore报错: module() takes at most 2 arguments (3 given)": "https://www.hiascend.com/developer/blog/details/0232108042887341161",
    "For ScatterNdAdd, the 3-th value of indices7 is out of range4scatterNdAdd算子报错解决": "https://www.hiascend.com/developer/blog/details/0213105346170669020",
    "使用ops.nonzero算子报错TypeError: Type Join Failed: dtype1 = Float32, dtype2 = Int64.": "https://www.hiascend.com/developer/blog/details/02114179635563096169",
    "调用MindSpore内部函数时的语法错误TypeError: module object is not callable": "https://www.hiascend.com/developer/blog/details/0227105419215279021",
    "MindSpore在静态图模式下使用try语法报错RuntimeError: Unsupported statement Try.": "https://www.hiascend.com/developer/blog/details/0213105603535368027",
    "报错: ValueError: For 'MatMul', the input dimensions must be equal, but got 'x1_col': 817920 and 'x2_row': 272640.": "https://www.hiascend.com/developer/blog/details/0231106902510576082",
    "MindSpore 报错提示 DropoutGrad 的bprop反向未定义：quot;Illegal primitive: Primitive DropoutGrad’s bprop not defined.quot;": "https://www.hiascend.com/developer/blog/details/0215110357704399023",
    "MindSpore报错The graph generated form MindIR is not support to execute in the PynativeMode,please convert to the GraphMode": "https://www.hiascend.com/developer/blog/details/0231110446520641006",
    "MindSpore报错RuntimeError: For ’Optimizer’, the argument group params must not be empty.": "https://www.hiascend.com/developer/blog/details/0239110447949760007",
    "使用mindspore.ops.MaxPool3D算子设置为ceil_mode=True时，在MindSpore1.8.1和1.9.0版本中计算结果不一致": "https://www.hiascend.com/developer/blog/details/0231110548672670010",
    "Construct内报错和定位解决": "https://www.hiascend.com/developer/blog/details/0212110549624068009",
    "gather算子报错：TypeError以及定位解决": "https://www.hiascend.com/developer/blog/details/0231110549752882011",
    "报错：module() takes at most 2 arguments (3 given)": "https://www.hiascend.com/developer/blog/details/0232108042887341161",
    "MindSpore报错untimeError: Exceed function call depth limit 1000.": "https://www.hiascend.com/developer/blog/details/0223111589074862027",
    "MindSpore图编译报错TypeError: ‘int’ object is not iterable.": "https://www.hiascend.com/developer/blog/details/0241113398877032008",
    "MindSpore报错RuntimeError: The 'getitem' operation does not support the type [Func, Int64].": "https://www.hiascend.com/developer/blog/details/0231112700807230008",
    "MindSpore cpu版本源码编译失败": "https://www.hiascend.com/developer/blog/details/0228112931941430014",
    "MindSpore的Cell.insert_child_to_cell 添加层会出现参数名重复": "https://www.hiascend.com/developer/blog/details/0251114706638241015",
    "mindspore.numpy.unique() 不支持 0 shape tensor": "https://www.hiascend.com/developer/blog/details/0251114779515668017",
    "MindSpore直接将Tensor从布尔值转换为浮点数导致错误Error: IndexError: index 1 is out of bounds for dimension with size 1": "https://www.hiascend.com/developer/blog/details/0251115039544077037",
    "MindSpore中的mindspore.numpy.bincount 大数值情况下报ValueError定位与解决": "https://www.hiascend.com/developer/blog/details/0228115819164415011",
    "使用mindspore.numpy.broadcast_to 算子报错及解决": "https://www.hiascend.com/developer/blog/details/0256117362238941025",
    "使用MindSpore中的SoftMax()算子计算单一数据出错Run op inputs type is invalid!": "https://www.hiascend.com/developer/blog/details/0256118422456486046",
    "CSRTensor 矩阵乘法计算出错RuntimeError:CUDA Error: cudaMemcpy failed.|Error Number: 700 an illegal memory access was encountered": "https://www.hiascend.com/developer/blog/details/0247117440849171022",
    "Cell对象序列化失败-使用pickle.dumps保存到本地后重新加载失败": "https://www.hiascend.com/developer/blog/details/0253120450839270013",
    "TopK算子返回的全零的Tensor的解决": "https://www.hiascend.com/developer/blog/details/0257124701265019040",
    "在NPU上的切片操作x=x[:,::-1,:,:]不生效的分析解决": "https://www.hiascend.com/developer/blog/details/0236124702051337043",
    "张量运算失败报错RuntimeError:Malloc for kernel output failed, Memory isn’t enough": "https://www.hiascend.com/developer/blog/details/0256117871596468038",
    "MindSpore Dump功能使用经验": "https://www.hiascend.com/developer/blog/details/0264125380897665006",
    '使用SymbolTree.get_network处理conv2d算子时报错NameError:name "Cell" is not defined': "https://www.hiascend.com/developer/blog/details/0224126023966377014",
    "使用mindspore中Conv2dTranspose的outputpadding时，设置has_bias=True时失效": "https://www.hiascend.com/developer/blog/details/0239126024840653013",
    "使用mindspore.ops.pad算子报错位置有误": "https://www.hiascend.com/developer/blog/details/0224126176173018016",
    "使用mindspore.numpy.sqrt 计算结果不正确": "https://www.hiascend.com/developer/blog/details/0235126413799683017",
    "函数变换获得梯度计算函数时报错AttributeError: module 'mindspore' has no attribute 'value_and_grad'": "https://www.hiascend.com/developer/blog/details/0267128088161989092",
    "自定义ops.Custom报错TypeError: function output_tensor expects two inputs, but get 1": "https://www.hiascend.com/developer/blog/details/0267128331057078105",
    "报错ValueError: Input buffer_size is not within the required interval of [2, 2147483647].": "https://www.hiascend.com/developer/blog/details/0247128417050944094",
    "使用MindSpore的LayerNorm报错ValueError: For 'LayerNorm', gamma or beta shape must match input shape.": "https://www.hiascend.com/developer/blog/details/0213130200675591020",
    "使用nn.pad报错RuntimeError:For 'Pad', output buffer memset failed": "https://www.hiascend.com/developer/blog/details/0270130207924786004",
    "使用shard接口遇到空指针的报错RuntineError: The pointer [comm_lib_instance_] is null.": "https://www.hiascend.com/developer/blog/details/0225130387213446010",
    "AttributeError: Tensor[Int64] object has no attribute: asnumpy": "https://www.hiascend.com/developer/blog/details/0213130489958864030",
    "使用计算得到的Tensor进行slicing赋值时报错RuntimeError: The int64_t value(-1) is less than 0.": "https://www.hiascend.com/developer/blog/details/0213130490459935031",
    "总loss由多个loss组成时的组合": "https://www.hiascend.com/developer/blog/details/0213130673323759046",
    "使用classmindspore_rl.policy.EpsilonGreedyPolicy发现维度不匹配及解决": "https://www.hiascend.com/developer/blog/details/0213130673553388047",
    "自定义Callback重载函数调用顺序错误及解决": "https://www.hiascend.com/developer/blog/details/0270130695101188042",
    "MindSpore报错：all types should be same, but got mindspore.tensor[float64], mindspore.tensorfloat32": "https://www.hiascend.com/developer/blog/details/0216133001600878009",
    "MindSpore在GRAPH_MODE下初始化，报错提示当前的执行模式是禁用了任务下沉（TASK_SINK）": "https://www.hiascend.com/developer/blog/details/0272133348525646050",
    "使用vision.ToPIL在一定情况下无效": "https://www.hiascend.com/developer/blog/details/0257134813602522075",
    "MindSpore不能像torch的param.grad直接获取梯度问题": "https://www.hiascend.com/developer/blog/details/0276134982626373011",
    "MindSpore跑resnet50报错For 'MatMul' the input dimensions must be equal, but got 'x1_col': 32768 and 'x2_row': 2048": "https://www.hiascend.com/developer/blog/details/0257136262889674011",
    "使用piecewise_constant_lr造成梯度异常": "https://www.hiascend.com/developer/blog/details/0257136274617169013",
    "MindSpore报错AttributeError: module 'mindspore.ops' has no attribute 'mm'": "https://www.hiascend.com/developer/blog/details/0279136275022676011",
    "MindSpore报错'Resize' from mindspore.dataset.vision.c_transforms is deprecated": "https://www.hiascend.com/developer/blog/details/0207136276271855007",
    "MindSpore中的text_format.Merge和text_format.Parse的区别": "https://www.hiascend.com/developer/blog/details/0279136376126281013",
    "MindSpore如何将add_node函数添加节点信息到self.node中": "https://www.hiascend.com/developer/blog/details/0207136376681330011",
    "如何读取MindSpore中的.pb文件中的节点": "https://www.hiascend.com/developer/blog/details/0207136377065182012",
    "MindSpore报错Please try to reduce 'batch_size' or check whether exists extra large shape.及解决": "https://www.hiascend.com/forum/thread-0266136794357565026-1-1.html",
    "MindSpore报错RuntimeError: Load op info form json config failed, version: Ascend310": "https://www.hiascend.com/developer/blog/details/0281136796576917025",
    "使用MindSpore的ops中的矩阵相乘算子进行int8的相乘运算时报错": "https://www.hiascend.com/developer/blog/details/0214138362153680055",
    "使用Mindspore模型训练时出现梯度为0现象": "https://www.hiascend.com/developer/blog/details/0276138897352110086",
    "MindSpore开启summary报错ValueError: not enough values to unpack (expected 4, got 0)": "https://www.hiascend.com/developer/blog/details/0281142396597298011",
    "使用MindSpore实现梯度对数据求导retain_graph=True": "https://www.hiascend.com/developer/blog/details/0272153917054996049",
    "使用SummaryRecord记录计算图报错：Failed to get proto for graph.": "https://www.hiascend.com/developer/blog/details/0265153917951938043",
    "使用MindSpore的initializer生成的Tensor行为不符合预期": "https://www.hiascend.com/developer/blog/details/0261154916382977132",
    "MindSpore的VIT报错[OneHot] failed. OneHot: index values should not bigger than num classes: 100, but got: 100.": "https://www.hiascend.com/developer/blog/details/0272155137773334174",
    "MindSpore使用run_pyscf跑量子化学时报错Invalid cross-device link": "https://www.hiascend.com/developer/blog/details/0286155138206522144",
    "算子编译过程中报错A module that was compiled using NumPy 1.x cannot be run in Numpy 2.0.0 .": "https://www.hiascend.com/developer/blog/details/0286157010903876380",
    "MindSpore神经网络训练中的梯度消失问题": "https://www.hiascend.com/developer/blog/details/0286157125046479394",
    "MindSpore报错：The supported input and output data types for the current operator are: node is Default/BitwiseAnd": "https://www.hiascend.com/developer/blog/details/0286157629062831453",
    "MindSpore模型加载报错RuntimeError: build from file failed! Error is Common error code.": "https://www.hiascend.com/developer/blog/details/0215159160723191068",
    "MindSpore模型转换报错RuntimeError: Can not find key SiLU in convert nap. Exporting SiLU operator is not yet supported.": "https://www.hiascend.com/developer/blog/details/0238165246984594075",
    "MindSpore报错Kernel launch failed, msg: Acl compile and execute failed, op_type_:AvgPool3D": "https://www.hiascend.com/developer/blog/details/0296165511151876088",
    "MindSpore报错“ValueError: For 'MatMul', the input dimensions必须相等": "https://www.hiascend.com/developer/blog/details/0231106902510576082",
    "类型报错： 编译报错，编译时报错 “Shape Join Failed”": "https://www.hiascend.com/developer/blog/details/0230107244246395103",
    "Asttokens版本稳定性性的问题": "https://www.hiascend.com/developer/blog/details/0232107660727925132",
    "MindSpore报错ValueError: x rank in NLLLoss should be int and must in [1, 2], but got 4 with type int": "https://www.hiascend.com/developer/blog/details/0229107752780642158",
    "MindSpore PyNative模式下The pointer[top_cell_] is null错误": "https://www.hiascend.com/developer/blog/details/0232108039729499157",
    "MindSpore报错ERROR:PyNative Only support STAND_ALONE,DATA_PARALLEL and AUTO_PARALLEL under shard function for ParallelMode": "https://www.hiascend.com/developer/blog/details/0227112434275075015",
    "LeNet-5实际应用中报错以及调试过程": "https://www.hiascend.com/developer/blog/details/0215121781780172030",
    "使用自定义数据集运行模型，报错TypeError: The predict type and infer type is not match, predict type is Tuple": "https://www.hiascend.com/developer/blog/details/0231123754196992039",
    "MindSpore图算融合 GPU调试": "https://www.hiascend.com/developer/blog/details/0267128606615143119",
    "MindSpore静态图网络编译使用HyperMap优化编译性能": "https://www.hiascend.com/developer/blog/details/0244128693531775001",
    "MindSpore静态图网络编译使用Select算子优化编译性能": "https://www.hiascend.com/developer/blog/details/0268128785568521009",
    "MindSpore静态图网络编译使用编译缓存或者vmap优化性能": "https://www.hiascend.com/developer/blog/details/0242128787389474014",
    "MindSpore模型权重功能无法保存更新后的权重": "https://www.hiascend.com/developer/blog/details/0281144493431225096",
    "模型微调报错RuntimeError: Preprocess failed before run graph 1.": "https://www.hiascend.com/developer/blog/details/0239144330983413094",
    "Mindspore训练plog中算子GatherV2_xxx_high_precision_xx报错": "https://www.hiascend.com/developer/blog/details/0290166325908215125",
    "使用Profiler()函数，报错RuntimeError: The output path of profiler only supports alphabets(a-zA-Z)": "https://www.hiascend.com/developer/blog/details/0280167835111953075",
    "使用dataset.create_dict_iterator()后，计算前向网络报错：untimeError: Illegal AnfNode for evaluating, node: @Batch": "https://www.hiascend.com/developer/blog/details/0255167820586699045",
    "使用MindSpore静态图速度慢的问题": "https://www.hiascend.com/developer/blog/details/0255167820951696046",
    "MindSpore报错refer to Ascend Error Message": "https://www.hiascend.com/developer/blog/details/0236124622072169038",
    "Ascend上构建MindSpore报has no member named 'update output desc dpse' ;did you mean 'update_output_desc_dq'?": "https://www.hiascend.com/developer/blog/details/0281144493807853097",
    "用ADGEN数据集评估时报错not support in PyNative RunOp!": "https://www.hiascend.com/developer/blog/details/0297148719847621055",
    "使用mint.arctan2在图模式下报错RuntimeError: Compile graph kernel_graph0 failed.": "https://www.hiascend.com/developer/blog/details/0284179715052946002",
    "模型训练时报错RuntimeError: aclnnFlashAttentionScoreGetWorkspaceSize call failed, please check!": "https://www.hiascend.com/developer/blog/details/0292179932426875018",
    "运行MindCV案例报错Malloc for kernel input failed, Memory isn't enough, node:Default/ReduceMean-op0": "https://www.hiascend.com/developer/blog/details/0284179932332373011",
    "使用mindspore.ops.Bernoulli在昇腾设备上训练报错RuntimeError: Sync stream failed:Ascend_0": "https://www.hiascend.com/developer/blog/details/0275179932129252013",
    "MindCV训练报错ValueError: For 'context.set_context', the keyword argument jit_config is notrecognized!": "https://www.hiascend.com/developer/blog/details/0263179713972486001",
    "GRAPH_MODE下运行ms_tensor = mint.ones_like(input_tensor, dtype=dtype)报错The pointer[device_address] is null.": "https://www.hiascend.com/developer/blog/details/02112176366083118210",
    "使用mint.index_select 在图模式下求梯度报错AssertionError": "https://www.hiascend.com/developer/blog/details/0292176350339963195",
    "使用mindspore.mint.where()报错The supported input and output data types for the current operator are: node is Default/Bitwis": "https://www.hiascend.com/developer/blog/details/02113175743294887118",
    "使用mindspore.mint.gather函数计算出的结果错误": "https://www.hiascend.com/developer/blog/details/0292175741600489120",
    "使用Modelarts训练yolov5出现报错TypeError: modelarts_pre_process() missing 1 required positional argument:’args’": "https://www.hiascend.com/developer/blog/details/02108173088031009259",
    "使用mint.masked_select在图模式下报错Parse Lambda Function Fail. Node type must be Lambda, but got Call.": "https://www.hiascend.com/developer/blog/details/0246172980505037280",
    "导入TextClassifier接口报错ModuleNotFoundError: No module named ‘mindnlp.models’": "https://discuss.mindspore.cn/t/topic/821",
    "模型调用Pad接口填充报错For ‘Pad’, output buffer memset failed.": "https://discuss.mindspore.cn/t/topic/805",
    "模型初始化和加载时间过长解决": "https://discuss.mindspore.cn/t/topic/208",
    "静态式下报错TypeError: pynative模式不支持重新计算": "https://discuss.mindspore.cn/t/topic/207",
    "MindSpoer报错：The strategy is ((6, 4), (4,6)), the value of stategy must be the power of 2, but get 6.": "https://www.hiascend.com/developer/blog/details/0232108043454406162",
    "MindSpore并行模式配置报错解决：Parallel mode dose not support": "https://www.hiascend.com/developer/blog/details/0232108043969799163",
    "Ascend多卡训练报错davinci_model : load task fail, return ret xxx": "https://www.hiascend.com/developer/blog/details/0232108044100657164",
    "docker下运行分布式代码报nccl错误：connect returned Connection timed out，成功解决": "https://www.hiascend.com/developer/blog/details/0231108044285551163",
    "MindSpore报错Please try to reduce 'batch_size' or check whether exists extra large shape.": "https://www.hiascend.com/forum/thread-0280136730526815031-1-1.html",
    "MindSpore报错：wq.weight in the argument 'net' should have the same shape as wq.weight in the argument 'parameter_dict'.": "https://www.hiascend.com/developer/blog/details/0265154575900858095",
    "多机训练报错：import torch_npu._C ImportError: libascend_hal.so: cannot open shared object file: No such file or directory": "https://www.hiascend.com/developer/blog/details/0290165512295945073",
    "MindSpore微调qwen1.5 报错AllocDeviceMemByEagerFree failed, alloc size": "https://www.hiascend.com/developer/blog/details/0255167824538077048",
    '使用MindSpore的get_auto_parallel_context("device_num")识别设备信息错误': "https://www.hiascend.com/developer/blog/details/0246171875561164207",
    "docker执行报错：RuntimeError: Maybe you are trying to call 'mindspore.communication.init()' without using 'mpirun'": "https://www.hiascend.com/developer/blog/details/02115183802559475007",
    "Ascend环境运行mindspore脚本报：网络脚本的设备被占用，只支持每张卡运行一个网络脚本": "https://www.hiascend.com/developer/blog/details/0205183952419553026",
    "Mindspore网络精度自动比对功能中protobuf问题分析": "https://www.hiascend.com/developer/blog/details/0225131451875913067",
    "使用mindpsore.nn.conv3d在GPU上精度不足": "https://www.hiascend.com/developer/blog/details/0274134894788575088",
    "使用model仓库的YOLOV5训练没有混合精度配置": "https://www.hiascend.com/developer/blog/details/02113174383145592015",
    "将torch架构的模型迁移到mindspore架构中时精度不一致": "https://discuss.mindspore.cn/t/topic/806",
    "mindspore-Dump功能调试": "https://www.hiascend.com/developer/blog/details/0247126190994430001",
    "PyNative 调试体验": "https://www.hiascend.com/developer/blog/details/0244126193191286002",
    "mindspore之中间文件保存": "https://www.hiascend.com/developer/blog/details/0235126195029329001",
    "随机数生成函数导致模型速度越来越慢": "https://www.hiascend.com/developer/blog/details/0213131112044374072",
    "训练过程中推理精度不变问题定位思路": "https://www.hiascend.com/developer/blog/details/0215111067861669006",
    "MindSpore网络推理时使用Matmul矩阵乘法算子计算速度较慢": "https://www.hiascend.com/developer/blog/details/0210113014840819037",
    "mindspore推理报错NameError:The name 'LTM' is not defined, or not supported in graph mode.": "https://www.hiascend.com/developer/blog/details/0213130695587534051",
    "MindSpore推理报错：Load op info form json config failed, version: Ascend310P3": "https://www.hiascend.com/developer/blog/details/0281136796576917025",
    "使用converter_lite转换包含Dropout算子的模型至MindSpore模型失败": "https://www.hiascend.com/developer/blog/details/0233159162412458078",
    "使用mindsporelite推理，出现data size not equal 错误，tensor size 0": "https://www.hiascend.com/developer/blog/details/0241162719828609072",
    "mindyolo在ckpt模型转为ONNX模型时报错": "https://www.hiascend.com/developer/blog/details/0241162807324189079",
    "MindSpore Lite推理报错RuntimeError: data size not equal! Numpy size: 6144000, Tensor size: 0": "https://www.hiascend.com/developer/blog/details/0238165246338689074",
    "使用MindSpore Lite端侧模型转换工具将YOLOv8.onnx转为.ms报错Convert failed. Ret: Common error code.": "https://www.hiascend.com/developer/blog/details/0296165251233612072",
    "模型推理报错ValueError: For BatchMatMul, inputs shape cannot be broadcast on CPU/GPU.": "https://www.hiascend.com/developer/blog/details/0285141985298637155",
    "MindSpore Lite调用macBert模型报错": "https://www.hiascend.com/developer/blog/details/0255167823803365047",
    "MindSpore Lite模型加载报错RuntimeError: build from file failed! Error is Common error code.": "https://www.hiascend.com/developer/blog/details/0294147496774074004",
    "使用.om格式模型结合gradio框架进行推理出现模型执行错误": "https://www.hiascend.com/developer/blog/details/02112174382197637015",
    "qwen1.5-0.5b推理报错Launch kernel failed, kernel full name: Default/ScatterNdUpdate-op0": "https://www.hiascend.com/developer/blog/details/02112174382197637015",
    "mindformers推理qwen2.5-72b报显存不足及解决": "https://www.hiascend.com/developer/blog/details/0215171874324193227",
    "使用MindSpore将.ckpt转.air再转.om出现AttributeError: 'AclLiteModel' object has no attribute '_is_destroye": "https://www.hiascend.com/developer/blog/details/02108170514150798088",
    "MindSpore报错ValueError: For 'Mul', x.shape and y.shape are supposed to broadcast": "https://www.hiascend.com/developer/blog/details/0230108043108840149",
    "MindSpore报错：The sub operat ion does not support the type kMetaTypeNone, Tensor Float32.": "https://www.hiascend.com/developer/blog/details/0229108043250750168",
    "迁移pytorch代码时如何将torch.device映射 usability/api": "https://www.hiascend.com/developer/blog/details/0228105348053022015",
    "迁移网络tacotron2时遇到backbone中的FPN架构没有nn.ModuleDict": "https://www.hiascend.com/developer/blog/details/0253134471303383051",
    "迁移网络tacotron2时遇到mindspore没有对应torch的tensor.clone接口": "https://www.hiascend.com/developer/blog/details/0253134471639503052",
    "迁移网络tacotron2时遇到mindspore中缺少MultiScaleRoiAlign算子": "https://www.hiascend.com/developer/blog/details/0275134471981878061",
    "迁移网络tacotron2时遇到torch.max、torch.min可以传入2个tensor，但是ops.max不可以": "https://www.hiascend.com/developer/blog/details/0257134472825068048",
    "迁移网络tacotron2时遇到mindsporeAPI binary_cross_entropy_with_logits描述有问题": "https://www.hiascend.com/developer/blog/details/0275134473029992062",
    "迁移网络tacotron2时遇到grad_fn反向求导报错": "https://www.hiascend.com/developer/blog/details/0277134984496923009",
    "迁移网络tacotron2时遇到RuntimeError: The pointer[top_cell_] is null.": "https://www.hiascend.com/developer/blog/details/0276134985116882012",
    "迁移网络tacotron2时遇到Loss损失过高问题": "https://www.hiascend.com/developer/blog/details/0276134986778773013",
    "迁移网络tacotron2时mindspore的权重初始化与torch的不一致": "https://www.hiascend.com/developer/blog/details/0277134989219991012",
    "迁移网络tacotron2时遇到mindspore中没有Tensor.detach()方法及解决": "https://www.hiascend.com/developer/blog/details/0216133272861987036",
    "迁移tacotron2网络到MindSpore时遇到torch.Tensor.new_full()接口缺失": "https://www.hiascend.com/developer/blog/details/0272133273202262044",
    "迁移tacotron2网络到MindSpore时遇到torch.tensor.copy_函数缺失": "https://www.hiascend.com/developer/blog/details/0272133273382307045",
    "迁移tacotron2网络到MindSpore时ops.flip(image, -1)水平翻转图片出现报错": "https://www.hiascend.com/developer/blog/details/0271133273944352026",
    "MindSpore实现多输出模型的loss用LossBase类实现": "https://www.hiascend.com/developer/blog/details/0273133276202603046",
    "迁移网络时转化为静态图的时候报错": "https://www.hiascend.com/developer/blog/details/0275135776188332072",
    "MindSpore实现Swin Transformer时遇到tensor和numpy均不能采用.format经行格式化输出": "https://www.hiascend.com/developer/blog/details/0207136370632894010",
    "MindSpore实现Swin Transformer时遇到ms.common.initializer.Constant(0.0)(m.bias)不起初始化改变数值的作用": "https://www.hiascend.com/developer/blog/details/0279136370811823012",
    "使用MindSpore替换torch.distributions的Categorical函数": "https://www.hiascend.com/developer/blog/details/0239144229019370086",
    "MindSpore报错AttributeError: 'Parameter' object has no attribute 'uniform_'": "https://www.hiascend.com/developer/blog/details/0225144231115143078",
    "MindSpore报错AttributeError: The 'Controller' object has no attribute 'to'.": "https://www.hiascend.com/developer/blog/details/0225144232778629080",
    "如何使用MindSpore实现Torch的logsumexp函数": "https://www.hiascend.com/developer/blog/details/0225144233289736081",
    "使用MindSpore Cell的construct报错AttributeError: For 'Cell', the method 'construct' is not defined.": "https://www.hiascend.com/developer/blog/details/0225144233838227082",
    "使用MindSpore替换PyTorch的torch.nn.init": "https://www.hiascend.com/developer/blog/details/0281144237633248081",
    "使用MindSpore报错AttributeError: 'Parameter' object has no attribute 'uniform_'": "https://www.hiascend.com/developer/blog/details/0225144231115143078",
    "使用MindSpore实现pytorch中的前反向传播": "https://www.hiascend.com/developer/blog/details/0232146671292646041",
    "MindSpore如何实现pytoch中的detach()方法": "https://www.hiascend.com/developer/blog/details/0289146760766805055",
    "MindSpore报错TypeError: init() missing 2 required positional arguments: 'vocab_size' and 'embedding_size'": "https://www.hiascend.com/developer/blog/details/0289147066941139078",
    "使用Mindspore的embedding报错": "https://www.hiascend.com/developer/blog/details/0292147094643791001",
    "使用MindSpore报错TypeError:Invalid dtype": "https://www.hiascend.com/developer/blog/details/0265153918786565044",
    "MindSpore2.3设置了int64后，算子里不会默认更改了": "https://www.hiascend.com/developer/blog/details/0261153920597547038",
    "运行wizardcoder迁移代码报错broken pipe": "https://www.hiascend.com/developer/blog/details/0259147496649863001",
}

# 构建标准化映射表用于模糊匹配
_normalized_llm_mapping = {}
_normalized_docs_mapping = {}

for original_key, url in llm_docs_mapping.items():
    normalized_key = normalize_text_for_matching(original_key)
    _normalized_llm_mapping[normalized_key] = url

for original_key, url in docs_mapping.items():
    normalized_key = normalize_text_for_matching(original_key)
    _normalized_docs_mapping[normalized_key] = url


from typing import Optional, Callable, Awaitable, Dict, AsyncGenerator, Any
from pprint import pprint
import requests
import json
from dataclasses import dataclass
from typing import Dict
from pydantic import BaseModel
import os
from datetime import datetime


@dataclass
class DifySchema:
    """
    Schema for Dify API request
    """

    dify_type: str
    user_input_key: str
    response_mode: str
    user: str = ""

    def get_schema(self) -> Dict:
        if self.dify_type == "workflow":
            return {
                "inputs": {},
                "response_mode": self.response_mode,
                "user": self.user,
            }
        elif self.dify_type == "agent":
            return {
                "inputs": {},
                "query": self.user_input_key,
                "response_mode": self.response_mode,
                "user": self.user,
            }
        elif self.dify_type == "chat":
            return {
                "inputs": {},
                "query": "",
                "response_mode": self.response_mode,
                "user": self.user,
            }
        elif self.dify_type == "completion":
            return {
                "inputs": {},
                "response_mode": self.response_mode,
                "user": self.user,
            }
        else:
            raise ValueError(
                "Invalid dify_type. Must be 'completion', 'workflow', 'agent', or 'chat'"
            )


class Pipe:
    """
    OpenWebUI Pipe → Dify → OpenWebUI
    (with native citation chips via __event_emitter__)
    """

    class Valves(BaseModel):
        HOST_URL: str = "https://api.dify.ai"
        DIFY_API_KEY: str
        USER_INPUT_KEY: str = "input"
        USER_INPUTS: str = "{}"
        DIFY_TYPE: str = "workflow"  # 'workflow' | 'agent' | 'chat' | 'completion'
        RESPONSE_MODE: Optional[str] = "streaming"  # 'streaming' | 'blocking'
        VERIFY_SSL: Optional[bool] = False  # kept for compatibility; NOT used

    def __init__(self):
        self.type = "pipe"
        self.id = "dify_pipeline"
        self.name = "Dify Pipe"
        self.last_emit_time = 0
        # env → valves
        self.valves = self.Valves(
            **{
                "pipelines": ["*"],
                "HOST_URL": os.getenv("HOST_URL", "http://host.docker.internal"),
                "DIFY_API_KEY": os.getenv("DIFY_API_KEY", "YOUR_DIFY_API_KEY"),
                "USER_INPUT_KEY": os.getenv("USER_INPUT_KEY", "input"),
                "USER_INPUTS": (
                    os.getenv("USER_INPUTS") if os.getenv("USER_INPUTS") else "{}"
                ),
                "DIFY_TYPE": os.getenv("DIFY_TYPE", "workflow"),
                "RESPONSE_MODE": os.getenv("RESPONSE_MODE", "streaming"),
                "VERIFY_SSL": False,  # Fixed False as requested
            }
        )
        self.data_schema = DifySchema(
            dify_type=self.valves.DIFY_TYPE,
            user_input_key=self.valves.USER_INPUT_KEY,
            response_mode=self.valves.RESPONSE_MODE,
        ).get_schema()
        self.debug = False

    # --------------- utils ---------------

    def create_api_url(self) -> str:
        if self.valves.DIFY_TYPE == "workflow":
            return f"{self.valves.HOST_URL}/v1/workflows/run"
        elif self.valves.DIFY_TYPE in ("agent", "chat"):
            return f"{self.valves.HOST_URL}/v1/chat-messages"
        elif self.valves.DIFY_TYPE == "completion":
            return f"{self.valves.HOST_URL}/v1/completion-messages"
        else:
            raise ValueError(f"Invalid Dify type: {self.valves.DIFY_TYPE}")

    def set_data_schema(self, schema: dict):
        self.data_schema = schema

    async def on_startup(self):
        if self.debug:
            print(f"on_startup: {__name__}")

    async def on_shutdown(self):
        if self.debug:
            print(f"on_shutdown: {__name__}")

    async def inlet(self, body: dict, user: Optional[dict] = None) -> dict:
        if self.debug:
            print(f"inlet: {__name__} - body:")
            pprint(body)
            print(f"inlet: {__name__} - user:")
            pprint(user)
        return body

    async def outlet(self, body: dict, user: Optional[dict] = None) -> dict:
        if self.debug:
            print(f"outlet: {__name__} - body:")
            pprint(body)
            print(f"outlet: {__name__} - user:")
            pprint(user)
        return body

    # --------------- citation & metadata helpers ---------------

    async def _emit_retriever_citations(self, retriever_resources, __event_emitter__):
        """Emit OpenWebUI-native citation events from Dify metadata.retriever_resources."""
        if not __event_emitter__ or not retriever_resources:
            return

        # Track processed documents to avoid duplicates
        seen_documents = set()

        for i, resource in enumerate(retriever_resources, start=1):
            if not isinstance(resource, dict):
                continue

            # Extract citation info
            idx = resource.get("position") or i
            document_name = (
                resource.get("document_name")
                or resource.get("dataset_name")
                or f"Source {idx}"
            )
            content = resource.get("content", "").strip()

            if not content:
                continue

            # 先计算display_name（去除后缀）
            display_name = (
                document_name[:-3] if document_name.endswith(".md") else document_name
            )

            # 使用display_name进行URL匹配（因为映射字典中的键通常不包含后缀）
            # 首先尝试精确匹配
            url = llm_docs_mapping.get(display_name) or docs_mapping.get(display_name)

            # 如果精确匹配失败，尝试标准化匹配
            if not url:
                normalized_name = normalize_text_for_matching(display_name)
                url = _normalized_llm_mapping.get(
                    normalized_name
                ) or _normalized_docs_mapping.get(normalized_name)

            # 如果仍然匹配失败，使用默认URL
            if not url:
                url = "https://www.hiascend.com/developer/blog"

            # Skip if this document has already been processed
            if display_name in seen_documents:
                continue
            seen_documents.add(display_name)

            # Create citation event using OpenWebUI citation schema
            citation_event = {
                "type": "citation",
                "data": {
                    "document": [f"{url}"],
                    "metadata": [
                        {
                            "date_accessed": datetime.now().isoformat(),
                            "source": display_name,
                        }
                    ],
                    "source": {
                        "name": display_name,
                        **({"url": url} if url else {}),
                    },
                },
            }

            await __event_emitter__(citation_event)

    # --------------- main pipe ---------------

    async def pipe(
        self,
        body: dict,
        __user__: Optional[dict] = None,
        __event_emitter__: Callable[[dict], Awaitable[None]] = None,
        __event_call__: Callable[[dict], Awaitable[dict]] = None,
    ) -> AsyncGenerator[str, None]:

        messages = body.get("messages", [])
        user_message = messages[-1]["content"] if messages else ""

        if self.debug:
            print(f"pipe - incoming message: {user_message}")

        try:
            # headers
            self.headers = {
                "Authorization": f"Bearer {self.valves.DIFY_API_KEY}",
                "Content-Type": "application/json",
            }

            # payload
            data = self.data_schema.copy()
            if self.valves.DIFY_TYPE == "workflow":
                data["inputs"][self.valves.USER_INPUT_KEY] = user_message
            elif self.valves.DIFY_TYPE in ("agent", "chat"):
                data["query"] = user_message
            elif self.valves.DIFY_TYPE == "completion":
                data["inputs"]["query"] = user_message
            data["user"] = (__user__ or {}).get("email", "user")

            # extra inputs
            if self.valves.USER_INPUTS:
                try:
                    inputs_dict = json.loads(self.valves.USER_INPUTS)
                    data["inputs"].update(inputs_dict)
                except Exception:
                    pass

            # request (verify fixed False, as requested)
            response = requests.post(
                self.create_api_url(),
                headers=self.headers,
                json=data,
                verify=False,
                stream=self.valves.RESPONSE_MODE == "streaming",
            )

            if response.status_code != 200:
                yield f"API request failed with status code {response.status_code}: {response.text}"
                return

            # ---------- streaming ----------
            if self.valves.RESPONSE_MODE == "streaming":
                buffer_text, final_output = "", ""
                emitted = False

                for line in response.iter_lines():
                    if not line:
                        continue
                    decoded = line.decode("utf-8")
                    if not decoded.startswith("data: "):
                        continue

                    try:
                        evt = json.loads(decoded[6:])
                    except json.JSONDecodeError:
                        continue

                    et = evt.get("event")

                    if et == "text_chunk":
                        chunk = evt["data"]["text"]
                        buffer_text += chunk
                        yield chunk

                    elif et in ("agent_message", "message", "completion"):
                        chunk = evt.get("answer") or evt.get("data", {}).get("text", "")
                        buffer_text += chunk
                        if chunk:
                            yield chunk

                    elif et == "workflow_finished":
                        final_output = (
                            evt.get("data", {}).get("outputs", {}).get("output", "")
                        )
                        if final_output:
                            yield final_output
                        # do not return; wait for message_end to emit citations

                    elif et == "message_end":
                        # emit retriever_resources-based citations once
                        if not emitted:
                            metadata = evt.get("metadata", {}) or {}
                            retriever_resources = (
                                metadata.get("retriever_resources") or []
                            )
                            await self._emit_retriever_citations(
                                retriever_resources, __event_emitter__
                            )
                            emitted = True
                        # do not return; stream may continue

                    else:
                        # ignore other events (started/node_* etc.)
                        pass

                # stream ended without explicit finish event

                return

        except requests.exceptions.RequestException as e:
            yield f"API request failed: {str(e)}"
