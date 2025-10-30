

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
    normalized = re.sub(r"[^\u4E00-\u9FFF\u3400-\u4DBFa-zA-Z]", "", text)
    return normalized.lower()

llm_docs_mapping = {
    "Ascend 训练脚本刚运行就报错RuntimeError Initialize GE failed": "https://discuss.mindspore.cn/t/topic/1173",
    "Ascend上构建MindSpore报has no member named update output desc dpse did you mean updateoutputdescdq": "https://discuss.mindspore.cn/t/topic/1060",
    "Ascend上用ADGEN数据集评估时报错not support in PyNative RunOp": "https://discuss.mindspore.cn/t/topic/1097",
    "Ascend环境分离部署时请求超时": "https://discuss.mindspore.cn/t/topic/1202",
    "Dump工具应用算子执行报错输入数据值越界": "https://discuss.mindspore.cn/t/topic/213",
    "Dump工具应用网络训练溢出": "https://discuss.mindspore.cn/t/topic/214",
    "INFNAN模式溢出问题": "https://discuss.mindspore.cn/t/topic/1180",
    "Llama推理报参数校验错误TypeError The input value must be int but got NoneType": "https://discuss.mindspore.cn/t/topic/1219",
    "MIndformer训练plog中算子GatherVxxxhighprecisionxx报错": "https://discuss.mindspore.cn/t/topic/959",
    "MTP Ascend切换不同型号设备报错KeyErrorgrouplist": "https://discuss.mindspore.cn/t/topic/1213",
    "MTP任务卡死平台报错信息ROOTCLUSTER job failed": "https://discuss.mindspore.cn/t/topic/1249",
    "MTP使用多进程生成mindrecord报错RuntimeError Unexpected error Internal ERROR Failed to write mindrecord meta files": "https://discuss.mindspore.cn/t/topic/1237",
    "MTP数据集分布式读写锁死Failed to execute the sql SELECT NAME from SHARD NAME while verifying meta file database is locked": "https://discuss.mindspore.cn/t/topic/1206",   
    "MindFormers进行单机八卡调用时报错No parameter is entered Notice that the program will run on default  cards": "https://discuss.mindspore.cn/t/topic/1157",
    "MindSpore Lite模型加载报错RuntimeError build from file failed Error is Common error code": "https://discuss.mindspore.cn/t/topic/1084",
    "MindSpore ge图模式报错 Current execute mode is KernelByKernel the processes must be launched with OpenMPI or ": "https://discuss.mindspore.cn/t/topic/1208",
    "MindSporeMindFormerr微调qwen 报错": "https://discuss.mindspore.cn/t/topic/683",    
    "MindSpore使用Flash attention特性报错AttributeError module mindsporennhas no attributeFlashAttention": "https://discuss.mindspore.cn/t/topic/1207",
    "MindSpore保存模型提示need to checkwhether you is batch size and so on in the net and parameter dict are same": "https://discuss.mindspore.cn/t/topic/1212",
    "MindSpore分布式ckpt权重A转换为其他策略的分布式权重B": "https://discuss.mindspore.cn/t/topic/1229",
    "MindSpore分布式并行报错The strategy is XXX shape XXX cannot be divisible by strategy value XXX": "https://discuss.mindspore.cn/t/topic/1157",
    "MindSpore分布式模型并行报错operator Mul init failed或者CheckStrategy failed": "https://discuss.mindspore.cn/t/topic/1257",
    "MindSpore分布式节点报错Call GE RunGraphWithStreamAsync Failed ret is ": "https://discuss.mindspore.cn/t/topic/1158",
    "MindSpore卡报Socket times out问题": "https://discuss.mindspore.cn/t/topic/1238",   
    "MindSpore和tbe版本不匹配问题及解决": "https://discuss.mindspore.cn/t/topic/1214",  
    "MindSpore在yaml文件的callbacks中配置SummaryMonitor后开启summary功能失效": "https://discuss.mindspore.cn/t/topic/1196",
    "MindSpore多机运行Profiler报错ValueError not enough values to unpack expected  got ": "https://discuss.mindspore.cn/t/topic/999",
    "MindSpore大模型在线推理速度慢及解决方案": "https://discuss.mindspore.cn/t/topic/1160",
    "MindSpore大模型并行需要在对应的yaml里面做哪些配置": "https://discuss.mindspore.cn/t/topic/1201",
    "MindSpore大模型微调时报溢出及解决": "https://discuss.mindspore.cn/t/topic/1245",   
    "MindSpore大模型打开pp并行或者梯度累积之后loss不溢出也不收敛": "https://discuss.mindspore.cn/t/topic/1255",
    "MindSpore大模型报错 Inner Error EZ InferShape The kaxis of a and b tensors must be the same": "https://discuss.mindspore.cn/t/topic/1175",
    "MindSpore大模型报错xxx is not a supported default model or a valid path to checkpoint": "https://discuss.mindspore.cn/t/topic/1186",
    "MindSpore开启MSDISABLEREFMODE导致报错The device address type is wrongtype name in addressCPUtype name in contextAscend": "https://discuss.mindspore.cn/t/topic/1214",      
    "MindSpore开启profiler功能报错IndexErrorlist index out of range": "https://discuss.mindspore.cn/t/topic/1203",
    "MindSpore开启profile使用并行策略报错ValueError When dstrbuted loads are slced weghts snk mode must be set True": "https://discuss.mindspore.cn/t/topic/1187",
    "MindSpore报错BrokenPipeError Errno  Broken pipe": "https://discuss.mindspore.cn/t/topic/1256",
    "MindSpore报错ImportError cannot import name build dataset loader from mindformersdataset dataloader": "https://discuss.mindspore.cn/t/topic/1171",
    "MindSpore报错TypeError Multiply values for specific argument queryembeds": "https://discuss.mindspore.cn/t/topic/1200",
    "MindSpore数据并行报错Call GE RunGraphWithStreamAsync FailedEL Failed to allocate memory": "https://discuss.mindspore.cn/t/topic/1253",
    "MindSpore机自动切分模型权重报错NotADirectoryError outputtransformed chec kpointrank is not a real directory": "https://discuss.mindspore.cn/t/topic/1182",
    "MindSpore权重自动切分后报错ValueError Failed to read the checkpoint please check the correct of the file": "https://discuss.mindspore.cn/t/topic/1226",
    "MindSpore模型Pipeline并行发现有些卡的log中loss为": "https://discuss.mindspore.cn/t/topic/1238",
    "MindSpore模型Pipeline并行训练报错RuntimeError Stage  should has at least  parameter but got none": "https://discuss.mindspore.cn/t/topic/1260",
    "MindSpore模型报错Reason Memory resources are exhausted": "https://discuss.mindspore.cn/t/topic/1220",
    "MindSpore模型推理报错TypeError cellreuse takes  positional arguments but  was given": "https://discuss.mindspore.cn/t/topic/1231",
    "MindSpore模型推理报错memory isnt enough and alloc failed kernel name kernelgraph HostDSActor alloc size B": "https://discuss.mindspore.cn/t/topic/1235",
    "MindSpore模型权重功能无法保存更新后的权重": "https://discuss.mindspore.cn/t/topic/942",
    "MindSpore模型正向报错Sync stream failedAscendplog显示Gather算子越界": "https://discuss.mindspore.cn/t/topic/1242",
    "MindSpore的ckpt格式完整权重和分布式权重互转": "https://discuss.mindspore.cn/t/topic/1232",
    "MindSpore的离线权重转换接口说明及转换过程": "https://discuss.mindspore.cn/t/topic/1204",
    "MindSpore盘古模型报错Failed to allocate memoryPossible Cause Available memory is insufficient": "https://discuss.mindspore.cn/t/topic/1236",
    "MindSpore训练大模型报错BrokenPipeError Errno  Broken pipe EOFError": "https://discuss.mindspore.cn/t/topic/1191",
    "MindSpore训练异常中止Try to send request before OpenTry to get response before OpenResponse is empty": "https://discuss.mindspore.cn/t/topic/1241",
    "MindSpore训练报错TypeError Invalid Kernel Build Info Kernel type AICPUKERNEL node DefaultConcatop": "https://discuss.mindspore.cn/t/topic/1240",
    "MindSpore跑分布式报错TypeError The parameters number of the function is  but the number of provided arguments is ": "https://discuss.mindspore.cn/t/topic/1252",
    "MindSpore跑模型并行报错ValueError array split does not result in an equal division": "https://discuss.mindspore.cn/t/topic/1193",
    "Mindformers模型启动时因为host侧OOM导致任务被kill": "https://discuss.mindspore.cn/t/topic/1215",
    "Mindrecoder 格式转换报错ValueError For Mul xshape and yshape need to broadcast": "https://discuss.mindspore.cn/t/topic/1192",
    "MindsporeMIndformer训练plog报错halMemAlloc faileddrvRetCode": "https://discuss.mindspore.cn/t/topic/1244",
    "Mindspore在自回归推理时的精度对齐设置": "https://discuss.mindspore.cn/t/topic/1176",
    "Mindspore并行策略下hccltools工具使用报错": "https://discuss.mindspore.cn/t/topic/1179",
    "Mixtral B 大模型精度问题总结": "https://discuss.mindspore.cn/t/topic/1168",        
    "NFS上生成mindrecord报错Failed to write mindrecord meta files": "https://discuss.mindspore.cn/t/topic/1211",
    "Tokenizer指向报错TypeError GPTTokenizer init  missing  required positional arguments vocabfile and mergesfile": "https://discuss.mindspore.cn/t/topic/1209",
    "Tokenizer文件缺失报错TypeErrorinit missing  required positional arguments vocabfile and mergefile": "https://discuss.mindspore.cn/t/topic/1170",
    "Transformers报错googleprotobufmessageDecodeError Wrong wire type in tag": "https://discuss.mindspore.cn/t/topic/1210",
    "baichaunb 在Ascend上持续溢出": "https://discuss.mindspore.cn/t/topic/1177",        
    "baichuanb算子溢出 loss跑飞问题和定位": "https://discuss.mindspore.cn/t/topic/1221",
    "docker执行报错RuntimeError Maybe you are trying to call mindsporecommunicationinit without using mpirun": "https://discuss.mindspore.cn/t/topic/1216",
    "jit编译加速功能开启时如何避免多次重新编译": "https://discuss.mindspore.cn/t/topic/286",
    "llamab的lora微调不开启权重转换会导致维度不匹配开启了之后会报错找不到rank的ckpt但是strategy目录里面是全的": "https://discuss.mindspore.cn/t/topic/1222",
    "llama模型转换报错ImportError cannot import name swapcache from mindsporecexpression": "https://discuss.mindspore.cn/t/topic/1205",
    "mindformers进行Lora微调后的权重合并": "https://discuss.mindspore.cn/t/topic/1195", 
    "modeltrain报错Exception in training The input value must be int and must   but got  with type int": "https://discuss.mindspore.cn/t/topic/1197",
    "msprobe工具应用网络训练溢出": "https://discuss.mindspore.cn/t/topic/297",
    "msprobe精度定位工具常见问题整理": "https://discuss.mindspore.cn/t/topic/211",      
    "pangub k集群线性度问题定位": "https://discuss.mindspore.cn/t/topic/1239",
    "qwenB推理出现回答混乱问题及解决": "https://discuss.mindspore.cn/t/topic/1161",     
    "使用jit编译加速时编译流程中的常见if控制流问题": "https://discuss.mindspore.cn/t/topic/276",
    "使用opsnonzero算子报错TypeError": "https://discuss.mindspore.cn/t/topic/886",      
    "使用单卡Ascend进行LLaMAB推理速度缓慢": "https://discuss.mindspore.cn/t/topic/1243",
    "单机卡分布式推理失报错RuntimeError Ascend kernel runtime initialization failed The details refer to Ascend Error Message": "https://discuss.mindspore.cn/t/topic/1307",    
    "增加数据并行数之后模型占用显存增加": "https://discuss.mindspore.cn/t/topic/1172",  
    "多机训练报错import torchnpuC ImportError libascendhalso cannot open shared object file No such file or directory": "https://discuss.mindspore.cn/t/topic/682",
    "大模型内存占用调优": "https://discuss.mindspore.cn/t/topic/99",
    "大模型动态图训练内存优化": "https://discuss.mindspore.cn/t/topic/898",
    "大模型动态图训练性能调优指南": "https://discuss.mindspore.cn/t/topic/897",
    "大模型精度收敛分析和调优": "https://discuss.mindspore.cn/t/topic/189",
    "大模型网络算法参数和输出对比": "https://discuss.mindspore.cn/t/topic/145",
    "大模型获取性能数据方法": "https://discuss.mindspore.cn/t/topic/190",
    "工作目录问题from mindformers import Trainer报错ModuleNotFoundErrorNo module named  mindformers": "https://discuss.mindspore.cn/t/topic/1165",
    "并行策略为时报错RuntimeError May you need to check if the batch size etc in your net and parameter dict are same": "https://discuss.mindspore.cn/t/topic/1218",
    "报错日志不完整": "https://discuss.mindspore.cn/t/topic/1166",
    "数据处理成Mindrecord数据时出现ImportError cannot import name tik from te": "https://discuss.mindspore.cn/t/topic/1233",
    "数据集处理结果精细对比": "https://discuss.mindspore.cn/t/topic/144",
    "数据集异常导致编译modelbuild或者训练modeltrain卡住": "https://discuss.mindspore.cn/t/topic/1167",
    "日志显示没有成功加载预训练模型model built but weights is unloaded since the config has no attribute or is None": "https://discuss.mindspore.cn/t/topic/1198",
    "昇腾FlashAttention适配alibi问题": "https://discuss.mindspore.cn/t/topic/1234",     
    "昇腾上CodeLlama导出mindir模型报错rankTablePath is invalid": "https://discuss.mindspore.cn/t/topic/1169",
    "昇腾上CodeLlama报错Session options is not equal in diff config infos when models weights are shared last session options": "https://discuss.mindspore.cn/t/topic/1189",    
    "昇腾上CodeLlama推理报错get fail deviceLogicId": "https://discuss.mindspore.cn/t/topic/1223",
    "昇腾上moxing拷贝超时导致Notify超时": "https://discuss.mindspore.cn/t/topic/1217",  
    "昇腾上算子溢出问题分析": "https://discuss.mindspore.cn/t/topic/1188",
    "权重文件被异常修改导致加载权重提示Failed to read the checkpoint file": "https://discuss.mindspore.cn/t/topic/1159",
    "模型启动时报Malloc device memory failed": "https://discuss.mindspore.cn/t/topic/1304",
    "模型并行显示内存溢出": "https://discuss.mindspore.cn/t/topic/1305",
    "模型并行策略为  时报错RuntimeError Stage num is  is not equal to stage used ": "https://discuss.mindspore.cn/t/topic/1178",
    "模型推理报错RuntimeError A model class needs to define a prepare inputs fordgeneration method in order to use generate": "https://discuss.mindspore.cn/t/topic/1228",      
    "模型编译的性能优化总结": "https://discuss.mindspore.cn/t/topic/866",
    "模型解析性能数据": "https://discuss.mindspore.cn/t/topic/192",
    "模型训练长稳性能抖动或劣化问题经验总结": "https://discuss.mindspore.cn/t/topic/250",
    "流水线并行报错Reshape op cant be a border": "https://discuss.mindspore.cn/t/topic/1199",
    "流水线并行没开Cell共享导致编译时间很长": "https://discuss.mindspore.cn/t/topic/1259",
    "盘古智子B在昇腾上greedy模式下无法固定输出": "https://discuss.mindspore.cn/t/topic/1224",
    "编译时报错ValueError Please set a unique name for the parameter": "https://discuss.mindspore.cn/t/topic/292",
    "训练过程中评测超时导致训练过程发生中断": "https://discuss.mindspore.cn/t/topic/1156",
    "运行wizardcoder迁移代码报错broken pipe": "https://discuss.mindspore.cn/t/topic/1099",
    "通过优化数据来加速训练速度": "https://discuss.mindspore.cn/t/topic/867",
    "集成通信库初始化init报错要求使用mpirun启动多卡训练": "https://discuss.mindspore.cn/t/topic/1254",
}



docs_mapping = {
    "Ascend print数据落盘使用": "https://discuss.mindspore.cn/t/topic/984",
    "Ascend上构建MindSpore报has no member named update output desc dpse did you mean updateoutputdescdq": "https://discuss.mindspore.cn/t/topic/1060",
    "Ascend多卡训练报错davincimodel  load task fail return ret xxx": "https://discuss.mindspore.cn/t/topic/526",
    "Ascend环境使用mindspore报Total stream number xxx exceeds the limit of  secrch details information in mindspores FAQ": "https://discuss.mindspore.cn/t/topic/1038",
    "Ascend环境运行mindspore脚本报网络脚本的设备被占用只支持每张卡运行一个网络脚本": "https://discuss.mindspore.cn/t/topic/1293",
    "Asttokens版本稳定性性的问题": "https://discuss.mindspore.cn/t/topic/905",
    "AttributeError TensorInt object has no attribute asnumpy": "https://discuss.mindspore.cn/t/topic/1124",
    "CSRTensor 矩阵乘法计算出错RuntimeErrorCUDA Error cudaMemcpy failedError Number  an illegal memory access was encountered": "https://discuss.mindspore.cn/t/topic/1125",
    "Cell对象序列化失败使用pickledumps保存到本地后重新加载失败": "https://discuss.mindspore.cn/t/topic/924",
    "Construct内报错和定位解决": "https://discuss.mindspore.cn/t/topic/985",
    "For ScatterNdAdd the th value of indices is out of rangescatterNdAdd算子报错解决": "https://discuss.mindspore.cn/t/topic/1130",
    "GPU环境运行MindSpore报错设卡失败 SetDevice failed": "https://discuss.mindspore.cn/t/topic/957",
    "GPU训练提示分配流失败cudaStreamCreate failed": "https://discuss.mindspore.cn/t/topic/1074",
    "GPU设备算力不足导致计算结果错误cublasGemmEx failed": "https://discuss.mindspore.cn/t/topic/1132",
    "GRAPHMODE下运行mstensor  mintoneslikeinputtensor dtypedtype报错The pointerdeviceaddress is null": "https://discuss.mindspore.cn/t/topic/986",
    "ImageFolderDataset读取图片在进行PIL转化的时候出现报错": "https://discuss.mindspore.cn/t/topic/960",
    "LeNet实际应用中报错以及调试过程": "https://discuss.mindspore.cn/t/topic/958",    
    "LoadTask Distribute Task Failed 报错解决": "https://discuss.mindspore.cn/t/topic/1106",
    "MindCV训练报错ValueError For contextsetcontext the keyword argument jitconfig is notrecognized": "https://discuss.mindspore.cn/t/topic/1126",
    "MindData如何将自有数据高效的生成MindRecord格式数据集并且防止爆内存": "https://discuss.mindspore.cn/t/topic/309",
    "MindRecordWindows下中文路径问题Unexpected error Failed to open file": "https://discuss.mindspore.cn/t/topic/324",
    "MindRecord数据集格式Windows下数据集报错Invalid file DB file can not match": "https://discuss.mindspore.cn/t/topic/325",
    "MindSpoer报错The strategy is    the value of stategy must be the power of  but get ": "https://discuss.mindspore.cn/t/topic/530",
    "MindSpore Dataset在使用Dataset处理数据过程中内存占用高怎么优化": "https://discuss.mindspore.cn/t/topic/310",
    "MindSpore Dump功能使用经验": "https://discuss.mindspore.cn/t/topic/912",
    "MindSpore Lite推理报错RuntimeError data size not equal Numpy size  Tensor size ": "https://discuss.mindspore.cn/t/topic/852",
    "MindSpore Lite模型加载报错RuntimeError build from file failed Error is Common error code": "https://discuss.mindspore.cn/t/topic/1084",
    "MindSpore Lite调用macBert模型报错": "https://discuss.mindspore.cn/t/topic/878",  
    "MindSpore PyNative模式下The pointer topcell is null错误": "https://discuss.mindspore.cn/t/topic/954",
    "MindSpore PyNative模式下The pointertopcell is null错误": "https://discuss.mindspore.cn/t/topic/954",
    "MindSpore cpu版本源码编译失败": "https://discuss.mindspore.cn/t/topic/839",      
    "MindSpore 报错提示 DropoutGrad 的bprop反向未定义quotIllegal primitive Primitive DropoutGrads bprop not definedquot": "https://discuss.mindspore.cn/t/topic/1121",      
    "MindSporeAscend环境运行mindspore脚本报网络脚本的设备被占用当前MindSpore框架在Ascend环境只支持每张卡运行一个网络脚本": "https://discuss.mindspore.cn/t/topic/1293",     
    "MindSporeFor Optimizerthe argument parameters must be Iterable typebut gotclassmindsporecommontensorTensor": "https://discuss.mindspore.cn/t/topic/1119",
    "MindSporeGeneratorDataset报错误Unexpected error Invalid data type": "https://discuss.mindspore.cn/t/topic/327",
    "MindSpore不能像torch的paramgrad直接获取梯度问题": "https://discuss.mindspore.cn/t/topic/875",
    "MindSpore中的mindsporenumpybincount 大数值情况下报ValueError定位与解决": "https://discuss.mindspore.cn/t/topic/963",
    "MindSpore中的textformatMerge和textformatParse的区别": "https://discuss.mindspore.cn/t/topic/907",
    "MindSpore使用runpyscf跑量子化学时报错Invalid crossdevice link": "https://discuss.mindspore.cn/t/topic/955",
    "MindSpore图算融合 GPU调试": "https://discuss.mindspore.cn/t/topic/1148",
    "MindSpore图编译报错TypeError int object is not iterable": "https://discuss.mindspore.cn/t/topic/1065",
    "MindSpore在GRAPHMODE下初始化报错提示当前的执行模式是禁用了任务下沉TASKSINK": "https://discuss.mindspore.cn/t/topic/1056",
    "MindSpore在construct中进行标量计算": "https://discuss.mindspore.cn/t/topic/1035",
    "MindSpore在静态图模式下使用try语法报错RuntimeError Unsupported statement Try": "https://discuss.mindspore.cn/t/topic/1007",
    "MindSpore如何实现pytoch中的detach方法": "https://discuss.mindspore.cn/t/topic/889",
    "MindSpore如何对使用了自定义采样器的数据集进行分布式采样": "https://discuss.mindspore.cn/t/topic/1089",
    "MindSpore如何将addnode函数添加节点信息到selfnode中": "https://discuss.mindspore.cn/t/topic/968",
    "MindSpore实现Swin Transformer时遇到mscommoninitializerConstantmbias不起初始化改变数值的作用": "https://discuss.mindspore.cn/t/topic/993",
    "MindSpore实现Swin Transformer时遇到tensor和numpy均不能采用format经行格式化输出": "https://discuss.mindspore.cn/t/topic/919",
    "MindSpore实现多输出模型的loss用LossBase类实现": "https://discuss.mindspore.cn/t/topic/1096",
    "MindSpore对visionSlicePatches的数据集切分": "https://discuss.mindspore.cn/t/topic/1292",
    "MindSpore并行模式配置报错解决Parallel mode dose not support": "https://discuss.mindspore.cn/t/topic/531",
    "MindSpore开启summary报错ValueError not enough values to unpack expected  got ": "https://discuss.mindspore.cn/t/topic/999",
    "MindSpore微调qwen 报错AllocDeviceMemByEagerFree failed alloc size": "https://discuss.mindspore.cn/t/topic/683",
    "MindSpore报RuntimeErrorReduceSum算子不支持维及以上的输入而报错": "https://discuss.mindspore.cn/t/topic/668",
    "MindSpore报类型错误TypeError For Tensor the type of inputdata should be one of Tensor ndarray": "https://discuss.mindspore.cn/t/topic/665",
    "MindSpore报运行时错误 x shape的Cin除以group应等于weight的Cin": "https://discuss.mindspore.cn/t/topic/1004",
    "MindSpore报错 Ascend 环境下ReduceMean不支持维及其以上的输入": "https://discuss.mindspore.cn/t/topic/995",
    "MindSpore报错 ConvD第三维输出数据类型必须是正整数或者SHPANY but got ": "https://discuss.mindspore.cn/t/topic/1011",
    "MindSpore报错 For primitive TensorSummary the v rank 必须大于等于": "https://discuss.mindspore.cn/t/topic/970",
    "MindSpore报错 ScatterNdUpdate这个算子在Ascend硬件上不支持input是int的数据类型": "https://discuss.mindspore.cn/t/topic/666",
    "MindSpore报错 ValueErrorMinimum inputs size  does not match the requires signature size ": "https://discuss.mindspore.cn/t/topic/945",
    "MindSpore报错 module takes at most  arguments  given": "https://discuss.mindspore.cn/t/topic/883",
    "MindSpore报错AttributeError Parameter object has no attribute uniform": "https://discuss.mindspore.cn/t/topic/871",
    "MindSpore报错AttributeError The Controller object has no attribute to": "https://discuss.mindspore.cn/t/topic/1034",
    "MindSpore报错AttributeError module mindsporeops has no attribute mm": "https://discuss.mindspore.cn/t/topic/836",
    "MindSpore报错AttributeErrorNoneType has no attribute": "https://discuss.mindspore.cn/t/topic/436",
    "MindSpore报错AttributeErrorTensor has no attribute": "https://discuss.mindspore.cn/t/topic/435",
    "MindSpore报错ERRORPyNative Only support STANDALONEDATAPARALLEL and AUTOPARALLEL under shard function for ParallelMode": "https://discuss.mindspore.cn/t/topic/1031",   
    "MindSpore报错Kernel launch failed msg Acl compile and execute failed optypeAvgPoolD": "https://discuss.mindspore.cn/t/topic/1076",
    "MindSpore报错MRMOpenError MindRecord File could not open successfully": "https://discuss.mindspore.cn/t/topic/928",
    "MindSpore报错Please try to reduce batchsize or check whether exists extra large shape": "https://discuss.mindspore.cn/t/topic/1295",
    "MindSpore报错Please try to reduce batchsize or check whether exists extra large shape及解决": "https://discuss.mindspore.cn/t/topic/1294",
    "MindSpore报错Resize from mindsporedatasetvisionctransforms is deprecated": "https://discuss.mindspore.cn/t/topic/1047",
    "MindSpore报错RuntimeError Call runtime rtStreamSynchronize failed Op name DefaultCTCGreedyDecoderop": "https://discuss.mindspore.cn/t/topic/1013",
    "MindSpore报错RuntimeError Exceed function call depth limit ": "https://discuss.mindspore.cn/t/topic/434",
    "MindSpore报错RuntimeError Exception thrown from PyFunc": "https://discuss.mindspore.cn/t/topic/903",
    "MindSpore报错RuntimeError For Optimizer the argument group params must not be empty": "https://discuss.mindspore.cn/t/topic/961",
    "MindSpore报错RuntimeError Invalid data the number of schema should be positive but got  Please checkthe input schema": "https://discuss.mindspore.cn/t/topic/1045",    
    "MindSpore报错RuntimeError Invalid python function the source of GeneratorDataset should return same number ": "https://discuss.mindspore.cn/t/topic/966",
    "MindSpore报错RuntimeError Load op info form json config failed version Ascend": "https://discuss.mindspore.cn/t/topic/716",
    "MindSpore报错RuntimeError Net parameters weight shape xxx i": "https://discuss.mindspore.cn/t/topic/803",
    "MindSpore报错RuntimeError Syntax error Invalid data Page size  is too small to save a blob row": "https://discuss.mindspore.cn/t/topic/881",
    "MindSpore报错RuntimeError The add operation does not support the type kMetaTypeNone Int": "https://discuss.mindspore.cn/t/topic/865",
    "MindSpore报错RuntimeError The getitem operation does not support the type Func Int": "https://discuss.mindspore.cn/t/topic/908",
    "MindSpore报错RuntimeError Thread ID  Unexpected error": "https://discuss.mindspore.cn/t/topic/988",
    "MindSpore报错RuntimeErrorInvalid data the number of schema should be positive but got  Please check the input schema": "https://discuss.mindspore.cn/t/topic/1045",    
    "MindSpore报错RuntimeErrorPrimitive ScatterAddaposs bprop not defined": "https://discuss.mindspore.cn/t/topic/987",
    "MindSpore报错RuntimeErrorThe sub operation does not support the type TensorFloat None": "https://discuss.mindspore.cn/t/topic/1123",
    "MindSpore报错RuntimeErrorthe size of columnnames is and number of returned NumPy array is": "https://discuss.mindspore.cn/t/topic/1083",
    "MindSpore报错Select GPU kernel op BatchNorm fail Incompatible data type": "https://discuss.mindspore.cn/t/topic/1152",
    "MindSpore报错The graph generated form MindIR is not support to execute in the PynativeModeplease convert to the GraphMode": "https://discuss.mindspore.cn/t/topic/939",
    "MindSpore报错The sub operat ion does not support the type kMetaTypeNone Tensor Float": "https://discuss.mindspore.cn/t/topic/774",
    "MindSpore报错The supported input and output data types for the current operator are node is DefaultBitwiseAnd": "https://discuss.mindspore.cn/t/topic/1063",
    "MindSpore报错The value parameterits name xxxx already exsts please set a unique name for the parameter ": "https://discuss.mindspore.cn/t/topic/894",
    "MindSpore报错TypeError For CellList each cell should be subclass of Cell but got NoneType": "https://discuss.mindspore.cn/t/topic/934",
    "MindSpore报错TypeError ScalarAdd不支持bool类型": "https://discuss.mindspore.cn/t/topic/667",
    "MindSpore报错TypeError init missing  required positional arguments vocabsize and embeddingsize": "https://discuss.mindspore.cn/t/topic/974",
    "MindSpore报错TypeError parse missing  required positional argumentself": "https://discuss.mindspore.cn/t/topic/920",
    "MindSpore报错TypeError 对于TopK的输入类型必须是int float或者float 而实际得到的是float": "https://discuss.mindspore.cn/t/topic/991",
    "MindSpore报错TypeErrorparse missing  required positional argument self": "https://discuss.mindspore.cn/t/topic/920",
    "MindSpore报错Unsupported parameter type for python primitive the parameter value is KeywordArg key  axis value": "https://discuss.mindspore.cn/t/topic/1075",
    "MindSpore报错ValueError Currently halfpixelcentersTrue only support in Ascend devicetarget but got CPU": "https://discuss.mindspore.cn/t/topic/856",
    "MindSpore报错ValueError For MatMul the input dimensions must be equal but got xcol  and xrow ": "https://discuss.mindspore.cn/t/topic/874",
    "MindSpore报错ValueError For MatMul the input dimensions必须相等": "https://discuss.mindspore.cn/t/topic/1110",
    "MindSpore报错ValueError For MirrorPad paddings must be a Tensor with type of int but got None": "https://discuss.mindspore.cn/t/topic/1068",
    "MindSpore报错ValueError For Mul xshape and yshape are supposed to broadcast": "https://discuss.mindspore.cn/t/topic/773",
    "MindSpore报错ValueError For Pad all elements of paddings must be gt ": "https://discuss.mindspore.cn/t/topic/438",
    "MindSpore报错ValueError Please input the correct checkpoint": "https://discuss.mindspore.cn/t/topic/1028",
    "MindSpore报错ValueError seed in StandardNormal should be int and must gt  but got  with type int": "https://discuss.mindspore.cn/t/topic/1094",
    "MindSpore报错ValueError when lossfn is not None traindataset should return two elements but got ": "https://discuss.mindspore.cn/t/topic/1061",
    "MindSpore报错ValueError x rank in NLLLoss should be int and must in   but got  with type int": "https://discuss.mindspore.cn/t/topic/873",
    "MindSpore报错ValueErrorFor xxthe x shapexx must be equal to xxx": "https://discuss.mindspore.cn/t/topic/664",
    "MindSpore报错ValueErrorpaddingidx in Embedding超出范围的报错": "https://discuss.mindspore.cn/t/topic/1021",
    "MindSpore报错ValueErrorxshape和yshape不能广播得到ixshapesyshape": "https://discuss.mindspore.cn/t/topic/884",
    "MindSpore报错ValueError输出形状的每一个值都应该大于零 实际出现了负数": "https://discuss.mindspore.cn/t/topic/1064",
    "MindSpore报错When eval Tensorselfdata mindsporefloat by using Fallback feature": "https://discuss.mindspore.cn/t/topic/1057",
    "MindSpore报错all types should be same but got mindsporetensorfloat mindsporetensorfloat": "https://discuss.mindspore.cn/t/topic/1107",
    "MindSpore报错module mindsporedatasetvision has no attribute Normalize": "https://discuss.mindspore.cn/t/topic/860",
    "MindSpore报错refer to Ascend Error Message": "https://discuss.mindspore.cn/t/topic/1128",
    "MindSpore报错should be initialized as a Parameter type in the init function but got  with type float": "https://discuss.mindspore.cn/t/topic/926",
    "MindSpore报错untimeError Exceed function call depth limit ": "https://discuss.mindspore.cn/t/topic/956",
    "MindSpore报错wqweight in the argument net should have the same shape as wqweight in the argument parameterdict": "https://discuss.mindspore.cn/t/topic/528",
    "MindSpore报错无法在AI CORE或AI CPU内核信息候选列表中为DefaultPowop选择有效的内核 信息": "https://discuss.mindspore.cn/t/topic/948",
    "MindSpore报错算子AddN的输入类型kNumberTypeBoolkNumberTypeBool和输出类型kNumberTypeBool不支持": "https://discuss.mindspore.cn/t/topic/870",
    "MindSpore拆分dataset输入给多输入模型": "https://discuss.mindspore.cn/t/topic/936",
    "MindSpore推理报错Load op info form json config failed version AscendP": "https://discuss.mindspore.cn/t/topic/716",
    "MindSpore数据加载报错too many open files": "https://discuss.mindspore.cn/t/topic/358",
    "MindSpore数据增强后内存不足自动退出": "https://discuss.mindspore.cn/t/topic/820",
    "MindSpore数据增强报错TypeError Invalid with type": "https://discuss.mindspore.cn/t/topic/635",
    "MindSpore数据增强报错Use Decode for encoded data or ToPILfor decoded data": "https://discuss.mindspore.cn/t/topic/355",
    "MindSpore数据类型转换结果不符合预期": "https://discuss.mindspore.cn/t/topic/1061",
    "MindSpore数据集加载GeneratorDataset功能及常见问题": "https://discuss.mindspore.cn/t/topic/357",
    "MindSpore数据集加载GeneratorDataset卡住卡死": "https://discuss.mindspore.cn/t/topic/326",
    "MindSpore数据集加载GeneratorDataset数据处理报错The pointer cnode is null": "https://discuss.mindspore.cn/t/topic/328",
    "MindSpore数据集加载baocDictIterator has no attribute getnext": "https://discuss.mindspore.cn/t/topic/638",
    "MindSpore数据集加载报错IdentitySampler object has no attribute childsampler": "https://discuss.mindspore.cn/t/topic/637",
    "MindSpore数据集加载报错IndexError list index out of range": "https://discuss.mindspore.cn/t/topic/639",
    "MindSpore数据集加载调试小工具 pyspy": "https://discuss.mindspore.cn/t/topic/308",
    "MindSpore数据集报错The data pipeline is not a tree": "https://discuss.mindspore.cn/t/topic/354",
    "MindSpore数据集格式报错MindRecord File could not open successfully": "https://discuss.mindspore.cn/t/topic/636",
    "MindSpore模型加载报错RuntimeError build from file failed Error is Common error code": "https://discuss.mindspore.cn/t/topic/1010",
    "MindSpore模型权重功能无法保存更新后的权重": "https://discuss.mindspore.cn/t/topic/942",
    "MindSpore模型转换报错RuntimeError Can not find key SiLU in convert nap Exporting SiLU operator is not yet supported": "https://discuss.mindspore.cn/t/topic/1133",     
    "MindSpore的Cellinsertchildtocell 添加层会出现参数名重复": "https://discuss.mindspore.cn/t/topic/946",
    "MindSpore的VIT报错OneHot failed OneHot index values should not bigger than num classes  but got ": "https://discuss.mindspore.cn/t/topic/1072",
    "MindSpore直接将Tensor从布尔值转换为浮点数导致错误Error IndexError index  is out of bounds for dimension with size ": "https://discuss.mindspore.cn/t/topic/1033",      
    "MindSpore神经网络训练中的梯度消失问题": "https://discuss.mindspore.cn/t/topic/1030",
    "MindSpore网络推理时使用Matmul矩阵乘法算子计算速度较慢": "https://discuss.mindspore.cn/t/topic/714",
    "MindSpore自定义数据增强报错args should be Numpy narrayGot class tuple": "https://discuss.mindspore.cn/t/topic/356",
    "MindSpore设置了int后算子里不会默认更改了": "https://discuss.mindspore.cn/t/topic/859",
    "MindSpore调用Datasetbatch中perbatchmap函数出错": "https://discuss.mindspore.cn/t/topic/922",
    "MindSpore跑resnet报错For MatMul the input dimensions must be equal but got xcol  and xrow ": "https://discuss.mindspore.cn/t/topic/1142",
    "MindSpore静态图网络编译使用HyperMap优化编译性能": "https://discuss.mindspore.cn/t/topic/976",
    "MindSpore静态图网络编译使用Select算子优化编译性能": "https://discuss.mindspore.cn/t/topic/925",
    "MindSpore静态图网络编译使用编译缓存或者vmap优化性能": "https://discuss.mindspore.cn/t/topic/869",
    "Mindspore 报错the dimension of logits must be equal to  but got ": "https://discuss.mindspore.cn/t/topic/437",
    "Mindspore网络精度自动比对功能中protobuf问题分析": "https://discuss.mindspore.cn/t/topic/685",
    "Mindspore训练plog中算子GatherVxxxhighprecisionxx报错": "https://discuss.mindspore.cn/t/topic/959",
    "PyNative 调试体验": "https://discuss.mindspore.cn/t/topic/700",
    "Tensor张量shape不匹配导致执行报错ValueErrorxshape和yshape不能广播": "https://discuss.mindspore.cn/t/topic/888",
    "TopK算子返回的全零的Tensor的解决": "https://discuss.mindspore.cn/t/topic/940",   
    "construct方法名称错误引起损失函数执行报错The sub operation does not support the type TensorFloat None": "https://discuss.mindspore.cn/t/topic/1092",
    "cvimwrite保存Tensor引起类型报错cverror OpenCV  error Bad argument in function imwrite": "https://discuss.mindspore.cn/t/topic/1029",
    "cv保存图片类型错误执行报错由于没有将tensor转换为numpy导致cvimwrite运行失败": "https://discuss.mindspore.cn/t/topic/1117",
    "docker下运行分布式代码报nccl错误connect returned Connection timed out成功解决": "https://discuss.mindspore.cn/t/topic/527",
    "docker执行报错RuntimeError Maybe you are trying to call mindsporecommunicationinit without using mpirun": "https://discuss.mindspore.cn/t/topic/1216",
    "gather算子报错TypeError以及定位解决": "https://discuss.mindspore.cn/t/topic/994",
    "mindformers推理qwenb报显存不足及解决": "https://discuss.mindspore.cn/t/topic/1018",
    "mindsporeDump功能调试": "https://discuss.mindspore.cn/t/topic/699",
    "mindsporedatasetDatasetsplit切分数据集时randomizeTrue时分割出的数据不够随机问题": "https://discuss.mindspore.cn/t/topic/879",
    "mindsporenumpyunique 不支持  shape tensor": "https://discuss.mindspore.cn/t/topic/983",
    "mindspore之中间文件保存": "https://discuss.mindspore.cn/t/topic/702",
    "mindspore推理报错NameErrorThe name LTM is not defined or not supported in graph mode": "https://discuss.mindspore.cn/t/topic/715",
    "mindyolo在ckpt模型转为ONNX模型时报错": "https://discuss.mindspore.cn/t/topic/887",
    "ms报错ValueError Please input the correct checkpoint": "https://discuss.mindspore.cn/t/topic/932",
    "qwenb推理报错Launch kernel failed kernel full name DefaultScatterNdUpdateop": "https://discuss.mindspore.cn/t/topic/1296",
    "return回来的参数承接问题导致执行报错AttributeError tuple object has no attribute asnumpy": "https://discuss.mindspore.cn/t/topic/901",
    "使用Dataset处理数据过程中如何优化内存占用高的问题": "https://discuss.mindspore.cn/t/topic/310",
    "使用ImageFolderDataset读取图片在进行PIL转化的时候出现报错": "https://discuss.mindspore.cn/t/topic/845",
    "使用MindSpore Cell的construct报错AttributeError For Cell the method construct is not defined": "https://discuss.mindspore.cn/t/topic/937",
    "使用MindSpore Lite端侧模型转换工具将YOLOvonnx转为ms报错Convert failed Ret Common error code": "https://discuss.mindspore.cn/t/topic/882",
    "使用MindSpore中的SoftMax算子计算单一数据出错Run op inputs type is invalid": "https://discuss.mindspore.cn/t/topic/913",
    "使用MindSpore实现pytorch中的前反向传播": "https://discuss.mindspore.cn/t/topic/1081",
    "使用MindSpore实现梯度对数据求导retaingraphTrue": "https://discuss.mindspore.cn/t/topic/1023",
    "使用MindSpore对visionSlicePatches的数据集切分和合并": "https://discuss.mindspore.cn/t/topic/877",
    "使用MindSpore将ckpt转air再转om出现AttributeError AclLiteModel object has no attribute isdestroye": "https://discuss.mindspore.cn/t/topic/911",
    "使用MindSpore报错AttributeError Parameter object has no attribute uniform": "https://discuss.mindspore.cn/t/topic/871",
    "使用MindSpore报错TypeErrorInvalid dtype": "https://discuss.mindspore.cn/t/topic/1066",
    "使用MindSpore替换PyTorch的torchnninit": "https://discuss.mindspore.cn/t/topic/1131",
    "使用MindSpore替换torchdistributions的Categorical函数": "https://discuss.mindspore.cn/t/topic/850",
    "使用MindSpore的LayerNorm报错ValueError For LayerNorm gamma or beta shape must match input shape": "https://discuss.mindspore.cn/t/topic/1077",
    "使用MindSpore的getautoparallelcontextdevicenum识别设备信息错误": "https://discuss.mindspore.cn/t/topic/684",
    "使用MindSpore的initializer生成的Tensor行为不符合预期": "https://discuss.mindspore.cn/t/topic/1070",
    "使用MindSpore的ops中的矩阵相乘算子进行int的相乘运算时报错": "https://discuss.mindspore.cn/t/topic/909",
    "使用MindSpore读取数据报错RuntimeErrorException thrown from dataset pipeline Refer to Dataset Pipline Error Message": "https://discuss.mindspore.cn/t/topic/930",       
    "使用MindSpore静态图速度慢的问题": "https://discuss.mindspore.cn/t/topic/1017",   
    "使用Mindspore模型训练时出现梯度为现象": "https://discuss.mindspore.cn/t/topic/902",
    "使用Mindspore的embedding报错": "https://discuss.mindspore.cn/t/topic/917",       
    "使用Modelarts训练yolov出现报错TypeError modelartspreprocess missing  required positional argumentargs": "https://discuss.mindspore.cn/t/topic/981",
    "使用Profiler函数报错RuntimeError The output path of profiler only supports alphabetsazAZ": "https://discuss.mindspore.cn/t/topic/864",
    "使用SummaryRecord记录计算图报错Failed to get proto for graph": "https://discuss.mindspore.cn/t/topic/978",
    "使用SymbolTreegetnetwork处理convd算子时报错NameErrorname Cell is not defined": "https://discuss.mindspore.cn/t/topic/1086",
    "使用classmindsporerlpolicyEpsilonGreedyPolicy发现维度不匹配及解决": "https://discuss.mindspore.cn/t/topic/982",
    "使用converterlite转换包含Dropout算子的模型至MindSpore模型失败": "https://discuss.mindspore.cn/t/topic/717",
    "使用datasetcreatedictiterator后计算前向网络报错untimeError Illegal AnfNode for evaluating node Batch": "https://discuss.mindspore.cn/t/topic/1055",
    "使用mindpsorennconvd在GPU上精度不足": "https://discuss.mindspore.cn/t/topic/686",
    "使用mindsporelite推理出现data size not equal 错误tensor size ": "https://discuss.mindspore.cn/t/topic/1109",
    "使用mindsporemintgather函数计算出的结果错误": "https://discuss.mindspore.cn/t/topic/1005",
    "使用mindsporemintwhere报错The supported input and output data types for the current operator are node is DefaultBitwis": "https://discuss.mindspore.cn/t/topic/977",   
    "使用mindsporenumpybroadcastto 算子报错及解决": "https://discuss.mindspore.cn/t/topic/1003",
    "使用mindsporenumpysqrt 计算结果不正确": "https://discuss.mindspore.cn/t/topic/965",
    "使用mindsporeopsBernoulli在昇腾设备上训练报错RuntimeError Sync stream failedAscend": "https://discuss.mindspore.cn/t/topic/979",
    "使用mindsporeopsMaxPoolD算子设置为ceilmodeTrue时在MindSpore和版本中计算结果不一致": "https://discuss.mindspore.cn/t/topic/1139",
    "使用mindsporeopsinterpolate报错ValueErrorFor scalefactor option cannot currentiy be set with the mode  bilinear  D": "https://discuss.mindspore.cn/t/topic/847",       
    "使用mindsporeopspad算子报错位置有误": "https://discuss.mindspore.cn/t/topic/921",
    "使用mindspore中ConvdTranspose的outputpadding时设置hasbiasTrue时失效": "https://discuss.mindspore.cn/t/topic/1127",
    "使用mintarctan在图模式下报错RuntimeError Compile graph kernelgraph failed": "https://discuss.mindspore.cn/t/topic/947",
    "使用mintindexselect 在图模式下求梯度报错AssertionError": "https://discuss.mindspore.cn/t/topic/953",
    "使用mintmaskedselect在图模式下报错Parse Lambda Function Fail Node type must be Lambda but got Call": "https://discuss.mindspore.cn/t/topic/935",
    "使用model仓库的YOLOV训练没有混合精度配置": "https://discuss.mindspore.cn/t/topic/687",
    "使用nnpad报错RuntimeErrorFor Pad output buffer memset failed": "https://discuss.mindspore.cn/t/topic/952",
    "使用om格式模型结合gradio框架进行推理出现模型执行错误": "https://discuss.mindspore.cn/t/topic/1020",
    "使用opsnonzero算子报错TypeError Type Join Failed dtype  Float dtype  Int": "https://discuss.mindspore.cn/t/topic/886",
    "使用piecewiseconstantlr造成梯度异常": "https://discuss.mindspore.cn/t/topic/972",
    "使用shard接口遇到空指针的报错RuntineError The pointer commlibinstance is null": "https://discuss.mindspore.cn/t/topic/929",
    "使用visionToPIL在一定情况下无效": "https://discuss.mindspore.cn/t/topic/1146",   
    "使用自定义数据集运行模型报错TypeError The predict type and infer type is not match predict type is Tuple": "https://discuss.mindspore.cn/t/topic/1151",
    "使用计算得到的Tensor进行slicing赋值时报错RuntimeError The intt value is less than ": "https://discuss.mindspore.cn/t/topic/980",
    "函数变换获得梯度计算函数时报错AttributeError module mindspore has no attribute valueandgrad": "https://discuss.mindspore.cn/t/topic/814",
    "加载checkpoint的时候报warning日志 quotxxx parameters in the net are not": "https://discuss.mindspore.cn/t/topic/1138",
    "图像类型错误导致执行报错TypeError img should be PIL image or NumPy array Got class list": "https://discuss.mindspore.cn/t/topic/975",
    "在NPU上的切片操作xx不生效的分析解决": "https://discuss.mindspore.cn/t/topic/827",
    "多机训练报错import torchnpuC ImportError libascendhalso cannot open shared object file No such file or directory": "https://discuss.mindspore.cn/t/topic/682",
    "如何使用MindSpore实现Torch的logsumexp函数": "https://discuss.mindspore.cn/t/topic/872",
    "如何处理GPU训练过程中出现内存申请大小为的错误The memory alloc size is ": "https://discuss.mindspore.cn/t/topic/1080",
    "如何处理数据集加载多进程multiprocessing错误": "https://discuss.mindspore.cn/t/topic/311",
    "如何读取MindSpore中的pb文件中的节点": "https://discuss.mindspore.cn/t/topic/1135",
    "导入TextClassifier接口报错ModuleNotFoundError No module named mindnlpmodels": "https://discuss.mindspore.cn/t/topic/821",
    "将torch架构的模型迁移到mindspore架构中时精度不一致": "https://discuss.mindspore.cn/t/topic/806",
    "张量运算失败报错RuntimeErrorMalloc for kernel output failed Memory isnt enough": "https://discuss.mindspore.cn/t/topic/944",
    "形参与实参的不对应导致opsGradOperation执行报错The parameters number of the function is  but the number of provided arguments is ": "https://discuss.mindspore.cn/t/topic/938",
    "总loss由多个loss组成时的组合": "https://discuss.mindspore.cn/t/topic/1062",      
    "执行时遇到 For contextsetcontext package type xxx support devic": "https://discuss.mindspore.cn/t/topic/964",
    "报错 ValueError For MatMul the input dimensions must be equal but got xcol  and xrow ": "https://discuss.mindspore.cn/t/topic/941",
    "报错ValueError Input buffersize is not within the required interval of  ": "https://discuss.mindspore.cn/t/topic/1145",
    "报错mindsporecexpressiontypingTensorType object is not callable": "https://discuss.mindspore.cn/t/topic/969",
    "报错module takes at most  arguments  given": "https://discuss.mindspore.cn/t/topic/883",
    "昇思报错The function construct need xx positional argument 怎么办": "https://discuss.mindspore.cn/t/topic/1120",
    "昇思报错inputx形状的乘积应等于inputshape的乘积但inputx形状的积为inputsshape的积为": "https://discuss.mindspore.cn/t/topic/992",
    "模型初始化和加载时间过长解决": "https://discuss.mindspore.cn/t/topic/208",       
    "模型微调报错RuntimeError Preprocess failed before run graph ": "https://discuss.mindspore.cn/t/topic/915",
    "模型推理报错ValueError For BatchMatMul inputs shape cannot be broadcast on CPUGPU": "https://discuss.mindspore.cn/t/topic/931",
    "模型训练时报错RuntimeError aclnnFlashAttentionScoreGetWorkspaceSize call failed please check": "https://discuss.mindspore.cn/t/topic/819",
    "模型调用Pad接口填充报错For Pad output buffer memset failed": "https://discuss.mindspore.cn/t/topic/805",
    "没有ckpt文件导致模型加载执行报错ckpt does not exist please check whether the ckptfilename is correct": "https://discuss.mindspore.cn/t/topic/890",
    "注释不当报错There are incorrect indentations in definition or comment of function Netconstruct": "https://discuss.mindspore.cn/t/topic/923",
    "用ADGEN数据集评估时报错not support in PyNative RunOp": "https://discuss.mindspore.cn/t/topic/1097",
    "算子编译过程中报错A module that was compiled using NumPy x cannot be run in Numpy  ": "https://discuss.mindspore.cn/t/topic/892",
    "类型报错 编译报错编译时报错 Shape Join Failed": "https://discuss.mindspore.cn/t/topic/1019",
    "维度数错误引起模型输入错误For primitive ConvD the x shape size must be equal to  but got ": "https://discuss.mindspore.cn/t/topic/1147",
    "自定义Callback重载函数调用顺序错误及解决": "https://discuss.mindspore.cn/t/topic/1082",
    "自定义loss没有继承nnCell导致执行报错ParseStatement Unsupported statement Try": "https://discuss.mindspore.cn/t/topic/880",
    "自定义opsCustom报错TypeError function outputtensor expects two inputs but get ": "https://discuss.mindspore.cn/t/topic/1012",
    "训练过程中推理精度不变问题定位思路": "https://discuss.mindspore.cn/t/topic/713", 
    "调用MindSpore内部函数时的语法错误TypeError module object is not callable": "https://discuss.mindspore.cn/t/topic/1112",
    "迁移pytorch代码时如何将torchdevice映射 usabilityapi": "https://discuss.mindspore.cn/t/topic/775",
    "迁移tacotron网络到MindSpore时opsflipimage 水平翻转图片出现报错": "https://discuss.mindspore.cn/t/topic/996",
    "迁移tacotron网络到MindSpore时遇到torchTensornewfull接口缺失": "https://discuss.mindspore.cn/t/topic/900",
    "迁移tacotron网络到MindSpore时遇到torchtensorcopy函数缺失": "https://discuss.mindspore.cn/t/topic/916",
    "迁移网络tacotron时mindspore的权重初始化与torch的不一致": "https://discuss.mindspore.cn/t/topic/1129",
    "迁移网络tacotron时遇到Loss损失过高问题": "https://discuss.mindspore.cn/t/topic/895",
    "迁移网络tacotron时遇到RuntimeError The pointertopcell is null": "https://discuss.mindspore.cn/t/topic/1036",
    "迁移网络tacotron时遇到backbone中的FPN架构没有nnModuleDict": "https://discuss.mindspore.cn/t/topic/776",
    "迁移网络tacotron时遇到gradfn反向求导报错": "https://discuss.mindspore.cn/t/topic/943",
    "迁移网络tacotron时遇到mindsporeAPI binarycrossentropywithlogits描述有问题": "https://discuss.mindspore.cn/t/topic/893",
    "迁移网络tacotron时遇到mindspore中没有Tensordetach方法及解决": "https://discuss.mindspore.cn/t/topic/962",
    "迁移网络tacotron时遇到mindspore中缺少MultiScaleRoiAlign算子": "https://discuss.mindspore.cn/t/topic/1122",
    "迁移网络tacotron时遇到mindspore没有对应torch的tensorclone接口": "https://discuss.mindspore.cn/t/topic/777",
    "迁移网络tacotron时遇到torchmaxtorchmin可以传入个tensor但是opsmax不可以": "https://discuss.mindspore.cn/t/topic/906",
    "迁移网络时转化为静态图的时候报错": "https://discuss.mindspore.cn/t/topic/1113",  
    "运行MindCV案例报错Malloc for kernel input failed Memory isnt enough nodeDefaultReduceMeanop": "https://discuss.mindspore.cn/t/topic/971",
    "运行wizardcoder迁移代码报错broken pipe": "https://discuss.mindspore.cn/t/topic/1099",
    "通道顺序错误引起matplotlibimageimsave执行报错raise ValueErrorThird dimension must be  or ": "https://discuss.mindspore.cn/t/topic/1071",
    "随机数生成函数导致模型速度越来越慢": "https://discuss.mindspore.cn/t/topic/701", 
    "静态图执行卡死问题For MakeTuple the inputs should not be emptynodexxx": "https://discuss.mindspore.cn/t/topic/868",
    "静态式下报错TypeError pynative模式不支持重新计算": "https://discuss.mindspore.cn/t/topic/207",
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
            content = resource.get("content", "")

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
                    "document": [content],
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
