import warnings
import mindspore as ms
import os
from collections import defaultdict
from mindspore import load_param_into_net,ops
import numpy as np
from datasets import init 
from evaluation.exporters import build as build_exporter
from evaluation.metric import build_metric
from vis import DetectionVisualizer


from models import build 
from utils.config import load_config

warnings.filterwarnings("ignore", category=RuntimeWarning)  # 或其他 Warning 子类
os.environ['MS_DEV_JIT_SYNTAX_LEVEL'] = '2' 
ms.set_context(
    jit_syntax_level = ms.STRICT,
    jit_level='O0'  
)

config=load_config('./config/kradar.json')
param_dict= ms.load_checkpoint("adapted_v1.ckpt")
model=build('dpft',config)
load_param_into_net(model,param_dict)


src='./processed_data'
test_dataset = init(dataset=config['dataset'],src=src,split='train', config=config)

inputs=defaultdict(list)
for i in range(6):
    for k,v in test_dataset[i][0].items():
        inputs[k].append(v)
for k,v in inputs.items():
    inputs[k]=ops.stack(v,axis=0)

model.set_train(False)

# ms.export(
#     model,
#     inputs,
#     file_name="20250818_222534_112",
#     file_format="MINDIR"
# )

res=model(inputs)
print("inference completed")

targets=defaultdict(list)
for i in range(6):
    for k,v in test_dataset[i][1].items():
        targets[k].append(v)
for k,v in targets.items():
    targets[k]=ops.stack(v,axis=0)

metric=build_metric(config['evaluate'])
score,idx=metric(res,targets)


sample_detections = {}
for k,v in res.items():
    sample_detections[k]=np.array(v[0,idx[0],:])

sample_detections = [sample_detections]
print(sample_detections)
visualizer = DetectionVisualizer()
test_image = './processed_data/train/1/00033_00001/mono.jpg'
result_image = visualizer.visualize_detections(test_image, sample_detections, "output_3d_detections.jpg")
