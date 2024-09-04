from data_process import get_precision_str
import json
import numpy as np
import re
import json
import numpy as np
import re
result = np.load('./mindformers/research/result_npy.npy', allow_pickle=True)
result_list = []
for batch in result:
    for sample in batch['text_generation_text']:
        result_list.append(re.findall('-?\d+\/\d+|-?\d+\.?\d*', sample)[-1])
        
        
labels = []
question = []
input_path = './valid-data-list.json'
with open(input_path, 'r', encoding='utf-8') as file:
    for line in file:
        line = json.loads(line)
        question.append(line['problem'])
        labels.append(line['answer'])

result = [get_precision_str(x) if len(re.findall('-?\d+\.?\d*', x))==1 else x for x in result_list]
labels = [get_precision_str(x) if len(re.findall('-?\d+\.?\d*', x))==1 else x for x in labels]
print(np.mean([ x[0].startswith(x[1]) or x[1].startswith(x[0]) for x in list(zip(result, labels))]))