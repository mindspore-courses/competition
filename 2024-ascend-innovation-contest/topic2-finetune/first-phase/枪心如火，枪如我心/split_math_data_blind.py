import os
import re
import json
import pickle
import random
from tqdm import tqdm

random.seed(0)
base_data_path = 'xxx'

def read_data(meta_data_path):
    if isinstance(meta_data_path, list):
        meta_data = []
        for file in meta_data_path:
            meta_data.extend(read_data(file))
    elif os.path.isdir(meta_data_path):
        meta_data = []
        for file in tqdm(os.listdir(meta_data_path)):
            meta_data.extend(read_data(os.path.join(meta_data_path, file)))
    elif meta_data_path.endswith('.json'):
        with open(meta_data_path, "r") as f:
            meta_data = json.load(f)
    elif meta_data_path.endswith('.txt'):
        meta_data = []
        with open(meta_data_path, 'r') as f:
            for line in tqdm(f, desc='读取数据ing'):
                line = line.strip()
                # meta_data.append(line)
                meta_data.append(eval(line))
    elif meta_data_path.endswith('.jsonl'):
        meta_data = []
        with open(meta_data_path, 'r') as f:
            for line in f:
            # for line in tqdm(f, desc='读取数据ing'):
                line = json.loads(line)
                meta_data.append(line)
    elif meta_data_path.endswith('.pkl'):
        with open(meta_data_path, 'rb') as f:
            meta_data = pickle.load(f)
    else:
        meta_data = []
        with open(meta_data_path, 'r') as f:
            for line in f:
                line = line.strip()
                meta_data.append(line)
    return meta_data

def write_data(meta_data_path, data):
    if meta_data_path.endswith('.json'):
        with open(meta_data_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
    elif meta_data_path.endswith('.txt'):
        with open(meta_data_path, 'w', encoding="utf-8") as f:
            for item in data:
                f.write(str(item) + '\n')
    elif meta_data_path.endswith('.jsonl'):
        with open(meta_data_path, 'w', encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

def dedupe_by_key(data, key):
    """
    根据字典中指定的 key 进行去重.
    
    参数:
    data (list of dict) - 需要去重的列表,每个元素都是一个字典.
    key (str) - 用于去重的字典 key.
    
    返回:
    list of dict - 去重后的列表.
    """
    seen = set()
    result = []
    for d in data:
        value = d[key]
        if value not in seen:
            seen.add(value)
            result.append(d)
    return result

def extract_last_number(string):
    # 使用正则表达式匹配字符串中的最后一个数字
    match = re.search(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?$', string)
    
    if match:
        return float(match.group())
    else:
        return None

def get_ops_prompt(num1, num2, op):
    num1 = float(num1)
    num2 = float(num2)
    if num1==0 or num2==0:
        return ''

    num1_sign = '正数' if num1>0 else '负数'
    num2_sign = '正数' if num2>0 else '负数'

    if op=='*' or op=='/':
        if num1_sign==num2_sign:
            result_sign = '正数'
        else:
            result_sign = '负数'
        op = '乘积' if op=='*' else '商'

        prompt = f'第一个数字 {num1} 为{num1_sign}，第二个数字 {num2} 为{num2_sign}，因此两个数字的{op}为{result_sign}。'
    elif op=='+':
        op = '和'
        if num1>0 and num2>0:
            result_sign = '正数'
            prompt = f'第一个数字 {num1} 为{num1_sign}，第二个数字 {num2} 为{num2_sign}，因此两个数字的{op}为{result_sign}。'
        elif num1<0 and num2<0:
            result_sign = '负数'
            prompt = f'第一个数字 {num1} 为{num1_sign}，第二个数字 {num2} 为{num2_sign}，因此两个数字的{op}为{result_sign}。'
        else:
            result_sign = '正数' if num1+num2>0 else '负数' if num1+num2<0 else '0'
            abs_compare = '大于' if abs(num1)>abs(num2) else '小于' if abs(num1)<abs(num2) else '等于'
            prompt = f'第一个数字 {num1} 为{num1_sign}，第二个数字 {num2} 为{num2_sign}，并且第一个数字的绝对值 {abs(num1)} {abs_compare}第二个数字的绝对值 {abs(num2)}，因此两个数字的{op}为 {result_sign}。'
    elif op=='-':
        op = '差'
        if num1>0 and num2<0:
            result_sign = '正数'
            prompt = f'第一个数字 {num1} 为{num1_sign}，第二个数字 {num2} 为{num2_sign}，因此两个数字的{op}为{result_sign}。'
        elif num1<0 and num2>0:
            result_sign = '负数'
            prompt = f'第一个数字 {num1} 为{num1_sign}，第二个数字 {num2} 为{num2_sign}，因此两个数字的{op}为{result_sign}。'
        else:
            result_sign = '正数' if num1-num2>0 else '负数' if num1-num2<0 else '0'
            abs_compare = '大于' if abs(num1)>abs(num2) else '小于' if abs(num1)<abs(num2) else '等于'
            prompt = f'第一个数字 {num1} 为{num1_sign}，第二个数字 {num2} 为{num2_sign}，并且第一个数字的绝对值 {abs(num1)} {abs_compare}第二个数字的绝对值 {abs(num2)}，因此两个数字的{op}为 {result_sign}。'
    return prompt

data = read_data(f'{base_data_path}/train.jsonl')
print(f'length of all data before filter: {len(data)}')
data = dedupe_by_key(data, 'problem')
print(f'length of all data after filter: {len(data)}')

math_templates = {
    "add": [
        "计算 {} + {} 等于多少？",
        "计算 {} + {} 等于多少？",
        "计算 {} + {} 等于多少？",
        "{} + {}的结果是什么？",
        "将{}与{}相加，会得到什么？",
        "现在计算{} + {}的总和。",
        "求出{} + {} 的值。",
        "将{}与{}相加，结果是多少？",
        "计算{} + {}的和。",
        "{} + {}等于多少？"
    ],
    "minus": [
        "计算 {} - {} 等于多少？",
        "计算 {} - {} 等于多少？",
        "计算 {} - {} 等于多少？",
        "{} - {} 的差是多少？",
        "如果从 {} 中减去 {}，会得到什么？",
        "现在求 {} - {} 的差。",
        "求 {} - {} 的值。",
        "将 {} 与 {} 相减，结果是多少？",
        "计算 {} - {} 的差。",
        "{} - {}等于多少？"
    ],
    "multi": [
        "计算 {} * {} 等于多少？",
        "计算 {} * {} 等于多少？",
        "计算 {} * {} 等于多少？",
        "{} * {}的积是多少？",
        "如果 {} 与 {} 相乘，会得到什么？",
        "现在计算 {} 和 {} 的乘积。",
        "求 {} * {} 的值。",
        "将 {} 与 {} 相乘，结果是多少？",
        "计算 {} * {} 的乘积。",
        "{} * {} 等于多少？"
    ],
    "div": [
        "计算 {} / {} 等于多少？",
        "计算 {} / {} 等于多少？",
        "计算 {} / {} 等于多少？",
        "计算 {} / {} 的结果是什么？",
        "{} / {} 的商是多少？",
        "现在求 {} / {} 的商。",
        "求 {} / {} 的值。",
        "将 {} 除以 {} ，结果是多少？",
        "计算 {} / {} 的商。",
        "{} / {} 等于多少？"
    ]
}

equation_templates = {
    "problem": [
        "解方程 {}x + {} = {}",
        "解方程 {}x + {} = {}",
        "方程 {}x + {} = {} 的解是什么？",
        "找出方程 {}x + {} = {} 的解",
        "求解方程 {}x + {} = {} 的根",
        "计算方程 {}x + {} = {} 的解",
        "对方程 {}x + {} = {} 进行求解"
    ],
    "solution": [
        "方程的解为：{}",
        "得到的解是：{}",
        "方程的根为：{}",
        "解得：{}",
        "方程的解是：{}",
        "求解方程后得到：{}",
        "最终解为：{}"
    ]
}

equation_data = []
add_data = []
minus_data = []
multi_data = []
div_data = []
complex_data = []

for item in data:
    pattern = r'^解方程\s([-+]?\d+)x\s\+\s([-+]?\d+)\s=\s(\d+)$'
    match = re.match(pattern, item['problem'])
    item['initial_problem'] = item['problem']
    item['initial_solution'] = item['solution']
    if match:
        numbers = match.groups()
        item['problem'] = random.choice(equation_templates['problem']).format(numbers[0], numbers[1], numbers[2])
        output_number = item['solution'].split('：')[1]
        item['solution'] = f'方程的解为：{round(float(output_number), 1)}'
        # item['solution'] = random.choice(equation_templates['solution']).format(round(float(output_number), 1))
        equation_data.append(item)
        continue
    
    pattern = r'^计算\s([-+]?\d+(?:\.\d+)?)\s\+\s([-+]?\d+(?:\.\d+)?) 等于多少？$'
    match = re.match(pattern, item['problem'])
    if match:
        numbers = match.groups()
        item['problem'] = random.choice(math_templates['add']).format(numbers[0], numbers[1])
        input_number, output_number = item['solution'].split('=')
        ops_prompt = get_ops_prompt(numbers[0], numbers[1], '+')
        # item['solution'] = f'{ops_prompt}计算结果为：{output_number.strip()}'
        item['solution'] = f'计算结果为：{output_number.strip()}'
        add_data.append(item)
        continue
    
    pattern = r'^计算\s([-+]?\d+(?:\.\d+)?)\s\-\s([-+]?\d+(?:\.\d+)?) 等于多少？$'
    match = re.match(pattern, item['problem'])
    if match:
        numbers = match.groups()
        item['problem'] = random.choice(math_templates['minus']).format(numbers[0], numbers[1])
        input_number, output_number = item['solution'].split('=')
        ops_prompt = get_ops_prompt(numbers[0], numbers[1], '-')
        # item['solution'] = f'{ops_prompt}计算结果为：{output_number.strip()}'
        item['solution'] = f'计算结果为：{output_number.strip()}'
        minus_data.append(item)
        continue
    
    pattern = r'^计算\s([-+]?\d+(?:\.\d+)?)\s\*\s([-+]?\d+(?:\.\d+)?) 等于多少？$'
    match = re.match(pattern, item['problem'])
    if match:
        numbers = match.groups()
        item['problem'] = random.choice(math_templates['multi']).format(numbers[0], numbers[1])
        input_number, output_number = item['solution'].split('=')
        ops_prompt = get_ops_prompt(numbers[0], numbers[1], '*')
        item['solution'] = f'{ops_prompt}计算结果为：{output_number.strip()}'
        # item['solution'] = f'计算结果为：{round(float(output_number), 0)}'
        multi_data.append(item)
        continue
    
    pattern = r'^计算\s([-+]?\d+(?:\.\d+)?)\s\/\s([-+]?\d+(?:\.\d+)?) 等于多少？$'
    match = re.match(pattern, item['problem'])
    if match:
        numbers = match.groups()
        item['problem'] = random.choice(math_templates['div']).format(numbers[0], numbers[1])
        input_number, output_number = item['solution'].split('=')
        ops_prompt = get_ops_prompt(numbers[0], numbers[1], '/')
        item['solution'] = f'{ops_prompt}计算结果为：{round(float(output_number), 1)}'
        # item['solution'] = f'计算结果为：{round(float(output_number), 1)}'
        div_data.append(item)
        continue
    
    complex_data.append(item)

print(f'length of equation data: {len(equation_data)}')
print(f'length of add data: {len(add_data)}')
print(f'length of minus data: {len(minus_data)}')
print(f'length of multi data: {len(multi_data)}')
print(f'length of div data: {len(div_data)}')
print(f'length of complex data: {len(complex_data)}')

write_data(f'{base_data_path}/equation_data.jsonl', equation_data)
write_data(f'{base_data_path}/add_data.jsonl', add_data)
write_data(f'{base_data_path}/minus_data.jsonl', minus_data)
write_data(f'{base_data_path}/multi_data.jsonl', multi_data)
write_data(f'{base_data_path}/div_data.jsonl', div_data)
# import ipdb;ipdb.set_trace()

train_sample_num = 9525
val_sample_num = 100
total_sample_num = train_sample_num + val_sample_num

data_dict = {
    'equation_data': random.sample(equation_data, total_sample_num),
    'add_data': random.sample(add_data, total_sample_num),
    'minus_data': random.sample(minus_data, total_sample_num),
    # 'multi_data': random.sample(multi_data, total_sample_num),
    'div_data': random.sample(div_data, total_sample_num),
}

train_data = []
val_data = []

for single_data in data_dict.values():
    train_data += single_data[:train_sample_num]
    val_data += single_data[train_sample_num:total_sample_num]

complex_data_dict = {}
for item in complex_data:
    complex_data_dict[item['problem']] = extract_last_number(item['solution'])
complex_cot_data = read_data(f'{base_data_path}/complex_data_cot')
complex_cot_data = dedupe_by_key(complex_cot_data, 'problem')
complex_cot_data_filtered = []
yi_ci_fang = 0
for item in complex_cot_data:
    if complex_data_dict[item['problem']] is None:
        continue
    if str(complex_data_dict[item['problem']]) in item['solution']:
        if '1 次方' in item['problem']:
            if yi_ci_fang > 10:
                continue
            else:
                yi_ci_fang += 1
                complex_cot_data_filtered.append(item)
        else:
            complex_cot_data_filtered.append(item)
write_data(f'{base_data_path}/complex_cot_data_filtered.jsonl', complex_cot_data_filtered)

def get_sublist_by_key(origin_list, key, sample_num, reformat=False):
    res = [x for x in origin_list if key in x['problem']]
    if sample_num > len(res):
        print(f'数据量为{len(res)}, 不足要采样的数量{sample_num}, 取全部数据')
    else:
        res = random.sample(res, sample_num)

    if key == '打折':
        temp = []
        for x in res:
            match = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', x['problem'])
            if float(match[0]) >= float(match[1]):
                solution = f'商品原价 {match[0]} 大于打折后的价格 {match[1]}，因此折扣比例为正数，结果为{x["solution"]}'
            else:
                solution = f'商品原价 {match[0]} 小于打折后的价格 {match[1]}，因此折扣比例为负数，结果为{x["solution"]}'
            temp.append({
                'problem': x['problem'],
                'solution': solution,
                'initial_problem': x['initial_problem'],
                'initial_solution': x['initial_solution']
            })
        res = temp
    if key == '平方根':
        temp = []
        for x in res:
            match = re.search(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?$', x['solution'])
            if match:
                last_number = float(match.group())
            
                # 将最后一个数字格式化为保留一位小数的形式
                formatted_number = "{:.2f}".format(last_number)
                
                # 将格式化后的数字替换回原字符串中
                solution = x['solution'].replace(match.group(), formatted_number)+', 保留一位小数为 '+"{:.1f}".format(last_number)
            temp.append({
                'problem': x['problem'],
                'solution': solution,
                'initial_problem': x['initial_problem'],
                'initial_solution': x['initial_solution']
            })
        res = temp
    if key == '面积':
        res = [{'problem': x['problem'], 'solution': x['solution'].replace('面积为', '面积等于长乘宽，结果为'), 'initial_problem': x['initial_problem'], 'initial_solution': x['initial_solution']} for x in res]
    if key == '质量':
        res = [{'problem': x['problem'], 'solution': '质量等于密度乘体积，结果为 ' + x['solution'], 'initial_problem': x['initial_problem'], 'initial_solution': x['initial_solution']} for x in res]
    if reformat:
        for i in range(len(res)):
            match = re.search(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?$', res[i]['solution'])
            if match:
                last_number = float(match.group())
            
                # 将最后一个数字格式化为保留一位小数的形式
                formatted_number = "{:.1f}".format(last_number)
                
                # 将格式化后的数字替换回原字符串中
                res[i]['solution'] = res[i]['solution'].replace(match.group(), formatted_number)
    return res

complex_train_data = []
complex_pingjunzhi_data = [x for x in complex_cot_data_filtered if '平均值' in x['problem']]
complex_xiaoshoue_data = [x for x in complex_cot_data_filtered if '销售额' in x['problem']]
complex_sample_dict = {'打折': 450-76+6, '平均值': 0+6, '销售额': 0+6, '面积': 450-76+6, '0 次方': 10+2, '1 次方': 10+2, '2 次方': 10+2, '平方根': 500-76+6, '函数': 0, '质量': 90+6, '简化': 36+6}
complex_val_sample = [6, 6, 6, 6, 2, 2, 2, 6, 0, 6, 6]
reformat_list = ['打折', '平方根']

complex_train_data.extend(random.sample(complex_pingjunzhi_data, 300))
complex_train_data.extend(random.sample(complex_xiaoshoue_data, 300))
for i, (k,v) in enumerate(complex_sample_dict.items()):
    sublist = get_sublist_by_key(complex_data, k, v, reformat=k in reformat_list)
    complex_train_data.extend(sublist[:-8])
    val_data.extend(sublist[-complex_val_sample[i]:])

print(f'length of complex train data: {len(complex_train_data)}')
write_data(f'{base_data_path}/complex_train_data.jsonl', complex_train_data)
train_data.extend(complex_train_data)

# write_data(f'{base_data_path}/train_data.jsonl', train_data)
for i in range(len(val_data)):
    val_data[i] = {
        'problem': val_data[i]['initial_problem'],
        'solution': val_data[i]['initial_solution'],
        'train_solution': val_data[i]['solution']
    }
print(f'length of all val data: {len(val_data)}')
write_data(f'{base_data_path}/val_data.jsonl', val_data)

train_data_conv = []
for id, item in enumerate(train_data):
    train_data_conv.append({
        "id": str(id),
        "conversations": [
        {
            "from": "human",
            "value": f'Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{item["problem"]}\n\n### Response:'
        },
        {
            "from": "gpt",
            "value": item['solution']
        }
        ]
    })

# val_data_conv = []
# for id, item in enumerate(val_data):
#     val_data_conv.append({
#         "id": str(id),
#         "conversations": [
#         {
#             "from": "human",
#             "value": f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{item['problem']}\n\n### Response:"
#         },
#         {
#             "from": "gpt",
#             "value": item['solution']
#         }
#         ]
#     })

random.shuffle(train_data_conv)
print(f'length of all train data: {len(train_data_conv)}')
write_data(f'{base_data_path}/train_data_conv.json', train_data_conv)
# write_data(f'{base_data_path}/val_data_conv.json', val_data_conv)