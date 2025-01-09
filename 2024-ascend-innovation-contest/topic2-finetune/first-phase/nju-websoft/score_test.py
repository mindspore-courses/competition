import re
import numpy as np
import json
from tqdm import tqdm


class CustomError(Exception):
    def __init__(self, message, context_info):
        super().__init__(message)
        self.context_info = context_info

    def __str__(self):
        return f'{super().__str__()}, Context: {self.context_info}'


def extract_last_number(text):
    # 正则表达式模式，用于匹配整数、负数、小数和科学计数法
    pattern = r'-?\d+\.?\d*(?:[eE][-+]?\d+)?'
    # 在文本中找到所有匹配项
    matches = re.findall(pattern, text)
    # 如果没有找到匹配项，返回None
    if not matches:
        return 0.0000001
        # raise CustomError('not match last number', text)
    # 返回最后一个匹配项
    return float(matches[-1])


def count_decimal_places(number):
    """计算浮点数的小数点后位数，包括科学计数法"""
    str_number = f"{number:.16e}"  # 将浮点数转换为科学计数法表示
    match = re.search(r'\d\.(\d+)e', str_number)
    if match:
        return len(match.group(1))
    return 0


def round_to_fewer_decimal_places(num1, num2):
    """根据位数较少的数四舍五入另一个数"""
    decimal_places_num1 = count_decimal_places(num1)
    decimal_places_num2 = count_decimal_places(num2)

    # 取较少的位数
    fewer_decimal_places = min(decimal_places_num1, decimal_places_num2)

    # 对两个数进行四舍五入
    rounded_num1 = round(num1, fewer_decimal_places)
    rounded_num2 = round(num2, fewer_decimal_places)

    return rounded_num1, rounded_num2


tmprediction = np.load('./lr4-v2/lr4_v2_3905.npy', allow_pickle=True)
prediction = []
for d in tmprediction:
    prediction.extend(d['text_generation_text'])
with open('./lr4-v2/prediction.json', 'w', encoding='utf-8') as f:
    json.dump(prediction, f, ensure_ascii=False, indent=4)
problems = []
answer = []
with open('test-v2.json', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        problems.append(data['problem'])
        answer.append(data['solution'])

cnt = 0
for i in tqdm(range(len(answer))):
    pred = extract_last_number(prediction[i])
    ans = extract_last_number(answer[i])
    ans, pred = round_to_fewer_decimal_places(ans, pred)
    if ans == pred:
        cnt += 1

print(cnt / len(answer))
