import re
import json
import numpy as np

def read_npy(path):
    """读取npy文件，并将每个问题的response写成列表"""
    data = np.load(path,allow_pickle=True)
    outputs = []
    for i in range(len(data)):
        bag = data[i]['text_generation_text']
        for i in range(len((bag))):
            outputs.append(extract_response(bag[i]))
    return outputs

def extract_solutions_from_file(file_path):
    """从给定的JSON文件中提取所有'solution'对应的值，并按顺序返回一个列表"""
    # 初始化一个列表来存储'solution'的值
    solutions = []

    # 打开并读取JSON文件
    with open(file_path, 'r', encoding='utf-8') as file:
        # 遍历文件中的每一行
        for line in file:
            # 解析每一行的JSON数据
            data = json.loads(line)
            # 提取'solution'的值并添加到列表中
            solutions.append(data['solution'])
    
    return solutions


def extract_number(solution):
    """提取数学表达式中的数值的整数部分"""
    try:
        ret = float(re.findall(r"[-+]?\d*\.\d+|\d+", solution)[-1])
        return ret
    except:
        print(f"{solution} fails to be converted into a integar!")
        return 0
    
def lcs(X, Y):
    """计算两个序列的最长公共子序列 (LCS)"""
    m = len(X)
    n = len(Y)
    L = [[0] * (n + 1) for i in range(m + 1)]
    
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0 or j == 0:
                L[i][j] = 0
            elif X[i - 1] == Y[j - 1]:
                L[i][j] = L[i - 1][j - 1] + 1
            else:
                L[i][j] = max(L[i - 1][j], L[i][j - 1])
    
    return L[m][n]

def extract_response(text):
    """从给定的文本中提取Response后面的字符"""
    match = re.search(r'### Response: (.*)', text)
    if match:
        return match.group(1)
    return '0'


def round_to_min_precision(num1, num2):
    def get_decimal_places(number):
        # 将数字转换为字符串
        str_num = str(number)
        # 分割整数部分和小数部分
        if '.' in str_num:
            return len(str_num.split('.')[1])
        else:
            return 0
    
    # 获取两个数的小数部分位数
    dec_places_num1 = get_decimal_places(num1)
    dec_places_num2 = get_decimal_places(num2)
    
    # 确定最小的小数位数
    min_dec_places = min(dec_places_num1, dec_places_num2)
    
    # 四舍五入两个数
    rounded_num1 = round(num1, min_dec_places)
    rounded_num2 = round(num2, min_dec_places)
    
    return rounded_num1, rounded_num2


def calculate_lcs_score(pred, target):
    """计算 LCS 分数"""
    pred_str = str(pred)
    target_str = str(target)
    lcs_length = lcs(pred_str, target_str)
    return lcs_length / max(len(pred_str), len(target_str))

def evaluate_predictions(predictions, ground_truths):
    total_samples = len(predictions)
    correct_predictions = 0
    total_lcs_score = 0

    for pred, gt in zip(predictions, ground_truths):
        pred_value = extract_number(pred)
        gt_value = extract_number(gt)
        pred_value, gt_value = round_to_min_precision(pred_value, gt_value)

        if pred_value == gt_value:
            correct_predictions += 1

        lcs_score = calculate_lcs_score(pred_value, gt_value)
        total_lcs_score += lcs_score

    accuracy = correct_predictions / total_samples
    avg_lcs_score = total_lcs_score / total_samples

    return accuracy, avg_lcs_score

def full_test_pipeline(npy_path:str, json_path:str):
    responses = read_npy(npy_path)
    truths = extract_solutions_from_file(json_path)
    acc, rouge_lcs = evaluate_predictions(responses, truths)
    print(f"Accuracy: {acc}, Rouge-L: {rouge_lcs}")
    #return acc, rouge_lcs


full_test_pipeline(r'E:\DianWork\DianGPT\昇腾AI\output_data\result_731_floatformat_insert_template_v3_npy.npy', r'E:\DianWork\DianGPT\昇腾AI\dataset\testset.json')