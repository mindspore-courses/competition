import random
import json
# from transformers import AutoTokenizer
# import pandas as pd
import re
from tqdm import tqdm
import math
# tokenizer = AutoTokenizer.from_pretrained("./tokenizer.model", use_fast=False)
tokenizer = None


class CustomError(Exception):
    def __init__(self, message, context_info):
        super().__init__(message)
        self.context_info = context_info

    def __str__(self):
        return f'{super().__str__()}, Context: {self.context_info}'


def is_chinese_char(char):
    """判断字符是否为中文字符"""
    return '\u4e00' <= char <= '\u9fff'


def is_english_char(char):
    """判断字符是否为英文字符"""
    return ('a' <= char <= 'z') or ('A' <= char <= 'Z')


def detect_language(text):
    """检测字符串是中文描述还是英文描述"""
    chinese_count = sum(1 for char in text if is_chinese_char(char))
    # english_count = sum(1 for char in text if is_english_char(char))

    if chinese_count > 0:
        return "Chinese"
    else:
        return "English"


def read_data(path):
    cn_data = []
    eg_data = []

    with open(path, encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            if detect_language(data['problem']) == 'Chinese':
                cn_data.append(data)
            else:
                eg_data.append(data)
    return cn_data, eg_data


def remove_duplicates(lst, key):
    """基于字典中指定的键对列表中的字典进行去重"""
    seen = set()
    unique_list = []
    for item in lst:
        value = item.get(key)
        if value not in seen:
            unique_list.append(item)
            seen.add(value)
    return unique_list


# def analyze(data):
#     seq = [len(tokenizer(sample['problem'])) for sample in data]
#     # 转换为 pandas Series
#     series = pd.Series(seq)

#     # 使用 pandas 的描述性统计函数
#     statistics = series.describe()
#     print(statistics)

#     # 检测离群值
#     q1 = series.quantile(0.25)
#     q3 = series.quantile(0.75)
#     iqr = q3 - q1
#     lower_bound = q1 - 1.5 * iqr
#     upper_bound = q3 + 1.5 * iqr
#     outliers = series[(series < lower_bound) | (series > upper_bound)]
#     print("离群值:", outliers)
#     return seq


def classification(text):
    if re.search(r"计算([\s\d\.\+\-\*/=]+)等于多少", text):
        return 'direct computation'
    elif "解方程" in text:
        return 'eq solve'
    elif "求以下数据的平均值：" in text:
        return 'mean computation'
    elif re.search(r"将分数([\s\d\.\+\-\*/]+)进行简化", text):
        return 'fraction simplify'
    elif "某物体的密度为" in text:
        return 'mass computation'
    elif '请计算打折的折扣比例' in text:
        return 'discount percentage'
    elif '请计算其面积' in text:
        return 'area computation'
    elif re.search(r"计算([\s\d\.]+)的平方根", text):
        return 'sqrt computation'
    elif re.search(r"计算([\s\d\.\-]+)的([\s\d]+)次方", text):
        return 'power computation'
    elif '请计算今年的销售额' in text:
        return 'sales revenue'
    elif re.search(r"求函数([\s\d\.\+\-\*/=xy^]+)的值", text):
        return 'function computation'
    else:
        raise CustomError('otherClassification', str(text))


def excute_classification(data):
    ret = {}
    for sample in data:
        tag = classification(sample['problem'])
        if tag not in ret.keys():
            ret[tag] = []
        ret[tag].append(sample)
    return ret


def custom_round(num, decimal_places=4):
    if isinstance(num, (int, float)):
        if int(num) == num:
            return int(num)  # 如果是整数，直接返回
        else:
            ret = round(num, decimal_places)  # 如果是浮点数，保留四位小数并四舍五入
            return int(ret) if int(ret) == ret else ret
    else:
        raise TypeError("Input must be an int or float.")


def direct_computation_process(sample):
    def extract_solution(text):
        matches = re.findall(r'-?(\d+\.\d+|\d+)', text)
        if len(matches) == 3:
            return matches[2]
        else:
            raise CustomError(
                'direct computation extract solution fail', text)
    result_str = extract_solution(sample['solution'])
    result = custom_round(float(result_str))
    return sample['solution'].replace(result_str, str(result))


def eq_solve_process(sample):
    def extract_equation(equation_str):
        pattern = r'(-?\d+)x\s+\+\s+(-?\d+)\s+=\s+0'
        match = re.search(pattern, equation_str)
        if match:
            return int(match.group(1)), int(match.group(2))
        else:
            raise CustomError('eq solve extract equation fail', equation_str)

    def extract_solution(solution_str):
        solution_pattern = r'-?[\d\.]+'
        match = re.search(solution_pattern, solution_str)
        if match:
            return match.group()
        else:
            raise CustomError('eq solve extract solution fail', solution_str)
    equation = extract_equation(sample['problem'])
    solution = extract_solution(sample['solution'])

    process = [str(equation[0])+'x'+' = '+str(-equation[1])]
    process.append('x = '+str(-equation[1])+' / '+str(equation[0]))
    result = -equation[1] / equation[0]
    if abs(result-float(solution)) >= 1e-6:
        raise CustomError('eq not equal', '\n'.join(
            [str(sample), str(result), str(solution)]))
    process.append('x = ' + str(custom_round(result, 4)))
    return ', '.join(process)


def mean_computation_process(sample):
    def extract_list(list_str):
        match = re.search(r'\[(.*?)\]', list_str)
        if match:
            data_part = match.group(1)
            data_list = list(map(int, data_part.split(', ')))
            return data_list
        else:
            raise CustomError('mean computation extract list fail', list_str)

    def extract_solution(solution_str):
        solution_pattern = r'-?[\d\.]+'
        match = re.search(solution_pattern, solution_str)
        if match:
            return match.group()
        else:
            raise CustomError(
                'mean computation extract solution fail', solution_pattern)
    list_data = extract_list(sample['problem'])
    process = '(' + ' + '.join([str(item)
                                for item in list_data]) + ') / ' + str(len(list_data))
    result = sum(list_data) / len(list_data)
    if abs(result-float(extract_solution(sample['solution']))) >= 1e-6:
        raise CustomError('mean not equal', '\n'.join(
            [str(sample), str(result)]))
    result = custom_round(result, 4)
    process += ' = ' + str(result)
    return process


def fraction_simplify_process(sample):
    def extract_fraction(fraction_str):
        match = re.search(r'(\d+)/(\d+)', fraction_str)
        if match:
            numerator = int(match.group(1))
            denominator = int(match.group(2))
            return numerator, denominator
        else:
            raise CustomError('fraction simplify extract fail', fraction_str)
    problem = extract_fraction(sample['problem'])
    gcd = math.gcd(problem[0], problem[1])
    process = ['gcd(' + ', '.join([str(item)
                                   for item in problem]) + ') = ' + str(gcd)]
    for num in problem:
        process.append(str(num) + ' / ' + str(gcd) +
                       ' = ' + str(int(num / gcd)))
    problem = tuple(item / gcd for item in problem)
    solu = extract_fraction(sample['solution'])
    if problem != solu:
        raise CustomError('fraction simplift not eq', '\n'.join(
            [str(sample), str(problem), str(solu)]))
    process.append(sample['solution'])
    return ', '.join(process)


def mass_computation_process(sample):
    """{"problem": "某物体的密度为 9 克/立方厘米，体积为 3 立方厘米，请计算该物体的质量。", "solution": "27 克"}"""
    def extract_nums(text):
        density = re.search(r'密度为 (\d+) 克/立方厘米', text)
        volume = re.search(r'体积为 (\d+) 立方厘米', text)
        if density and volume:
            density_value = int(density.group(1))
            volume_value = int(volume.group(1))
            if density_value != float(density.group(1)):
                raise CustomError('mass couputation density not int', text)
            if volume_value != float(volume.group(1)):
                raise CustomError('mass couputation volume not int', text)
            return density_value, volume_value
        else:
            raise CustomError('mass computation extract fail', text)

    def extract_solution(solution_str):
        solution_pattern = r'[\d]+'
        match = re.search(solution_pattern, solution_str)
        if match:
            return match.group()
        else:
            raise CustomError(
                'mass computation extract solution fail', solution_pattern)

    den, vol = extract_nums(sample['problem'])
    result = den * vol
    solu = extract_solution(sample['solution'])
    if abs(result-float(solu)) >= 1e-2:
        raise CustomError('mass computation not equal', '\n'.join(
            [str(sample), str(result), solu]))
    process = str(den) + ' * ' + str(vol) + ' = ' + str(result)
    return process


def discount_percentage_process(sample):
    """{"problem": "商品原价为 60 元，打折后的价格为 24 元，请计算打折的折扣比例。", "solution": "60.0"}
    {"problem": "商品原价为 72 元，打折后的价格为 2 元，请计算打折的折扣比例。", "solution": "97.22222222222221"}"""
    def extract_nums(text):
        original = re.search(r'商品原价为 (\d+) 元', text)
        discount = re.search(r'打折后的价格为 (\d+) 元', text)
        if original and discount:
            original_value = int(original.group(1))
            discount_value = int(discount.group(1))
            return original_value, discount_value
        else:
            raise CustomError(
                'discount percentage computation extract fail', text)

    original, discount = extract_nums(sample['problem'])
    result = (original - discount) / original * 100
    solu = sample['solution']
    if abs(result-float(solu)) >= 1e-4:
        raise CustomError('discount percentage computation not equal', '\n'.join(
            [str(sample), str(result), solu]))
    process = '(' + str(original) + ' - ' + str(discount) + ') / ' + \
        str(original) + ' = ' + str(custom_round(result, 2))
    return process


def area_computation_process(sample):
    """{"problem": "一个长方形的长为 74 厘米，宽为 40 厘米，请计算其面积。", "solution": "面积为 2960 平方厘米"}"""
    def extract_nums(text):
        length = re.search(r'一个长方形的长为 (\d+) 厘米', text)
        width = re.search(r'宽为 (\d+) 厘米', text)
        if length and width:
            length_value = int(length.group(1))
            width_value = int(width.group(1))
            if length_value != float(length.group(1)):
                raise CustomError('area couputation length not int', text)
            if width_value != float(width.group(1)):
                raise CustomError('area couputation width not int', text)
            return length_value, width_value
        else:
            raise CustomError('area computation extract fail', text)

    def extract_solution(solution_str):
        solution_pattern = r'[\d]+'
        match = re.search(solution_pattern, solution_str)
        if match:
            return match.group()
        else:
            raise CustomError(
                'area computation extract solution fail', solution_pattern)

    length, width = extract_nums(sample['problem'])
    result = length * width
    solu = extract_solution(sample['solution'])
    if abs(result-int(solu)) >= 1e-2:
        raise CustomError('mass computation not equal', '\n'.join(
            [str(sample), str(result), solu]))
    process = str(length) + ' * ' + str(width) + \
        ' = ' + str(result)
    return process


def sqrt_computation_process(sample):
    """{"problem": "计算 6145.00 的平方根？", "solution": "√6145.00 = 78.39005038906404479030928703"}"""
    def extract_solution(text):
        matches = re.findall(r'-?(\d+\.\d+|\d+)', text)
        if len(matches) == 2:
            return matches[1]
        else:
            raise CustomError(
                'sqrt computation extract solution fail', text)
    result_str = extract_solution(sample['solution'])
    result = custom_round(float(result_str))
    return sample['solution'].replace(result_str, str(result))


def power_computation_process(sample):
    "0-5"
    """{"problem": "计算 -787.36 的 4 次方？", "solution": "-787.36^4 = 384320358429.54428416"}"""
    return sample['solution']


def sales_revenue_process(sample):
    """{"problem": "去年销售额为 90 万元，今年销售额增加了 65%，请计算今年的销售额。", "solution": "148.5"}"""
    def extract_nums(text):
        amount = re.search(r'去年销售额为 (\d+) 万元', text)
        inc_percentage = re.search(r'今年销售额增加了 (\d+)%', text)
        if amount and inc_percentage:
            amount_value = int(amount.group(1))
            inc_percentage_value = int(inc_percentage.group(1))
            if amount_value != float(amount.group(1)):
                raise CustomError(
                    'sales_revenue couputation amount not int', text)
            if inc_percentage_value != float(inc_percentage.group(1)):
                raise CustomError(
                    'sales_revenue couputation inc_percentage not int', text)
            return amount_value, inc_percentage_value
        else:
            raise CustomError('sales_revenue computation extract fail', text)

    def extract_solution(solution_str):
        solution_pattern = r'[\d\.]+'
        match = re.search(solution_pattern, solution_str)
        if match:
            return match.group()
        else:
            raise CustomError(
                'sales_revenue computation extract solution fail', solution_pattern)

    amount, inc_percentage = extract_nums(sample['problem'])
    inc_percentage /= 100
    result = amount * (1 + inc_percentage)
    solu = extract_solution(sample['solution'])
    if abs(result-float(solu)) >= 1e-2:
        raise CustomError('sales_revenue computation not equal', '\n'.join(
            [str(sample), str(result), solu]))
    process = str(amount) + ' * ' + '(1 + ' + str(round(inc_percentage, 2)
                                                  ) + ') = ' + str(custom_round(result, 4))
    return process


def function_computation_process(sample):
    """{"problem": "当 x = 2.50 时，求函数 y = 9x^49 的值", "solution": "函数的值为：283989925879564249948.2222835"}"""
    def extract_nums(text):
        matches = re.findall(r'(\d+\.\d+|\d+)', text)
        x_value = matches[0]
        y_coefficient = matches[1]
        y_exponent = matches[2]
        if len(matches) == 3:
            para1 = int(y_coefficient)
            para2 = int(y_exponent)
            return float(x_value), para1, para2
        else:
            raise CustomError(
                'function computation extract solution fail', text)

    def extract_solution(solution_str):
        solution_pattern = r'(\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)'
        match = re.search(solution_pattern, solution_str)
        if match:
            return match.group()
        else:
            raise CustomError(
                'function computation extract solution fail', solution_str)

    x, p1, p2 = extract_nums(sample['problem'])
    solu = extract_solution(sample['solution'])
    result = p1 * x ** p2
    scientific_notation = f"{result:.1e}"  # 转换为科学计数法
    exponent_part = scientific_notation.split('e')[-1]  # 提取右边部分
    if abs(result-float(solu)) >= 1e-6 * (10 ** int(exponent_part)):
        raise CustomError('function computation not equal',
                          '\n'.join([str(sample), str(result), solu]))
    process = str(p1) + ' * ' + str(x) + ' ^ ' + str(p2) + \
        ' = ' + solu
    return process


process_functions = {
    'direct computation': direct_computation_process,
    'eq solve': eq_solve_process,
    'mean computation': mean_computation_process,
    'fraction simplify': fraction_simplify_process,
    'mass computation': mass_computation_process,
    'discount percentage': discount_percentage_process,
    'area computation': area_computation_process,
    'sqrt computation': sqrt_computation_process,
    'power computation': power_computation_process,
    'sales revenue': sales_revenue_process,
    'function computation': function_computation_process
}


def sample_and_shuffle(data_dict, target_size=50000):
    total_samples = sum(len(samples) for samples in data_dict.values())
    sample_ratio = target_size / total_samples

    sampled_data = []

    for key, samples in data_dict.items():
        num_samples = int(len(samples) * sample_ratio)
        sampled_data.extend(random.sample(samples, num_samples))

    random.shuffle(sampled_data)

    return sampled_data


def count_direct_computation(lis):
    add = sum([1 for x in [s for s in lis if '+' in s['problem']]])
    mul = sum([1 for x in [s for s in lis if '*' in s['problem']]])
    div = sum([1 for x in [s for s in lis if '/' in s['problem']]])
    sub = len(lis) - add - mul - div
    print(
        f"add: {add}, sub: {sub}, mul: {mul}, div: {div}, total: {len(lis)}")


if __name__ == "__main__":
    cn, eg = read_data('./train.json')
    print("cn count {0}, eg count {1}".format(len(cn), len(eg)))
    cn, eg = remove_duplicates(cn, 'problem'), remove_duplicates(eg, 'problem')
    print("after remove duplicate")
    print("cn count {0}, eg count {1}".format(len(cn), len(eg)))
    classed_data = excute_classification(cn)
    count_classed_data = {k: len(v) for k, v in classed_data.items()}
    print(json.dumps(count_classed_data, indent=4))
    for k in process_functions:
        for s in tqdm(classed_data[k], desc=k):
            s['solution'] = "计算过程：" + process_functions[k](s)
    # classed_data['eg'] = eg
    count_classed_data = {k: len(v) for k, v in classed_data.items()}
    print(json.dumps(count_classed_data, indent=4))
    total_samples = sum(
        sample_num for sample_num in count_classed_data.values())
    count_classed_data = {k: v / total_samples for k,
                          v in count_classed_data.items()}
    print(json.dumps(count_classed_data, indent=4))
    count_direct_computation(classed_data['direct computation'])
    sampled_and_shuffled = sample_and_shuffle(classed_data, 100000)
    with open('train-v2.json', 'w', encoding='utf-8') as f:
        for item in sampled_and_shuffled:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')
