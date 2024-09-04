import argparse
import os
import pandas as pd
import json
import re
import numpy as np
from decimal import Decimal

#pip install pandas

#获取保留小数点后最多6位的字符串
def get_precision_str(answer):
    answer = Decimal(answer)
    answer_str = str(answer)
    splits = str(answer).split('.')
    if len(splits) == 2:
        if len(splits[1]) < 6:
            answer_str = splits[0] + '.' + splits[1][:5]
        else:
            splits = str(answer + np.sign(answer)*Decimal('0.000005')).split('.')
            answer_str = splits[0] + '.' + splits[1][:5]
            
    return answer_str   

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        if s.replace('/', '').isnumeric():
            return True
        return False
        
#获取原始solution中数字结果（暂时不包括科学计数法数字）
def get_clean_answer(row):
    if re.search(r'计算\s?-?\d+\.?\d*\s?\+\s?-?\d+\.?\d*', row['problem']) is not None \
        or re.search(r'计算\s?-?\d+\.?\d*\s?\-\s?-?\d+\.?\d*', row['problem']) is not None \
        or re.search(r'计算\s?-?\d+\.?\d*\s?\*\s?-?\d+\.?\d*', row['problem']) is not None \
        or re.search(r'计算\s?-?\d+\.?\d*\s?\/\s?-?\d+\.?\d*', row['problem']) is not None \
        or re.search(r'解方程 -?\d+x \+ -?\d+ = 0', row['problem']) is not None \
        or re.search(r'一个长方形的长为 \d+ 厘米，宽为 \d+ 厘米，请计算其面积', row['problem']) is not None \
        or re.search(r'某物体的密度为 \d+ 克/立方厘米，体积为 \d+ 立方厘米，请计算该物体的质量', row['problem']) is not None \
        or re.search(r'解方程 -?\d+x \+ -?\d+ = 0', row['problem']) is not None \
        or re.search(r'计算\s?-?\d+.?\d*\s?的\s?\d+\s?次方?', row['problem']) is not None \
        or re.search(r'计算\s?\d+.?\d*\s?的平方根', row['problem']) is not None \
        or re.search(r'求以下数据的平均值：\[.+\]', row['problem']) is not None:
        return re.findall('-?\d+\.?\d*',row['solution'])[-1]
    elif re.search(r'将分数 \d+/\d+ 进行简化', row['problem']) is not None \
        or re.search(r'当 x = \d+.?\d* 时，求函数 y = \d*x\^\d+ 的值', row['problem']) is not None:
        return row['solution'].split('：')[-1]
    else: 
        return row['solution']

#判断样本在目前的方案下模型是否可以学习
def can_learn(row):
    if re.search(r'计算\s?-?\d+\.?\d*\s?\+\s?-?\d+\.?\d*', row['problem']) is not None \
        or re.search(r'计算\s?-?\d+\.?\d*\s?\-\s?-?\d+\.?\d*', row['problem']) is not None \
        or re.search(r'计算\s?-?\d+\.?\d*\s?\*\s?-?\d+\.?\d*', row['problem']) is not None \
        or re.search(r'计算\s?-?\d+\.?\d*\s?\/\s?-?\d+\.?\d*', row['problem']) is not None \
        or re.search(r'解方程 -?\d+x \+ -?\d+ = 0', row['problem']) is not None \
        or re.search(r'一个长方形的长为 \d+ 厘米，宽为 \d+ 厘米，请计算其面积', row['problem']) is not None \
        or re.search(r'某物体的密度为 \d+ 克/立方厘米，体积为 \d+ 立方厘米，请计算该物体的质量', row['problem']) is not None \
        or re.search(r'商品原价为 \d+ 元，打折后的价格为 \d+ 元，请计算打折的折扣比例', row['problem']) is not None \
        or re.search(r'去年销售额为 \d+ 万元，今年销售额增加了 \d+%，请计算今年的销售额', row['problem']) is not None \
        or re.search(r'求以下数据的平均值：\[.+\]', row['problem']) is not None \
        or re.search(r'将分数 \d+/\d+ 进行简化', row['problem']) is not None:
        return True
    elif re.search(r'计算\s?-?\d+.?\d*\s?的\s?\d+\s?次方?', row['problem']) is not None:
        result_list =  re.findall('-?\d+\.?\d*',row['problem'])
        _, b = result_list
        if Decimal(b) <= Decimal(1):
            return True
    return False

from decimal import Decimal
import numpy as np
#分解一个数字到非0的有效位
def get_number_split(input_num):
    splits =  str(input_num).split('.')
    if len(splits) == 2:
        integer_part, decimal_part = splits
    else:
        integer_part, decimal_part = splits[0], '0'

    integer_len = len(integer_part)
    integer_split = [Decimal(10**(integer_len-i-1))*Decimal(integer_part[i]) for i in range(integer_len)]
    decimal_split = [Decimal('0.'+'0'*i + decimal_part[i]) for i in range(len(decimal_part))]
    integer_split.extend(decimal_split)
    integer_split = list(filter(lambda x:x != 0, integer_split))
    return integer_split

#乘法CoT
def get_mul_cot(num1, num2, answer):
    num1, num2, answer = np.abs(num1), np.abs(num2), np.abs(answer)
    question = f"{num1} * {num2}"
    num2_split = get_number_split(num2)
    
    if len(num2_split) == 1:
        cot = question + " = " + str(answer)
    else:
        split = f"""{num1} * ({" + ".join(str(x) for x in num2_split)})"""
        expansion = " + ".join([f"{num1} * {x}" for x in num2_split])
        summation_terms = [num1 * x for x in num2_split]
        summation = " + ".join(str(x) for x in summation_terms)
        step = ""
        while summation_terms:
            first = summation_terms.pop(0)
            if not summation_terms:
                output = first
                break
            summation_terms[0] = first + summation_terms[0]
            if len(summation_terms) == 1:
                summation_terms = [str(answer)]
            step = step + " + ".join([f"{x}" for x in summation_terms]) 
            if len(summation_terms)>=2:
                step = step + " = "
    
        cot = question + " = " + f"{split} = {expansion} = {summation} = " + step
    return cot

#除法CoT
def get_div_cot(num1, num2, answer = None):
    base = Decimal('10')**6
    num1, num2 = np.abs(num1), np.abs(num2)
    num1 = num1*base
    quotient = num1 // num2
    remainder = num1 % num2

    if quotient == 0:
        cot = f"{num1} / {num2} = {quotient} R {remainder}"
    elif num1 == num2:
        cot = f"{num1} / {num2} = {quotient}"
    else:
        step = ""
        cot = ""
        left = num1
    
        i = 0
        computed_q = 0
    
        while left>=num2:
            if int(str(quotient)[i])!=0:
    
                intermediate = int(str(quotient)[i] + "0" * (len(str(quotient))-1-i))
    
                answer = num2 * intermediate
    
                new_left = left - answer
    
                step = f"{left} - {num2} * {intermediate} = {left} - {answer} = {new_left}\n"
    
                cot = cot + step
                
                left = new_left
    
                computed_q = computed_q + intermediate
    
            i = i+1
        #print(left, remainder)
        assert(left == remainder)
        assert(computed_q == quotient)
    
        if remainder!=0:
            cot = cot + f"Therefore, {num1} / {num2} = {quotient} R {remainder}"
        else:
            cot = cot + f"Therefore, {num1} / {num2} = {quotient}"
    return cot

def apply_cot(row):
    if re.search(r'计算\s?-?\d+\.?\d*\s?\*\s?-?\d+\.?\d*', row['problem']) is not None:
        result_list =  re.findall('-?\d+\.?\d*',row['solution'])
        result_list = [Decimal(x) for x in result_list]
        num1, num2, answer = result_list
        cot =  get_mul_cot(num1, num2, answer)
        cot = cot+ f"\nfinal, {num1} * {num2} = {answer}"
        return cot
    elif re.search(r'计算\s?-?\d+\.?\d*\s?\/\s?-?\d+\.?\d*', row['problem']) is not None:
        result_list =  re.findall('-?\d+\.?\d*',row['solution'])
        #print(result_list)
        result_list = [Decimal(x) for x in result_list]
        num1, num2, answer = result_list
        splits = str(answer).split('.')
        answer_str = str(answer)
        if len(splits) == 2:
            answer_str = splits[0] + '.' + splits[1][:6]
        cot = get_div_cot(num1, num2, None)
        answer_str = get_precision_str(answer_str)
        cot = cot + f"\nfinal, {num1} / {num2} ~= {answer_str}"
        return cot
    elif re.search(r'解方程 -?\d+x \+ -?\d+ = 0', row['problem']) is not None:
        result_list =  re.findall('-?\d+\.?\d*',row['problem'])
        result_list = [Decimal(x) for x in result_list]
        a, b, c = result_list
        num1, num2 = -b + c, a
        answer = num1/num2
        splits = str(answer).split('.')
        answer_str = str(answer)
        if len(splits) == 2:
            answer_str = splits[0] + '.' + splits[1][:6]
      
        cot = get_div_cot(num1, num2, None)
        answer_str = get_precision_str(answer_str)
        cot = cot + f"\nfinal, {num1} / {num2} ~= {answer_str}"
        return cot
    elif re.search(r'商品原价为 \d+ 元，打折后的价格为 \d+ 元，请计算打折的折扣比例', row['problem']) is not None:
        result_list =  re.findall('-?\d+\.?\d*',row['problem'])
        result_list = [Decimal(x) for x in result_list]
        a, b = result_list
        cot_head = f"({a} - {b}) / {a} = "
        num1, num2 = a - b, a
        answer = num1*100/num2
        splits = str(answer).split('.')
        answer_str = str(answer)
        if len(splits) == 2:
            answer_str = splits[0] + '.' + splits[1][:6]
        cot = cot_head + get_div_cot(num1, num2, None)
        answer_str = get_precision_str(answer_str)
        cot = cot + f"\nfinal, {num1} / {num2} ~= {answer_str} 折"
        return cot
    elif re.search(r'一个长方形的长为 \d+ 厘米，宽为 \d+ 厘米，请计算其面积', row['problem']) is not None:
        result_list =  re.findall('-?\d+\.?\d*',row['problem'])
        result_list = [Decimal(x) for x in result_list]
        num1, num2 = result_list
        answer = num1*num2
        cot =  get_mul_cot(num1, num2, answer)
        cot = cot+ f"\nfinal, {num1} * {num2} = {answer} 平方厘米"
        return cot
    elif re.search(r'某物体的密度为 \d+ 克/立方厘米，体积为 \d+ 立方厘米，请计算该物体的质量', row['problem']) is not None:
        result_list =  re.findall('-?\d+\.?\d*',row['problem'])
        result_list = [Decimal(x) for x in result_list]
        num1, num2 = result_list
        answer = num1*num2
        cot =  get_mul_cot(num1, num2, answer)
        cot = cot+ f"\nfinal, {num1} * {num2} = {answer} 克"
        return cot
    elif re.search(r'去年销售额为 \d+ 万元，今年销售额增加了 \d+%，请计算今年的销售额', row['problem']) is not None:
        result_list =  re.findall('-?\d+\.?\d*',row['problem'])
        result_list = [Decimal(x) for x in result_list]
        a, b = result_list
        num1, num2 = a, Decimal('1') + b/Decimal('100')
        answer = num1*num2
        cot =  get_mul_cot(num1, num2, answer)
        cot = cot+ f"\nfinal, {num1} * {num2} = {answer} 万元"
        return cot
    elif re.search(r'求以下数据的平均值：\[.+\]', row['problem']) is not None:
        value_list = re.findall(r'\[.*\]' ,row['problem'])[0]
        value_list = eval(value_list)
        value_len = len(value_list)
        cum_list = list(np.cumsum(value_list))
        step = ""
        cot = ' + '.join([str(x) for x in value_list])
        if len(value_list) == 1:
            cot = str(value_list[0])
            return cot
        for idx in range(1, value_len):
            cot += ' = ' + ' + '.join([str(x) for x in ([cum_list[idx]] + value_list[idx+1:])])
        
        num1, num2 = Decimal(str(cum_list[-1])), Decimal(str(value_len))
        answer = num1/num2
        cot += f'\n计算 {cum_list[-1]} / {value_len}'
        cot += '\n' + get_div_cot(Decimal(str(cum_list[-1])), Decimal(str(value_len)), None)
        splits = str(answer).split('.')
        answer_str = str(answer)
        if len(splits) == 2:
            answer_str = splits[0] + '.' + splits[1][:6]
        answer_str = get_precision_str(answer_str)
        cot = cot + f"\nfinal, {num1} / {num2} ~= {answer_str}"
        return cot
    else:
        return row['answer']

if __name__ == "__main__":
    work_path = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_path',
        default="./train.json",
        required=True,
        help='org data path')
    parser.add_argument(
        '--train_len', default=50000, type=int,
        help='sample train data number, if train_len == -1 ,all data use for train')

    parser.add_argument(
        '--valid_len', default=-1, type=int,
        help='valid data number, if valid_len == -1 ,all data except train use for valid')
    
    parser.add_argument(
        '--out_dir',
        default="./",
        required=True,
        help='output dir for train/valid data')
    args_, rest_args_ = parser.parse_known_args()
    
    data = [json.loads(line) for line in open(args_.data_path, 'r').readlines()]
    df = pd.DataFrame.from_records(data)
    print('原始数据数量:', df.shape)
    df['id'] = df.index
    df['answer'] = df.apply(lambda x:get_clean_answer(x), axis = 1)
    df['can_learn'] = df.apply(lambda x:can_learn(x), axis = 1)
    print('根据目前规则认为模型可以学习的样本数量:', df[df['can_learn']].shape)
    
    df['output'] = df.apply(lambda row:apply_cot(row), axis = 1)
    org_df = df
    df = df.drop_duplicates(['problem'])
    print('去重后样本数量:', df.shape)
    df['is_number'] = df['answer'].apply(lambda x:is_number(x))
    df = df[df['is_number']]
    learn_df = df[df['can_learn']]
    print('去重后认为模型可以学习的样本数量:', learn_df.shape)

    if args_.train_len > 0:
        train_df =  learn_df.sample(args_.train_len)
    else:
        train_df = learn_df
    if args_.valid_len > 0:
        valid_df = df[~df['id'].isin(train_df['id'])].sample(args_.valid_len)
    else:
        valid_df = df[~df['id'].isin(train_df['id'])]
    if not os.path.exists(args_.out_dir):
        os.makedirs(args_.out_dir, exist_ok = True) 
    print('train_len:',train_df.shape)
    print('valid_len:', valid_df.shape)
    json.dump(train_df[['id', 'problem', 'solution', 'answer', 'output']].to_dict(orient='records'), open(os.path.join(args_.out_dir, 'train-data.json'), 'w'), indent=2)
    json.dump(valid_df[['id', 'problem', 'solution', 'answer']].to_dict(orient='records'), open(os.path.join(args_.out_dir,'valid-data.json'), 'w'), indent=2)

    valid_ms_data = valid_df.to_dict(orient='records')
    with open(os.path.join(args_.out_dir,'valid-data-list.json'), 'w') as f:
        for line in valid_ms_data:
            json.dump(line, f)
            f.write('\n')