import json
import argparse
import json
import numpy as np

def read_text_to_json_lines(input_file_path):
    # 打开输入文件并读取所有行
    with open(input_file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    # 在每一行的 JSON 数据的 'problem' 前添加指定的文本
    solutions = []
    for line in lines:
        data = json.loads(line)
        if 'solution' in data:
            solutions.append(data['solution'])
    return solutions

def read_npy_to_list(input_file_path):
    array = np.load(input_file_path,allow_pickle=True)
    result_list =[]
    for i in range(len(array)):
        for j in range(len(array[i]['text_generation_text'])):
            result_list.append(array[i]['text_generation_text'][j])
    return result_list

def calculate_accuracy(test_path,answer_path):
    solutions = read_text_to_json_lines(test_path)
    resluts  = read_npy_to_list(answer_path)
    acc =0

    if len(solutions)==len(resluts):
        for i in range(len(solutions)):
            if solutions[i] in resluts[i]:
                acc+=1
    else:
        print('长度不一致')
    print(f'模型准确性:{acc/len(solutions)*100}%')
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--test_path', default='train_150k.json', type=str,
                        help='传入原始训练文件路径')
    parser.add_argument('--answer_path', default='train_150k_with_prompt.json', type=str,
                        help='输出新的文件路径')
    args = parser.parse_args()
    calculate_accuracy(args.test_path,args.answer_path)


