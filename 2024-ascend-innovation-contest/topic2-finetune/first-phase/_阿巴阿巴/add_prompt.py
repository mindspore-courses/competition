import json
import argparse
def prepend_text_to_json_lines(input_file_path, output_file_path):
    # 打开输入文件并读取所有行
    with open(input_file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    # 在每一行的 JSON 数据的 'problem' 前添加指定的文本
    modified_lines = []
    for line in lines:
        data = json.loads(line)
        if 'problem' in data:
            original_problem = data['problem']
            #original_solution = data['solution']
            data['problem'] = f"你正在进行LoRA微调以加强你的数学处理能力，接下来，你会收到一个数学问题，你必须精准的给出这个数学问题的答案。\n{original_problem}"
            #data['solution'] = f"这个问题的答案为，{original_solution}"
            modified_lines.append(json.dumps(data, ensure_ascii=False))

    # 写入新的文本文件
    with open(output_file_path, 'w', encoding='utf-8') as file:
        file.write('\n'.join(modified_lines))
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='train_150k.json', type=str,
                        help='传入原始训练文件路径')
    parser.add_argument('--output', default='train_150k_with_prompt.json', type=str,
                        help='输出新的文件路径')
    args = parser.parse_args()
    prepend_text_to_json_lines(args.input,args.output)

# prepend_text_to_json_lines("train_150k.json","train_150k_with_prompt.json")