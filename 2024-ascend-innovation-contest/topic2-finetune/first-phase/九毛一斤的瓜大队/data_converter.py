
"""
fastchat stanford alpaca data convert tools.
"""

import argparse
import random
import json
import re


def get_first_three_chars(original_list):
    # 创建一个临时字典来辅助分类
    temp_dict = {}

    # 遍历原始列表
    for item in original_list:
        # 获取“problem”值的前三个字
        first_three_chars = item["problem"][:3]
        # 检查这三个字是否包含汉字
        if any('\u4e00' <= char <= '\u9fff' for char in first_three_chars):
            # 如果包含汉字，检查这三个字是否已经在临时字典中
            if first_three_chars in temp_dict:
                # 如果是，将当前字典添加到对应的列表中
                temp_dict[first_three_chars].append(item)
            else:
                # 如果不是，创建一个新的列表，并将当前字典添加进去
                temp_dict[first_three_chars] = [item]
                # 检查这三个字是否全是非中文字符
        elif all(not '\u4e00' <= char <= '\u9fff' for char in first_three_chars):
            # 如果是，检查'Eng'是否已经在临时字典中
            if 'Eng' in temp_dict:
                # 如果是，将当前字典添加到对应的列表中
                temp_dict['Eng'].append(item)
            else:
                # 如果不是，创建一个新的列表，并将当前字典添加进去
                temp_dict['Eng'] = [item]
        else:
            # 如果不是，打印警告信息
            print(f"Warning: {first_three_chars} is not a valid first three chars.")

    new_dict = {}
    for key, value in temp_dict.items():
        # 创建一个新的字典，键是三个字，值是对应的字典列表
        new_dict[key] = value

    return new_dict


def normalize_numbers(text):
    # 定义一个正则表达式来匹配数字，包括整数、小数和科学计数法表示的数字
    number_pattern = re.compile(r'\b\d+(\.\d*)?([eE][-+]?\d+)?\b')

    def normalize_number(match):
        number_str = match.group(0)
        try:
            # 将数字字符串转换为浮点数
            if '.' in number_str:
                number = float(number_str)
            else:
                number = int(number_str)
            # 如果数字是小数，则四舍五入到小数点后三位
            if isinstance(number, float):
                return f"{round(number, 3)}"
                # 如果数字是整数，则直接返回
            else:
                return str(int(number))
        except ValueError:
            # 如果转换失败，则返回原始数字字符串
            return number_str

            # 使用正则表达式替换文本中的数字

    normalized_text = number_pattern.sub(normalize_number, text)

    return normalized_text


# Prompt from stanford alpaca's training script
PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),

    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{problem}\n\n### Response:"
    ),

}


def main(args_param):
    prompt_input, prompt_no_input = (
        PROMPT_DICT["prompt_input"],
        PROMPT_DICT["prompt_no_input"],
    )

    data_path = args_param.data_path

    sources = []
    targets = []

    with open(data_path, 'r', encoding='utf-8') as f:
        data = f.readlines()
        news_data = []
        for d in data:
            news_data.append(normalize_numbers(d.strip()))
        news_data_list = [json.loads(line.strip()) for line in news_data]

        pro_dat_dict = get_first_three_chars(news_data_list)

        # json.dump(news_data_list, open("temp_data.json", "w", encoding='utf-8'), ensure_ascii=False, indent=2)

        # 打印 pro_dat_dict 的所有 keys
        for key, value in pro_dat_dict.items():
            print(key, len(value))

        # 设置随机种子
        random.seed(randon_seed)  # 42
        sample_data_list = []
        for key, value in pro_dat_dict.items():
            # 随机采样1000个样例
            sampled_list = random.sample(value, 1000)
            sample_data_list = sample_data_list + sampled_list
        
        # test_file = open("test_data_2000.json", "w", encoding='utf-8')
        # for item in sample_data_list:
        #     # 将item序列化为JSON字符串，并确保它是一行
        #     json_str = json.dumps(item, ensure_ascii=False)
        #     # 写入文件，每个item占一行
        #     test_file.write(json_str + '\n')
        # test_file.close()

        for item in sample_data_list:
            a = prompt_no_input.format_map(item)
            # a = item["problem"]
            b = item["solution"]
            sources.append(a)
            targets.append(b)

    new_data = []
    print("Total data size:", len(sources))
    cnt = 1

    for s, t in zip(sources, targets):
        new_data.append(
            {
                "id": str(cnt),
                "conversations": [
                    {
                        "from": "human",
                        "value": s,
                    },
                    {
                        "from": "gpt",
                        "value": t,
                    },
                ],
            }
        )

        cnt += 1

    json.dump(new_data, open(args_param.output_path, "w", encoding='utf-8'), ensure_ascii=False, indent=2)


if __name__ == "__main__":
    randon_seed = 42
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="train.json")
    parser.add_argument(
        "--output_path", type=str, default=f"qa-2000-{randon_seed}-train-data-conversation.json"
    )
    args = parser.parse_args()
    main(args)
    print("Done!")
