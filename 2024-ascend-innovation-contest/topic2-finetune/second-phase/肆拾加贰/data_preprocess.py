import json
import re
import random
from itertools import groupby
from transformers import AutoTokenizer
random.seed(42)


def shuffle_options_and_update_output(data_list):
    tokenizer = AutoTokenizer.from_pretrained("/path/to/internlm-7b", trust_remote_code=True)
    train_data_list = []
    train_shuffle_data_list = []
    valid_data_list = []
    all_sample_list = []
    new_data_list = []
    max_length = 0
    for entry in data_list:
        instruction = entry['instruction']
        question = entry['input']
        correct_option = entry['output'][-2]

        # 使用正则表达式匹配 A-D 选项
        pattern = r'(?<=\n)[A-D]\.\s*(.*?)(?=\n[A-D]\.|$)'
        matches = re.findall(pattern, question, re.S)

        # 检查是否获取了四个选项
        if len(matches) != 4:
            print(matches)
            matches = matches[1:]

        options = ["A", "B", "C", "D"]
        option_map = {options[i]: matches[i] for i in range(4)}

        new_data_list.append(
            {
                "instruction": instruction,
                "input": question,
                "output": "The right option is " + correct_option + "." + option_map[correct_option].strip()
            }
        )

        generated_permutations = set()
        shuffled_options = options[:]
        generated_permutations.add(tuple(shuffled_options))

        # 生成5次不同的随机排列
        sample_list = []
        for shuffle_round in range(5):
            while True:
                shuffled_options = options[:]
                random.shuffle(shuffled_options)
                # 确保排列是新的
                if tuple(shuffled_options) not in generated_permutations:
                    generated_permutations.add(tuple(shuffled_options))
                    break

            new_correct_id = shuffled_options.index(correct_option)

            new_input = question.split("\nA.")[0]
            for i in range(4):
                new_input = new_input + "\n" + options[i] + "." + option_map[shuffled_options[i]]
            new_output = "The right option is " + options[new_correct_id] + "." + option_map[shuffled_options[new_correct_id]].strip()

            input_ids = tokenizer.encode(new_output, return_tensors="pt", truncation=True)
            max_length = len(input_ids[0]) if len(input_ids[0]) > max_length else max_length

            new_data = {
                "instruction": instruction,
                "input": new_input,
                "output": new_output
            }
            sample_list.append(new_data)
        all_sample_list.append(sample_list)

    train_data_list.extend(new_data_list)

    shuffle_data_list = new_data_list
    random.shuffle(shuffle_data_list)
    train_shuffle_data_list.extend(shuffle_data_list)
    
    for i in range(4):
        epoch_list = []
        for j in range(len(all_sample_list)):
            epoch_list.append(all_sample_list[j][i])
        train_data_list.extend(epoch_list)
        shuffle_epoch_list = epoch_list
        random.shuffle(shuffle_epoch_list)
        train_shuffle_data_list.extend(shuffle_epoch_list)

    for i in range(len(all_sample_list)):
        valid_data_list.append(all_sample_list[i][4])
    
    print("train_len: ", len(all_sample_list))
    print("max_output_tokens: ", max_length)
    return train_data_list, train_shuffle_data_list, valid_data_list
        

if __name__ == "__main__":
    data_path = "/path/to/origin_train_alpaca_format.json"
    train_path = "/path/to/train_alpaca_format.json"
    train_shuffle_path = "/path/to/train_shuffle_alpaca_format.json"
    valid_path = "/path/to/valid_alpaca_format.json"
    test_path = "/path/to/test_alpaca_format.json"
    train_list, train_shuffle_list, valid_list, test_list = [], [], [], []

    with open(data_path, "r", encoding="utf-8") as file:
        data_list = json.load(file)

    train_data_list, train_shuffle_data_list, valid_data_list = shuffle_options_and_update_output(data_list)

    train_list = train_data_list
    train_shuffle_list = train_shuffle_data_list

    grouped_data = []
    for key, group in groupby(valid_data_list, key=lambda x: x["instruction"]):
        grouped_data.append(list(group))

    for data in grouped_data:
        random.shuffle(data)
        valid_list.extend(data[0:20])
        test_list.extend(data[20:40])

    with open(train_path, "w", encoding="utf-8") as json_file:
        json.dump(train_list, json_file, ensure_ascii=False, indent=4)
    with open(train_shuffle_path, "w", encoding="utf-8") as json_file:
        json.dump(train_shuffle_list, json_file, ensure_ascii=False, indent=4)
    with open(valid_path, "w", encoding="utf-8") as json_file:
        json.dump(valid_list, json_file, ensure_ascii=False, indent=4)
    with open(test_path, "w", encoding="utf-8") as json_file:
        json.dump(test_list, json_file, ensure_ascii=False, indent=4)