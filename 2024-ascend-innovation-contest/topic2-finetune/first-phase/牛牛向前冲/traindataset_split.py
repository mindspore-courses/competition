import argparse
import json
import pathlib
import random
import os
import re

os.chdir(os.path.dirname(os.path.abspath(__file__)))
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
    file_path = pathlib.Path(args_param.data_path)

    # Read the data from the file
    with open(file_path, 'r', encoding='utf-8') as file:
        data = [json.loads(line) for line in file]

    def contains_chinese(text):
        # 使用正则表达式匹配中文字符
        return re.search('[\u4e00-\u9fff]', text) is not None

    additions = []
    subtractions = []
    multiplications = []
    divisions = []
    equations = []
    functions = []
    applications = []
    exponents_roots = []
    simplifications = []
    avg_value = []
    english_problem = []
    for item in data:
        problem = item['problem']
        '''x
        if contains_chinese(problem) is False:
            english_problem.append({
                "problem": PROMPT_DICT["prompt_no_input"].format(problem=problem, input=""),
                "solution": item["solution"]
            })
        el
        '''
        if "解方程" in problem:
            equations.append({
                "problem": PROMPT_DICT["prompt_no_input"].format(problem=problem, input=""),
                "solution": item["solution"]
            })
        elif "函数" in problem:
            functions.append({
                "problem": PROMPT_DICT["prompt_no_input"].format(problem=problem, input=""),
                "solution": item["solution"]
            })
        elif "次方" in problem or "方根" in problem:
            exponents_roots.append({
                "problem": PROMPT_DICT["prompt_no_input"].format(problem=problem, input=""),
                "solution": item["solution"]
            })
        elif "简化" in problem:
            simplifications.append({
                "problem": PROMPT_DICT["prompt_no_input"].format(problem=problem, input=""),
                "solution": item["solution"]
            })
        elif "平均值" in problem:
            avg_value.append({
                "problem": PROMPT_DICT["prompt_no_input"].format(problem=problem, input=""),
                "solution": item["solution"]
            })
        elif " + " in problem:
            additions.append({
                "problem": PROMPT_DICT["prompt_no_input"].format(problem=problem, input=""),
                "solution": item["solution"]
            })
        elif " - " in problem:
            subtractions.append({
                "problem": PROMPT_DICT["prompt_no_input"].format(problem=problem, input=""),
                "solution": item["solution"]
            })
        elif " * " in problem:
            multiplications.append({
                "problem": PROMPT_DICT["prompt_no_input"].format(problem=problem, input=""),
                "solution": item["solution"]
            })
        elif " / " in problem:
            divisions.append({
                "problem": PROMPT_DICT["prompt_no_input"].format(problem=problem, input=""),
                "solution": item["solution"]
            })
        else:
            applications.append({
                "problem": PROMPT_DICT["prompt_no_input"].format(problem=problem, input=""),
                "solution": item["solution"]
            })

    ## 开始拼接数据集，训练数据集为9w数据，enlish全部，add等基础运算1w，equation 2.2w，function 7k，app 1.8w, exp 1.5w, simple 5k, awg 5k
    train_samples= (random.sample(equations,25000)).copy()
    train_samples.extend(random.sample(additions,5000))
    train_samples.extend(random.sample(subtractions,5000))
    train_samples.extend(random.sample(multiplications,5000))
    train_samples.extend(random.sample(divisions,5000))
    train_samples.extend(random.sample(functions,10000))
    train_samples.extend(random.sample(exponents_roots,17000))
    train_samples.extend(random.sample(simplifications,7000))
    train_samples.extend(random.sample(avg_value,7000))
    train_samples.extend(random.sample(applications,20000))



    # Create new data for training and testing
    def create_data(samples, output_path):
        new_data = []
        cnt = 1
        for line in samples:
            s, t = line['problem'], line['solution']
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
        json.dump(new_data, open(output_path, "w", encoding='utf-8'), ensure_ascii=False, indent=2)

    create_data(train_samples, args_param.train_output_path)
    #create_data(test_samples, args_param.test_output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="train.json")
    parser.add_argument("--train_output_path", type=str, default="110k-train-data-conversation-v3.json")
    parser.add_argument("--test_output_path", type=str, default="test-data-conversation.json")
    args = parser.parse_args()
    main(args)
