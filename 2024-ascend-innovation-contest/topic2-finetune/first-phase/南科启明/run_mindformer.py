
"""
fastchat stanford alpaca data convert tools.
"""

import argparse

import json

import pathlib

# Prompt from stanford alpaca's training script

PROMPT_TEMPLATES = {  
    "basic_arithmetic": "To solve this basic arithmetic problem, please follow the order of operations, paying close attention to the presence of negative signs to ensure neither adding an extra one nor missing one. Remember that two negative signs adjacent to each other are equivalent to a plus sign. Round the result to one decimal place. Do not output the code example; instead, output the final result.",  
    "exponentiation": "To find the power of a number, one must multiply the base by itself the number of times indicated by the exponent. Note that the even power of a negative number results in a positive number. Round the result to one decimal place. Do not output the code example; instead, output the final result.",  
    "square_root": "To find the square root of a number, one needs to locate a number that, when multiplied by itself, equals the given number. Round the result to one decimal place. Do not output the code example or any intermediate steps; simply output the final result.", 
    "equation_solution": "To find the solution to an equation, one needs to locate a number that, when substituted into the equation, makes the equation true or equalizes both sides of the equation. Round the result to one decimal place. Do not output the code example or any intermediate steps; simply present the final result.",
    "size": "To find the area of a rectangle, multiply the two input values (typically the length and width of the rectangle). Do not output the code example or any intermediate steps; simply present the final result.",
    "function": "To find the value of a function, substitute the input value into the function. Round the result to one decimal place. Do not output the code example or any intermediate steps; simply present the final result.",
    "rate": "To find the discount rate, one needs to calculate the percentage of the difference between the original price and the discounted price, relative to the original price. Round the result to one decimal place. Do not output the code example or any intermediate steps; simply present the final result.",
    "mass": "To find the mass, multiply the density by the volume. Do not output the code example or any intermediate steps; simply present the final result.",
    "mean": "To find the average of a series of numbers, you need to divide the sum of all the numbers by the total number of numbers. Round the result to one decimal place. Do not output the code example or any intermediate steps; simply present the final result.",
    "sale": "To find this year's sales, you need to calculate it based on last year's sales and the growth rate. Round the result to one decimal place. Do not output the code example or any intermediate steps; simply present the final result.",
    "fraction": "To simplify a fraction, you need to divide both the numerator and the denominator by their greatest common divisor. Do not output the code example or any intermediate steps; simply present the final simplified fraction.",
} 

def determine_problem_type(problem):  
    if "计算" and "次方" in problem:  
        return "exponentiation"  
    elif "计算" and "平方根" in problem:  
        return "square_root"   
    elif "解方程" in problem:  
        return "equation_solution"
    elif "长方形" and "面积" in problem:  
        return "size"
    elif "函数" in problem:  
        return "function"
    elif "折扣比例" in problem:  
        return "rate"
    elif "质量" in problem:  
        return "mass"
    elif "数据的平均值" in problem:  
        return "mean"
    elif "销售额" in problem:  
        return "sale"
    elif "分数" and "简化" in problem:  
        return "fraction"
    elif "计算" and any(op in problem for op in ["+", "-", "*", "/"]):  
        return "basic_arithmetic" 
    else:  
        return None  # Or you could return a default prompt template 

def main(args_param):

    data_path = pathlib.Path(args_param.data_path)

    sources = []
    targets = []
    problems=[]
    with data_path.open(encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            a = PROMPT_TEMPLATES[determine_problem_type(data['problem'])]
            b = data["solution"]
            c=data["problem"]
            sources.append(a)
            targets.append(b)
            problems.append(c)
    new_data = []

    cnt = 1

    for s, t, p in zip(sources, targets,problems):
        new_data.append(
            {
                "id": str(cnt),
                "conversations": [
                    {
                        "from": "human",
                        "value": s+"\n\n### Instruction:\n"+p+"\n\n### Response:",
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

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="train.json")
    parser.add_argument(
        "--output_path", type=str, default="train-data-conversation.json"
    )
    args = parser.parse_args()
    main(args)