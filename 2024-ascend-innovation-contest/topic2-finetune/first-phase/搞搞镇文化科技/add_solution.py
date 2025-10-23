import json


# 定义函数以添加详细的解题步骤
def add_solution_steps(problem, solution):
    detailed_solution = solution  # 默认使用原始 solution
    try:
        if "计算" in problem:
            operation = problem.split("计算 ")[1].split(" 等于")[0]
            if "+" in operation:
                num1, num2 = map(float, operation.split(" + "))
                detailed_solution = f"首先，将 {num1} 和 {num2} 相加。\n{num1} + {num2} = {num1 + num2}\n最终结果为：{num1 + num2}"
            elif "-" in operation:
                num1, num2 = map(float, operation.split(" - "))
                detailed_solution = f"首先，将 {num1} 和 {num2} 相减。\n{num1} - {num2} = {num1 - num2}\n最终结果为：{num1 - num2}"
            elif "*" in operation:
                num1, num2 = map(float, operation.split(" * "))
                detailed_solution = f"首先，将 {num1} 和 {num2} 相乘。\n{num1} * {num2} = {num1 * num2}\n最终结果为：{num1 * num2}"
            elif "/" in operation:
                num1, num2 = map(float, operation.split(" / "))
                detailed_solution = f"首先，将 {num1} 和 {num2} 相除。\n{num1} / {num2} = {num1 / num2}\n最终结果为：{num1 / num2}"
            elif "^" in operation:
                base, exponent = map(float, operation.split("^"))
                detailed_solution = f"首先，将 {base} 提到 {exponent} 次方。\n{base}^{exponent} = {base ** exponent}\n最终结果为：{base ** exponent}"
        elif "解方程" in problem:
            equation = problem.split("解方程 ")[1]
            if "x + " in equation:
                a, b = map(float, equation.split("x + "))
                x = -b / a
                detailed_solution = f"首先，将方程化为标准形式：{a}x + ({b}) = 0\n将 {b} 移项得：{a}x = {-b}\n最后，x = {-b} / {a} = {x}"
            elif "x - " in equation:
                a, b = map(float, equation.split("x - "))
                x = b / a
                detailed_solution = f"首先，将方程化为标准形式：{a}x - {b} = 0\n将 {b} 移项得：{a}x = {b}\n最后，x = {b} / {a} = {x}"
        elif "平均值" in problem:
            numbers = list(map(float, problem.split("平均值：[")[1][:-1].split(", ")))
            avg = sum(numbers) / len(numbers)
            detailed_solution = f"首先，将所有数字相加：{' + '.join(map(str, numbers))} = {sum(numbers)}\n然后，将总和除以数字的个数：{sum(numbers)} / {len(numbers)} = {avg}\n最终结果为：{avg}"
        elif "面积" in problem:
            dimensions = list(map(float, problem.split("，")))
            length, width = dimensions
            area = length * width
            detailed_solution = f"首先，计算长方形的面积：长度 {length} 乘以 宽度 {width}\n{length} * {width} = {area}\n最终结果为：{area} 平方厘米"
        elif "平方根" in problem:
            number = float(problem.split("计算 ")[1].split(" 的平方根")[0])
            sqrt = number ** 0.5
            detailed_solution = f"首先，计算 {number} 的平方根。\n√{number} = {sqrt}\n最终结果为：{sqrt}"
    except Exception as e:
        print(f"Error processing problem '{problem}': {e}")

    return detailed_solution


# 读取文件并处理每一行数据
processed_data = []
with open("json/train_filter_choose_1w.json", "r", encoding="utf-8") as file:
    for line in file:
        item = json.loads(line)
        item["solution"] = add_solution_steps(item["problem"], item["solution"])
        processed_data.append(item)

# 将处理后的数据保存到新文件
with open("json/updated_train_filter_choose_1w.json", "w", encoding="utf-8") as file:
    for item in processed_data:
        file.write(json.dumps(item, ensure_ascii=False) + "\n")

print("数据处理完成，已保存到 updated_train_filter_choose_1w.json")
