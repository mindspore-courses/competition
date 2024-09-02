def extract_lines(input_file, output_file, interval=2000):
    with open(input_file, 'r', encoding='utf-8') as infile:
        lines = infile.readlines()

    selected_lines = [line for index, line in enumerate(lines) if index % interval == 0]

    with open(output_file, 'w', encoding='utf-8') as outfile:
        outfile.writelines(selected_lines)


# 调用函数
input_file = r'C:\Users\49470\Downloads\train.json'  # 替换为你的输入文件路径
output_file = 'radom2000_1.json'  # 替换为你的输出文件路径
extract_lines(input_file, output_file)
