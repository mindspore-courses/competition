import os
import pandas as pd
import json

# 文件夹路径
base_dir = os.path.dirname(__file__)  # 获取当前文件的目录
path_list = [os.path.join(base_dir, 'cmmlu/'),os.path.join(base_dir, 'data/dev/'),os.path.join(base_dir, 'data/test/'),os.path.join(base_dir, 'data/val/')]

def read_csv(folder_path):
    # 获取文件夹中所有后缀名为 .csv 的文件
    csv_files = [file for file in os.listdir(folder_path) if file.endswith('.csv')]

    # 创建一个空的列表，用于存储所有 CSV 文件的内容
    all_data = []

    id = 0
    # 逐个读取 CSV 文件并存储到列表中
    for file in csv_files:
        file_path = os.path.join(folder_path, file)
        df = pd.read_csv(file_path)
        if 'cmmlu' not in file_path:
            df.columns = ['Question', 'A', 'B', 'C', 'D', 'Answer']
        # 将 DataFrame 转换为字典格式，并添加到列表中
        dict_data = df.to_dict(orient='records')
        for item in dict_data:
            domain =  file.replace('_sum', '').replace('_dev', '').replace('_test', '').replace('_val', '').replace('_', ' ').replace('.csv', '')
            all_data.append({
                'instruction': f"Here is a question about {domain}, the correct answer is one of the options A/B/C/D. Please use the knowledge about {domain} to select the correct option and answer the question with 'The right option is'.",
                'input': 'Question: ' + item['Question'] + ' \nA.' + str(item['A']) + '\nB.' + str(item['B']) + '\nC.' + str(item['C']) + '\nD.' + str(item['D']),
                'output': 'The right option is ' + item['Answer'] + '.'
            })
    print(f"已读取 {len(csv_files)} 个 CSV 文件, 共 {len(all_data)} 条数据")
    return all_data
    

json_data = []
for folder_path in path_list:
    json_data.extend(read_csv(folder_path))
print(f"合计 {len(json_data)} 条数据")
# 写入到 JSON 文件中
json_output_path = os.path.join(base_dir, 'r4_mmlu_alpaca_format.json')
with open(json_output_path, 'w', encoding='utf-8') as json_file:
    json.dump(json_data, json_file, ensure_ascii=False, indent=4)
