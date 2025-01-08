import os
import json
import random
import pandas as pd
random.seed(42)


if __name__ == "__main__":
    save_path = "/path/to/origin_train_alpaca_format.json"
    cmmlu_path_list = ["/path/to/cmmlu/dev/", "/path/to/cmmlu/test/"]
    mmlu_path_list = ["/path/to/mmlu/data/dev", "/path/to/mmlu/data/test", "/path/to/mmlu/data/val"]

    train_data = []

    csv_files = [file for file in os.listdir(cmmlu_path_list[0]) if file.endswith(".csv")]
    for file in csv_files:
        data_list = []
        for folder_path in cmmlu_path_list:
            file_path = os.path.join(folder_path, file)
            df = pd.read_csv(file_path)
            if "cmmlu" not in file_path:
                df.columns = ["Question", "A", "B", "C", "D", "Answer"]
            # 将 DataFrame 转换为字典格式，并添加到列表中
            dict_data = df.to_dict(orient="records")
            for item in dict_data:
                domain =  file.replace("_dev", "").replace("_test", "").replace("_val", "").replace("_", " ").replace(".csv", "")
                data_list.append({
                    "instruction": f"Here is a question about {domain}, the correct answer is one of the options A/B/C/D. Please select the correct option and answer the question with 'The right option is'.",
                    "input": "Question: " + item["Question"] + " \nA." + str(item["A"]) + "\nB." + str(item["B"]) + "\nC." + str(item["C"]) + "\nD." + str(item["D"]),
                    "output": "The right option is " + item["Answer"] + "."
                })
        random.shuffle(data_list)
        train_data.extend(data_list)
        print("cmmlu: ", domain, len(data_list))

    csv_files = [file for file in os.listdir(mmlu_path_list[0]) if file.endswith(".csv")]
    for file in csv_files:
        data_list = []
        i = 0
        for folder_path in mmlu_path_list:
            i += 1
            if i == 2:
                file = file.replace("_dev", "_test")
            elif i == 3:
                file = file.replace("_test", "_val")
            file_path = os.path.join(folder_path, file)
            df = pd.read_csv(file_path)
            if "cmmlu" not in file_path:
                df.columns = ["Question", "A", "B", "C", "D", "Answer"]
            # 将 DataFrame 转换为字典格式，并添加到列表中
            dict_data = df.to_dict(orient="records")
            for item in dict_data:
                domain =  file.replace("_dev", "").replace("_test", "").replace("_val", "").replace("_", " ").replace(".csv", "")
                data_list.append({
                    "instruction": f"Here is a question about {domain}, the correct answer is one of the options A/B/C/D. Please select the correct option and answer the question with 'The right option is'.",
                    "input": "Question: " + item["Question"] + " \nA." + str(item["A"]) + "\nB." + str(item["B"]) + "\nC." + str(item["C"]) + "\nD." + str(item["D"]),
                    "output": "The right option is " + item["Answer"] + "."
                })
        random.shuffle(data_list)
        train_data.extend(data_list)
        print("mmlu: ", domain, len(data_list))

    with open(save_path, "w", encoding="utf-8") as json_file:
        json.dump(train_data, json_file, ensure_ascii=False, indent=4)

    print("train_data: ", len(train_data))