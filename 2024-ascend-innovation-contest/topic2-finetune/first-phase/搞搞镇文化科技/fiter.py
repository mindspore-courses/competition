import json


def remove_key_from_json(file_path, output_path, key_to_remove):
    entries = []

    # 读取并处理文件中的每一行
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            entry = json.loads(line)

            # 删除指定的键
            if key_to_remove in entry:
                del entry[key_to_remove]

            entries.append(entry)

    # 将处理后的数据写入新文件
    with open(output_path, 'w', encoding='utf-8') as file:
        for entry in entries:
            file.write(json.dumps(entry, ensure_ascii=False) + '\n')


def main():
    input_path = 'train.json'
    output_path = 'test.json'
    key_to_remove = 'solution'

    remove_key_from_json(input_path, output_path, key_to_remove)

    print(f"数据已从 {input_path} 复制并去除了 '{key_to_remove}' 键，结果保存在 {output_path}")


if __name__ == "__main__":
    main()
