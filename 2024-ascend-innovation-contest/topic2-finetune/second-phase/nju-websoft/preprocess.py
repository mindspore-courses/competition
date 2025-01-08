import json
import hashlib

original_path = '..\data\mmlu_alpaca_format.json'


def get_unique(lis):
    print(f"original list has length of {len(lis)}")  # 27604
    seen = set()
    unique_list = []
    for item in lis:
        hsh = json.dumps(item, ensure_ascii=False)
        if hsh not in seen:
            unique_list.append(item)
            seen.add(hsh)
    print(f"unique list has length of {len(unique_list)}")  # 27562
    return unique_list


with open(original_path, 'r', encoding='utf-8') as f:
    datalis = json.load(f)

datalis = get_unique(datalis)

with open("mmlu_train_unique.json", 'w', encoding='utf-8') as f:
    json.dump(datalis, f, ensure_ascii=False, indent=4)
