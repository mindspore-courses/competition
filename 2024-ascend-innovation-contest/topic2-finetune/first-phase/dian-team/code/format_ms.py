import json
import jsonlines

with jsonlines.open(r"E:\DianWork\DianGPT\昇腾AI\dataset\trainset_v2.json", mode="r") as reader:
    data = [i for i in reader]
    
formatted_data = []

for n, item in enumerate(data):
    formatted_sample = {"id": f"{n}", "conversations": []}
    template_q = """Below is an instruction that describes a task. Write a response that appropriately completes the request. ### Instruction: {query} ### Response:"""
    q = template_q.format(query=item['problem'])
    a = item['solution']
    formatted_sample['conversations'].append(
        {
            "from": "human",
            "value": q,
        }
    )
    formatted_sample['conversations'].append(
        {
            "from": "gpt",
            "value": a,
        }
    )
    formatted_data.append(formatted_sample)

json_file = open(r"E:\DianWork\DianGPT\昇腾AI\dataset\trainset_conversation_v2.json", "w", encoding="utf-8")

json.dump(formatted_data, json_file, ensure_ascii=False, indent=4)
