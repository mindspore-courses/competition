# Finetue

----

### baseline finetune lora setting

32 layers of `wq` and `wv`, each weight split to W_a and W_b, 32 * 4 = 128 matrices in total

```python
"model.layers.{idx}.attention.(wq|wv).mindpet_delta_lora_(a|b)",
```


### our finetune

⚪ run-0

see [run-0/text_generation_result.txt](./run-0/text_generation_result.txt)

```
prompt: Below is an instruction that describes a task. Write a response that appropriately completes the request.
data: uniform_pick_17145
epoch: 5
bs: 32
lr: 1e-5
target_modules: .*wq|.*wv
n_param: 3407872

finetune speed: 5.66398 samples/s/p on NPU*4(32G)
finetune runtime: 1h23m57s = 1.3992h = 83.95min = 5037s
predict runtime: 15.15min = 909s

[EPS=1e-2]
samples_count: 200
correct rate: 7.50% (15)
bingo_cnt:
{0: [14, 137], 1: [0, 17], 2: [0, 12], 4: [1, 7], 5: [0, 13], 6: [0, 3], 7: [0, 4], 8: [0, 1], 9: [0, 5], 10: [0, 1]}
```

⚪ run-1

```
prompt: Below is a simple grade school math problem. Directly show the correct answer.
data: arith_15000
epoch: 5
bs: 64
lr: 1e-4
lora_dropout: 0.05
target_modules: .*wq|.*wv|.*wk
n_param: 4718592

finetune speed: 5.04060 samples/s/p on NPU*4(32G)
finetune runtime: 1h24min9s = 1.4025h = 84.15min = 5049s
predict runtime: 14.05min = 843s

[EPS=1e-2]
correct rate: 0.00% (0)
bingo_cnt:
{0: [0, 137], 1: [0, 17], 2: [0, 12], 4: [0, 7], 5: [0, 13], 6: [0, 3], 7: [0, 4], 8: [0, 1], 9: [0, 5], 10: [0, 1]}
```

⚪ run-2

```
prompt: Below is an instruction that describes a grade school math problem. Write a response that gives the correct answer.\n\n### Instruction:\n{problem}\n\n### Response:
output reprocess: CoT
data: easy_5000
epoch: 1
bs: 4
lr: 3e-5
target_modules: .*wq|.*wv
n_param: 3407872

F1 score: 70.45845460279635
Em score: 54.66860183841316
```

⚪ run-3

```
prompt: Below is an instruction that describes a grade school math problem. Write a response that gives the correct answer.\n\n### Instruction:\n{problem}\n\n### Response:
output reprocess: CoT_v2
data: easy_7500
epoch: 2
bs: 4
lr: 3e-5
target_modules: .*wq|.*wv
n_param: 3407872

F1 score: 71.18689963833116
Em score: 54.0396710208031
```
