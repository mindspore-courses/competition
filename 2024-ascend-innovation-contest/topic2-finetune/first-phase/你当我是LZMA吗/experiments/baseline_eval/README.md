# Baseline Eval

----

### original task: SQuAD

⚠ runtime 1h7min = 67min = 4020s

see [test_eval_base.log](./test_eval_base.log)

```
Generation Config is: {
  'max_length': 227, 
  'max_new_tokens': 20, 
  'min_length': 0, 
  'min_new_tokens': None, 
  'num_beams': 1, 
  'do_sample': False, 
  'use_past': True, 
  'temperature': 1.0, 
  'top_k': 0, 
  'top_p': 1, 
  'repetition_penalty': 1, 
  'encoder_repetition_penalty': 1.0, 
  'renormalize_logits': False, 
  'pad_token_id': 128002, 
  'bos_token_id': 128000, 
  'eos_token_id': 128001, 
  '_from_model_config': True
}

F1 score: 59.87023775108988, Em score: 44.17029511369134, total_count: 2067
```


### new task: math

⚠ runtime 16.76min = 1006s

see [test_eval_base_math.log](./test_eval_base_math.log)

```
Generation Config is: {
  'max_length': 512, 
  'max_new_tokens': 20, 
  'min_length': 0, 
  'min_new_tokens': None, 
  'num_beams': 1, 
  'do_sample': False, 
  'use_past': True, 
  'temperature': 1.0, 
  'top_k': 0, 
  'top_p': 1, 
  'repetition_penalty': 1, 
  'encoder_repetition_penalty': 1.0, 
  'renormalize_logits': False, 
  'pad_token_id': 128002, 
  'bos_token_id': 128000, 
  'eos_token_id': 128001, 
  '_from_model_config': True
}

samples_count: 200
match rate: 0.00% (0)

[EPS=1e-8]
correct rate: 14.50% (29)
bingo_cnt:
{
   0: [28, 137], 
   1: [0, 17],
   2: [0, 12],
   4: [0, 7], 
   5: [0, 13], 
   6: [0, 3], 
   7: [0, 4], 
   8: [1, 1], 
   9: [0, 5], 
  10: [0, 1],
}

[EPS=1e-4]
correct rate: 15.50% (31)
bingo_cnt: {0: [30, 137], 1: [0, 17], 2: [0, 12], 4: [0, 7], 5: [0, 13], 6: [0, 3], 7: [0, 4], 8: [1, 1], 9: [0, 5], 10: [0, 1]}

[EPS=1e-2]
correct rate: 20.00% (40)
bingo_cnt: {0: [39, 137], 1: [0, 17], 2: [0, 12], 4: [0, 7], 5: [0, 13], 6: [0, 3], 7: [0, 4], 8: [1, 1], 9: [0, 5], 10: [0, 1]}
```
