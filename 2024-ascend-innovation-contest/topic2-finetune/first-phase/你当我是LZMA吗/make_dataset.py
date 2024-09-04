#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/07/04 

# 挑选训练数据
# - 中文 80w + 英文 9993 = 总计 809993
# - 中文都是小学数学模板式简短问答
# - 英文是长问答，基本定论不可能回答正确

from argparse import ArgumentParser
from utils import *
from judger import get_problem_template


PROMPTS = {
  'alpaca': "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{problem}\n\n### Response:",
  'math': "Below is an instruction that describes a grade school math problem. Write a response that gives the correct answer.\n\n### Instruction:\n{problem}\n\n### Response:",
  'none': "{problem}",
}

def render_prompt(samples:QA, tmpl:str='math') -> List[Dict]:
  prompt_no_input = PROMPTS[tmpl]
  sources = []
  targets = []
  for data in samples:
    sources.append(prompt_no_input.format_map({'problem': data[0]}))
    targets.append(data[1])

  samples_rendered = []
  cnt = 1
  for s, t in zip(sources, targets):
    samples_rendered.append({
      "id": str(cnt),
      "conversations": [
        { "from": "human", "value": s },
        { "from": "gpt",   "value": t },
      ],
    })
    cnt += 1
  return samples_rendered


def stats_pattern_problem_count():
  # [359347, 39207, 39227, 45, 19709, 40080, 20028, 8665, 100, 6275, 6266]
  pairs_raw_ch = load_dataset_raw()
  pattern_problem_count = [0] * len(PROBLEM_TEMPLATES)
  for prb, ans in pairs_raw_ch:
    matched = False
    for idx, tmpl in enumerate(PROBLEM_TEMPLATES):
      if tmpl['Q'].match(prb) and tmpl['A'].match(ans):
        pattern_problem_count[idx] += 1
        matched = True
        break
    if not matched:
      print(prb, ans)
  print('[pattern_problem_count]')
  print(pattern_problem_count)


def round_s(s:str, n_prec:int=2) -> str:
  return str(round(float(s), n_prec))

def neg_s(s:str) -> str:
  return str(-float(s))

def make_CoT(samples:QA) -> QA:
  samples_CoT = []
  for prb, ans in samples:
    tid, tmpl = get_problem_template(prb, ans)

    # 在答案中追加算法步骤，并修正答案精度
    if tid == 0:
      # 计算 ([\d\. \+\-\*\/]+) 等于多少？
      left, right = tmpl['A'].findall(ans)[0]
      a, op, b = left.split(' ')
      a_sign = str(a).startswith('-')
      b_sign = str(b).startswith('-')

      # handle right
      right = round_s(right, 0 if op == '*' else 2)
      # fix sign
      if op in ['*', '/'] and a_sign and b_sign:
        a = a[1:]
        b = b[1:]
        ans_CoT = f'{left} = {a} {op} {b} = {right}'
      elif op == '-' and b_sign:    # 减负数
        b = b[1:]
        op = '+'
        ans_CoT = f'{left} = {a} {op} {b} = {right}'
      elif op == '+' and b_sign:    # 加负数
        b = b[1:]
        op = '-'
        ans_CoT = f'{left} = {a} {op} {b} = {right}'
      else:
        ans_CoT = f'{left} = {right}'

    elif tid == 1:
      # 计算 (-?[\d\.]+) 的 (\d+) 次方？
      base, index, result = tmpl['A'].findall(ans)[0]
      result = round_s(result, 0)
      ans_CoT = f'{base}^{index} = {result}'

    elif tid == 2:
      # 计算 ([\d\.]+) 的平方根？
      a, b = tmpl['A'].findall(ans)[0]
      b = round_s(b, 1)
      ans_CoT = f'√{a} = {b}'

    elif tid == 3:
      # 将分数 (\d+)/(\d+) 进行简化。
      ans_CoT = ans

    elif tid == 4:
      # 求以下数据的平均值：(\[[\d, ]+\])
      nums = eval(tmpl['Q'].findall(prb)[0])
      nums_sum = sum(nums)
      v = round_s(tmpl['A'].findall(ans)[0])
      CoT = [
        ' + '.join([str(e) for e in nums]) + f' = {nums_sum}',
        f'{nums_sum} / {len(nums)} = {v}', 
      ]
      ans_CoT = '\n'.join(CoT) + f'\n平均值为: {v}'

    elif tid == 5:
      # 解方程 (-?\d+)x \+ (-?\d+) = 0
      k, b = tmpl['Q'].findall(prb)[0]
      v = round_s(tmpl['A'].findall(ans)[0])
      b_neg = neg_s(b)
      ans_CoT = f'方程为 {k}x = {b_neg}，因此 x = {b_neg} / {k}  = {v}。方程的解为：{v}'

    elif tid == 6:
      # 当 x = (-?[\d\.]+) 时，求函数 y = (-?\d+)x\^(\d+) 的值
      x, k, b = tmpl['Q'].findall(prb)[0]
      v = round_s(tmpl['A'].findall(ans)[0])
      CoT = [
        f'y = {k}*{x}^{b}',
        f'y = {k}*{round_s(float(x)**float(b))}',
        f'y = {round_s(int(k)*(float(x)**float(b)))}',
      ]
      ans_CoT = '\n'.join(CoT) + f'\n函数的值为：{v}'

    elif tid == 7:
      # 一个长方形的长为 (\d+) 厘米，宽为 (\d+) 厘米，请计算其面积。
      a, b = tmpl['Q'].findall(prb)[0]
      ans_CoT = f'{a} * {b} = {int(a) * int(b)}\n{ans}'

    elif tid == 8:
      # 某物体的密度为 (\d+) 克/立方厘米，体积为 (\d+) 立方厘米，请计算该物体的质量。
      a, b = tmpl['Q'].findall(prb)[0]
      ans_CoT = f'{a} * {b} = {int(a) * int(b)}\n{ans}'

    elif tid == 9:
      # 商品原价为 (\d+) 元，打折后的价格为 (\d+) 元，请计算打折的折扣比例。
      a, b = tmpl['Q'].findall(prb)[0]
      CoT = [
        f'= ({a} - {b}) / {a} * 100',
        f'= {int(a) - int(b)} / {a} * 100',
        f'= {round_s((int(a) - int(b)) / int(a), 4)} * 100',
        f'= {round_s(((int(a) - int(b)) / int(a)) * 100)}',
      ]
      ans_CoT = '\n'.join(CoT)

    elif tid == 10:
      # 去年销售额为 (\d+) 万元，今年销售额增加了 (\d+)%，请计算今年的销售额。
      a, b = tmpl['Q'].findall(prb)[0]
      ans_CoT = f'= {a} * 1.{b} = {round_s(ans)}'

    samples_CoT.append((prb, ans_CoT))
  return samples_CoT


def write_data_subset(subset, fp:Path, is_test:bool=False):
  print(f'>> write file: {fp}')
  with open(fp, 'w', encoding='utf-8') as fh:
    if is_test:
      for s in subset:
        qa = {
          'problem':  s['conversations'][0]['value'],
          'solution': s['conversations'][1]['value'],
        }
        fh.write(json.dumps(qa, ensure_ascii=False, indent=None))
        fh.write('\n')
    else:
      json.dump(subset, fh, ensure_ascii=False, indent=None)


def make_testset(N:int=200):
  if DATASET_TEST_FILE.exists():
    print('>> ignore due to file exists')
    return
  samples = load_dataset_raw()
  subset = random.sample(samples, N)

  subset = make_CoT(subset)
  subset = render_prompt(subset)
  write_data_subset(subset, DATASET_TEST_FILE, is_test=True)


def make_trainset_uniform_pick(N:int=5000):
  # [359347, 39207, 39227, 45, 19709, 40080, 20028, 8665, 100, 6275, 6266]
  pairs_raw_ch = load_dataset_raw()
  pattern_problems: Dict[int, List] = {}
  for prb, ans in pairs_raw_ch:
    idx, tmpl = get_problem_template(prb, ans)
    if idx not in pattern_problems:
      pattern_problems[idx] = []
    pattern_problems[idx].append((prb, ans))

  subset = []
  for i in range(11):
    subprbs = pattern_problems[i]
    if i == 0:
      st = random.sample(subprbs, 5000)
    elif len(subprbs) < 1000:
      st = subprbs
    else:
      st = random.sample(subprbs, 1500)
    st.sort()
    subset.extend(st)
  nlen = len(subset)
  print('len(subset):', nlen)

  subset = make_CoT(subset)
  subset = render_prompt(subset)
  write_data_subset(subset, BASE_PATH / f'data_uniform_pick_{nlen}.json')


def make_trainset_arith(N:int=15000):
  # [359347], 加减乘除 only
  pairs_raw_ch = load_dataset_raw()
  arith_problems: List = []
  for prb, ans in pairs_raw_ch:
    idx, tmpl = get_problem_template(prb, ans)
    if idx != 0: continue
    arith_problems.append((prb, ans))

  subset = random.sample(arith_problems, N)
  nlen = len(subset)
  print('len(subset):', nlen)

  subset = make_CoT(subset)
  subset = render_prompt(subset)
  write_data_subset(subset, BASE_PATH / f'data_arith_{nlen}.json')


def make_trainset_easy(N:int=7500):
  # 忽略困难的题:
  # [tid=0] 算数乘法
  # [tid=1] 次方
  # [tid=2] 开根
  # [tid=3] 分数化简
  # [tid=6] 已知自变量求函数值

  pairs_raw_ch = load_dataset_raw()
  easy_problems: List = []
  for prb, ans in pairs_raw_ch:
    idx, tmpl = get_problem_template(prb, ans)
    if idx in [1, 2, 3, 6]: continue
    if idx == 0 and '*' in ans: continue
    easy_problems.append((prb, ans))

  subset = random.sample(easy_problems, N)
  nlen = len(subset)
  print('len(subset):', nlen)

  subset = make_CoT(subset)
  subset = render_prompt(subset)
  write_data_subset(subset, BASE_PATH / f'data_easy_{nlen}.json')


if __name__ == '__main__':
  TRAINSET_MAKER = [name[len('make_trainset_'):] for name in globals() if name.startswith('make_trainset_')]

  parser = ArgumentParser()
  parser.add_argument('--split', default='train', choices=['test', 'train'])
  parser.add_argument('--maker', default='easy', choices=TRAINSET_MAKER + ['none'])
  args = parser.parse_args()

  suffix = f'_{args.maker}' if args.maker != 'none' else ''
  maker = globals()[f'make_{args.split}set{suffix}']
  maker()
