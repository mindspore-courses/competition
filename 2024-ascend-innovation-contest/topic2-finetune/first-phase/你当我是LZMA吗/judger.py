#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/07/05 

# FUCK: 加了 prompt 和 CoT 之后自动判题有点困难了...

from re import Pattern
from argparse import ArgumentParser
import numpy as np
from utils import *

# 浮点数误差容忍阈值
EPS = 1e-2
isclose = lambda x, y: np.isclose(x, y, atol=EPS)


def get_problem_template(problem:str, solution:str) -> Tuple[int, Tuple[Pattern, Pattern]]:
  for idx, tmpl in enumerate(PROBLEM_TEMPLATES):
    if tmpl['Q'].match(problem) and tmpl['A'].match(solution):
      return idx, tmpl
  breakpoint()
  raise ValueError('unknown problem template')


def check_correct(problem:str, solution:str, predict:str) -> Tuple[int, bool]:
  if predict.startswith(problem):
    predict = predict[len(problem):].strip()

  tid, tmpl = get_problem_template(problem, solution)
  m_solt = tmpl['A'].findall(solution)[0]
  try:
    m_pred = tmpl['A'].findall(predict)[0]
  except IndexError:
    return tid, False

  ok = False
  try:
    if tid in [0, 1, 2]:
      a_res = float(m_solt[-1])
      p_res = float(m_pred[-1])
      ok = isclose(a_res, p_res)
    elif tid == 3:
      a_res1, a_res2 = [int(e) for e in m_solt]
      p_res1, p_res2 = [int(e) for e in m_pred]
      ok = a_res1 == p_res1 and a_res2 == p_res2
    elif tid in [4, 5, 6, 7, 8, 9, 10]:
      a_res = float(m_solt)
      p_res = float(m_pred)
      ok = isclose(a_res, p_res)
  except ValueError:
    return tid, False

  if not ok and 'show':
    print(f'solution: #{solution}#')
    print(f'predict: #{predict}#')

  return tid, ok


def get_acc(problems:List[str], solutions:List[str], predicts:List[str]) -> Tuple[float, Dict[int, Tuple[int, int]]]:
  ok, total = 0, 0
  bingo_cnt = {}
  for problem, solution, predict in zip(problems, solutions, predicts):
    tid, bingo = check_correct(problem, solution, predict)
    if tid not in bingo_cnt: 
      bingo_cnt[tid] = [0, 0]
    bingo_cnt[tid][0] += bingo
    bingo_cnt[tid][1] += 1
    ok += bingo
    total += 1
  return (ok / total), bingo_cnt


def run(args):
  pairs = load_testset()
  problems  = [p for p, a in pairs]
  solutions = [a for p, a in pairs]
  predict_list = np.load(args.R, allow_pickle=True)
  predicts = [it['text_generation_text'][0].strip() for it in predict_list]

  mcr, bingo_cnt = get_acc(problems, solutions, predicts)

  N = len(predicts)
  print('samples_count:', N)
  print(f'correct rate: {mcr:.2%} ({round(N * mcr)})')
  print('bingo_cnt:')
  print({k: bingo_cnt[k] for k in sorted(bingo_cnt)})


def run_show(args):
  pairs = load_testset()
  problems  = [p for p, a in pairs]
  solutions = [a for p, a in pairs]
  predict_list = np.load(args.R, allow_pickle=True)
  predicts = [it['text_generation_text'][0].strip() for it in predict_list]

  for prb, ans, slt in zip(problems, solutions, predicts):
    print('prb:', prb[134:-15])
    print('ans:', ans)
    print('slt:', slt[len(prb)+1:])
    print()


def _unittest_():
  problems = [
    '解方程 -1x + -17 = 0',
    '计算 -7431.41 / 6769.29 等于多少？',
    '求以下数据的平均值：[70, 18, 94]',
    '去年销售额为 32 万元，今年销售额增加了 28%，请计算今年的销售额。',
  ]
  solutions = [
    '方程的解为：-17.0',
    '-7431.41 / 6769.29 = -1.097812325960329665297246831',
    '平均值为 60.666666666666664',
    '40.96',
  ]
  predicts = [
    '方程的解为：-17.0',
    '-7431.41 / 6769.29 = -1.097888888',
    '平均值为 60.666666666666666',
    '40.96000',
  ]
  assert isclose(get_acc(problems, solutions, predicts)[0], 3/4)


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('-I', default=DATASET_TEST_FILE, type=Path, help='predict dataset')
  parser.add_argument('-R', default=(BASE_PATH / 'experiments' / 'test_eval_base_math' / 'result_npy.npy'), type=Path, help='predict result.npy file')
  args = parser.parse_args()

  #run(args)
  run_show(args)
