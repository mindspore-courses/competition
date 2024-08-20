# 推理调优赛题

## 赛题介绍

### 第一阶段

基于给定数据集及后处理方法，跑通baseline，并对MindFormers中LLaMA2-7b模型进行推理调优，调优算法不限，在精度无损下（对比输出logits的误差，千分之五以内），推理性能相比baseline有提升，对推理总时间进行排名，推理时间越短排名越靠前。

1. 精度无损：此评价方法以对比推理单个token的logits为准，要求偏差在千分之五以内的作品方可视为有效作品，请选手按照官方提供的推理脚本获取特定token的logits，并保存为npy文件，如何获取logits及保存npy文件请参考指导手册-logits文件获取（待更新）

2. 推理总时间：因上述保存logits文件会增加额外耗时，所以建议选手运行两次：一次保存logits文件，一次不进行保存文件操作，仅作推理，推理总时间以后者为准，如何进行两次运行的配置，请参考指导手册-推理时长获取

3. 选手提交作品后，审核老师会检查代码是否包含前处理-推理-后处理全流程，且选手并没有通过如事先保存推理结果文件，然后直接读取文件进行推理等不正当方式缩短推理时间，一经发现有不正当手段即刻取消参赛资格


### 获奖作品展示

第一阶段推理调优赛题已完成，获奖团队如下所示：

| 作品名称 | 团队名称 | 推理时长（秒） |  排名  | 作品链接 |
|-------|-------|-------|  -------| -------| 
| moon.zip         | moon       | 669.8692       | 1    | https://github.com/mindspore-courses/competition/tree/master/2024-ascend-innovation-contest/topic3-inference/first-phase/moon  |
| 美滋滋学编程.zip | 美滋滋学编程 | 674.2927       | 2     |  https://github.com/mindspore-courses/competition/tree/master/2024-ascend-innovation-contest/topic3-inference/first-phase/美滋滋学编程   |
| faster.zip       | faster     | 679.9595       | 3    |   https://github.com/mindspore-courses/competition/tree/master/2024-ascend-innovation-contest/topic3-inference/first-phase/faster  |
| 西电超算.zip     | 西电超算   | 788.13775      | 4     |  https://github.com/mindspore-courses/competition/tree/master/2024-ascend-innovation-contest/topic3-inference/first-phase/西电超算  | 
| Introspection.zip| Introspection| 788.8821       |5      |  https://github.com/mindspore-courses/competition/tree/master/2024-ascend-innovation-contest/topic3-inference/first-phase/Introspection   |
| ccdd.zip         | ccdd       | 791.0843       | 6     |  https://github.com/mindspore-courses/competition/tree/master/2024-ascend-innovation-contest/topic3-inference/first-phase/ccdd   |
| 向日葵.zip       | 向日葵     | 791.3588       | 7     | https://github.com/mindspore-courses/competition/tree/master/2024-ascend-innovation-contest/topic3-inference/first-phase/向日葵  |  
| submit.zip       | 夏日弥光   | 794.8383       | 8     |  https://github.com/mindspore-courses/competition/tree/master/2024-ascend-innovation-contest/topic3-inference/first-phase/夏日弥光   |
| Mindspore-track3-URobot.zip | URobot   | 797.0908 |  9    |  https://github.com/mindspore-courses/competition/tree/master/2024-ascend-innovation-contest/topic3-inference/first-phase/URobot  | 
| hack_ai2.zip     | hack_ai2   | 797.8816       |  10    |  https://github.com/mindspore-courses/competition/tree/master/2024-ascend-innovation-contest/topic3-inference/first-phase/hack_ai2   |
| 汪汪队.zip   | 汪汪队     | 801.0472       | 11   |  https://github.com/mindspore-courses/competition/tree/master/2024-ascend-innovation-contest/topic3-inference/first-phase/汪汪队   |
| debias_world2.zip   | debias_world2 | 802.7824    | 12       |  https://github.com/mindspore-courses/competition/tree/master/2024-ascend-innovation-contest/topic3-inference/first-phase/debias_world2   |
| carryhjr.zip                  | carryhjr     | 835.1026    | 13 |  https://github.com/mindspore-courses/competition/tree/master/2024-ascend-innovation-contest/topic3-inference/first-phase/carryhjr   |
| 勇敢牛牛yyds.zip              | 勇敢牛牛yyds | 839.1604    | 14  |  https://github.com/mindspore-courses/competition/tree/master/2024-ascend-innovation-contest/topic3-inference/first-phase/勇敢牛牛yyds   |
| Ascend chenyun.pdf.zip        | Oops         | 857.1035    | 15   |  https://github.com/mindspore-courses/competition/tree/master/2024-ascend-innovation-contest/topic3-inference/first-phase/Oops   |
| 不羁混分战队.zip              | 不羁混分战队 | 1890.3587   | 16  | https://github.com/mindspore-courses/competition/tree/master/2024-ascend-innovation-contest/topic3-inference/first-phase/不羁混分战队    |
| 推理赛道_G.zip                | 摸鱼         | 2106.2045   | 17   |  https://github.com/mindspore-courses/competition/tree/master/2024-ascend-innovation-contest/topic3-inference/first-phase/摸鱼   |
| 啪的一下，很快啊，就寄了.zip | 啪的一下，很快啊，就寄了 | 3076.7487   | 18  |  https://github.com/mindspore-courses/competition/tree/master/2024-ascend-innovation-contest/topic3-inference/first-phase/啪的一下，很快啊，就寄了 |


### 第二阶段

基于给定数据集及后处理方法（Greedy Search），跑通baseline，并对MindFormers中LLaMA2-7b模型进行推理调优，优化技术不限，在精度无损下，推理性能相比baseline有提升，推理总时间越短越好。推理时间指数据集里所有的prompt全部推理结束所需要的时间。
1. 精度无损：此评价方法以对比推理单个token的logits为准，要求偏差在千分之五以内的作品方可视为有效作品，请选手提供指定tokens的logits，并保存为npy文件。如何获取logits及保存npy文件请参考操作指导手册。

2. 推理总时间：因上述保存logits文件会增加额外耗时，所以建议选手运行两次：一次保存logits文件，一次不进行保存文件操作，仅作推理，推理总时间以后者为准，如何进行两次运行的配置，请参考第一阶段操作指导手册。

3. 参赛选手可以参考入围赛中其他获奖选手已提交的作品进行优化，提升推理性能，入围赛的参赛作品获取地址：https://github.com/mindspore-courses/competition/tree/master/2024-ascend-innovation-contest/topic3-inference

4. 要求提交的作品①基于给定数据集及后处理方法， 选取数据集中前1500条数据进行单卡推理，推理性能（总时长）不得低于如下基线：600秒；②必须提供优化技术方案（例如投机推理等），仅修改超参将被视为无效作品。

5. 选手提交作品后，会对选手作品进行验收，审核老师也会检查代码是否包含前处理-推理-后处理全流程，且选手并没有通过，如事先保存推理结果文件，然后直接读取文件进行推理等不正当方式缩短推理时间，一经发现有不正当手段即刻取消参赛资格。

6.根据验收结果确定答辩入围名单。入围的选手需要做线上答辩，答辩20分钟左右，由评审专家对作品中的优化技术做综合评估后，进行得分排名，最终评选出金银铜奖。答辩PPT包括以下三个方面：
1）创新性：方案能提升推理性能，且相比于领域内同类方案有优势，鼓励从0到1的创新；
2）系统复杂性/技术复杂性：方案涉及的代码量，方案的泛化性好，鲁棒性强；
3）功能完备性：能正确跑通过代码工程，无返修和报错。


***后续操作详见[2024昇腾AI创新大赛MindSpore赛道实验指导手册](../2024昇腾AI创新大赛MindSpore赛道实验指导手册.pdf)***
