#项目总结：
MindSpore生成式套件提供了ASR（自动语音识别）、LM（语言模型）、TTS（文本转语音）等全链条语音交互能力，结合Dify多Agent协作引擎，可构建支持多业务智能体联动的金融语音客服系统，实现从语音导航到复杂业务处理的全流程智能化。
1. 应用MindSporeNLP和生成式套件的ASR + lm +TTS功能：
   应用mindspore框架训练asr和tts功能，并结合开源的华为云语音服务等api实现输入的语音转文字和输出的文本转换为语音服务。
2. 搭建了多个业务智能体，包括认证，查询，投资顾问，投诉等；完成多个智能体交互和意图识别功能：
   搭建认证、查询、投资顾问、投诉等业务智能体，实现跨智能体任务流转与信息共享。
<img width="1489" height="914" alt="WPS图片(1)" src="https://github.com/user-attachments/assets/19658032-4f09-4576-87e1-9f09652aaac3" />
3.搭建后端服务层：大数据引擎用于用户画像；风控引擎用于规则和安全：
当用户输入的要求涉及风险时，自动识别触发风控引擎；同时还建立了用户画像工具，为每一个的用户信用等级等评分，在调用理财推荐等功能
时可以调用该工具。
<img width="498" height="535" alt="image" src="https://github.com/user-attachments/assets/4dcc5267-61be-4260-a0cd-243f3b36deec" />

#环境要求
Python 3.9+
FastAPI
Uvicorn

#运行结果演示
<img width="346" height="254" alt="image" src="https://github.com/user-attachments/assets/f9062331-9432-42d6-b933-e219ca4a1e01" />
<img width="373" height="564" alt="image" src="https://github.com/user-attachments/assets/d5901fec-de79-4d64-972d-5b8590723293" />
<img width="364" height="387" alt="image" src="https://github.com/user-attachments/assets/fb2e7158-c828-48a6-aff4-50408b9d16af" />
<img width="335" height="409" alt="image" src="https://github.com/user-attachments/assets/ab9ba678-8891-4959-9abf-4c473e8cb4da" />

项目完成时间：2024年暑期 | 技术栈：Python + FastAPI + 规则引擎 + 语音技术




