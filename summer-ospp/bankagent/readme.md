# 项目总结：

MindSpore生成式套件提供了ASR（自动语音识别）、LM（语言模型）、TTS（文本转语音）等全链条语音交互能力，结合Dify多Agent协作引擎，可构建支持多业务智能体联动的金融语音客服系统，实现从语音导航到复杂业务处理的全流程智能化。

1. 应用MindSporeNLP和生成式套件的ASR + lm +TTS功能：
   应用mindspore框架训练asr和tts功能，并结合开源的华为云语音服务等api实现输入的语音转文字和输出的文本转换为语音服务。
2. 搭建了多个业务智能体，包括认证，查询，投资顾问，投诉等；完成多个智能体交互和意图识别功能：

   搭建认证、查询、投资顾问、投诉等业务智能体，实现跨智能体任务流转与信息共享。
   <img width="1489" height="914" alt="WPS图片(1)" src="https://github.com/user-attachments/assets/19658032-4f09-4576-87e1-9f09652aaac3" />

3. 搭建后端服务层：大数据引擎用于用户画像；风控引擎用于规则和安全：
当用户输入的要求涉及风险时，自动识别触发风控引擎；同时还建立了用户画像工具，为每一个的用户信用等级等评分，在调用理财推荐等功能
时可以调用该工具。
   <img width="498" height="535" alt="image" src="https://github.com/user-attachments/assets/4dcc5267-61be-4260-a0cd-243f3b36deec" />

# 环境要求

Python 3.9+
FastAPI
Uvicorn

### 安装步骤

1. **准备模型文件**

```bash
pip install mindspore
```

2. **配置环境**

```bash
cd ../../
pip install -r requestment.txt
```

3. **启动服务**

```bash
npm run dev
cd ../llm_service
python app.py
```
### 项目结构

```text
─BankAI-Assistant
│  ├─bankagent              # 银行代理服务模块
│  │  ├─app.py              # 代理服务主程序
│  │  ├─asr.py              # 语音识别服务
│  │  ├─tts.py              # 文本转语音服务
│  │  ├─luyin.py            # 录音功能模块
│  │  ├─huaweicloud_sis.py   # 华为云语音接口
│  │  ├─utils_audio.py       # 音频处理工具
│  │  ├─env.example          # 环境配置示例
│  │  └─requirements.txt     # Python依赖包列表
│  ├─bank-user              # 银行用户管理模块
│  │  ├─app.py              # 用户服务主程序
│  │  ├─account_manager.py   # 账户管理功能
│  │  ├─asr.py              # 语音识别模块
│  │  ├─recommend.py         # 智能推荐功能
│  │  ├─dify_integration.py  # Dify平台集成
│  │  ├─generate_test_data.py # 测试数据生成
│  │  ├─test_asr.py          # 语音识别测试
│  │  ├─accounts.json        # 账户数据文件
│  │  ├─bank_users.json      # 用户信息数据
│  │  └─bank_users_en.json   # 英文用户数据
│  ├─tencent_tts            # 腾讯语音服务模块
│  │  ├─app.py              # TTS服务主程序
│  │  ├─asr.py              # 语音识别功能
│  │  ├─tts.py              # 文本转语音核心
│  │  ├─utils_audio.py      # 音频工具函数
│  │  ├─requirements.txt    # 依赖包配置
│  │  ├─audio_storage       # 音频文件存储
│  │  │  ├─tts_*.mp3        # 生成的语音文件
│  │  │  └─english_output.mp3 # 英文语音输出
│  │  └─__pycache__         # Python编译缓存
│  │      └─speech_client.cpython-39.pyc # 语音客户端缓存
│  ├─shared                 # 共享工具模块
│  │  ├─account_manager.py  # 通用账户管理
│  │  ├─import_requests.py  # 请求处理工具
│  │  ├─test_encoding.py    # 编码测试工具
│  │  ├─accounts.json       # 共享账户数据
│  │  ├─bank_users.json     # 共享用户数据
│  │  ├─tickets.json       # 业务票据数据
│  │  └─tts_output          # 共享语音输出
│  │      ├─tts_*.mp3      # 语音输出文件
│  └─docs                   # 项目文档
│      ├─README.md          # 项目说明文档
```

# 运行结果演示

<img width="346" height="254" alt="image" src="https://github.com/user-attachments/assets/f9062331-9432-42d6-b933-e219ca4a1e01" />
<img width="373" height="564" alt="image" src="https://github.com/user-attachments/assets/d5901fec-de79-4d64-972d-5b8590723293" />
<img width="364" height="387" alt="image" src="https://github.com/user-attachments/assets/fb2e7158-c828-48a6-aff4-50408b9d16af" />
<img width="335" height="409" alt="image" src="https://github.com/user-attachments/assets/ab9ba678-8891-4959-9abf-4c473e8cb4da" />

项目完成时间：2025年暑期 | 技术栈：Python + FastAPI + 规则引擎 + 语音技术
