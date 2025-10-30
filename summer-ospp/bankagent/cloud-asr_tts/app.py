from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import uuid
import time
import base64
import logging
from datetime import datetime
from tencentcloud.common import credential
from tencentcloud.common.profile.client_profile import ClientProfile
from tencentcloud.tts.v20190823 import tts_client, models as tts_models
from tencentcloud.asr.v20190614 import asr_client, models as asr_models
from dotenv import load_dotenv

# ========================
# 初始化配置
# ========================
load_dotenv()
app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO)

# 音频存储目录
AUDIO_DIR = "audio_storage"
os.makedirs(AUDIO_DIR, exist_ok=True)

# 音色映射表（英文专用）
VOICE_MAP = {
    "standard_female": 1001,  # 标准女声
    "standard_male": 1002,  # 标准男声
    "news_female": 1015,  # 新闻女声
    "news_male": 1016  # 新闻男声
}


# ========================
# 核心服务类
# ========================
class SpeechService:
    def __init__(self):
        self.tts_client = self._init_tts_client()
        self.asr_client = self._init_asr_client()

    def _init_tts_client(self):
        """初始化腾讯云TTS客户端"""
        cred = credential.Credential(
            os.getenv('TENCENT_SECRET_ID'),
            os.getenv('TENCENT_SECRET_KEY')
        )
        cp = ClientProfile()
        cp.httpProfile.endpoint = "tts.tencentcloudapi.com"
        return tts_client.TtsClient(cred, "ap-shanghai", cp)

    def _init_asr_client(self):
        """初始化腾讯云ASR客户端"""
        cred = credential.Credential(
            os.getenv('TENCENT_SECRET_ID'),
            os.getenv('TENCENT_SECRET_KEY')
        )
        cp = ClientProfile()
        cp.httpProfile.endpoint = "asr.tencentcloudapi.com"
        return asr_client.AsrClient(cred, "ap-shanghai", cp)

    def text_to_speech(self, text, voice_type="standard_female"):
        """
        文本转语音（完整修复版）
        :param text: 要合成的文本
        :param voice_type: 音色类型
        :return: (文件路径, 文件名)
        """
        try:
            if not text or not isinstance(text, str):
                raise ValueError("Text must be a non-empty string")

            req = tts_models.TextToVoiceRequest()
            req.Text = text
            req.VoiceType = VOICE_MAP.get(voice_type, 1001)  # 默认标准女声
            req.PrimaryLanguage = 2  # 2=English

            # === 关键修复：添加所有必需参数 ===
            req.SessionId = f"dify_{uuid.uuid4().hex[:16]}"  # 唯一会话ID
            req.ModelType = 1  # 1=基础模型
            req.SampleRate = 16000  # 16kHz采样率
            # ================================

            resp = self.tts_client.TextToVoice(req)
            audio_data = base64.b64decode(resp.Audio)

            # 生成唯一文件名
            timestamp = int(time.time())
            random_str = uuid.uuid4().hex[:8]
            filename = f"tts_{timestamp}_{random_str}.mp3"
            filepath = os.path.join(AUDIO_DIR, filename)

            # 保存音频文件
            with open(filepath, "wb") as f:
                f.write(audio_data)

            return filepath, filename

        except Exception as e:
            logging.error(f"[TTS ERROR] {str(e)}", exc_info=True)
            raise

    def speech_to_text(self, audio_path):
        """
        语音转文本（支持WAV/MP3）
        :param audio_path: 音频文件路径
        :return: 识别文本
        """
        try:
            # 读取并编码音频
            with open(audio_path, "rb") as f:
                audio_data = base64.b64encode(f.read()).decode('utf-8')

            # 调用ASR接口
            req = asr_models.SentenceRecognitionRequest()
            req.EngSerViceType = "16k_zh"  # 中文普通话
            req.SourceType = 1  # 1=本地音频
            req.VoiceFormat = "wav" if audio_path.endswith(".wav") else "mp3"
            req.UsrAudioKey = os.path.basename(audio_path)
            req.Data = audio_data
            req.DataLen = len(audio_data)

            resp = self.asr_client.SentenceRecognition(req)
            return resp.Result

        except Exception as e:
            logging.error(f"[ASR ERROR] {str(e)}", exc_info=True)
            raise


# ========================
# Flask路由
# ========================
speech_service = SpeechService()


@app.route('/health', methods=['GET'])
def health_check():
    """健康检查接口"""
    return jsonify({
        "status": "running",
        "timestamp": datetime.now().isoformat(),
        "services": ["TTS", "ASR"]
    })


@app.route('/tts/generate', methods=['POST'])
def generate_tts():
    """
    TTS生成接口（Dify调用入口）
    请求格式：
    {
        "text": "Hello world",
        "voice": "standard_female"
    }
    """
    try:
        # 参数验证
        data = request.get_json()
        if not data:
            return jsonify({"error": "JSON body is required"}), 400

        text = data.get('text')
        if not text or not isinstance(text, str):
            return jsonify({"error": "Valid 'text' parameter is required"}), 400

        voice = data.get('voice', 'standard_female')
        if voice not in VOICE_MAP:
            return jsonify({"error": f"Invalid voice type. Allowed: {list(VOICE_MAP.keys())}"}), 400

        # 调用TTS服务
        _, filename = speech_service.text_to_speech(text, voice)

        # 构造响应
        return jsonify({
            "status": "success",
            "audio_url": f"{request.host_url}audio/{filename}",
            "text": text,
            "voice": voice,
            "timestamp": datetime.now().isoformat()
        })

    except Exception as e:
        logging.error(f"[API ERROR] {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e),
            "solution": "Check server logs for details"
        }), 500


@app.route('/asr/recognize', methods=['POST'])
def recognize_speech():
    """
    ASR识别接口（Dify调用入口）
    请求格式：form-data上传音频文件
    """
    try:
        if 'file' not in request.files:
            return jsonify({"error": "Audio file is required"}), 400

        audio_file = request.files['file']
        if audio_file.filename == '':
            return jsonify({"error": "Empty file"}), 400

        # 保存临时文件
        temp_filename = f"asr_temp_{uuid.uuid4().hex[:8]}.wav"
        temp_path = os.path.join(AUDIO_DIR, temp_filename)
        audio_file.save(temp_path)

        # 调用ASR
        text = speech_service.speech_to_text(temp_path)

        # 清理临时文件
        os.remove(temp_path)

        return jsonify({
            "status": "success",
            "text": text,
            "language": "zh-CN",
            "timestamp": datetime.now().isoformat()
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/audio/<filename>', methods=['GET'])
def get_audio(filename):
    """音频文件下载接口"""
    try:
        # 安全验证（防止路径遍历）
        if not (filename.startswith("tts_") or filename.startswith("asr_")) or not filename.endswith((".mp3", ".wav")):
            return jsonify({"error": "Invalid filename format"}), 400

        filepath = os.path.join(AUDIO_DIR, filename)
        if not os.path.exists(filepath):
            return jsonify({"error": "File not found"}), 404

        return send_file(filepath, mimetype='audio/mpeg')

    except Exception as e:
        logging.error(f"[AUDIO DOWNLOAD ERROR] {str(e)}")
        return jsonify({"error": "Internal server error"}), 500


# ========================
# 启动服务
# ========================
if __name__ == '__main__':
    # 检查环境变量
    if not os.getenv('TENCENT_SECRET_ID') or not os.getenv('TENCENT_SECRET_KEY'):
        logging.error("Missing Tencent Cloud credentials in .env file")
        exit(1)

    # 启动前清理旧文件
    for f in os.listdir(AUDIO_DIR):
        if f.startswith("tts_") or f.startswith("asr_"):
            filepath = os.path.join(AUDIO_DIR, f)
            if os.path.getmtime(filepath) < time.time() - 86400:  # 24小时
                os.remove(filepath)

    # 运行服务
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=False,  # 生产环境设为False
        threaded=True
    )