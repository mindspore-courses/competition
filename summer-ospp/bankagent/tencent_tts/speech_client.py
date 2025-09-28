import os
import base64
import logging
from tencentcloud.common import credential
from tencentcloud.common.profile.client_profile import ClientProfile
from tencentcloud.asr.v20190614 import asr_client, models as asr_models
from tencentcloud.tts.v20190823 import tts_client, models as tts_models
from dotenv import load_dotenv

load_dotenv()


class TencentSpeech:
    def __init__(self):
        # 初始化认证
        self.secret_id = os.getenv('TENCENT_SECRET_ID')
        self.secret_key = os.getenv('TENCENT_SECRET_KEY')
        self.region = os.getenv('TENCENT_REGION', 'ap-shanghai')

        # 创建客户端
        cred = credential.Credential(self.secret_id, self.secret_key)
        self.asr_client = self._init_asr_client(cred)
        self.tts_client = self._init_tts_client(cred)

    def _init_asr_client(self, cred):
        """初始化ASR客户端"""
        cp = ClientProfile()
        cp.httpProfile.endpoint = "asr.tencentcloudapi.com"
        return asr_client.AsrClient(cred, self.region, cp)

    def _init_tts_client(self, cred):
        """初始化TTS客户端"""
        cp = ClientProfile()
        cp.httpProfile.endpoint = "tts.tencentcloudapi.com"
        return tts_client.TtsClient(cred, self.region, cp)

    def recognize(self, audio_path):
        """语音识别（ASR）
        :param audio_path: 音频文件路径（支持wav/mp3）
        :return: 识别文本
        """
        try:
            # 读取并编码音频
            with open(audio_path, "rb") as f:
                audio_data = base64.b64encode(f.read()).decode('utf-8')

            req = asr_models.SentenceRecognitionRequest()
            req.ProjectId = 0
            req.SubServiceType = 2  # 实时识别
            req.EngSerViceType = "16k_zh"  # 中文普通话
            req.SourceType = 1  # 本地音频
            req.VoiceFormat = "wav" if audio_path.endswith(".wav") else "mp3"
            req.UsrAudioKey = os.path.basename(audio_path)
            req.Data = audio_data
            req.DataLen = len(audio_data)

            resp = self.asr_client.SentenceRecognition(req)
            return resp.Result
        except Exception as e:
            logging.error(f"ASR识别失败: {str(e)}")
            raise

    def synthesize(self, text, voice_type=1):
        """语音合成（TTS）
        :param text: 待合成文本
        :param voice_type: 音色类型（1-6）
        :return: 音频二进制数据
        """
        try:
            req = tts_models.TextToVoiceRequest()
            req.Text = text
            req.SessionId = "pycharm-session"
            req.ModelType = 1  # 基础模型
            req.VoiceType = voice_type
            req.PrimaryLanguage = 1  # 中文
            req.SampleRate = 16000

            resp = self.tts_client.TextToVoice(req)
            return base64.b64decode(resp.Audio)
        except Exception as e:
            logging.error(f"TTS合成失败: {str(e)}")
            raise