import os
import base64
import logging
from tencentcloud.common import credential
from tencentcloud.common.profile.client_profile import ClientProfile
from tencentcloud.tts.v20190823 import tts_client, models
from dotenv import load_dotenv

# 初始化
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TTS-Test")


def synthesize_english_text(text, voice_type=1001):
    """合成英文语音（使用英文专用音色）

    :param text: 英文文本
    :param voice_type: 英文音色类型 (1001-1016)
    :return: 音频二进制数据
    """
    try:
        cred = credential.Credential(
            os.getenv('TENCENT_SECRET_ID'),
            os.getenv('TENCENT_SECRET_KEY')
        )
        cp = ClientProfile()
        cp.httpProfile.endpoint = "tts.tencentcloudapi.com"
        client = tts_client.TtsClient(cred, "ap-shanghai", cp)

        req = models.TextToVoiceRequest()
        req.Text = text
        req.SessionId = "english-test"
        req.ModelType = 1
        req.VoiceType = voice_type  # 关键修改：使用英文音色ID
        req.PrimaryLanguage = 2  # 2=English
        req.SampleRate = 16000

        resp = client.TextToVoice(req)
        return base64.b64decode(resp.Audio)

    except Exception as e:
        logger.error(f"TTS Error: {str(e)}")
        raise


def save_audio(audio_data, filename="english_output.mp3"):
    with open(filename, "wb") as f:
        f.write(audio_data)
    logger.info(f"Audio saved to: {os.path.abspath(filename)}")


if __name__ == "__main__":
    # 测试参数
    test_text = "123453434Hello, this is a test of Tencent Cloud English TTS service."
    voice_type = 1001  # 英文女声 (1001-1016)

    try:
        logger.info(f"Starting English TTS: '{test_text}'")
        audio = synthesize_english_text(test_text, voice_type)
        save_audio(audio)
        logger.info("Test succeeded!")
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")