# tts.py
import base64
from huaweicloud_sis import tts_text_to_wav

def tts_wav_bytes(text: str) -> bytes:
    return tts_text_to_wav(text, lang="en_us")

def tts_wav_base64(text: str) -> str:
    return base64.b64encode(tts_wav_bytes(text)).decode("utf-8")
