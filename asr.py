# asr.py
from huaweicloud_sis import asr_short_sentence_wav16k

def asr_recognize_bytes(wav16k_bytes: bytes) -> str:
    return asr_short_sentence_wav16k(wav16k_bytes, lang="en_us")
