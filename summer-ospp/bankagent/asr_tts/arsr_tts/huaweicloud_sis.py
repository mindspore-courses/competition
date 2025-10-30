# huaweicloud_sis.py
import os, hmac, hashlib, base64, json, datetime, requests
from typing import Tuple

AK = os.getenv("HUAWEI_AK")
SK = os.getenv("HUAWEI_SK")
PROJECT_ID = os.getenv("HUAWEI_PROJECT_ID")
REGION = os.getenv("HUAWEI_REGION", "cn-north-4")
SIS_ENDPOINT = os.getenv("HUAWEI_SIS_ENDPOINT", f"https://sis-ext.{REGION}.myhuaweicloud.com").rstrip("/")

if not (AK and SK and PROJECT_ID):
    raise RuntimeError("请先设置 HUAWEI_AK / HUAWEI_SK / HUAWEI_PROJECT_ID（必填），HUAWEI_REGION 可选，HUAWEI_SIS_ENDPOINT 可选")

def _utc_iso() -> str:
    # 形如：20250829T080102Z
    return datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

def _canonical_request(method: str, path: str, query: str, headers: dict, body: bytes) -> Tuple[str, str]:
    # 规范化 header：host + x-sdk-date 必须；其余按需增加
    # 注意：Host 必须与实际域名一致
    host = SIS_ENDPOINT.replace("https://", "").replace("http://", "")
    x_sdk_date = headers.get("X-Sdk-Date") or _utc_iso()
    headers["Host"] = host
    headers["X-Sdk-Date"] = x_sdk_date

    # 参与签名的头（小写、按字典序）
    signed_header_keys = ["host", "x-sdk-date"]
    canonical_headers = f"host:{host}\n" + f"x-sdk-date:{x_sdk_date}\n"
    signed_headers = ";".join(signed_header_keys)

    # body sha256
    payload_hash = hashlib.sha256(body or b"").hexdigest()

    # path & query 已经是规范形式（path 形如 /v1/{project_id}/tts）
    canonical = "\n".join([
        method.upper(),
        path,
        query or "",
        canonical_headers,
        signed_headers,
        payload_hash
    ])
    return canonical, signed_headers

def _sign(method: str, path: str, query: str, body: bytes, extra_headers: dict = None) -> dict:
    """返回带 Authorization 的 headers；采用华为云 APIG V2 简化签名"""
    headers = {"Content-Type": "application/json"}
    if extra_headers:
        headers.update(extra_headers)
    canonical, signed_headers = _canonical_request(method, path, query, headers, body)
    string_to_sign = canonical.encode("utf-8")
    signature = hmac.new(SK.encode("utf-8"), string_to_sign, hashlib.sha256).hexdigest()
    auth = f"HMAC-SHA256 Credential={AK}, SignedHeaders={signed_headers}, Signature={signature}"
    headers["Authorization"] = auth
    return headers

def _request_json(method: str, url: str, path: str, body: dict, timeout: int = 60) -> dict:
    body_bytes = json.dumps(body, ensure_ascii=False).encode("utf-8")
    headers = _sign(method, path, "", body_bytes)
    resp = requests.request(method, url, headers=headers, data=body_bytes, timeout=timeout)
    if resp.status_code >= 300:
        raise RuntimeError(f"SIS HTTP {resp.status_code}: {resp.text}")
    return resp.json()

# ===================== 一句话识别（短音频，<=1min，<=10MB） =====================
def asr_short_sentence_wav16k(wav_bytes: bytes, lang="en_us") -> str:
    # 参考属性：英文 16k
    prop = "english_16k" if lang.lower().startswith("en") else "chinese_16k_general"
    b64 = base64.b64encode(wav_bytes).decode("utf-8")
    path = f"/v1/{PROJECT_ID}/short-audio"
    url = f"{SIS_ENDPOINT}{path}"
    body = {
        "config": {
            "audio_format": "wav",
            "property": prop,
            "add_punc": "yes"
        },
        "data": b64
    }
    data = _request_json("POST", url, path, body, timeout=90)
    # 返回结构通常为 {"result":{"text":"..."}}
    return (data.get("result") or {}).get("text", "")

# ===================== 文本转语音（TTS） =====================
def tts_text_to_wav(text: str, lang="en_us") -> bytes:
    prop = "english_common" if lang.lower().startswith("en") else "chinese_xiaoyan_common"
    path = f"/v1/{PROJECT_ID}/tts"
    url = f"{SIS_ENDPOINT}{path}"
    body = {
        "text": text,
        "config": {
            "audio_format": "wav",
            "sample_rate": "16000",
            "property": prop
        }
    }
    data = _request_json("POST", url, path, body, timeout=90)
    # 返回结构通常为 {"result":{"data":"base64", "format":"wav"}}
    b64 = (data.get("result") or {}).get("data", "")
    if not b64:
        raise RuntimeError(f"TTS返回空：{data}")
    return base64.b64decode(b64)
