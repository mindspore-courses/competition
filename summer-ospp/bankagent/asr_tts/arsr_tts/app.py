from fastapi import FastAPI, UploadFile, File, Header, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
import io
from utils_audio import ensure_wav16k_mono
from asr import asr_recognize_bytes
from tts import tts_wav_bytes, tts_wav_base64

# 固定 API Key，直接在代码里写死
_API_KEY = "super_secret_12345"

def _auth(x_api_key: str = Header(default=None, alias="X-API-Key")):
    if x_api_key != _API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

app = FastAPI(title="Bank Agent ASR/TTS", version="0.1.0")

@app.post("/asr")
async def asr_endpoint(
    file: UploadFile = File(...),
    x_api_key: str = Header(default=None, alias="X-API-Key")
):
    _auth(x_api_key)
    raw = await file.read()
    wav16k = ensure_wav16k_mono(raw)
    text = asr_recognize_bytes(wav16k)
    return JSONResponse({"text": text})

@app.post("/tts/wav")
async def tts_wav_endpoint(
    text: str,
    x_api_key: str = Header(default=None, alias="X-API-Key")
):
    _auth(x_api_key)
    wav = tts_wav_bytes(text)
    return StreamingResponse(io.BytesIO(wav), media_type="audio/wav")

@app.post("/tts/base64")
async def tts_b64_endpoint(
    text: str,
    x_api_key: str = Header(default=None, alias="X-API-Key")
):
    _auth(x_api_key)
    b64 = tts_wav_base64(text)
    return JSONResponse({"audio_base64": b64, "mime": "audio/wav"})

@app.get("/healthz")
def healthz():
    return {"ok": True}
