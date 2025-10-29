# utils_audio.py  —— 无需 soundfile；优先使用 ffmpeg，失败则纯 Python WAV 兜底
import io, os, shutil, subprocess, numpy as np, wave

def _has_ffmpeg():
    return shutil.which("ffmpeg") is not None

def _resample_linear(x: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return x.astype(np.float32)
    t_old = np.linspace(0, len(x)/src_sr, num=len(x), endpoint=False)
    t_new = np.linspace(0, len(x)/src_sr, num=int(len(x)*dst_sr/src_sr), endpoint=False)
    y = np.interp(t_new, t_old, x).astype(np.float32)
    return y

def _wav_bytes_to_np(raw_bytes: bytes):
    # 兜底方案，仅支持 PCM WAV
    bio = io.BytesIO(raw_bytes)
    with wave.open(bio, 'rb') as wf:
        n_channels = wf.getnchannels()
        sampwidth  = wf.getsampwidth()
        framerate  = wf.getframerate()
        n_frames   = wf.getnframes()
        pcm = wf.readframes(n_frames)
    if sampwidth == 2:
        dtype = np.int16
        data = np.frombuffer(pcm, dtype=dtype).astype(np.float32) / 32768.0
    else:
        data = np.frombuffer(pcm, dtype=np.uint8).astype(np.float32)
        data = (data - 128.0) / 128.0
    if n_channels > 1:
        data = data.reshape(-1, n_channels).mean(axis=1)
    return data, int(framerate)

def ensure_wav16k_mono(raw_bytes: bytes) -> bytes:
    """将任意输入音频转成 16kHz/mono 的 WAV（bytes）"""
    if _has_ffmpeg():
        try:
            p = subprocess.run(
                ["ffmpeg", "-hide_banner", "-loglevel", "error",
                 "-i", "pipe:0", "-f", "wav", "-ar", "16000", "-ac", "1", "pipe:1"],
                input=raw_bytes, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True
            )
            return p.stdout
        except subprocess.CalledProcessError:
            pass

    # 没有 ffmpeg，兜底：只能处理 WAV
    data, sr = _wav_bytes_to_np(raw_bytes)
    data = _resample_linear(data, sr, 16000)
    bio = io.BytesIO()
    with wave.open(bio, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)   # 16-bit PCM
        wf.setframerate(16000)
        pcm = np.clip(data * 32767.0, -32768, 32767).astype(np.int16).tobytes()
        wf.writeframes(pcm)
    return bio.getvalue()
