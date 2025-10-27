import sounddevice as sd
import soundfile as sf

fs = 16000  # 采样率
seconds = 3  # 录音时长

print("开始录音...")
audio = sd.rec(int(seconds * fs), samplerate=fs, channels=1, dtype='int16')
sd.wait()
print("录音完成，保存为 sample.wav")

sf.write("sample.wav", audio, fs)
