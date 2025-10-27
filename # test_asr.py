# test_asr.py
import requests
import os

def test_asr():
    """测试ASR功能"""
    print("测试语音识别功能...")
    
    # 你可以录制一个英文语音文件来测试，或者使用现有的
    test_audio_file = "E:\guolei\Documents\bank-user\tts_output\tts_1756829756_5376c557.wav"  # 替换为你的测试文件
    
    if not os.path.exists(test_audio_file):
        print("请先创建一个测试音频文件")
        return
    
    try:
        with open(test_audio_file, 'rb') as f:
            files = {'audio': (test_audio_file, f, 'audio/wav')}
            data = {'language': 'en'}
            
            response = requests.post(
                "http://127.0.0.1:8080/asr/transcribe",
                files=files,
                data=data,
                timeout=30
            )
        
        print(f"状态码: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"识别结果: {result}")
        else:
            print(f"错误: {response.text}")
            
    except Exception as e:
        print(f"测试失败: {e}")

if __name__ == "__main__":
    test_asr()