# tts_asr_server.py
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import edge_tts
import whisper
import torch
import uuid
import time
import asyncio
import logging
import os
from pathlib import Path
import tempfile

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# 创建输出目录
output_dir = Path("audio_output")
output_dir.mkdir(exist_ok=True)

# Edge-TTS 语音映射
VOICE_MAPPING = {
    'xiaoxiao': 'zh-CN-XiaoxiaoNeural',
    'yunyang': 'zh-CN-YunyangNeural', 
    'aria': 'en-US-AriaNeural',
    'guy': 'en-US-GuyNeural',
    'jenny': 'en-US-JennyNeural'
}

# 初始化 Whisper ASR 模型
def load_whisper_model():
    """加载Whisper语音识别模型"""
    try:
        logger.info("正在加载Whisper模型...")
        # 使用中等模型，平衡精度和速度
        model = whisper.load_model("medium")
        logger.info("Whisper模型加载成功")
        return model
    except Exception as e:
        logger.error(f"加载Whisper模型失败: {e}")
        return None

# 全局变量
whisper_model = load_whisper_model()

class AudioService:
    """音频服务类，包含TTS和ASR功能"""
    
    def get_voice_name(self, voice_input):
        """获取完整的语音名称"""
        if not voice_input:
            return 'en-US-AriaNeural'
        
        voice_lower = voice_input.lower()
        if voice_lower in VOICE_MAPPING:
            return VOICE_MAPPING[voice_lower]
        
        if 'zh-CN-' in voice_input or 'en-US-' in voice_input:
            return voice_input
        
        return 'en-US-AriaNeural'
    
    # TTS 功能
    async def generate_speech_async(self, text, voice, output_path):
        """异步生成语音"""
        try:
            full_voice_name = self.get_voice_name(voice)
            communicate = edge_tts.Communicate(text, full_voice_name)
            await communicate.save(output_path)
            return True
        except Exception as e:
            logger.error(f"TTS生成失败: {e}")
            return False
    
    def generate_speech(self, text, voice="aria", output_path=None):
        """同步生成语音"""
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(
                self.generate_speech_async(text, voice, output_path)
            )
            loop.close()
            return result
        except Exception as e:
            logger.error(f"TTS运行错误: {e}")
            return False
    
    # ASR 功能
    def transcribe_audio(self, audio_file_path, language="en"):
        """语音识别转文字"""
        try:
            if not whisper_model:
                return None, "Whisper模型未加载"
            
            # 设置语言参数
            if language.lower() in ['en', 'english']:
                lang = "en"
            else:
                lang = None  # 自动检测语言
            
            # 进行语音识别
            result = whisper_model.transcribe(
                audio_file_path,
                language=lang,
                fp16=torch.cuda.is_available(),  # 使用GPU加速（如果可用）
                verbose=False
            )
            
            text = result["text"].strip()
            language_detected = result["language"]
            
            logger.info(f"ASR识别结果: {text} (语言: {language_detected})")
            return text, language_detected
            
        except Exception as e:
            logger.error(f"ASR识别失败: {e}")
            return None, str(e)

# 创建服务实例
audio_service = AudioService()

@app.route('/tts/generate', methods=['POST'])
def generate_tts():
    """TTS生成接口（直接返回音频）"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        voice = data.get('voice', 'standard_female')
        
        if not text:
            return jsonify({"error": "Text parameter is required"}), 400

        # 调用TTS服务
        filepath, filename = speech_service.text_to_speech(text, voice)
        
        # ⭐ 关键修改：直接返回音频文件，而不是JSON
        return send_file(
            filepath,
            mimetype='audio/mpeg',
            as_attachment=False,  # 不下载，直接播放
            download_name=filename
        )
            
    except Exception as e:
        logging.error(f"[TTS ERROR] {str(e)}")
        return jsonify({"error": str(e)}), 500
            


# ASR 路由
@app.route('/asr/transcribe', methods=['POST', 'OPTIONS'])
def transcribe_audio():
    if request.method == 'OPTIONS':
        return '', 200
        
    try:
        # 检查是否有文件上传
        if 'audio' not in request.files:
            return jsonify({'status': 'error', 'message': 'No audio file provided'}), 400
        
        audio_file = request.files['audio']
        if audio_file.filename == '':
            return jsonify({'status': 'error', 'message': 'No file selected'}), 400
        
        # 获取语言参数
        language = request.form.get('language', 'en')
        
        # 保存上传的音频文件
        filename = f"asr_{int(time.time())}_{uuid.uuid4().hex[:8]}_{audio_file.filename}"
        input_path = output_dir / filename
        audio_file.save(input_path)
        
        logger.info(f"开始语音识别: {filename}")
        
        # 进行语音识别
        text, detected_language = audio_service.transcribe_audio(str(input_path), language)
        
        if text:
            return jsonify({
                'status': 'success',
                'message': 'Transcription successful',
                'text': text,
                'detected_language': detected_language,
                'filename': filename,
                'confidence': 'high'  # Whisper精度很高
            })
        else:
            return jsonify({'status': 'error', 'message': 'Transcription failed'}), 500
            
    except Exception as e:
        logger.error(f"ASR error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

# 文件下载路由
@app.route('/audio/download/<filename>', methods=['GET'])
def download_audio(filename):
    try:
        file_path = output_dir / filename
        if file_path.exists():
            return send_file(file_path, as_attachment=True)
        return jsonify({'status': 'error', 'message': 'File not found'}), 404
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

# 健康检查
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'service': 'TTS + ASR Server',
        'tts_available': True,
        'asr_available': whisper_model is not None,
        'supported_voices': list(VOICE_MAPPING.keys()),
        'supported_languages': ['en', 'auto']
    })

# 获取支持的语音列表
@app.route('/voices', methods=['GET'])
def list_voices():
    voices = [
        {"name": "Aria (English Female)", "value": "aria"},
        {"name": "Guy (English Male)", "value": "guy"},
        {"name": "Jenny (English Female)", "value": "jenny"},
        {"name": "Xiaoxiao (Chinese Female)", "value": "xiaoxiao"},
        {"name": "Yunyang (Chinese Male)", "value": "yunyang"}
    ]
    return jsonify({'status': 'success', 'voices': voices})

if __name__ == '__main__':
    print("启动 TTS + ASR 服务器...")
    print("功能:")
    print("  TTS: POST /tts/generate")
    print("  ASR: POST /asr/transcribe")
    print("  下载: GET /audio/download/<filename>")
    print("  健康检查: GET /health")
    print("  语音列表: GET /voices")
    
    # 检查模型加载情况
    if whisper_model:
        print("✓ Whisper ASR 模型加载成功")
    else:
        print("⚠ Whisper ASR 模型加载失败，ASR功能不可用")
    
    output_dir.mkdir(exist_ok=True)
    app.run(host='0.0.0.0', port=8080, debug=True)