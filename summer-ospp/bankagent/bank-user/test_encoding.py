# edge_tts_server_fixed.py
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import edge_tts
import uuid
import time
import asyncio
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

output_dir = Path("tts_output")
output_dir.mkdir(exist_ok=True)

class EdgeTTSWrapper:
    """Edge-TTS 包装类，避免命名冲突"""
    
    async def generate_speech_async(self, text, voice="zh-CN-XiaoxiaoNeural", output_path=None):
        """异步生成语音"""
        try:
            communicate = edge_tts.Communicate(text, voice)
            if output_path:
                await communicate.save(output_path)
                return True
            else:
                return await communicate.get_audio_data()
        except Exception as e:
            logger.error(f"Edge TTS错误: {e}")
            return None
    
    def generate_speech(self, text, voice="zh-CN-XiaoxiaoNeural", output_path=None):
        """同步生成语音"""
        try:
            # 创建新的事件循环
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(
                self.generate_speech_async(text, voice, output_path)
            )
            loop.close()
            return result
        except Exception as e:
            logger.error(f"Edge TTS运行错误: {e}")
            return None

# 创建实例
tts_service = EdgeTTSWrapper()

@app.route('/tts/generate', methods=['POST', 'OPTIONS'])
def generate_tts():
    if request.method == 'OPTIONS':
        return '', 200
        
    try:
        data = request.get_json()
        logger.info(f"收到请求: {data}")
        
        if not data or 'text' not in data:
            return jsonify({'status': 'error', 'message': '缺少文本参数'}), 400
        
        text = data['text'].strip()
        if not text:
            return jsonify({'status': 'error', 'message': '文本内容不能为空'}), 400
        
        voice = data.get('voice', 'zh-CN-XiaoxiaoNeural')
        filename = f"tts_{int(time.time())}_{uuid.uuid4().hex[:8]}.mp3"
        output_path = output_dir / filename
        
        logger.info(f"开始生成语音: {text[:50]}...")
        
        # 生成语音
        success = tts_service.generate_speech(text, voice, str(output_path))
        
        if success and output_path.exists():
            file_size = output_path.stat().st_size
            logger.info(f"语音生成成功: {filename} ({file_size} bytes)")
            
            audio_url = f"http://172.16.22.115:8080/tts/download/{filename}"
            
            return jsonify({
                'status': 'success',
                'message': '语音生成成功',
                'audio_url': audio_url,
                'filename': filename,
                'text': text,
                'voice': voice,
                'file_size': file_size
            })
        else:
            logger.error("语音生成失败")
            return jsonify({'status': 'error', 'message': '语音生成失败'}), 500
            
    except Exception as e:
        logger.error(f"处理请求时出错: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/tts/download/<filename>', methods=['GET'])
def download_audio(filename):
    try:
        # 安全检查
        if '..' in filename or '/' in filename:
            return jsonify({'status': 'error', 'message': '无效文件名'}), 400
            
        file_path = output_dir / filename
        
        if not file_path.exists():
            return jsonify({'status': 'error', 'message': '文件不存在'}), 404
        
        return send_file(
            file_path,
            as_attachment=True,
            download_name=filename,
            mimetype='audio/mpeg'
        )
        
    except Exception as e:
        logger.error(f"下载失败: {e}")
        return jsonify({'status': 'error', 'message': '下载失败'}), 500

@app.route('/tts/health', methods=['GET'])
def health_check():
    """健康检查端点"""
    return jsonify({
        'status': 'healthy',
        'service': 'Edge TTS Server',
        'timestamp': time.time()
    })

@app.route('/tts/voices', methods=['GET'])
def list_voices():
    """获取支持的语音列表"""
    voices = [
        {"name": "晓晓（女声）", "value": "zh-CN-XiaoxiaoNeural"},
        {"name": "云扬（男声）", "value": "zh-CN-YunyangNeural"},
        {"name": "晓辰（女声）", "value": "zh-CN-XiaochenNeural"},
        {"name": "晓悠（女声）", "value": "zh-CN-XiaoyouNeural"},
        {"name": "云希（男声）", "value": "zh-CN-YunxiNeural"},
        {"name": "英语女声", "value": "en-US-AriaNeural"},
        {"name": "英语男声", "value": "en-US-GuyNeural"}
    ]
    return jsonify({'status': 'success', 'voices': voices})

if __name__ == '__main__':
    print("启动修复版 Edge TTS 服务器...")
    print("服务器地址: http://172.16.22.115:8080")
    print("可用端点:")
    print("  POST /tts/generate - 生成语音")
    print("  GET  /tts/download/<filename> - 下载语音")
    print("  GET  /tts/health - 健康检查")
    print("  GET  /tts/voices - 获取语音列表")
    
    # 确保输出目录存在
    output_dir.mkdir(exist_ok=True)
    
    # 测试Edge-TTS是否正常工作
    try:
        import edge_tts
        print("✓ Edge-TTS 导入成功")
    except ImportError:
        print("✗ 请安装: pip install edge-tts")
        exit(1)
    
    app.run(host='0.0.0.0', port=8080, debug=True)