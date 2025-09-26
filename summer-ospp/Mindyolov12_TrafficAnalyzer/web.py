import os
from pathlib import Path
from werkzeug.utils import secure_filename
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, flash, jsonify
from datetime import datetime
import subprocess
import tempfile
import json

# 本地导入
from lane_detect import run_lane_detection

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / 'web_data' / 'uploads'
OUTPUT_DIR = BASE_DIR / 'web_data' / 'outputs'
CONFIGS_DIR = BASE_DIR / 'configs'
DEFAULT_LANE_CONFIG = BASE_DIR / 'YOLOv12' / 'lane_config.json'

ALLOWED_VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.mkv'}
ALLOWED_CONFIG_EXTS = {'.yaml', '.yml'}
ALLOWED_JSON_EXTS = {'.json'}

app = Flask(__name__)
app.secret_key = 'mindyolo-secret-key'
# 限制上传大小（默认 512MB，可用环境变量覆盖）
max_mb = int(os.getenv('MAX_UPLOAD_MB', '512'))
app.config['MAX_CONTENT_LENGTH'] = max_mb * 1024 * 1024


def ensure_dirs():
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def list_model_configs():
    res = []
    if CONFIGS_DIR.exists():
        for p in CONFIGS_DIR.rglob('*.yaml'):
            # 排除数据集yaml
            if p.name.lower() in {'dataset.yaml', 'coco.yaml'}:
                continue
            res.append(str(p.relative_to(BASE_DIR)))
    return sorted(res)


def save_uploaded(file_storage, allow_exts, subdir: Path) -> Path:
    if not file_storage:
        return None
    filename = secure_filename(file_storage.filename)
    if not filename:
        return None
    ext = Path(filename).suffix.lower()
    if ext not in allow_exts:
        raise ValueError(f'不支持的文件类型: {ext}')
    ensure_dirs()
    dest = subdir / filename
    file_storage.save(str(dest))
    return dest

# 自定义 strftime 过滤器
@app.template_filter('strftime')
def strftime_filter(fmt: str):
    """
    用法: {{ '%Y'|strftime }} 或 {{ some_datetime|strftime('%Y-%m-%d') }}
    若传入的是 datetime 则按其格式化；若是格式字符串，则对当前时间格式化。
    """
    if isinstance(fmt, datetime):
        # 如果模板调用形式是 {{ some_dt|strftime('%Y') }} jinja会把参数当作第2个，不适用这里的简单形式
        return fmt.strftime('%Y-%m-%d %H:%M:%S')
    # 当前时间按传入格式
    try:
        return datetime.now().strftime(fmt)
    except Exception:
        return datetime.now().strftime('%Y')

# 车辆类型映射过滤器
@app.template_filter('vehicle_type_icon')
def vehicle_type_icon(vehicle_type):
    """为车辆类型添加图标"""
    icons = {
        'car': '🚗',
        'truck': '🚚', 
        'bus': '🚌',
        'motorcycle': '🏍️',
        'bicycle': '🚲',
        'person': '🚶',
        'unknown': '❓'
    }
    return icons.get(vehicle_type, '🚗')

# 百分比计算过滤器
@app.template_filter('percentage')
def percentage_filter(value, total):
    """计算百分比"""
    if total == 0:
        return "0.0"
    return f"{(value / total * 100):.1f}"

@app.route('/', methods=['GET'])
def index():
    configs = list_model_configs()
    default_lane = str(DEFAULT_LANE_CONFIG.relative_to(BASE_DIR)) if DEFAULT_LANE_CONFIG.exists() else ''
    return render_template('index.html', configs=configs, default_lane=default_lane)

def convert_to_h264(input_path: Path, output_path: Path = None) -> Path:
    """
    将视频转换为浏览器兼容的H.264格式
    """
    if output_path is None:
        # 确保输出文件名不同，避免覆盖
        output_path = input_path.parent / f"{input_path.stem}_h264_converted.mp4"
    
    try:
        # 检查ffmpeg是否可用
        print("检查FFmpeg可用性...")
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        print("FFmpeg可用")
        
        # 检查输入文件
        print(f"输入文件: {input_path}, 存在: {input_path.exists()}, 大小: {input_path.stat().st_size if input_path.exists() else 'N/A'}")
        
        # 转换命令：强制H.264编码，兼容性最好
        cmd = [
            'ffmpeg', '-i', str(input_path),
            '-c:v', 'libx264',  # 强制H.264视频编码
            '-c:a', 'aac',      # AAC音频编码
            '-movflags', '+faststart',  # 优化网络播放
            '-pix_fmt', 'yuv420p',      # 兼容性像素格式
            '-preset', 'fast',          # 更快的编码速度
            '-crf', '23',              # 质量控制
            '-y',                      # 覆盖输出文件
            str(output_path)
        ]
        
        print(f"转换命令: {' '.join(cmd)}")
        print(f"转换视频: {input_path} -> {output_path}")
        
        # 执行转换
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)  # 增加超时时间
        
        print(f"转换返回码: {result.returncode}")
        if result.stdout:
            print(f"转换输出: {result.stdout}")
        if result.stderr:
            print(f"转换错误: {result.stderr}")
        
        # 验证转换结果
        if result.returncode == 0 and output_path.exists():
            input_size = input_path.stat().st_size if input_path.exists() else 0
            output_size = output_path.stat().st_size
            print(f"转换成功: {output_path}")
            print(f"文件大小: {input_size} -> {output_size} bytes")
            
            # 验证输出文件格式
            try:
                import cv2
                cap = cv2.VideoCapture(str(output_path))
                if cap.isOpened():
                    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
                    codec = ''.join([chr((fourcc >> (8 * i)) & 0xFF) for i in range(4)])
                    cap.release()
                    print(f"转换后编码: {codec}")
                else:
                    print("警告: 无法验证转换后的视频")
            except Exception as e:
                print(f"验证转换结果时出错: {e}")
            
            return output_path
        else:
            print(f"转换失败，返回原文件")
            return input_path  # 返回原文件
            
    except subprocess.TimeoutExpired:
        print(f"转换超时，返回原文件")
        return input_path
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"FFmpeg不可用或转换失败: {e}")
        return input_path  # 返回原文件
    except Exception as e:
        print(f"转换过程出现未知错误: {e}")
        return input_path

@app.route('/process', methods=['POST'])
def process():
    try:
        ensure_dirs()
        # 1) 视频
        video_file = request.files.get('video_file')
        if not video_file or not video_file.filename:
            flash('请上传视频文件')
            return redirect(url_for('index'))
        video_path = save_uploaded(video_file, ALLOWED_VIDEO_EXTS, UPLOAD_DIR)

        # 2) 模型配置：优先使用上传文件，否则使用下拉选择
        cfg_upload = request.files.get('config_file')
        if cfg_upload and cfg_upload.filename:
            config_path = save_uploaded(cfg_upload, ALLOWED_CONFIG_EXTS, UPLOAD_DIR)
        else:
            selected_cfg = request.form.get('selected_config', '').strip()
            if not selected_cfg:
                flash('请选择或上传模型配置文件')
                return redirect(url_for('index'))
            config_path = BASE_DIR / selected_cfg

        # 3) 权重：文本输入或上传
        weight_upload = request.files.get('weight_file')
        weight_path = None
        if weight_upload and weight_upload.filename:
            weight_path = save_uploaded(weight_upload, {'.ckpt', '.pt', '.bin'}, UPLOAD_DIR)
        else:
            weight_text = request.form.get('weight_path', '').strip()
            if weight_text:
                weight_path = Path(weight_text)
            else:
                weight_path = Path('')  # 允许空权重（随机初始化）

        # 4) 车道配置：优先上传，否则文本，最后默认
        lane_upload = request.files.get('lane_config_file')
        if lane_upload and lane_upload.filename:
            lane_config_path = save_uploaded(lane_upload, ALLOWED_JSON_EXTS, UPLOAD_DIR)
        else:
            lane_text = request.form.get('lane_config_path', '').strip()
            if lane_text:
                lane_config_path = Path(lane_text)
            elif DEFAULT_LANE_CONFIG.exists():
                lane_config_path = DEFAULT_LANE_CONFIG
            else:
                flash('请提供 lane_config.json')
                return redirect(url_for('index'))

        # 输出文件名
        out_name = f"lane_out_{video_path.stem}.mp4"
        output_path = OUTPUT_DIR / out_name

        # 调用处理
        result_ret = run_lane_detection(
            config_path=str(config_path),
            weight_path=str(weight_path) if str(weight_path) else '',
            lane_config_path=str(lane_config_path),
            video_path=str(video_path),
            output_path=str(output_path),
        )

        # 兼容返回 (path, summary) 或仅 path
        if isinstance(result_ret, tuple) and len(result_ret) == 2:
            result_file, summary = result_ret
        else:
            result_file = result_ret
            summary = {}

        # 添加调试和验证
        print(f"Expected output path: {output_path}")
        print(f"Returned result file: {result_file}")
        
        # 确保文件存在且在正确位置
        result_path = Path(result_file)
        if not result_path.exists():
            flash(f'输出文件未生成: {result_file}')
            return redirect(url_for('index'))

        # 移动到输出目录
        if result_path.parent != OUTPUT_DIR:
            import shutil
            final_name = f"lane_out_{video_path.stem}.mp4"
            final_path = OUTPUT_DIR / final_name
            shutil.move(str(result_path), str(final_path))
            result_path = final_path

        # 转换为浏览器兼容格式
        print("开始视频格式转换...")
        compatible_path = convert_to_h264(result_path)
        
        # 确保返回的是转换后的文件
        if compatible_path != result_path:
            print(f"使用转换后的文件: {compatible_path}")
        else:
            print(f"转换失败或未执行，使用原文件: {result_path}")

        # 写入统计
        summary_filename = ""
        if isinstance(summary, dict) and summary:
            summary_filename = f"{Path(compatible_path).stem}_summary.json"
            summary_path = OUTPUT_DIR / summary_filename
            try:
                with open(summary_path, 'w', encoding='utf-8') as f:
                    json.dump(summary, f, ensure_ascii=False, indent=2)
                print(f"统计信息已写入: {summary_path}")
            except Exception as e:
                print(f"写入统计信息失败: {e}")
                summary_filename = ""

        return redirect(url_for('result', filename=Path(compatible_path).name, summary=summary_filename))

    except Exception as e:
        print(f"处理失败异常: {e}")
        flash(f'处理失败: {e}')
        return redirect(url_for('index'))

@app.route('/result/<filename>')
def result(filename):
    summary_file = request.args.get('summary', '')
    summary_data = {}
    if summary_file:
        sf = OUTPUT_DIR / summary_file
        if sf.exists():
            try:
                with open(sf, 'r', encoding='utf-8') as f:
                    summary_data = json.load(f)
            except Exception as e:
                print(f"读取统计文件失败: {e}")

    # 处理统计数据，确保所有字段都存在
    default_summary = {
        'datetime': '未知',
        'total_frames': 0,
        'total_time_sec': 0,
        'fps_estimated': 0,
        'lanes': [],
        'emergency_violations_count': 0,
        'emergency_violations': [],
        'suspicious_vehicles_count': 0,
        'suspicious_vehicles': [],
        'vehicle_classification': {
            'cumulative_counts': {},
            'total_vehicles_detected': 0,
            'current_frame_counts': {},
            'current_frame_total': 0
        }
    }
    
    # 合并默认值和实际数据
    for key, default_value in default_summary.items():
        if key not in summary_data:
            summary_data[key] = default_value

    return render_template('result.html', video_filename=filename, summary=summary_data)


@app.route('/download/<filename>')
def download(filename):
    return send_from_directory(OUTPUT_DIR, filename, as_attachment=False)

@app.route('/test_video/<filename>')
def test_video(filename):
    file_path = OUTPUT_DIR / filename
    return f"""
    <html>
    <body>
        <h3>视频测试页面</h3>
        <p>文件: {filename}</p>
        <p>路径: {file_path}</p>
        <p>存在: {file_path.exists()}</p>
        <p>大小: {file_path.stat().st_size if file_path.exists() else 'N/A'} bytes</p>
        <video width="640" height="480" controls>
            <source src="/download/{filename}" type="video/mp4">
            您的浏览器不支持视频标签。
        </video>
        <br><br>
        <a href="/download/{filename}" target="_blank">直接下载链接</a>
        <br><br>
        <a href="/video_info/{filename}" target="_blank">视频信息</a>
    </body>
    </html>
    """

@app.route('/video_info/<filename>')
def video_info(filename):
    file_path = OUTPUT_DIR / filename
    if not file_path.exists():
        return jsonify({'error': '文件不存在', 'filename': filename}), 404
    info = {
        'filename': filename,
        'size_bytes': file_path.stat().st_size,
        'size_mb': round(file_path.stat().st_size / (1024*1024), 2),
        'path': str(file_path),
    }
    try:
        import cv2
        cap = cv2.VideoCapture(str(file_path))
        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS)
            frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
            codec = ''.join([chr((fourcc >> (8 * i)) & 0xFF) for i in range(4)])
            cap.release()
            info.update({
                'width': w,
                'height': h,
                'fps': fps,
                'frames': frames,
                'duration_sec': round(frames / fps, 2) if fps and fps > 0 else None,
                'codec': codec
            })
        else:
            info['warn'] = '无法打开视频（可能编码不兼容或文件未写完）'
    except Exception as e:
        info['error'] = f'读取失败: {e}'
    return jsonify(info)

@app.route('/video_stats/<base>')
def video_stats(base):
    """AJAX 获取统计信息（可选）"""
    p = OUTPUT_DIR / f"{Path(base).stem}_summary.json"
    if not p.exists():
        return jsonify({"error": "统计文件不存在"}), 404
    try:
        return jsonify(json.load(open(p, 'r', encoding='utf-8')))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    ensure_dirs()
    host = os.getenv('FLASK_HOST', '0.0.0.0')  # 远程部署推荐 0.0.0.0
    port = int(os.getenv('FLASK_PORT', '5001'))
    debug = os.getenv('FLASK_DEBUG', '0') == '1'
    app.run(host=host, port=port, debug=debug)