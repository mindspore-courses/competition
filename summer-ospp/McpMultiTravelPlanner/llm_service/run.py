from app import create_app
from flask_cors import CORS

app = create_app()
# 配置跨域：允许所有域名访问（开发环境常用）
CORS(app, resources=r'/*')  # 允许所有路径的跨域请求

# 2. 配置 JSON 序列化：禁用 Unicode 转义，支持中文
app.config['JSON_AS_ASCII'] = False  # 关键配置：关闭 ASCII 编码，保留中文
app.config['JSONIFY_MIMETYPE'] = 'application/json;charset=utf-8'  # 可选：指定响应编码为 UTF-8

if __name__ == '__main__':
    
    app.run(debug=False)