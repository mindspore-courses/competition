# -*- coding: utf-8 -*-
# @Time    : 2024-08-25 18:08
# @Author  : Jiang Liu


# 导入Flask库
from flask import Flask
from flask_cors import CORS


# 创建Flask应用函数
def create_app():
    app = Flask(__name__)
    CORS(app, origins="*")

    # 导入路由模块
    from .routers.detection import blueprint as detection_blueprint
    from .routers.points import blueprint as points_blueprint
    from .routers.visualization import blueprint as visualization_blueprint

    # 注册蓝图
    app.register_blueprint(detection_blueprint)
    app.register_blueprint(points_blueprint)
    app.register_blueprint(visualization_blueprint)

    return app
