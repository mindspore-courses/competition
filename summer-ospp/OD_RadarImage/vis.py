import cv2
import numpy as np

class DetectionVisualizer:
    def __init__(self, camera_matrix=None, dist_coeffs=None):
        """初始化可视化器
        Args:
            camera_matrix: 相机内参矩阵，如果为None则使用默认值
            dist_coeffs: 相机畸变系数，如果为None则使用默认值
        """
        # 如果没有提供相机参数，使用默认参数
        if camera_matrix is None:
            # 默认相机内参矩阵 (假设图像尺寸为1280x720)
            self.camera_matrix = np.array([
                [567.7, 0, 628.7],
                [0, 577.2, 369.3],
                [0, 0, 1]
            ])
        else:
            self.camera_matrix = camera_matrix
        
        if dist_coeffs is None:
            # 默认畸变系数
            self.dist_coeffs = np.array([-0.028873818023371287, 0.0006023302214797655, 0.0039573086622276855, -0.005047176298643093, 0.0])
        else:
            self.dist_coeffs = dist_coeffs
            
        # 设置不同类别的颜色
        self.colors = {
            0: (0, 255, 0),    # 绿色
            1: (255, 0, 255),    # 蓝色
            2: (0, 0, 255),    # 红色
        }
        
        # 设置类别名称
        self.class_names = {
            0: "Other",
            1: "Sedan"
        }
    
    def draw_3d_bbox(self, image, center, size, theta, class_idx, score=1.0):
        """在图像上绘制3D边界框
        Args:
            image: 输入图像
            center: 中心点坐标 (x, y, z)
            size: 目标尺寸 (w, h, l)
            theta: 方向角元组 (sin(theta), cos(theta))
            class_idx: 类别索引
            score: 置信度分数
        Returns:
            绘制了边界框的图像
        """
        # 确保输入图像是可修改的
        img = image.copy()
        
        # 从中心点计算8个顶点
        center = (center[0],-center[1],center[2])
        l, w, h  = size
        sin_theta,cos_theta = theta
        
        # 3D边界框的8个顶点 (相对于中心点)

        vertices = np.array([
            [l/2, -w/2, h/2],  # 前下左
            [l/2, w/2, h/2],   # 前下右
            [l/2, w/2, -h/2],    # 前上右
            [l/2, -w/2, -h/2],   # 前上左
            [-l/2, -w/2, h/2],   # 后下左
            [-l/2, w/2, h/2],    # 后下右
            [-l/2, w/2, -h/2],     # 后上右
            [-l/2, -w/2, -h/2]     # 后上左
        ])
        
        # 创建旋转矩阵（绕z轴旋转theta角，现在z是竖直方向）
        rotation_matrix = np.array([
            [cos_theta,  -sin_theta,0], 
            [sin_theta, cos_theta, 0],
            [0, 0, 1]
        ])
        
        # 应用旋转并平移到中心点
        rotated_vertices = np.dot(vertices, rotation_matrix.T) + np.array(center)

        
        # 将3D点投影到2D图像平面
        img_pts, _ = cv2.projectPoints(
            rotated_vertices, 
            np.array([-1.209, 1.209, 1.209]),  # 旋转向量（假设相机已校准）
            np.zeros(3),  # 平移向量（假设相机已校准）
            self.camera_matrix, 
            self.dist_coeffs
        )
        
        # 转换为整数坐标
        img_pts = img_pts.reshape(-1, 2).astype(int)

        # 选择颜色
        color = self.colors.get(class_idx % len(self.colors), (255, 255, 255))
        thickness = 1
        
        # 绘制边界框的边
        # 前面和后面
        for i in range(4):
            cv2.line(img, tuple(img_pts[i]), tuple(img_pts[(i+1)%4]), color, thickness)
            cv2.line(img, tuple(img_pts[i+4]), tuple(img_pts[((i+1)%4)+4]), color, thickness)
        
        # 连接前面和后面
        for i in range(4):
            cv2.line(img, tuple(img_pts[i]), tuple(img_pts[i+4]), color, thickness)
        
        # 绘制中心点
        center_2d = tuple(img_pts[0]+img_pts[-2])  # 使用第一个点作为标签位置
        center_2d = (int(center_2d[0]/2),int(center_2d[1]/2))
        cv2.circle(img, center_2d, 3, (0, 0, 255), -1)
        
        # 绘制类别标签和置信度
        class_name = self.class_names.get(class_idx, f"Class {class_idx}")
        label = f"{class_name}: {score:.2f}"
        cv2.putText(img, label, (center_2d[0]+10, center_2d[1]+10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # 绘制方向指示线（指示物体前进方向）
        # 从中心点向前延伸 l/2 距离
        front_dir_3d = np.array([l/2, 0, 0])  # 前方方向（现在y轴是深度方向）
        front_dir_3d = np.dot(front_dir_3d, rotation_matrix.T) + np.array(center)  # 应用旋转
        
        # 投影到2D
        front_dir_2d, _ = cv2.projectPoints(
            np.array([front_dir_3d]), 
            np.array([-1.209, 1.209, 1.209]), 
            np.zeros(3), 
            self.camera_matrix, 
            self.dist_coeffs
        )
        front_dir_2d = front_dir_2d.reshape(-1, 2).astype(int)
        # 绘制方向线
        # 为了避免arrowedLine函数的类型问题，我们暂时使用普通的line函数
        try:
            pt1 = (int(center_2d[0]), int(center_2d[1]))
            pt2 = (int(front_dir_2d[0][0]), int(front_dir_2d[0][1]))
            # 使用line函数替代arrowedLine
            cv2.line(img, pt1, pt2, (255, 0, 255), thickness*2)  # 使用更粗的线来表示方向
        except Exception as e:
            print(f"绘制方向线时出错: {e}")
            # 如果出错，我们就跳过绘制方向线，继续执行其他部分
        
        return img
    
    def visualize_detections(self, image_path, detections, output_path=None):
        """可视化检测结果
        Args:
            image_path: 图像路径或图像数组
            detections: 检测结果列表，每个元素是包含检测信息的字典
            output_path: 输出图像路径，如果为None则显示图像
        Returns:
            绘制了所有检测结果的图像
        """
        # 加载图像
        if isinstance(image_path, str):
            image = cv2.imread(image_path)
            if image is None:
                raise FileNotFoundError(f"无法加载图像: {image_path}")
        else:
            image = image_path.copy()
        
        # 为每个检测结果绘制3D边界框
        for det in detections:
            # 从字典中提取信息
            center = det['center'] if det.get('center').any() else det['gt_center']
            size = det['size'] if det.get('size').any() else det['gt_size']
            theta = det['angle'] if det.get('angle').any() else det['gt_angle']
            
            # 确定类别索引
            class_probs = det['class'] if det.get('class').any() else det['gt_class']
            class_idx = np.argmax(class_probs)
            score = class_probs[class_idx]
            
            # 绘制边界框
            image = self.draw_3d_bbox(image, center, size, theta, class_idx, score)
        
        # 保存或显示图像
        if output_path:
            cv2.imwrite(output_path, image)
            print(f"结果已保存到: {output_path}")
        else:
            # 显示图像
            cv2.imshow("3D Detection Visualization", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return image

