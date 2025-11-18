from src.yolo import YOLOV5s

def create_network(name, *args, **kwargs):
    if name == "yolov5s":
        yolov5s_net = YOLOV5s(is_training=True)
        return yolov5s_net
    raise NotImplementedError(f"{name} is not implemented in the repo")
