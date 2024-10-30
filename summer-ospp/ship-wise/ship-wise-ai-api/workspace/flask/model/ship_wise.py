from demo.predict import get_parser_infer, set_default_infer, detect
from mindyolo.utils.config import load_config, Config
import os
import mindspore as ms
from mindyolo.models import create_model
from mindyolo.utils.utils import draw_result, set_seed


class NetworkSingleton:
    _instance = None
    _args = None

    def __new__(cls, args):
        if cls._instance is None:
            cls._instance = super(NetworkSingleton, cls).__new__(cls)
            cls._instance.init_network(args)
            cls._args = args
        return cls._instance

    def init_network(self, args):
        set_seed(args.seed)
        set_default_infer(args)
        self.network = create_model(
            model_name=args.network.model_name,
            model_cfg=args.network,
            num_classes=args.data.nc,
            sync_bn=False,
            checkpoint_path=args.weight,
        )
        self.network.set_train(False)
        ms.amp.auto_mixed_precision(self.network, amp_level=args.ms_amp_level)

    def get_network(self):
        return self.network

    def get_args(self):
        return self._args


def infer(args, network, img):
    is_coco_dataset = "coco" in args.data.dataset_name
    # 默认任务为 Detection
    result_dict = detect(
        network=network,
        img=img,
        conf_thres=args.conf_thres,
        iou_thres=args.iou_thres,
        conf_free=args.conf_free,
        nms_time_limit=args.nms_time_limit,
        img_size=args.img_size,
        stride=max(max(args.network.stride), 32),
        num_class=args.data.nc,
        is_coco_dataset=is_coco_dataset,
    )
    if args.save_result:
        save_path = os.path.join(args.save_dir, "detect_results")
        draw_result(args.image_path, result_dict, args.data.names, is_coco_dataset=is_coco_dataset,
                    save_path=save_path)
    return result_dict


def init(user_config=None):
    parser = get_parser_infer()
    test_img_path = r"H:\Library\Datasets\HRSC\HRSC2016_dataset\HRSC2016\FullDataSet-YOLO-Split\test\100000630.bmp"
    if user_config is None:
        user_config = {
            "config": "./workspace/configs/ship-wise/ship-wise-s.yaml",
            "weight": "./runs/2024.09.15-22.56.30/weights/ship-wise-s-153_422.ckpt",
            "device_target": "CPU",
        }
    cfg, _, _ = load_config(user_config["config"])
    cfg = Config(cfg)
    parser.set_defaults(**cfg)
    parser.set_defaults(**user_config)
    args = parser.parse_args()
    args = Config(vars(args))
    network = NetworkSingleton(args).get_network()
    return args, network
