from mindspore.dataset import GeneratorDataset,Dataset
import numpy as np
import os
from PIL import Image
from datasets.data_io import *
# from data_io import *# 如果要运行最底部的main，同级运行则解

def encode_scanid(scan):
    if isinstance(scan, str) and scan.startswith('scan'):
        return int(scan.replace('scan', ''))
    try:
        return int(scan)
    except Exception:
        return scan
class MVSDataset():
    def __init__(self, datapath, listfile, mode, nviews, ndepths=192, interval_scale=1.06, **kwargs):
        super(MVSDataset, self).__init__()
        self.datapath = datapath
        self.listfile = listfile
        self.mode = mode
        self.nviews = nviews
        self.ndepths = ndepths
        self.interval_scale = interval_scale
        self.kwargs = kwargs
        print("mvsdataset kwargs", self.kwargs)

        assert self.mode in ["train", "val", "test"]
        self.metas = self.build_list()

    def build_list(self):
        metas = []
        with open(self.listfile) as f:
            scans = f.readlines()
            scans = [line.rstrip() for line in scans]

        # scans
        for scan in scans:
            with open(os.path.join(self.datapath, 'Cameras', 'pair.txt')) as f:
                num_viewpoint = int(f.readline())
                # viewpoints (49)
                for view_idx in range(num_viewpoint):
                    ref_view = int(f.readline().rstrip())
                    src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
                    # light conditions 0-6
                    for light_idx in range(7):
                        metas.append((scan, light_idx, ref_view, src_views))
        print("dataset", self.mode, "metas:", len(metas))
        return metas

    def __len__(self):
        return len(self.metas)

    def read_cam_file(self, filename):
        with open(filename) as f:
            lines = f.readlines()
            lines = [line.rstrip() for line in lines]
        extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ').reshape((4, 4))
        intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ').reshape((3, 3))
        depth_min = float(lines[11].split()[0])
        depth_interval = float(lines[11].split()[1]) * self.interval_scale
        return intrinsics, extrinsics, depth_min, depth_interval

    def read_img(self, filename):
        img = Image.open(filename)
        # scale 0~255 to 0~1
        np_img = np.array(img, dtype=np.float32) / 255.
        return np_img

    def read_depth(self, filename):
        # read pfm depth file
        return np.array(read_pfm(filename)[0], dtype=np.float32)

    def prepare_img(self, hr_img):
        #w1600-h1200-> 800-600 ; crop -> 640, 512; downsample 1/4 -> 160, 128
        #downsample
        h, w = hr_img.shape
        hr_img_ds = cv2.resize(hr_img, (w//2, h//2), interpolation=cv2.INTER_NEAREST)
        #crop
        h, w = hr_img_ds.shape
        target_h, target_w = 512, 640
        start_h, start_w = (h - target_h)//2, (w - target_w)//2
        hr_img_crop = hr_img_ds[start_h: start_h + target_h, start_w: start_w + target_w]
        # #downsample
        # lr_img = cv2.resize(hr_img_crop, (target_w//4, target_h//4), interpolation=cv2.INTER_NEAREST)

        return hr_img_crop

    def read_mask_hr(self, filename):
        img = Image.open(filename)
        np_img = np.array(img, dtype=np.float32)
        np_img = (np_img > 10).astype(np.float32)
        np_img = self.prepare_img(np_img)

        h, w = np_img.shape
        np_img_ms = {
            "stage1": cv2.resize(np_img, (w//4, h//4), interpolation=cv2.INTER_NEAREST),
            "stage2": cv2.resize(np_img, (w//2, h//2), interpolation=cv2.INTER_NEAREST),
            "stage3": np_img,
        }
        return np_img_ms

    def read_depth_hr(self, filename):
        # read pfm depth file
        #w1600-h1200-> 800-600 ; crop -> 640, 512; downsample 1/4 -> 160, 128
        depth_hr = np.array(read_pfm(filename)[0], dtype=np.float32)
        depth_lr = self.prepare_img(depth_hr)

        h, w = depth_lr.shape
        depth_lr_ms = {
            "stage1": cv2.resize(depth_lr, (w//4, h//4), interpolation=cv2.INTER_NEAREST),
            "stage2": cv2.resize(depth_lr, (w//2, h//2), interpolation=cv2.INTER_NEAREST),
            "stage3": depth_lr,
        }
        return depth_lr_ms

    def __getitem__(self, idx):
        meta = self.metas[idx]
        scan, light_idx, ref_view, src_views = meta
        view_ids = [ref_view] + src_views[:self.nviews - 1]

        imgs = []
        mask = None
        depth_values = None
        depth_ms = None
        proj_matrices = []

        for i, vid in enumerate(view_ids):
            img_filename = os.path.join(
                self.datapath,
                "Rectified", f"{scan}_train", f"rect_{vid + 1:0>3}_{light_idx}_r5000.png"
            )
            mask_filename_hr = os.path.join(
                self.datapath,
                "Depths", f"{scan}_train", f"depth_visual_{vid:0>4}.png"
            )
            depth_filename_hr = os.path.join(
                self.datapath,
                "Depths", f"{scan}_train", f"depth_map_{vid:0>4}.pfm"
            )
            proj_mat_filename = os.path.join(
                self.datapath,
                "Cameras", "train", f"{vid:0>8}_cam.txt"
            )

            img = self.read_img(img_filename)
            intrinsics, extrinsics, depth_min, depth_interval = self.read_cam_file(proj_mat_filename)
            proj_mat = np.zeros(shape=(2, 4, 4), dtype=np.float32)
            proj_mat[0, :4, :4] = extrinsics
            proj_mat[1, :3, :3] = intrinsics
            proj_matrices.append(proj_mat)

            if i == 0:
                mask_read_ms = self.read_mask_hr(mask_filename_hr)
                depth_ms = self.read_depth_hr(depth_filename_hr)
                depth_max = depth_interval * self.ndepths + depth_min
                depth_values = np.arange(depth_min, depth_max, depth_interval, dtype=np.float32)
                mask = mask_read_ms
            imgs.append(img)
        imgs = np.stack(imgs).transpose([0, 3, 1, 2])
        proj_matrices = np.stack(proj_matrices)
        stage1_proj = proj_matrices
        stage2_proj = proj_matrices.copy()
        stage2_proj[:, 1, :2, :] = proj_matrices[:, 1, :2, :] * 2
        stage3_proj = proj_matrices.copy()
        stage3_proj[:, 1, :2, :] = proj_matrices[:, 1, :2, :] * 4
        stage1_mask = mask["stage1"] if mask is not None else None
        stage2_mask = mask["stage2"] if mask is not None else None
        stage3_mask = mask["stage3"] if mask is not None else None
        stage1_depth = depth_ms["stage1"] if depth_ms is not None else None
        stage2_depth = depth_ms["stage2"] if depth_ms is not None else None
        stage3_depth = depth_ms["stage3"] if depth_ms is not None else None

        return (
            imgs,                   # (nviews, 3, H, W)
            stage1_proj,            # (nviews, 2, 4, 4)
            stage2_proj,
            stage3_proj,
            stage1_depth,           # (H, W)
            stage2_depth,
            stage3_depth,
            stage1_mask,            # (H, W)
            stage2_mask,
            stage3_mask,
            depth_values,           # (ndepths,)
            self.encode_scanid(scan), # scanid int
            view_ids[0]             # ref_view id
        )
        
# ================== 测试代码 =====================
if __name__ == "__main__":
    def decode_scanid(scanid_int):
        """将 88 转为 'scan88'"""
        return f"scan{scanid_int}"

    DTU_TRAINING = "/media/outbreak/68E1-B517/Dataset/DTU_ZIP/dtu_training/mvs_training/dtu_training"
    dataset = MVSDataset(
        datapath=DTU_TRAINING,
        listfile="lists/dtu/train.txt",
        mode="train",
        nviews=5,
        ndepths=384,
        interval_scale=1.06
    )
    # 测试1: 检查数据集长度
    print(f"数据集长度: {len(dataset)}")
    assert len(dataset) > 0, "数据集为空"

    # 测试2: 获取第一个样本并检查结构
    sample = dataset.__getitem__(0)
    print("样本结构类型:", type(sample))
    (
        imgs, stage1_proj, stage2_proj, stage3_proj,
        stage1_depth, stage2_depth, stage3_depth,
        stage1_mask, stage2_mask, stage3_mask,
        depth_values, scanid, viewid
    ) = sample
    print(f"图像形状: {imgs.shape}")          # (5, 3, H, W)
    print(f"stage1投影矩阵形状: {stage1_proj.shape}")
    print(f"stage2投影矩阵形状: {stage2_proj.shape}")
    print(f"stage3投影矩阵形状: {stage3_proj.shape}")
    print(f"stage1深度图形状: {stage1_depth.shape if stage1_depth is not None else None}")
    print(f"stage2深度图形状: {stage2_depth.shape if stage2_depth is not None else None}")
    print(f"stage3深度图形状: {stage3_depth.shape if stage3_depth is not None else None}")
    print(f"stage1掩码形状: {stage1_mask.shape if stage1_mask is not None else None}")
    print(f"stage2掩码形状: {stage2_mask.shape if stage2_mask is not None else None}")
    print(f"stage3掩码形状: {stage3_mask.shape if stage3_mask is not None else None}")
    print(f"深度值形状: {depth_values.shape}")  # (384,)
    print(f"组id类型: {type(scanid)}")
    print(f"组id: {scanid}")
    print(f"组id: {decode_scanid(scanid)}")
    print(f"参考视角id类型: {type(viewid)}")
    print(f"参考视角id: {viewid}")
    filename = decode_scanid(scanid) + '/{}/' + '{:0>8}'.format(viewid) + "{}"
    print("filename:", filename)

    # 测试4: 通过GeneratorDataset加载
    from mindspore.dataset import GeneratorDataset
    minds_dataset = GeneratorDataset(
        dataset,
        column_names=[
            "imgs", "stage1_proj", "stage2_proj", "stage3_proj",
            "stage1_depth", "stage2_depth", "stage3_depth",
            "stage1_mask", "stage2_mask", "stage3_mask",
            "depth_values", "scanid", "viewid"
        ],
        shuffle=True
    )
    batched_dataset = minds_dataset.batch(batch_size=4)
    iterator = batched_dataset.create_dict_iterator()
    for item in iterator:
        print(type(item))
        print(item.keys())
        print("imgs:", item['imgs'].shape)
        print("stage1_proj:", item['stage1_proj'].shape)
        print("stage2_proj:", item['stage2_proj'].shape)
        print("stage3_proj:", item['stage3_proj'].shape)
        print("stage1_depth:", item['stage1_depth'][0].shape if item['stage1_depth'] is not None else None)
        print("stage2_depth:", item['stage2_depth'][0].shape if item['stage2_depth'] is not None else None)
        print("stage3_depth:", item['stage3_depth'][0].shape if item['stage3_depth'] is not None else None)
        print("stage1_mask:", item['stage1_mask'][0].shape if item['stage1_mask'] is not None else None)
        print("stage2_mask:", item['stage2_mask'][0].shape if item['stage2_mask'] is not None else None)
        print("stage3_mask:", item['stage3_mask'][0].shape if item['stage3_mask'] is not None else None)
        print("depth_values:", item['depth_values'].shape)
        print("scanid:", item['scanid'])
        print("viewid:", item['viewid'])
        print(type(item['scanid'][0]))
        print(decode_scanid(int(item['scanid'][0].asnumpy())))
        print(type(item['viewid'][0]))
        print((item['viewid'][0].asnumpy()))
        viewid = item['viewid'][0].asnumpy()
        scanid = decode_scanid(item['scanid'][0].asnumpy())
        filename = str(scanid) + '/{}/' + '{:0>8}'.format(viewid) + "{}"
        print("filename:", filename)
        print("type:", type(filename))
        break
    print("基础测试通过!")