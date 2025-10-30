from mindspore.dataset import GeneratorDataset
import numpy as np
import os, cv2, time
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
s_h, s_w = 0, 0
class MVSDataset():
    def __init__(self, datapath, listfile, mode, nviews, ndepths=192, interval_scale=1.06, **kwargs):
        super(MVSDataset, self).__init__()
        self.datapath = datapath
        self.listfile = listfile
        self.mode = mode
        self.nviews = nviews
        self.ndepths = ndepths
        self.interval_scale = interval_scale
        self.max_h, self.max_w = kwargs["max_h"], kwargs["max_w"]
        self.fix_res = kwargs.get("fix_res", False)  #whether to fix the resolution of input image.
        self.fix_wh = False

        assert self.mode == "test"
        self.metas = self.build_list()

    def build_list(self):
        metas = []
        scans = self.listfile

        interval_scale_dict = {}
        # scans
        for scan in scans:
            # determine the interval scale of each scene. default is 1.06
            if isinstance(self.interval_scale, float):
                interval_scale_dict[scan] = self.interval_scale
            else:
                interval_scale_dict[scan] = self.interval_scale[scan]

            pair_file = "{}/pair.txt".format(scan)
            # read the pair file
            with open(os.path.join(self.datapath, pair_file)) as f:
                num_viewpoint = int(f.readline())
                # viewpoints
                for view_idx in range(num_viewpoint):
                    ref_view = int(f.readline().rstrip())
                    src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
                    # filter by no src view and fill to nviews
                    if len(src_views) > 0:
                        if len(src_views) < self.nviews:
                            print("{}< num_views:{}".format(len(src_views), self.nviews))
                            src_views += [src_views[0]] * (self.nviews - len(src_views))
                        metas.append((scan, ref_view, src_views, scan))

        self.interval_scale = interval_scale_dict
        print("dataset", self.mode, "metas:", len(metas), "interval_scale:{}".format(self.interval_scale))
        return metas

    def __len__(self):
        return len(self.metas)

    def read_cam_file(self, filename, interval_scale):
        with open(filename) as f:
            lines = f.readlines()
            lines = [line.rstrip() for line in lines]
        # extrinsics: line [1,5), 4x4 matrix
        extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ').reshape((4, 4))
        # intrinsics: line [7-10), 3x3 matrix
        intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ').reshape((3, 3))
        intrinsics[:2, :] /= 4.0
        # depth_min & depth_interval: line 11
        depth_min = float(lines[11].split()[0])
        depth_interval = float(lines[11].split()[1])

        if len(lines[11].split()) >= 3:
            num_depth = lines[11].split()[2]
            depth_max = depth_min + int(float(num_depth)) * depth_interval
            depth_interval = (depth_max - depth_min) / self.ndepths

        depth_interval *= interval_scale

        return intrinsics, extrinsics, depth_min, depth_interval

    def read_img(self, filename):
        img = Image.open(filename)
        # scale 0~255 to 0~1
        np_img = np.array(img, dtype=np.float32) / 255.

        return np_img

    def read_depth(self, filename):
        # read pfm depth file
        return np.array(read_pfm(filename)[0], dtype=np.float32)

    def scale_mvs_input(self, img, intrinsics, max_w, max_h, base=32):
        h, w = img.shape[:2]
        if h > max_h or w > max_w:
            scale = 1.0 * max_h / h
            if scale * w > max_w:
                scale = 1.0 * max_w / w
            new_w, new_h = scale * w // base * base, scale * h // base * base
        else:
            new_w, new_h = 1.0 * w // base * base, 1.0 * h // base * base

        scale_w = 1.0 * new_w / w
        scale_h = 1.0 * new_h / h
        intrinsics[0, :] *= scale_w
        intrinsics[1, :] *= scale_h

        img = cv2.resize(img, (int(new_w), int(new_h)))

        return img, intrinsics

    def __getitem__(self, idx):
        global s_h, s_w
        meta = self.metas[idx]
        scan, ref_view, src_views, scene_name = meta
        view_ids = [ref_view] + src_views[:self.nviews - 1]

        imgs = []
        depth_values = None
        proj_matrices = []

        for i, vid in enumerate(view_ids):
            img_filename = os.path.join(self.datapath, '{}/images_post/{:0>8}.jpg'.format(scan, vid))
            if not os.path.exists(img_filename):
                img_filename = os.path.join(self.datapath, '{}/images/{:0>8}.jpg'.format(scan, vid))

            proj_mat_filename = os.path.join(self.datapath, '{}/cams/{:0>8}_cam.txt'.format(scan, vid))
            if not os.path.exists(proj_mat_filename):
                proj_mat_filename = os.path.join(self.datapath, '{}/cams_1/{:0>8}_cam.txt'.format(scan, vid))

            img = self.read_img(img_filename)
            intrinsics, extrinsics, depth_min, depth_interval = self.read_cam_file(proj_mat_filename, interval_scale=self.interval_scale[scene_name])
            img, intrinsics = self.scale_mvs_input(img, intrinsics, self.max_w, self.max_h)

            if self.fix_res:
                s_h, s_w = img.shape[:2]
                self.fix_res = False
                self.fix_wh = True
            if i == 0:
                if not self.fix_wh:
                    s_h, s_w = img.shape[:2]
            c_h, c_w = img.shape[:2]
            if (c_h != s_h) or (c_w != s_w):
                scale_h = 1.0 * s_h / c_h
                scale_w = 1.0 * s_w / c_w
                img = cv2.resize(img, (s_w, s_h))
                intrinsics[0, :] *= scale_w
                intrinsics[1, :] *= scale_h
            imgs.append(img)
            proj_mat = np.zeros(shape=(2, 4, 4), dtype=np.float32)
            proj_mat[0, :4, :4] = extrinsics
            proj_mat[1, :3, :3] = intrinsics
            proj_matrices.append(proj_mat)
            if i == 0:
                depth_values = np.arange(depth_min, depth_interval * (self.ndepths - 0.5) + depth_min, depth_interval, dtype=np.float32)
        imgs = np.stack(imgs).transpose([0, 3, 1, 2])
        proj_matrices = np.stack(proj_matrices)
        stage1_proj = proj_matrices
        stage2_proj = proj_matrices.copy()
        stage2_proj[:, 1, :2, :] = proj_matrices[:, 1, :2, :] * 2
        stage3_proj = proj_matrices.copy()
        stage3_proj[:, 1, :2, :] = proj_matrices[:, 1, :2, :] * 4
        return (
            imgs,
            stage1_proj, 
            stage2_proj, 
            stage3_proj, 
            depth_values, 
            encode_scanid(scan), 
            view_ids[0], 
        )

def decode_scanid(scanid_int):
    return f"scan{scanid_int}"

if __name__ == "__main__":
    listfile = ["scan1", "scan4"]
    TESTPATH="/media/outbreak/68E1-B517/Dataset/DTU_ZIP/dtu"
    dataset = MVSDataset(
        datapath=TESTPATH,
        listfile=listfile,
        mode="test",
        nviews=5,
        ndepths=192,
        interval_scale=1.06,
        max_h=512,
        max_w=640,
        fix_res=False
    )
    print(f"数据集长度: {len(dataset)}")
    assert len(dataset) > 0, "数据集为空"
    sample = dataset[0]
    (
        imgs, stage1_proj, stage2_proj, stage3_proj, depth_values, scanid, viewid,
        # filename
    ) = sample
    print(f"图像形状: {imgs.shape}")
    print(f"stage1投影矩阵形状: {stage1_proj.shape}")
    print(f"stage2投影矩阵形状: {stage2_proj.shape}")
    print(f"stage3投影矩阵形状: {stage3_proj.shape}")
    print(f"深度值形状: {depth_values.shape}")
    print(f"组id: {scanid}")
    print(f"参考视角id: {viewid}")
    
    filename = decode_scanid(scanid) + '/{}/' + '{:0>8}'.format(viewid) + "{}"
    print(f"filename: {filename}")
    minds_dataset = GeneratorDataset(
        dataset,
        column_names=["imgs", "stage1_proj", "stage2_proj", "stage3_proj", "depth_values", "scanid", "viewid"],
        shuffle=True
    )
    batched_dataset = minds_dataset.batch(batch_size=2)
    iterator = batched_dataset.create_dict_iterator()
    for item in iterator:
        print(type(item))
        print(item.keys())
        print("imgs:", item['imgs'].shape)
        print("stage1_proj:", item['stage1_proj'].shape)
        print("stage2_proj:", item['stage2_proj'].shape)
        print("stage3_proj:", item['stage3_proj'].shape)
        print("depth_values:", item['depth_values'].shape)
        print("scanid:", item['scanid'])
        print("scanid:", item['scanid'][0])
        print("scanid:", item['scanid'][0].asnumpy())
        print("viewid:", item['viewid'])
        print("viewid:", item['viewid'][0])
        print("viewid:", item['viewid'][0].asnumpy())
        
        filename = decode_scanid(item['scanid'][0].asnumpy()) + '/{}/' + '{:0>8}'.format(int(item['viewid'][0].asnumpy())) + "{}"
        print(filename)
        break
    print("基础测试通过!")
