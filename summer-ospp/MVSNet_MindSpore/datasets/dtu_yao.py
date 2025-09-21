from mindspore.dataset import GeneratorDataset,Dataset
import numpy as np
import os
from PIL import Image
from datasets.data_io import *
# from data_io import *# 如果要运行最底部的main，同级运行则解

# the DTU dataset preprocessed by Yao Yao (only for training)
class MVSDataset():
    def __init__(self, datapath, listfile, mode, nviews, ndepths=192, interval_scale=1.06, **kwargs):
        super(MVSDataset, self).__init__()
        self.datapath = datapath
        self.listfile = listfile
        self.mode = mode
        self.nviews = nviews
        self.ndepths = ndepths
        self.interval_scale = interval_scale

        assert self.mode in ["train", "val", "test"]
        self.metas = self.build_list()

    def build_list(self):
        metas = []
        with open(self.listfile) as f:
            scans = f.readlines()
            scans = [line.rstrip() for line in scans]

        # scans
        for scan in scans:
            # pair_file = "Cameras/pair.txt"
            # read the pair file
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
        # extrinsics: line [1,5), 4x4 matrix
        extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ').reshape((4, 4))
        # intrinsics: line [7-10), 3x3 matrix
        intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ').reshape((3, 3))
        # depth_min & depth_interval: line 11
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

    def __getitem__(self, idx):
        meta = self.metas[idx]
        scan, light_idx, ref_view, src_views = meta
        # use only the reference view and first nviews-1 source views
        view_ids = [ref_view] + src_views[:self.nviews - 1]

        imgs = []
        mask = None
        depth = None
        depth_values = None
        proj_matrices = []

        for i, vid in enumerate(view_ids):
            # NOTE that the id in image file names is from 1 to 49 (not 0~48)
            # img_filename = os.path.join(self.datapath,'Rectified/{}_train/rect_{:0>3}_{}_r5000.png'.format(scan, vid + 1, light_idx))
            # mask_filename = os.path.join(self.datapath, 'Depths/{}_train/depth_visual_{:0>4}.png'.format(scan, vid))
            # depth_filename = os.path.join(self.datapath, 'Depths/{}_train/depth_map_{:0>4}.pfm'.format(scan, vid))
            # proj_mat_filename = os.path.join(self.datapath, 'Cameras/train/{:0>8}_cam.txt').format(vid)
            img_filename = os.path.join(
            self.datapath,
                "Rectified", f"{scan}_train", f"rect_{vid + 1:0>3}_{light_idx}_r5000.png"
            )
            mask_filename = os.path.join(
                self.datapath,
                "Depths", f"{scan}_train", f"depth_visual_{vid:0>4}.png"
            )
            depth_filename = os.path.join(
                self.datapath,
                "Depths", f"{scan}_train", f"depth_map_{vid:0>4}.pfm"
            )
            proj_mat_filename = os.path.join(
                self.datapath,
                "Cameras", "train", f"{vid:0>8}_cam.txt"
            )

            imgs.append(self.read_img(img_filename))
            intrinsics, extrinsics, depth_min, depth_interval = self.read_cam_file(proj_mat_filename)

            # multiply intrinsics and extrinsics to get projection matrix
            proj_mat = extrinsics.copy()
            # 这里逻辑不太对
            proj_mat[:3, :4] = np.matmul(intrinsics, proj_mat[:3, :4])
            proj_matrices.append(proj_mat)

            if i == 0:  # reference view
                depth_values = np.arange(depth_min, depth_interval * self.ndepths + depth_min, depth_interval,
                                         dtype=np.float32)
                mask = self.read_img(mask_filename)
                depth = self.read_depth(depth_filename)

        imgs = np.stack(imgs).transpose([0, 3, 1, 2])
        proj_matrices = np.stack(proj_matrices)
        
        def encode_scanid(scanid_str):
            """将 'scan88' 转为 88"""
            return int(scanid_str.replace("scan", ""))
        # return {"imgs": imgs,
        #         "proj_matrices": proj_matrices,
        #         "depth": depth,
        #         "depth_values": depth_values,
        #         "mask": mask}
        return (
            imgs,                   # "imgs" (nviews, 3, H, W)
            proj_matrices,        # "proj_matrices" (多尺度投影矩阵)
            depth,                # "depth" (H, W)
            depth_values,            # "depth_values" (ndepths,)
            mask,                    # "mask" (H, W)
            encode_scanid(scan),
            view_ids[0]
        )


if __name__ == "__main__":
    def decode_scanid(scanid_int):
        """将 88 转为 'scan88'"""
        return f"scan{scanid_int}"
    DTU_TRAINING="/media/outbreak/68E1-B517/Dataset/DTU_ZIP/dtu_training/mvs_training/dtu_training"
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
    sample = dataset[0]
    print("样本结构类型:", type(sample))
    # imgs, proj_matrices, depth, depth_values, mask, filename = sample
    imgs, proj_matrices, depth, depth_values, mask, scanid, viewid = sample
    print(f"图像形状: {imgs.shape}")          # (5, 3, H, W)
    print(f"投影矩阵类型: {type(proj_matrices)}") # dict
    print(f"深度图形状: {len(depth)}")        # (H, W)
    print(f"深度值形状: {depth_values.shape}")  # (384,)
    print(f"掩码形状: {len(mask)}")          # (H, W)
    print(f"组id类型: {type(scanid)}") # dict
    print(f"组id: {scanid}")          # (H, W)
    print(f"组id: {decode_scanid(scanid)}")          # (H, W)
    print(f"参考视角id类型: {type(scanid)}") # dict
    print(f"参考视角id: {viewid}")          # (H, W)
    filename = decode_scanid(scanid) + '/{}/' + '{:0>8}'.format(viewid) + "{}"
    print("filename:", filename)
    def decode_scanid(scanid_int):
        """将 88 转为 'scan88'"""
        return f"scan{scanid_int}"
    DTU_TRAINING="/media/outbreak/68E1-B517/Dataset/DTU_ZIP/dtu_training/mvs_training/dtu_training"
    dataset = MVSDataset(
        datapath=DTU_TRAINING,
        listfile="lists/dtu/train.txt",
        mode="train",
        nviews=5,
        ndepths=384,
        interval_scale=1.06
    )
    # 测试4: 通过GeneratorDataset加载
    minds_dataset = GeneratorDataset(
        dataset,
        column_names=["imgs", "proj_matrices","depth","depth_values","mask","viewid","scanid"], 
        shuffle=True
        )
    batched_dataset = minds_dataset.batch(batch_size=4)
    iterator = batched_dataset.create_dict_iterator()
    # create_dict_iterator返回字典
    # create_tuple_iterator返回列表
    for item in iterator:
        print(type(item))
        print(item.keys())
        print("imgs:", item['imgs'].shape)
        print("proj_matrices:", type(item['proj_matrices']))
        print("depth:", item['depth'][0].shape)
        print("depth_values:", item['depth_values'].shape)
        print("mask:", item['mask'][0].shape)
        
        print("viewid:", item['viewid'])
        print("scanid:", item['scanid'])
        print(type(item['scanid'][0]))
        print(decode_scanid(int(item['scanid'][0].asnumpy())))
        print(type(item['viewid'][0]))
        print((item['viewid'][0].asnumpy()))
        viewid = item['viewid'][0].asnumpy()# 获取第一个元素的值
        scanid = decode_scanid(item['scanid'][0].asnumpy()) 
        filename = str(scanid) + '/{}/' + '{:0>8}'.format(viewid) + "{}"
        print("filename:", filename)
        print("type:", type(filename))
        break
    print("基础测试通过!")