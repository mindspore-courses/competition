
from datasets.data_io import *
# from data_io import *# 如果要运行最底部的main，同级运行则解
import numpy as np
import cv2,os
from collections import defaultdict
from PIL import Image
from mindspore import dataset as Dataset
from mindspore.dataset import GeneratorDataset
import mindspore
class MVSDataset():
    def __init__(self, datapath, n_views=3, ndepths=192, img_wh=(1600, 1184), split='intermediate', scan=['Family']):

        self.datapath = datapath
        self.stages = 4
        self.img_wh = img_wh
        self.input_scans = scan
        self.split = split
        self.n_views = n_views
        self.ndepths = ndepths
        self.build_metas()

    def build_metas(self):
        self.metas = []
        if self.split == 'intermediate':
            self.scans = self.input_scans
            # self.scans = ['Family', 'Francis', 'Horse', 'Lighthouse',
            #               'M60', 'Panther', 'Playground', 'Train']
            # self.image_sizes = {'Panther': (2048, 1080)}
            self.image_sizes = {'Family': (1600, 1184),
                                'Francis': (1600, 1184),
                                'Horse': (1600, 1184),
                                'Lighthouse': (1600, 1184),
                                'M60': (1600, 1184),
                                'Panther': (1600, 1184),
                                'Playground': (1600, 1184),
                                'Train': (1600, 1184),
                                'Auditorium': (1600, 1184),
                                'Ballroom': (1600, 1184),
                                'Courtroom': (1600, 1184),
                                'Museum': (1600, 1184),
                                'Palace': (1600, 1184),
                                'Temple': (1600, 1184),
                                'Truck': (1600, 1184),
                                "Ignatius": (1600, 1184)
                                }

        elif self.split == 'advanced':
            self.scans = self.input_scans
            # self.scans = ['Auditorium', 'Ballroom', 'Courtroom',
            #               'Museum', 'Palace', 'Temple']
            self.image_sizes = {'Auditorium': (1600, 1184),
                                'Ballroom': (1600, 1184),
                                'Courtroom': (1600, 1184),
                                'Museum': (1600, 1184),
                                'Palace': (1600, 1184),
                                'Temple': (1600, 1184)}

        for scan in self.scans:
            if scan in ['Family', 'Francis', 'Horse', 'Lighthouse','M60', 'Panther', 'Playground', 'Train']:
                split = 'intermediate'
            elif scan in ['Auditorium', 'Ballroom', 'Courtroom','Museum', 'Palace', 'Temple']:
                split = 'advanced'
            else:
                split = ''

            with open(os.path.join(self.datapath, split, scan, 'pair.txt')) as f:
                num_viewpoint = int(f.readline())
                for view_idx in range(num_viewpoint):
                    ref_view = int(f.readline().rstrip())
                    src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
                    if len(src_views) != 0:
                        self.metas += [(scan, -1, ref_view, src_views, split)]

    def read_cam_file(self, filename):
        with open(filename) as f:
            lines = [line.rstrip() for line in f.readlines()]
        # extrinsics: line [1,5), 4x4 matrix
        extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ')
        extrinsics = extrinsics.reshape((4, 4))
        # intrinsics: line [7-10), 3x3 matrix
        intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ')
        intrinsics = intrinsics.reshape((3, 3))

        depth_values = lines[11].split()
        if len(depth_values) < 4:
            depth_min = float(lines[11].split()[0])
            depth_max = float(lines[11].split()[1])
        else:
            depth_min = float(lines[11].split()[0])
            depth_max = float(lines[11].split()[3])

        return intrinsics, extrinsics, depth_min, depth_max

    def read_img(self, filename, imsize):
        img = Image.open(filename)
        # scale 0~255 to 0~1
        np_img = np.array(img, dtype=np.float32) / 255.
        np_img = cv2.resize(np_img, imsize, interpolation=cv2.INTER_LINEAR)
        # print(np_img.shape)
        return np_img


    def center_img(self, img):  # this is very important for batch normalization
        var = np.var(img, axis=(0, 1), keepdims=True)
        mean = np.mean(img, axis=(0, 1), keepdims=True)
        return (img - mean) / (np.sqrt(var) + 0.00000001)

    def __len__(self):
        return len(self.metas)

    def __getitem__(self, idx):
        # cv2.setNumThreads(0)
        # cv2.ocl.setUseOpenCL(False)
        scan, _, ref_view, src_views, split = self.metas[idx]
        # use only the reference view and first nviews-1 source views
        view_ids = [ref_view] + src_views[:self.n_views - 1]
        img_w, img_h = self.image_sizes[scan]

        # depth = None
        imgs = []
        depth_values = None
        proj_matrices = []

        for i, vid in enumerate(view_ids):
            img_filename = os.path.join(self.datapath, split, scan, f'images/{vid:08d}.jpg')
            if split not in ['intermediate', 'advanced']:
                proj_mat_filename = os.path.join(self.datapath, split, scan, f'cams/{vid:08d}_cam.txt')
            else:
                proj_mat_filename = os.path.join(self.datapath, split, scan, f'cams_1/{vid:08d}_cam.txt')

            img = self.read_img(img_filename, (1600, 1184))
            # img = cv2.resize(img, (1920,1024), interpolation=cv2.INTER_LINEAR)
            intrinsics, extrinsics, depth_min_, depth_max_ = self.read_cam_file(proj_mat_filename)
            intrinsics[0] *= self.img_wh[0] / img_w
            intrinsics[1] *= self.img_wh[1] / img_h
            imgs.append(img)

            proj_mat = np.zeros(shape=(2, 4, 4), dtype=np.float32)  #
            proj_mat[0, :4, :4] = extrinsics
            proj_mat[1, :3, :3] = intrinsics
            proj_matrices.append(proj_mat)


            # depth_max_ = 1
            # depth_min_ = 0.5

            # if i == 0:  # reference view
            #     disp_min = 1 / depth_max_
            #     disp_max = 1 / depth_min_
            #     depth_values = np.linspace(disp_min, disp_max, self.ndepths, dtype=np.float32)
            if i == 0:  # reference view
                depth_values = np.linspace(depth_min_, depth_max_, self.ndepths, dtype=np.float32)

        # imgs: N*3*H0*W0, N is number of images
        imgs = np.stack(imgs).transpose([0, 3, 1, 2])
        proj_matrices = np.stack(proj_matrices)
        # stage0_pjmats = proj_matrices.copy()
        # stage0_pjmats[:, 1, :2, :] = proj_matrices[:, 1, :2, :] * 0.0625
        # stage1_pjmats = proj_matrices.copy()
        # stage1_pjmats[:, 1, :2, :] = proj_matrices[:, 1, :2, :] * 0.125
        # stage2_pjmats = proj_matrices.copy()
        # stage2_pjmats[:, 1, :2, :] = proj_matrices[:, 1, :2, :] * 0.25
        # stage3_pjmats = proj_matrices.copy()
        # stage3_pjmats[:, 1, :2, :] = proj_matrices[:, 1, :2, :] * 0.5
        # proj_matrices_ms = {
        #     "stage0":stage0_pjmats,
        #     "stage1":stage1_pjmats,
        #     "stage2":stage2_pjmats,
        #     "stage3":stage3_pjmats,
        #     "stage4":proj_matrices,
        # }

        imgs = mindspore.Tensor.from_numpy(imgs.copy()).contiguous().float()
        depth_values = mindspore.Tensor.from_numpy(depth_values.copy()).contiguous().float()
        # return {"imgs": imgs,
        #         "proj_matrices": proj_matrices_ms,
        #         "depth_values": depth_values,
        #         "filename": scan + '/{}/' + '{:0>8}'.format(view_ids[0]) + "{}"}
        filename = scan + '/{}/' + '{:0>8}'.format(view_ids[0]) + "{}"
        filename_np = np.array(filename.encode('utf-8'), dtype=np.bytes_)
        return (
            imgs,                   # "imgs" (nviews, 3, H, W)
            proj_matrices,        # "proj_matrices" (多尺度投影矩阵)
            depth_values,            # "depth_values" (ndepths,)
            filename_np,
            view_ids[0]
        )
if __name__ == "__main__":
    TANK_TESTING='/media/outbreak/68E1-B517/Dataset/TankandTemples/test_offline/'
    # testlist_path = "lists/dtu/test.txt"
    # with open(testlist_path) as f:
    #     content = f.readlines()
    #     testlist = [line.rstrip() for line in content]
    testlist = ['Family','Francis','Horse','Lighthouse','M60', 'Panther', 'Playground', 'Train', 'Auditorium', 'Ballroom', 'Courtroom','Museum', 'Palace', 'Temple']
    for scene in testlist:
        print(f"正在测试场景: {scene}")
        dataset = MVSDataset(
            datapath=TANK_TESTING,
            n_views=11,
            ndepths = 96,
            img_wh=(1600,1184),
            scan=[scene],
        )
        # 测试1: 检查数据集长度
        print(f"数据集长度: {len(dataset)}")
        assert len(dataset) > 0, "数据集为空"
        
        # 测试2: 获取第一个样本并检查结构
        sample = dataset[0]
        print("样本结构类型:", type(sample))
        imgs, proj_matrices, depth_values,filename_np,viewid = sample
        print(f"图像形状: {imgs.shape}")          # (5, 3, H, W)
        print(f"投影矩阵类型: {type(proj_matrices)}") # dict
        print(f"深度值形状: {depth_values.shape}")  # (384,)
        depth_values = depth_values.asnumpy()
        print(f"深度值: {depth_values}")
        print(f"scanid: {filename_np}")
        print(f"viewid: {viewid}")
        
        # 测试4: 通过GeneratorDataset加载
        minds_dataset = GeneratorDataset(
            dataset, 
            column_names=["imgs", "proj_matrices","depth_values","filename_np","viewid"], 
            shuffle=True
            )
        batched_dataset = minds_dataset.batch(batch_size=4)
        # create_tuple_iterator返回列表
        # create_dict_iterator返回字典
        iterator = batched_dataset.create_dict_iterator()
        for item in iterator:
            print(type(item))
            print(item.keys())
            break
        print("基础测试通过!")
        break