#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import mindspore as ms
from mindspore import Tensor
from mindspore.dataset import GeneratorDataset
import numpy as np
import sys
from datasets import find_dataset_def
from models import *
from utils import *
from datasets.data_io import read_pfm, save_pfm
import cv2
from plyfile import PlyData, PlyElement
from PIL import Image
from datasets.tank import MVSDataset as tank_MVSDataset

parser = argparse.ArgumentParser(description='Predict depth, filter, and fuse for Tanks & Temples style dataset')
parser.add_argument('--model', default='mvsnet', help='select model')

parser.add_argument('--dataset', default='tanks', help='select dataset (for compatibility)')
parser.add_argument('--testpath', help='testing data path (e.g. /path/to/TankandTemples/test_offline/)')
parser.add_argument('--testlist', help='testing scan list (file) or comma-separated scenes or leave empty to use default list)')

parser.add_argument('--batch_size', type=int, default=1, help='testing batch size')
parser.add_argument('--numdepth', type=int, default=96, help='the number of depth values')
parser.add_argument('--interval_scale', type=float, default=1.06, help='the depth interval scale')

parser.add_argument('--loadckpt', default=None, help='load a specific checkpoint')
parser.add_argument('--outdir', default='./outputs', help='output dir')
parser.add_argument('--display', action='store_true', help='display depth images and masks')

args = parser.parse_args()
print("argv:", sys.argv[1:])
print_args(args)


def read_camera_parameters(filename):
    with open(filename) as f:
        lines = f.readlines()
        lines = [line.rstrip() for line in lines]
    extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ').reshape((4, 4))
    intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ').reshape((3, 3))
    intrinsics[:2, :] /= 4
    return intrinsics, extrinsics


def read_img(filename):
    img = Image.open(filename)
    np_img = np.array(img, dtype=np.float32) / 255.
    return np_img


def read_mask(filename):
    return read_img(filename) > 0.5


def save_mask(filename, mask):
    assert mask.dtype == np.bool or mask.dtype == bool
    mask_u8 = mask.astype(np.uint8) * 255
    Image.fromarray(mask_u8).save(filename)


def read_pair_file(filename):
    data = []
    with open(filename) as f:
        num_viewpoint = int(f.readline())
        for view_idx in range(num_viewpoint):
            ref_view = int(f.readline().rstrip())
            src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
            data.append((ref_view, src_views))
    return data


def save_depth():
    if args.testlist is None:
        scenes = ['Family','Francis','Horse','Lighthouse','M60', 'Panther', 'Playground', 'Train',
                  'Auditorium', 'Ballroom', 'Courtroom','Museum', 'Palace', 'Temple']
    else:
        if os.path.isfile(args.testlist):
            with open(args.testlist) as f:
                content = f.readlines()
                scenes = [line.rstrip() for line in content]
        else:
            scenes = [s.strip() for s in args.testlist.split(',') if s.strip()]
    for scene in scenes:
        print(f"=== Processing scene: {scene} ===")
  
        dataset = tank_MVSDataset(
            datapath=args.testpath,
            n_views=11,
            ndepths=args.numdepth,
            img_wh=(1600, 1184),
            scan=[scene],
        )
        possible_columns =["imgs", "proj_matrices", "depth_values", "filename_np", "viewid"]
        ds = GeneratorDataset(dataset, column_names=possible_columns, shuffle=False)
        ds = ds.batch(args.batch_size, drop_remainder=False)

        model = MVSNet(refine=False)
        if args.loadckpt is None:
            raise ValueError("please use --loadckpt  checkpoint")
        param_dict = ms.load_checkpoint(args.loadckpt)
        ms.load_param_into_net(model, param_dict)
        model.set_train(False)

        print(f"Dataset size (batches): {ds.get_dataset_size()}")

        for batch_idx, data in enumerate(ds.create_dict_iterator()):
            imgs = Tensor(data["imgs"], ms.float32)
            proj_matrices = Tensor(data["proj_matrices"], ms.float32)
            depth_values = Tensor(data["depth_values"], ms.float32)

            outputs = model(imgs, proj_matrices, depth_values)
            depth_maps = outputs["depth"]
            conf_maps = outputs["photometric_confidence"]
            if isinstance(depth_maps, ms.Tensor):
                depth_maps = depth_maps.asnumpy()
            if isinstance(conf_maps, ms.Tensor):
                conf_maps = conf_maps.asnumpy()
            print(f"Iter {batch_idx+1}/{ds.get_dataset_size()} (scene {scene})")

            if "filename_np" in data:
                filename_np = data["filename_np"][0].asnumpy() if isinstance(data["filename_np"][0], ms.Tensor) else data["filename_np"][0]
                if isinstance(filename_np, bytes):
                    filename_str = filename_np.decode()
                else:
                    filename_str = str(filename_np)
                viewid_val = data["viewid"][0].asnumpy() if isinstance(data["viewid"][0], ms.Tensor) else data["viewid"][0]
                base_folder = filename_str
                def make_out_paths(base_folder, viewid_int):
                    depth_filename = os.path.join(args.outdir, base_folder, 'depth_est', '{:0>8}.pfm'.format(int(viewid_int)))
                    conf_filename = os.path.join(args.outdir, base_folder, 'confidence', '{:0>8}.pfm'.format(int(viewid_int)))
                    return depth_filename, conf_filename
            for i in range(depth_maps.shape[0]):
                depth_est = depth_maps[i]
                photometric_confidence = conf_maps[i]
                depth_filename, confidence_filename = make_out_paths(base_folder, viewid_val if np.ndim(viewid_val)==0 else viewid_val[i])
                os.makedirs(os.path.dirname(depth_filename), exist_ok=True)
                os.makedirs(os.path.dirname(confidence_filename), exist_ok=True)
                save_pfm(depth_filename, depth_est)
                save_pfm(confidence_filename, photometric_confidence)
                print("Saved:", depth_filename, confidence_filename)

        print(f"Finished scene: {scene}")

def reproject_with_depth(depth_ref, intrinsics_ref, extrinsics_ref, depth_src, intrinsics_src, extrinsics_src):
    width, height = depth_ref.shape[1], depth_ref.shape[0]
    x_ref, y_ref = np.meshgrid(np.arange(0, width), np.arange(0, height))
    x_ref, y_ref = x_ref.reshape([-1]), y_ref.reshape([-1])
    xyz_ref = np.matmul(np.linalg.inv(intrinsics_ref),
                        np.vstack((x_ref, y_ref, np.ones_like(x_ref))) * depth_ref.reshape([-1]))
    xyz_src = np.matmul(np.matmul(extrinsics_src, np.linalg.inv(extrinsics_ref)),
                        np.vstack((xyz_ref, np.ones_like(x_ref))))[:3]
    K_xyz_src = np.matmul(intrinsics_src, xyz_src)
    xy_src = K_xyz_src[:2] / K_xyz_src[2:3]

    x_src = xy_src[0].reshape([height, width]).astype(np.float32)
    y_src = xy_src[1].reshape([height, width]).astype(np.float32)
    sampled_depth_src = cv2.remap(depth_src, x_src, y_src, interpolation=cv2.INTER_LINEAR)

    xyz_src = np.matmul(np.linalg.inv(intrinsics_src),
                        np.vstack((xy_src, np.ones_like(x_ref))) * sampled_depth_src.reshape([-1]))
    xyz_reprojected = np.matmul(np.matmul(extrinsics_ref, np.linalg.inv(extrinsics_src)),
                                np.vstack((xyz_src, np.ones_like(x_ref))))[:3]
    depth_reprojected = xyz_reprojected[2].reshape([height, width]).astype(np.float32)
    K_xyz_reprojected = np.matmul(intrinsics_ref, xyz_reprojected)
    xy_reprojected = K_xyz_reprojected[:2] / K_xyz_reprojected[2:3]
    x_reprojected = xy_reprojected[0].reshape([height, width]).astype(np.float32)
    y_reprojected = xy_reprojected[1].reshape([height, width]).astype(np.float32)

    return depth_reprojected, x_reprojected, y_reprojected, x_src, y_src


def check_geometric_consistency(depth_ref, intrinsics_ref, extrinsics_ref, depth_src, intrinsics_src, extrinsics_src):
    width, height = depth_ref.shape[1], depth_ref.shape[0]
    x_ref, y_ref = np.meshgrid(np.arange(0, width), np.arange(0, height))
    depth_reprojected, x2d_reprojected, y2d_reprojected, x2d_src, y2d_src = reproject_with_depth(depth_ref, intrinsics_ref, extrinsics_ref,
                                                                                                  depth_src, intrinsics_src, extrinsics_src)
    dist = np.sqrt((x2d_reprojected - x_ref) ** 2 + (y2d_reprojected - y_ref) ** 2)
    depth_diff = np.abs(depth_reprojected - depth_ref)
    relative_depth_diff = depth_diff / depth_ref
    mask = np.logical_and(dist < 1, relative_depth_diff < 0.01)
    depth_reprojected[~mask] = 0
    return mask, depth_reprojected, x2d_src, y2d_src


def filter_depth(scan_folder, out_folder, plyfilename):
    pair_file = os.path.join(scan_folder, "pair.txt")
    pair_data = read_pair_file(pair_file)
    vertexs = []
    vertex_colors = []
    for ref_view, src_views in pair_data:
        ref_intrinsics, ref_extrinsics = read_camera_parameters(
            os.path.join(scan_folder, 'cams/{:0>8}_cam.txt'.format(ref_view)))
        ref_img = read_img(os.path.join(scan_folder, 'images/{:0>8}.jpg'.format(ref_view)))
        ref_depth_est = read_pfm(os.path.join(out_folder, 'depth_est/{:0>8}.pfm'.format(ref_view)))[0]
        confidence = read_pfm(os.path.join(out_folder, 'confidence/{:0>8}.pfm'.format(ref_view)))[0]
        photo_mask = confidence > 0.8

        all_srcview_depth_ests = []
        all_srcview_x = []
        all_srcview_y = []
        all_srcview_geomask = []

        geo_mask_sum = 0
        for src_view in src_views:
            src_intrinsics, src_extrinsics = read_camera_parameters(
                os.path.join(scan_folder, 'cams/{:0>8}_cam.txt'.format(src_view)))
            src_depth_est = read_pfm(os.path.join(out_folder, 'depth_est/{:0>8}.pfm'.format(src_view)))[0]

            geo_mask, depth_reprojected, x2d_src, y2d_src = check_geometric_consistency(ref_depth_est, ref_intrinsics, ref_extrinsics,
                                                                                         src_depth_est, src_intrinsics, src_extrinsics)
            geo_mask_sum += geo_mask.astype(np.int32)
            all_srcview_depth_ests.append(depth_reprojected)
            all_srcview_x.append(x2d_src)
            all_srcview_y.append(y2d_src)
            all_srcview_geomask.append(geo_mask)

        depth_est_averaged = (sum(all_srcview_depth_ests) + ref_depth_est) / (geo_mask_sum + 1)
        geo_mask = geo_mask_sum >= 3
        final_mask = np.logical_and(photo_mask, geo_mask)

        os.makedirs(os.path.join(out_folder, "mask"), exist_ok=True)
        save_mask(os.path.join(out_folder, "mask/{:0>8}_photo.png".format(ref_view)), photo_mask)
        save_mask(os.path.join(out_folder, "mask/{:0>8}_geo.png".format(ref_view)), geo_mask)
        save_mask(os.path.join(out_folder, "mask/{:0>8}_final.png".format(ref_view)), final_mask)

        print("processing {}, ref-view{:0>2}, photo/geo/final-mask:{}/{}/{}".format(scan_folder, ref_view,
                                                                                    photo_mask.mean(),
                                                                                    geo_mask.mean(), final_mask.mean()))

        height, width = depth_est_averaged.shape[:2]
        x, y = np.meshgrid(np.arange(0, width), np.arange(0, height))
        valid_points = final_mask
        print("valid_points", valid_points.mean())
        x, y, depth = x[valid_points], y[valid_points], depth_est_averaged[valid_points]
        color = ref_img[1:-16:4, 1::4, :][valid_points]
        xyz_ref = np.matmul(np.linalg.inv(ref_intrinsics),
                            np.vstack((x, y, np.ones_like(x))) * depth)
        xyz_world = np.matmul(np.linalg.inv(ref_extrinsics),
                              np.vstack((xyz_ref, np.ones_like(x))))[:3]
        vertexs.append(xyz_world.transpose((1, 0)))
        vertex_colors.append((color * 255).astype(np.uint8))

    vertexs = np.concatenate(vertexs, axis=0)
    vertex_colors = np.concatenate(vertex_colors, axis=0)
    vertexs = np.array([tuple(v) for v in vertexs], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    vertex_colors = np.array([tuple(v) for v in vertex_colors], dtype=[('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])

    vertex_all = np.empty(len(vertexs), vertexs.dtype.descr + vertex_colors.dtype.descr)
    for prop in vertexs.dtype.names:
        vertex_all[prop] = vertexs[prop]
    for prop in vertex_colors.dtype.names:
        vertex_all[prop] = vertex_colors[prop]

    el = PlyElement.describe(vertex_all, 'vertex')
    PlyData([el]).write(plyfilename)
    print("saving the final model to", plyfilename)


if __name__ == '__main__':
    # step1: save depth maps
    save_depth()

    # step2: filter and fuse
    with open(args.testlist) as f:
        scans = f.readlines()
        scans = [line.rstrip() for line in scans]
    for scan in scans:
        scan_folder = os.path.join(args.testpath, scan)
        out_folder = os.path.join(args.outdir, scan)
        filter_depth(scan_folder, out_folder, os.path.join(args.outdir, f'{scan}.ply'))
