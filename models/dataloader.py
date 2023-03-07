"""
This file (dataloader.py) is designed for:
    dataloader for pytorch
Copyright (c) 2021, Yongjie Duan. All rights reserved.
"""
import os.path as osp
import pickle
import random
from glob import glob

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage as sndi
from scipy.interpolate import RectBivariateSpline
from torch.utils.data import Dataset

from fptools import uni_image

cv2.setUseOptimized(True)


def array_histogram_equalization(array, number_bins=180, low=-90, high=90):
    # from http://www.janeriksolem.net/histogram-equalization-with-python-and.html

    # get image histogram
    image_histogram, bins = np.histogram(array.flatten(), number_bins, density=True)
    cdf = image_histogram.cumsum()  # cumulative distribution function
    cdf = (high - low) * cdf / cdf[-1] + low  # normalize

    # use linear interpolation of cdf to find new pixel values
    def func(src):
        if isinstance(src, np.ndarray):
            image_equalized = np.interp(src.flatten(), bins[:-1], cdf)
            return image_equalized.reshape(src.shape)
        elif isinstance(src, list):
            src = np.array(src)
            image_equalized = np.interp(src.flatten(), bins[:-1], cdf)
            return image_equalized.reshape(src.shape)
        else:
            assert isinstance(src, (float, int))
            value = np.interp(np.array([src]), bins[:-1], cdf)
            return value[0]

    return func


class MyDataset(Dataset):
    def __init__(
        self,
        prefix,
        pkl_path,
        img_ppi=500,
        ranges_2d=(-90, 90),
        with_bg=False,
        seg_zoom=4,
        do_aug=False,
        repeat=1,
        phase="train",
        middle_shape=(512, 512),
        seed=None,
    ) -> None:
        super().__init__()
        self.prefix = prefix
        self.pkl_path = pkl_path
        self.phase = phase
        self.img_ppi = img_ppi
        self.ranges_2d = ranges_2d
        self.with_bg = with_bg
        self.seg_zoom = seg_zoom
        self.do_aug = do_aug
        self.middle_shape = np.array(middle_shape)
        self.seed = seed

        self.scale = self.img_ppi * 1.0 / 500
        ppad = 128 if self.img_ppi == 30 else 64
        self.tar_shape = np.rint(np.maximum(np.ones(2), (self.middle_shape * self.scale + ppad / 2) // ppad) * ppad).astype(
            int
        )

        # self.pca_model = sio.loadmat("./fptools/PCAParameterForDistortedVideo.mat")
        with open(pkl_path, "rb") as fp:
            self.items = pickle.load(fp)
            self.items = self.items * repeat

        if not self.do_aug:
            # all_angles = np.array([x["pose_3d"] for x in self.items])
            # self.y_heq = array_histogram_equalization(all_angles[:, 0] % 180)
            # print("absolute histogram")
            self.rand_yaw = np.random.uniform(low=self.ranges_2d[0], high=self.ranges_2d[1], size=len(self.items))
            self.rand_shift = np.random.uniform(low=-0.2, high=0.2, size=(len(self.items), 2))
            self.rand_para = np.random.uniform(low=0, high=10, size=(len(self.items), 2))

        if self.with_bg:
            self.bg_lst = glob("/home/duanyongjie/data/finger/MSRA-TD500/image/**/*.JPG", recursive=True)
            self.bg_lst.sort()
            random.shuffle(self.bg_lst)

            self.rand_bg_scale = np.random.uniform(low=0.5, high=2, size=len(self.items))
            self.rand_bg_rotate = np.random.uniform(low=-180, high=180, size=len(self.items))
            self.rand_bg_center = np.random.uniform(low=0, high=1, size=(len(self.items), 2))
            self.rand_bg_lambda = np.random.uniform(low=0.2, high=0.8, size=len(self.items))

        # # mini-dataset
        # self.items = self.items[:100]

        if self.do_aug:
            random.shuffle(self.items)

    def load_img(self, img_path):
        # img = np.asarray(imageio.imread(img_path), dtype=np.float32)
        img = np.asarray(Image.open(img_path).convert("L"), dtype=np.float32)
        return img

    def padding_img(self, img, tar_shape, cval=0):
        src_shape = np.array(img.shape[:2])
        padding = np.maximum(src_shape, tar_shape) - src_shape
        img = np.pad(img, ((0, padding[0]), (0, padding[1])), constant_values=cval)
        return img[: tar_shape[0], : tar_shape[1]]

    def resize_img(self, img, ratio, order=1, cval=0):
        resized_img = sndi.zoom(img, ratio, order=order, cval=cval)
        return resized_img

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        item = self.items[index]

        img_format = ".bmp" if "NIST14" in item["img"] or "Hisign" in item["img"] else ".png"
        img = self.load_img(osp.join(self.prefix, item["img"] + img_format))
        seg = (
            self.load_img(osp.join(self.prefix, item["seg"] + ".png")) / 255
            if item["seg"] is not None
            else np.zeros_like(img)
        )
        pose_2d = np.loadtxt(osp.join(self.prefix, item["pose_2d"] + ".txt"))
        pose_3d = np.array(item["pose_3d"])[1:] if item["pose_3d"] is not None else np.zeros(2)  # yaw angle excluded
        seg_flags = 1 if item["seg"] is not None else 0
        pose_flags = 1 if item["pose_3d"] is not None else 0
        pose_2d[2] *= -1
        path = "/".join(item["img"].split("/")[-3:]).replace(".png", "")

        # cval = img[0].max()
        img = 255 - img

        # zoom to simulate low resolution image
        if self.scale < 1:
            win_size = min(15, np.rint(1.2 / self.scale))
            if self.do_aug:
                win_size = np.rint(win_size * np.random.uniform(1, 1.25)).astype(int)
            img = sndi.uniform_filter(img, win_size, mode="constant")
            img = self.resize_img(img, self.scale, order=1)
            seg = self.resize_img(seg, self.scale, order=0)
            pose_2d[:2] = pose_2d[:2] * self.scale

        center = self.tar_shape[::-1] / 2.0
        scale = 1.0

        if self.do_aug:
            if np.random.random() > 0.5:
                img = np.flip(img, axis=1)
                seg = np.flip(seg, axis=1)
                pose_2d[0] = img.shape[1] - pose_2d[0]
                pose_2d[2] = -pose_2d[2]
                pose_3d[-1] = -pose_3d[-1]

            rot = pose_2d[2] - np.random.uniform(low=self.ranges_2d[0], high=self.ranges_2d[1])
            shift = np.array(
                (
                    np.random.uniform(-0.2, 0.2) * self.tar_shape[1],
                    np.random.uniform(-0.2, 0.2) * self.tar_shape[0],
                )
            )
            para = 10 * (np.random.uniform(0, 10, 2) - 5)
        else:
            rot = pose_2d[2] - self.rand_yaw[index]
            shift = np.array(
                (
                    self.rand_shift[index, 1] * self.tar_shape[1],
                    self.rand_shift[index, 0] * self.tar_shape[0],
                )
            )
            para = 10 * (self.rand_para[index] - 5)
        seg_c = np.array(seg.shape[1::-1]) / 2.0

        T = affine_matrix(scale=scale, theta=-np.deg2rad(rot), trans=-seg_c, trans_2=center + shift)

        # no TPS deformation
        img = cv2.warpAffine(img, T[:2], dsize=tuple(self.tar_shape[::-1]), flags=cv2.INTER_LINEAR)
        seg = cv2.warpAffine(seg, T[:2], dsize=tuple(self.tar_shape[::-1]), flags=cv2.INTER_NEAREST)

        # # TPS deformation
        # flow = np.squeeze(self.pca_model["coeff"][:, :2].dot(para[:, None]) + self.pca_model["f_mean"])
        # matches = [cv2.DMatch(ii, ii, 0) for ii in range(len(flow) // 2)]
        # tps_pts = fast_tps_distortion(
        #     img.shape,
        #     self.tar_shape,
        #     flow,
        #     matches,
        #     center=pose_2d[:2],
        #     theta=np.deg2rad(pose_2d[2]),
        #     shift=shift,
        #     rotation=-np.deg2rad(rot),
        # )
        # img = cv2.remap(img, tps_pts[..., 0], tps_pts[..., 1], cv2.INTER_LINEAR)
        # seg = cv2.remap(seg, tps_pts[..., 0], tps_pts[..., 1], cv2.INTER_NEAREST)

        pose_2d[:2] = np.dot(T[:2, :2], pose_2d[:2]) + T[:2, 2]
        pose_2d[2] = (pose_2d[2] - rot + 180) % 360 - 180

        seg = self.resize_img(seg, 1.0 / self.seg_zoom, order=0)

        if self.with_bg:
            # load background image
            img_bg = self.load_img(self.bg_lst[index % len(self.bg_lst)])
            bg_cval = int(img_bg[0].max())
            if self.do_aug:
                bg_scale = np.random.uniform(low=0.5, high=2)
                bg_rot = np.random.uniform(-180, 180)
                bg_center = np.random.uniform(low=0, high=1, size=2) * np.array(img_bg.shape[1::-1])
                bg_lambda = np.random.uniform(low=0.2, high=0.8)
            else:
                bg_scale = self.rand_bg_scale[index]
                bg_rot = self.rand_bg_rotate[index]
                bg_center = self.rand_bg_center[index] * np.array(img_bg.shape[1::-1])
                bg_lambda = self.rand_bg_lambda[index]
            T = affine_matrix(scale=bg_scale, theta=-np.deg2rad(bg_rot), trans=-bg_center, trans_2=center)

            img_bg = 255 - cv2.warpAffine(
                img_bg, T[:2], dsize=tuple(self.tar_shape), flags=cv2.INTER_LINEAR, borderValue=bg_cval
            )
            # img_bg = cv2.equalizeHist(np.rint(img_bg).astype(np.uint8)) * 1.0
            # img = cv2.equalizeHist(np.rint(img).astype(np.uint8)) * 1.0

            img = bg_lambda * img + (1 - bg_lambda) * img_bg
            # img = img_bg + bg_lambda * img * (img - img_bg) / 255

        if self.do_aug and np.random.random() > 0.5:
            img = uni_image.add_noise(img, scale=1, sigma=0.1)

        return {
            "img": 255 - img[None].astype(np.float32),
            "seg": seg[None].astype(np.float32),
            "pose_2d": np.array(pose_2d).astype(np.float32),
            "pose_3d": np.array(pose_3d).astype(np.float32),
            "seg_flags": np.array(seg_flags).astype(np.float32),
            "pose_flags": np.array(pose_flags).astype(np.float32),
            "name": path,
        }


def affine_matrix(scale=1.0, theta=0.0, trans=np.zeros(2), trans_2=np.zeros(2)):
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]) * scale
    t = np.dot(R, trans) + trans_2
    return np.array([[R[0, 0], R[0, 1], t[0]], [R[1, 0], R[1, 1], t[1]], [0, 0, 1]])


def fast_tps_distortion(cur_shape, tar_shape, flow, matches, center=None, theta=0, shift=np.zeros(2), rotation=0, stride=16):
    cur_center = np.array([cur_shape[1], cur_shape[0]]) / 2
    tar_center = np.array([tar_shape[1], tar_shape[0]]) / 2
    if center is None:
        center = cur_center
    R_theta = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    R_rotation = np.array([[np.cos(rotation), -np.sin(rotation)], [np.sin(rotation), np.cos(rotation)]])

    src_x, src_y = np.meshgrid(np.linspace(-200, 200, 11), np.linspace(-160, 160, 9))
    src_x = src_x.T.reshape(-1)
    src_y = src_y.T.reshape(-1)
    src_cpts = np.stack((src_x, src_y), axis=-1)
    tar_cpts = src_cpts + flow.reshape(-1, 2)

    src_cpts = src_cpts.dot(R_theta.T) + center[None]
    tar_cpts = tar_cpts.dot(R_theta.T) + tar_center[None]

    tps = cv2.createThinPlateSplineShapeTransformer()
    tps.estimateTransformation(tar_cpts[None], src_cpts[None], matches)
    grid_x = np.arange(-stride, tar_shape[1] + stride * 2 - 2, step=stride)
    grid_y = np.arange(-stride, tar_shape[0] + stride * 2 - 2, step=stride)
    tar_pts = np.stack(np.meshgrid(grid_x, grid_y), axis=-1).astype(np.float32)
    src_pts = tps.applyTransformation(tar_pts.reshape(1, -1, 2))[1].reshape(*tar_pts.shape)

    bspline_x = RectBivariateSpline(grid_y, grid_x, src_pts[..., 0])
    bspline_y = RectBivariateSpline(grid_y, grid_x, src_pts[..., 1])
    tps_x, tps_y = np.meshgrid(np.arange(tar_shape[1]), np.arange(tar_shape[0]))
    tps_x = bspline_x.ev(tps_y, tps_x).astype(np.float32)
    tps_y = bspline_y.ev(tps_y, tps_x).astype(np.float32)
    tps_pts = np.stack((tps_x, tps_y), axis=-1)

    tps_pts = (tps_pts - center[None] - shift[None]).dot(R_rotation) + cur_center[None]
    # tps_pts = (np.stack((tps_x, tps_y), axis=-1) - center[None] - shift[None]).dot(R_rotation) + center[None]
    tps_pts = tps_pts.astype(np.float32)

    return tps_pts
