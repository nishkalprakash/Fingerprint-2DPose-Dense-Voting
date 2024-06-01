"""
This file (deploy.py) is designed for:
    deploy pose estimation for other dataset
Copyright (c) 2022, Yongjie Duan. All rights reserved.
"""
import argparse
import os.path as osp
from glob import glob
from types import SimpleNamespace
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn as nn
import yaml
from PIL import Image
from scipy import ndimage as sndi

from fptools import fp_draw, uni_io
from models.model_zoo import *
from utils.misc import load_model


def affine_matrix(scale=1.0, theta=0.0, trans_1=np.zeros(2), trans_2=np.zeros(2)):
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]) * scale
    t = np.dot(R, trans_1) + trans_2
    return np.array([[R[0, 0], R[0, 1], t[0]], [R[1, 0], R[1, 1], t[1]], [0, 0, 1]])


def process_img(img_ori, img_ppi, img_c=None):
    img = 255 - img_ori

    scale = img_ppi * 1.0 / 500
    ppad = 64

    if scale < 1:
        img = sndi.uniform_filter(img, min(15, np.rint(1.2 / scale)))
        img = sndi.zoom(img, scale, order=1)

    tar_shape = np.rint(np.maximum(np.ones(2), (np.array(img.shape[:2]) + ppad / 2) // ppad) * ppad).astype(int)
    center = tar_shape[::-1] / 2.0
    shift = np.zeros(2)
    img_c = np.array(img.shape[1::-1]) / 2.0 if img_c is None else img_c

    T = affine_matrix(scale=1, theta=0, trans_1=-img_c, trans_2=center + shift)
    img = cv2.warpAffine(img, T[:2], dsize=tuple(tar_shape[::-1]), flags=cv2.INTER_LINEAR)

    return img, scale, T


def deploy(args, model, model_params, prefix):
    img_ori = np.asarray(Image.open(osp.join(prefix, args.img_name)).convert("L"), dtype=np.float32)

    img_in = img_ori
    img_c = None

    img, scale, T = process_img(img_in, model_params.img_ppi, img_c)

    main_dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        output = model.module.get_prediction(torch.from_numpy(img).float()[None, None].to(main_dev))
    out_prefix = prefix.replace("test_", "fvc_pose_2d_gridnet4")
    pose_file = osp.join(out_prefix, args.img_name.split(".")[0] + ".txt")
    show_file = osp.join(out_prefix, args.img_name.split(".")[0] + ".png")
    # seg_file = osp.join(prefix + "_feature", "seg", model_params.exp_name, args.img_name.split(".")[0] + ".png")

    T_inv = np.linalg.inv(T)
    pose_2d = output["pose_2d"].squeeze().detach().cpu().numpy()
    pose_2d[:2] = np.dot(T_inv[:2, :2], pose_2d[:2]) + T_inv[:2, 2]
    pose_2d[0] = round(pose_2d[0] / scale)
    pose_2d[1] = round(pose_2d[1] / scale)
    pose_2d[2] = (-pose_2d[2] + 180) % 360 - 180

    Path(osp.dirname(pose_file)).mkdir( parents=True, exist_ok=True)
    np.savetxt(pose_file, [pose_2d], fmt="%g",)
    fp_draw.draw_img_with_pose(img_ori, pose_2d, show_file, color="red")

    seg = output["seg"].squeeze().detach().cpu().numpy()
    tar_shape = np.array(img.shape[:2]).astype(int)
    seg = cv2.warpAffine(
        sndi.zoom(seg, model_params.seg_zoom, order=1), T_inv[:2], dsize=tuple(tar_shape[::-1]), flags=cv2.INTER_LINEAR
    )
    if scale < 1:
        seg = sndi.zoom(seg, 1.0 / scale, order=1)
    # uni_io.imwrite(seg_file, seg[: img_ori.shape[0], : img_ori.shape[1]] * 255)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", "-g", default=[0], type=int, nargs="+")
    # parser.add_argument("--input-folder", "-f", default="input_folder", type=str, help="input folder containing images")
    args = parser.parse_args()

    input_folder = "test_tiff"

    # this is the trained model parameters for plain fingerprints, trained models for rolled and
    # latent fingerprints are coming soon...
    #
    # you can also tried the released model on rolled fingerprints and enhanced latent fingerprints
    # (using FingerNet), but the performance is not guaranteed
    ckp_date = "gridnet4"

    ckp_dir = osp.join("./logs", ckp_date)
    i2p_params = yaml.safe_load(open(osp.join(ckp_dir, "configs.yaml"), "r"))
    i2p_params = SimpleNamespace(**i2p_params)
    i2p_params.__dict__.update(args.__dict__)
    try:
        i2p_model = eval(i2p_params.exp_name.upper())(
            num_in=i2p_params.num_in,
            num_pose_2d=i2p_params.num_pose_2d,
            num_layers=i2p_params.num_layers,
            img_ppi=i2p_params.img_ppi,
            middle_shape=np.array(i2p_params.middle_shape),
            activate=i2p_params.activate,
            bin_type=i2p_params.bin_type,
            with_tv=i2p_params.with_tv,
            with_enh=i2p_params.with_enh,
        )
    except Exception as ex:
        raise ValueError(ex)
    name_lst = glob(osp.join("./", "logs", ckp_date, "e_*.pth.tar"))
    name_lst.sort()
    load_model(i2p_model, name_lst[-1])
    i2p_model = nn.DataParallel(i2p_model, device_ids=i2p_params.gpus)
    main_dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    i2p_model.to(main_dev)
    i2p_model.eval()

    input_folder_list = Path("~/datasets/fvc_fingerprint_datasets/").expanduser().rglob("FVC200*/Dbs/DB*/")
    input_folder_list = Path(".").expanduser().rglob("test_*")
    for input_folder in input_folder_list:
        image_files = [i.as_posix() for i in input_folder.glob("*") if i.as_posix().endswith(".tif") or i.as_posix().endswith(".bmp")]
        for image_file in image_files:
            args.img_name = osp.basename(image_file)
            deploy(args, i2p_model, i2p_params, input_folder.as_posix())
    # image_files = glob(osp.join(input_folder, "*.tif"))

    # for image_file in image_files:
    #     args.img_name = osp.basename(image_file)
    #     deploy(args, i2p_model, i2p_params, input_folder)

