"""
This file (visualize.py) is designed for:
    visualization
Copyright (c) 2021, Yongjie Duan. All rights reserved.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from pylab import subplots_adjust
from scipy.ndimage import zoom
import cv2
from collections import Iterable


def tensor_to_numpy(tensor):
    if torch.is_tensor(tensor):
        try:
            tensor = tensor.detach().squeeze(0).numpy()
        except:
            tensor = tensor.detach().squeeze(0).cpu().numpy()
    return tensor


def matplotlib_imshow(ax, img, alpha=1.0, reverse=False, one_channel=False, factor=1):
    npimg = tensor_to_numpy(img)
    npimg = (npimg - npimg.min()) / (npimg.max() - npimg.min()).clip(0.001, None)
    if reverse:
        npimg = 1 - npimg
    npimg[0], npimg[:, 0] = 0, 0
    npimg[-1], npimg[:, -1] = 0, 0
    if factor != 1:
        npimg = zoom(npimg, factor, order=1)
    npimg = (npimg * 255).astype(np.uint8)

    if one_channel:
        ax.imshow(npimg, alpha=alpha, cmap="gray", vmin=0, vmax=255)
    elif npimg.ndim > 2:
        ax.imshow(np.transpose(npimg, (1, 2, 0)), alpha=alpha)
    else:
        ax.imshow(npimg, alpha=alpha, cmap="gray", vmin=0, vmax=255)

    ax.axis("off")


def matplotlib_draw_pose(ax, pose_2d, length=100, color="green"):
    trans, theta = pose_2d[:2], pose_2d[2]

    start = trans
    theta = np.deg2rad(theta)
    end = (start[0] + length * np.sin(theta), start[1] - length * np.cos(theta))

    ax.plot(start[0], start[1], "o", color=color)
    ax.arrow(start[0], start[1], end[0] - start[0], end[1] - start[1], width=2, fc=color, ec=color)


def draw_on_image(name, img_lst, pose_2d_lst, pose_3d_lst, seg_lst):
    pose_3d_lst = [tensor_to_numpy(x) for x in pose_3d_lst]
    pose_2d_lst = [tensor_to_numpy(x) for x in pose_2d_lst]

    n_cols = max(len(img_lst), len(seg_lst))
    fig = plt.figure(figsize=(4 * n_cols, 4 * 2))

    suptitle = [f"{name[0]}"]
    suptitle = suptitle + [f"{np.round(x[0], 2)}" for x in pose_3d_lst]
    suptitle = suptitle + [f"{np.round(x[0], 2)}" for x in pose_2d_lst]
    suptitle = "\n".join(suptitle)
    plt.suptitle(suptitle, fontsize=10)

    for ii in range(len(img_lst)):
        ax = fig.add_subplot(2, n_cols, ii + 1, xticks=[], yticks=[])
        if ii < len(pose_2d_lst):
            matplotlib_imshow(ax, img_lst[ii][0], reverse=False)
            matplotlib_draw_pose(ax, pose_2d_lst[ii][0])
        else:
            matplotlib_imshow(ax, img_lst[ii][0], reverse=False)

    for ii in range(len(seg_lst)):
        ax = fig.add_subplot(2, n_cols, n_cols + ii + 1, xticks=[], yticks=[])
        matplotlib_imshow(ax, seg_lst[ii][0])

    plt.tight_layout()
    return fig


def draw_axis(img, yaw, pitch, roll, tdx=None, tdy=None, size=50, thickness=(2, 2, 2)):
    """
    Function used to draw y (headpose label) on Input Image x.
    Implemented by: shamangary
    https://github.com/shamangary/FSA-Net/blob/master/demo/demo_FSANET.py
    Modified by: Omar Hassan
    """
    pitch = pitch * np.pi / 180
    yaw = -(yaw * np.pi / 180)
    roll = roll * np.pi / 180

    if tdx != None and tdy != None:
        tdx = tdx
        tdy = tdy
    else:
        height, width = img.shape[:2]
        tdx = width / 2
        tdy = height / 2

    # X-Axis pointing to right. drawn in red
    x1 = size * (np.cos(yaw) * np.cos(roll)) + tdx
    y1 = size * (np.cos(pitch) * np.sin(roll) + np.cos(roll) * np.sin(pitch) * np.sin(yaw)) + tdy

    # Y-Axis | drawn in green
    #        v
    x2 = size * (-np.cos(yaw) * np.sin(roll)) + tdx
    y2 = size * (np.cos(pitch) * np.cos(roll) - np.sin(pitch) * np.sin(yaw) * np.sin(roll)) + tdy

    # Z-Axis (out of the screen) drawn in blue
    x3 = size * (np.sin(yaw)) + tdx
    y3 = size * (-np.cos(yaw) * np.sin(pitch)) + tdy

    cv2.line(img, (int(tdx), int(tdy)), (int(x1), int(y1)), (0, 0, 255), thickness[0])
    cv2.line(img, (int(tdx), int(tdy)), (int(x2), int(y2)), (0, 255, 0), thickness[1])
    cv2.line(img, (int(tdx), int(tdy)), (int(x3), int(y3)), (255, 0, 0), thickness[2])

    return img
