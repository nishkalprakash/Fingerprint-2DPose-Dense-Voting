"""
This file (metrics.py) is designed for:
    metric functions
Copyright (c) 2021, Yongjie Duan. All rights reserved.
"""
import os
import sys

# os.chdir(sys.path[0])
import os.path as osp
import numpy as np
from glob import glob
from scipy.spatial.transform import Rotation


def angle_metric(pred, gt, deg=True, err="l1"):
    if pred is None or gt is None or len(pred) == 0 or len(gt) == 0:
        return 0

    if pred.ndim > 2 and pred.shape[1] == 4:
        # quat
        pred = Rotation.from_quat(pred).as_euler("ZYX", degrees=True)

        assert pred.shape[1] == gt.shape[1]

    # euler
    error = np.abs(pred - gt)
    error = np.minimum(error, 360 - error)

    if err == "l1":
        error = np.abs(error).mean()
    elif err == "l2":
        error = (error ** 2).mean()
    else:
        raise ValueError(f"Unsupport metric type {err} for AngleMetric")

    return error


def angular_metric(pred, gt):
    if pred.ndim == 2:
        pred = Rotation.from_euler("ZYX", pred, degrees=True).as_matrix()
        gt = Rotation.from_euler("ZYX", gt, degrees=True).as_matrix()

    M = np.matmul(gt, pred.swapaxes(-2, -1))
    error = np.arccos((np.diagonal(M, axis1=-2, axis2=-1).sum(-1) - 1) / 2)
    error = np.abs(np.rad2deg(error)).mean()

    return error


def center_metric(pred, gt, err="l1"):
    assert pred.shape[1] == gt.shape[1]

    error = pred - gt
    if err == "l1":
        error = np.abs(error).mean()
    elif err == "l2":
        error = (error ** 2).mean()
    else:
        raise ValueError(f"Unsupport metric type {err} for CenterMetric")

    return error


def flatten(input):
    C = input.shape[1]
    axis_order = (1, 0) + tuple(range(2, input.ndim))
    out = input.transpose(axis_order)
    return out.reshape(C, -1)


def iou_metric(pred, gt, smooth=1.0):
    if pred is None or gt is None or len(pred) == 0 or len(gt) == 0:
        return 100

    pred = flatten(pred)
    gt = flatten(gt)
    inter = (pred * gt).sum(1)
    intra = (pred + gt).sum(1)
    iou = (inter + smooth) / (intra - inter + smooth)
    return iou * 100
