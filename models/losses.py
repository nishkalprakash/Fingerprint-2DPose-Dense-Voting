"""
This file (losses.py) is designed for:
    loss functions
Copyright (c) 2021, Yongjie Duan. All rights reserved.
"""
from glob import glob

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .units import *


class ImgBCELoss(nn.Module):
    def __init__(self, w_pos=1, w_neg=1, eps=1e-6) -> None:
        super().__init__()
        self.w_pos = w_pos
        self.w_neg = w_neg
        self.eps = eps

    def forward(self, input, target):
        input = torch.clamp(input, self.eps, 1 - self.eps)
        loss = self.w_pos * target * torch.log(input) + self.w_neg * (1 - target) * torch.log(1 - input)
        # loss /= np.log(2) * (self.w_pos + self.w_neg)
        return -loss.mean()


class MSELoss(nn.Module):
    def __init__(self, eps=1e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, input, target, is_angle=True, deg=True):
        if is_angle:
            loss = normalize_angle(input - target, deg=deg) ** 2
        else:
            loss = (input - target) ** 2
        return loss.mean()


class MAELoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, input, target, is_angle=True, deg=True):
        if is_angle:
            loss = normalize_angle(input - target, deg=deg)
        else:
            loss = (input - target).abs()
        return loss.mean()


def normalize_angle(theta, deg=True):
    p = 360 if deg else 2 * np.pi
    angle = theta.abs()
    return torch.minimum(angle, p - angle)


class CenterLoss(nn.Module):
    def __init__(self, num_out, img_ppi=500, middle_shape=(640, 640), bin_type="arcsin", sigma=5.0) -> None:
        super().__init__()
        self.num_out = num_out
        self.bin_type = bin_type
        self.max_range = img_ppi / 500 * np.array(middle_shape) / 2
        self.x_sigma = sigma / self.max_range[1]
        self.y_sigma = sigma / self.max_range[0]

        if self.num_out[0] > 1:
            self.register_buffer("x_linear", custom_linspace(self.num_out[0], bin_type).view(1, -1, 1, 1))
        if self.num_out[1] > 1:
            self.register_buffer("y_linear", custom_linspace(self.num_out[1], bin_type).view(1, -1, 1, 1))
        self.loss_bce = SmoothFocalLoss()

        self.loss_mse = MSELoss()

    def forward(self, input, target, img_H=None, img_W=None, mask=None):
        losses = 0
        prob_x, prob_y = input
        if self.num_out[0] > 1 or self.num_out[1] > 1:
            grid_y, grid_x = torch.meshgrid(
                torch.linspace(-1, 1, prob_x.size(-2)), torch.linspace(-1, 1, prob_x.size(-1)), indexing="ij"
            )

        if self.num_out[0] > 1:
            grid_x = (grid_x.type_as(prob_x)[None, None] + 1) / 2 * (img_W - 1)
            delta_x = (target[:, 0].view(-1, 1, 1, 1) - grid_x) / self.max_range[1]

            x_tar = delta_x.clamp(-1, 1) - self.x_linear
            x_tar = torch.exp(-(x_tar ** 2) / (2 * self.x_sigma ** 2))
            losses += self.loss_bce(torch.sigmoid(prob_x), x_tar, mask=mask)
        else:
            if mask is not None:
                area = mask > 0.5
                if area.sum() > 0:
                    losses += self.loss_mse(prob_x[area], target[:, 0, None][area], is_angle=False)
            else:
                losses += self.loss_mse(prob_x, target[:, 0, None], is_angle=False)

        if self.num_out[1] > 1:
            grid_y = (grid_y.type_as(prob_y)[None, None] + 1) / 2 * (img_H - 1)
            delta_y = (target[:, 1].view(-1, 1, 1, 1) - grid_y) / self.max_range[0]

            y_tar = delta_y.clamp(-1, 1) - self.y_linear
            y_tar = torch.exp(-(y_tar ** 2) / (2 * self.y_sigma ** 2))
            losses += self.loss_bce(torch.sigmoid(prob_y), y_tar, mask=mask)

        else:
            if mask is not None:
                area = mask > 0.5
                if area.sum() > 0:
                    losses += self.loss_mse(prob_y[area], target[:, 1, None][area], is_angle=False)
            else:
                losses += self.loss_mse(prob_y, target[:, 1, None], is_angle=False)

        losses = losses / 2
        return losses


class GridLoss(nn.Module):
    def __init__(self, num_out, img_ppi=500, middle_shape=(640, 640), bin_type="arcsin", sigma=5.0) -> None:
        super().__init__()
        self.num_out = num_out
        self.bin_type = bin_type
        self.max_range = img_ppi / 500 * np.array(middle_shape) / 2
        self.x_sigma = sigma / self.max_range[1]
        self.y_sigma = sigma / self.max_range[0]

        if self.num_out[0] > 1:
            self.register_buffer("x_linear", custom_linspace(self.num_out[0], bin_type).view(1, -1, 1, 1))
        if self.num_out[1] > 1:
            self.register_buffer("y_linear", custom_linspace(self.num_out[1], bin_type).view(1, -1, 1, 1))
        self.loss_bce = SmoothFocalLoss()

        self.loss_mse = MSELoss()

    def forward(self, input, pose_center, pose_theta, img_H=None, img_W=None, mask=None):
        losses = 0
        prob_x, prob_y = input
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, prob_x.size(-2)), torch.linspace(-1, 1, prob_x.size(-1)), indexing="ij"
        )
        grid_x = (grid_x.type_as(prob_x)[None, None] + 1) / 2 * (img_W - 1)
        grid_y = (grid_y.type_as(prob_y)[None, None] + 1) / 2 * (img_H - 1)

        # transform
        cos_theta = torch.cos(torch.deg2rad(pose_theta)).view(-1, 1, 1, 1)
        sin_theta = torch.sin(torch.deg2rad(pose_theta)).view(-1, 1, 1, 1)
        trans_x = pose_center[:, 0].view(-1, 1, 1, 1)
        trans_y = pose_center[:, 1].view(-1, 1, 1, 1)
        t_grid_x = sin_theta * (trans_y - grid_y) + cos_theta * (trans_x - grid_x)
        t_grid_y = cos_theta * (trans_y - grid_y) - sin_theta * (trans_x - grid_x)

        if self.num_out[0] > 1:
            delta_x = t_grid_x / self.max_range[1]

            x_tar = delta_x.clamp(-1, 1) - self.x_linear
            x_tar = torch.exp(-(x_tar ** 2) / (2 * self.x_sigma ** 2))
            losses += self.loss_bce(torch.sigmoid(prob_x), x_tar, mask=mask)
        else:
            if mask is not None:
                area = mask > 0.5
                if area.sum() > 0:
                    losses += self.loss_mse(prob_x[area], t_grid_x[:, 0, area], is_angle=False)
            else:
                losses += self.loss_mse(prob_x, t_grid_x[:, 0, None], is_angle=False)

        if self.num_out[1] > 1:
            delta_y = t_grid_y / self.max_range[0]

            y_tar = delta_y.clamp(-1, 1) - self.y_linear
            y_tar = torch.exp(-(y_tar ** 2) / (2 * self.y_sigma ** 2))
            losses += self.loss_bce(torch.sigmoid(prob_y), y_tar, mask=mask)

        else:
            if mask is not None:
                area = mask > 0.5
                if area.sum() > 0:
                    losses += self.loss_mse(prob_y[area], t_grid_y[:, 1, None][area], is_angle=False)
            else:
                losses += self.loss_mse(prob_y, t_grid_y[:, 1, None], is_angle=False)

        losses = losses / 2
        return losses


class ThetaLoss(nn.Module):
    def __init__(self, num_out, ranges=(-180, 179)) -> None:
        super().__init__()
        self.num_out = num_out
        if self.num_out > 1:
            self.register_buffer("t_bin", torch.linspace(ranges[0], ranges[1], self.num_out).view(1, -1))
            self.loss_bce = nn.CrossEntropyLoss(label_smoothing=0.1)
        else:
            self.loss_mse = MSELoss()

    def forward(self, input, target, is_angle=True, deg=True):
        if self.num_out > 1:
            t_tar = torch.argmin((target - self.t_bin).abs(), dim=1).clamp(0, self.num_out - 1).view(-1, 1)
            input = input.view(input.size(0), input.size(1), -1)
            t_tar = t_tar.expand(-1, input.size(2))
            loss_theta = self.loss_bce(F.log_softmax(input, dim=1), t_tar.long())  # / np.log(self.num_out)
        else:
            loss_theta = self.loss_mse(input, target, is_angle=is_angle, deg=deg)

        return loss_theta


class SmoothFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, beta=4.0, eps=1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.gamma = gamma
        self.beta = beta

    def forward(self, logit, target, mask=None):
        logit = logit.clamp(self.eps, 1 - self.eps)
        target = target / torch.max(target, dim=1, keepdim=True)[0].clamp_min(self.eps)
        loss = -torch.where(
            (target - 1).abs() <= 1e-3,
            (1 - logit) ** self.gamma * torch.log(logit),
            (1 - target) ** self.beta * logit ** self.gamma * torch.log(1 - logit),
        )
        if mask is not None:
            loss = (loss * mask).sum() / mask.sum().clamp_min(self.eps)
        else:
            loss = loss.mean()
        return loss


def flatten(input):
    C = input.size(1)
    axis_order = (1, 0) + tuple(range(2, input.ndim))
    out = input.permute(axis_order)
    return out.contiguous().view(C, -1)


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, input, target):
        """input, target: [batch, channel, width, height(, depth)]

        Parameters:
            [None]
        Returns:
            [None]
        """
        input = flatten(input)
        target = flatten(target)
        inter = (input * target).sum(1)
        intra = input.sum(1) + target.sum(1)
        dice = 1 - (2 * inter + self.smooth) / (intra + self.smooth)
        return dice.mean()
