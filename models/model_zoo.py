"""
This file (model_zoo.py) is designed for:
    models
Copyright (c) 2021, Yongjie Duan. All rights reserved.
"""
from glob import glob

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import resnet
from .units import *

MAX_RANGE = (640, 640)


def dense_hough_voting4(
    center_prob,
    grid_prob,
    center_att,
    theta_att,
    img_H,
    img_W,
    img_ppi=500,
    middle_shape=(640, 640),
    bin_type="arcsin",
    activate="sigmoid",
):
    c_prob_x, c_prob_y = center_prob
    g_prob_x, g_prob_y = grid_prob
    prob_H, prob_W = c_prob_x.shape[2:]
    max_range = img_ppi / 500 * np.array(middle_shape) / 2

    if c_prob_x.size(1) > 1:
        if activate == "sigmoid":
            c_prob_x = torch.sigmoid(c_prob_x)
            c_prob_y = torch.sigmoid(c_prob_y)
            g_prob_x = torch.sigmoid(g_prob_x)
            g_prob_y = torch.sigmoid(g_prob_y)
            c_prob_x = c_prob_x / c_prob_x.sum(dim=1, keepdim=True)
            c_prob_y = c_prob_y / c_prob_y.sum(dim=1, keepdim=True)
            g_prob_x = g_prob_x / g_prob_x.sum(dim=1, keepdim=True)
            g_prob_y = g_prob_y / g_prob_y.sum(dim=1, keepdim=True)
        elif activate == "softmax":
            c_prob_x = torch.softmax(c_prob_x, dim=1)
            c_prob_y = torch.softmax(c_prob_y, dim=1)
            g_prob_x = torch.softmax(g_prob_x, dim=1)
            g_prob_y = torch.softmax(g_prob_y, dim=1)

        # actual vector
        x_vec = custom_linspace(c_prob_x.size(1), bin_type).view(1, -1, 1, 1).type_as(c_prob_x) * max_range[1]
        y_vec = custom_linspace(c_prob_y.size(1), bin_type).view(1, -1, 1, 1).type_as(c_prob_y) * max_range[0]

        # calculate theta
        c_exp_x = (c_prob_x * x_vec).sum(dim=1, keepdim=True)
        c_exp_y = (c_prob_y * y_vec).sum(dim=1, keepdim=True)
        g_exp_x = (g_prob_x * x_vec).sum(dim=1, keepdim=True)
        g_exp_y = (g_prob_y * y_vec).sum(dim=1, keepdim=True)
    else:
        c_exp_x = c_prob_x * max_range[1]
        c_exp_y = c_prob_y * max_range[0]
        g_exp_x = g_prob_x * max_range[1]
        g_exp_y = g_prob_y * max_range[0]
    sin_theta = g_exp_x * c_exp_y.detach() - g_exp_y * c_exp_x.detach()
    cos_theta = g_exp_x * c_exp_x.detach() + g_exp_y * c_exp_y.detach()
    # norm = torch.sqrt(g_exp_x ** 2 + g_exp_y ** 2).clamp_min(1e-6)
    # sin_theta = sin_theta / norm
    # cos_theta = cos_theta / norm
    out_theta = torch.rad2deg(torch.atan2((sin_theta * theta_att).mean((1, 2, 3)), (cos_theta * theta_att).mean((1, 2, 3))))

    # calculate center
    grid_y, grid_x = torch.meshgrid(torch.linspace(-1, 1, prob_H), torch.linspace(-1, 1, prob_W), indexing="ij")
    grid_x = (grid_x.type_as(c_prob_x) + 1) / 2 * (img_W - 1)
    grid_y = (grid_y.type_as(c_prob_y) + 1) / 2 * (img_H - 1)

    if c_prob_x.size(1) > 1:
        x_width = custom_linspace(c_prob_x.size(1), bin_type, delta=True).view(1, -1, 1, 1).type_as(c_prob_x)
        y_width = custom_linspace(c_prob_y.size(1), bin_type, delta=True).view(1, -1, 1, 1).type_as(c_prob_y)
        c_prob_x = c_prob_x / x_width.clamp_min(1e-6)
        c_prob_y = c_prob_y / y_width.clamp_min(1e-6)

        x_bin = x_vec + grid_x[None, None]
        y_bin = y_vec + grid_y[None, None]
        out_x = (x_bin * c_prob_x * center_att).sum((1, 2, 3)) / (c_prob_x * center_att).sum((1, 2, 3)).clamp_min(1e-6)
        out_y = (y_bin * c_prob_y * center_att).sum((1, 2, 3)) / (c_prob_y * center_att).sum((1, 2, 3)).clamp_min(1e-6)
    else:
        x_bin = c_exp_x + grid_x[None, None]
        y_bin = c_exp_y + grid_y[None, None]
        out_x = (x_bin * center_att).sum((1, 2, 3)) / center_att.sum((1, 2, 3)).clamp_min(1e-6)
        out_y = (y_bin * center_att).sum((1, 2, 3)) / center_att.sum((1, 2, 3)).clamp_min(1e-6)

    return (out_x[:, None], out_y[:, None]), out_theta[:, None], (g_exp_x, g_exp_y, c_exp_x, c_exp_y)


class FingerNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.img_norm = NormalizeModule(m0=0, var0=1)

        # feature extraction VGG
        self.conv1 = nn.Sequential(ConvBnPRelu(1, 64, 3), ConvBnPRelu(64, 64, 3), nn.MaxPool2d(2, 2))
        self.conv2 = nn.Sequential(ConvBnPRelu(64, 128, 3), ConvBnPRelu(128, 128, 3), nn.MaxPool2d(2, 2))
        self.conv3 = nn.Sequential(
            ConvBnPRelu(128, 256, 3), ConvBnPRelu(256, 256, 3), ConvBnPRelu(256, 256, 3), nn.MaxPool2d(2, 2)
        )

        # multi-scale ASPP
        self.conv4_1 = ConvBnPRelu(256, 256, 3, padding=1, dilation=1)
        self.ori1 = nn.Sequential(ConvBnPRelu(256, 128, 1, stride=1, padding=0), nn.Conv2d(128, 90, 1, stride=1, padding=0))
        self.seg1 = nn.Sequential(ConvBnPRelu(256, 128, 1, stride=1, padding=0), nn.Conv2d(128, 1, 1, stride=1, padding=0))

        self.conv4_2 = ConvBnPRelu(256, 256, 3, padding=4, dilation=4)
        self.ori2 = nn.Sequential(ConvBnPRelu(256, 128, 1, stride=1, padding=0), nn.Conv2d(128, 90, 1, stride=1, padding=0))
        self.seg2 = nn.Sequential(ConvBnPRelu(256, 128, 1, stride=1, padding=0), nn.Conv2d(128, 1, 1, stride=1, padding=0))

        self.conv4_3 = ConvBnPRelu(256, 256, 3, padding=8, dilation=8)
        self.ori3 = nn.Sequential(ConvBnPRelu(256, 128, 1, stride=1, padding=0), nn.Conv2d(128, 90, 1, stride=1, padding=0))
        self.seg3 = nn.Sequential(ConvBnPRelu(256, 128, 1, stride=1, padding=0), nn.Conv2d(128, 1, 1, stride=1, padding=0))

        # enhance part
        gabor_cos, gabor_sin = gabor_bank(enh_ksize=25, ori_stride=2, Lambda=8)

        self.enh_img_real = nn.Conv2d(gabor_cos.size(1), gabor_cos.size(0), kernel_size=(25, 25), padding=12)
        self.enh_img_real.weight = nn.Parameter(gabor_cos, requires_grad=True)
        self.enh_img_real.bias = nn.Parameter(torch.zeros(gabor_cos.size(0)), requires_grad=True)

        self.enh_img_imag = nn.Conv2d(gabor_sin.size(1), gabor_sin.size(0), kernel_size=(25, 25), padding=12)
        self.enh_img_imag.weight = nn.Parameter(gabor_sin, requires_grad=True)
        self.enh_img_imag.bias = nn.Parameter(torch.zeros(gabor_sin.size(0)), requires_grad=True)

        # mnt part
        self.mnt_conv1 = nn.Sequential(ConvBnPRelu(2, 64, 9, padding=4), nn.MaxPool2d(2, 2))
        self.mnt_conv2 = nn.Sequential(ConvBnPRelu(64, 128, 5, padding=2), nn.MaxPool2d(2, 2))
        self.mnt_conv3 = nn.Sequential(ConvBnPRelu(128, 256, 3, padding=1), nn.MaxPool2d(2, 2))
        self.mnt_o = nn.Sequential(ConvBnPRelu(256 + 90, 256, 1, padding=0), nn.Conv2d(256, 180, 1, padding=0))
        self.mnt_w = nn.Sequential(ConvBnPRelu(256, 256, 1, padding=0), nn.Conv2d(256, 8, 1, padding=0))
        self.mnt_h = nn.Sequential(ConvBnPRelu(256, 256, 1, padding=0), nn.Conv2d(256, 8, 1, padding=0))
        self.mnt_s = nn.Sequential(ConvBnPRelu(256, 256, 1, padding=0), nn.Conv2d(256, 1, 1, padding=0))

    def forward(self, input):
        img_norm = self.img_norm(input)

        # feature extraction VGG
        conv1 = self.conv1(img_norm)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)

        # multi-scale ASPP
        conv4_1 = self.conv4_1(conv3)
        ori1 = self.ori1(conv4_1)
        seg1 = self.seg1(conv4_1)

        conv4_2 = self.conv4_2(conv3)
        ori2 = self.ori2(conv4_2)
        seg2 = self.seg2(conv4_2)

        conv4_3 = self.conv4_3(conv3)
        ori3 = self.ori3(conv4_3)
        seg3 = self.seg3(conv4_3)

        ori_out = torch.sigmoid(ori1 + ori2 + ori3)
        seg_out = torch.sigmoid(seg1 + seg2 + seg3)

        # enhance part
        enh_real = self.enh_img_real(input)
        enh_imag = self.enh_img_imag(input)
        ori_peak = orientation_highest_peak(ori_out)
        ori_peak = select_max_orientation(ori_peak)
        ori_up = F.interpolate(ori_peak, scale_factor=8, mode="nearest")
        seg_round = F.softsign(seg_out)
        seg_up = F.interpolate(seg_round, scale_factor=8, mode="nearest")
        enh_real = (enh_real * ori_up).sum(1, keepdim=True)
        enh_imag = (enh_imag * ori_up).sum(1, keepdim=True)
        phase_img = torch.atan2(enh_imag, enh_real)
        phase_seg_img = torch.cat((phase_img, seg_up), dim=1)

        # mnt part
        mnt_conv1 = self.mnt_conv1(phase_seg_img)
        mnt_conv2 = self.mnt_conv2(mnt_conv1)
        mnt_conv3 = self.mnt_conv3(mnt_conv2)

        mnt_o = torch.sigmoid(self.mnt_o(torch.cat((mnt_conv3, ori_out), dim=1)))
        mnt_w = torch.sigmoid(self.mnt_w(mnt_conv3))
        mnt_h = torch.sigmoid(self.mnt_h(mnt_conv3))
        mnt_s = torch.sigmoid(self.mnt_s(mnt_conv3))

        return {
            "enh": enh_real,
            "ori": ori_out,
            "seg": seg_out,
            "mnt_o": mnt_o,
            "mnt_w": mnt_w,
            "mnt_h": mnt_h,
            "mnt_s": mnt_s,
        }


class EnhanceNet(nn.Module):
    def __init__(self, requires_grad=False) -> None:
        super().__init__()
        self.requires_grad = requires_grad

        self.img_norm = NormalizeModule(m0=0, var0=1)

        # feature extraction VGG
        self.conv1 = nn.Sequential(ConvBnPRelu(1, 64, 3), ConvBnPRelu(64, 64, 3), nn.MaxPool2d(2, 2))
        self.conv2 = nn.Sequential(ConvBnPRelu(64, 128, 3), ConvBnPRelu(128, 128, 3), nn.MaxPool2d(2, 2))
        self.conv3 = nn.Sequential(
            ConvBnPRelu(128, 256, 3), ConvBnPRelu(256, 256, 3), ConvBnPRelu(256, 256, 3), nn.MaxPool2d(2, 2)
        )

        # multi-scale ASPP
        self.conv4_1 = ConvBnPRelu(256, 256, 3, padding=1, dilation=1)
        self.ori1 = nn.Sequential(ConvBnPRelu(256, 128, 1, stride=1, padding=0), nn.Conv2d(128, 90, 1, stride=1, padding=0))
        self.seg1 = nn.Sequential(ConvBnPRelu(256, 128, 1, stride=1, padding=0), nn.Conv2d(128, 1, 1, stride=1, padding=0))

        self.conv4_2 = ConvBnPRelu(256, 256, 3, padding=4, dilation=4)
        self.ori2 = nn.Sequential(ConvBnPRelu(256, 128, 1, stride=1, padding=0), nn.Conv2d(128, 90, 1, stride=1, padding=0))
        self.seg2 = nn.Sequential(ConvBnPRelu(256, 128, 1, stride=1, padding=0), nn.Conv2d(128, 1, 1, stride=1, padding=0))

        self.conv4_3 = ConvBnPRelu(256, 256, 3, padding=8, dilation=8)
        self.ori3 = nn.Sequential(ConvBnPRelu(256, 128, 1, stride=1, padding=0), nn.Conv2d(128, 90, 1, stride=1, padding=0))
        self.seg3 = nn.Sequential(ConvBnPRelu(256, 128, 1, stride=1, padding=0), nn.Conv2d(128, 1, 1, stride=1, padding=0))

        # enhance part
        gabor_cos, gabor_sin = gabor_bank(enh_ksize=25, ori_stride=2, Lambda=8)

        self.enh_img_real = nn.Conv2d(gabor_cos.size(1), gabor_cos.size(0), kernel_size=(25, 25), padding=12)
        self.enh_img_real.weight = nn.Parameter(gabor_cos, requires_grad=True)
        self.enh_img_real.bias = nn.Parameter(torch.zeros(gabor_cos.size(0)), requires_grad=True)

        self.enh_img_imag = nn.Conv2d(gabor_sin.size(1), gabor_sin.size(0), kernel_size=(25, 25), padding=12)
        self.enh_img_imag.weight = nn.Parameter(gabor_sin, requires_grad=True)
        self.enh_img_imag.bias = nn.Parameter(torch.zeros(gabor_sin.size(0)), requires_grad=True)

        for param in self.parameters():
            param.requires_grad = self.requires_grad

    def forward(self, input):
        img_norm = self.img_norm(input)

        # feature extraction VGG
        conv1 = self.conv1(img_norm)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)

        # multi-scale ASPP
        conv4_1 = self.conv4_1(conv3)
        ori1 = self.ori1(conv4_1)
        seg1 = self.seg1(conv4_1)

        conv4_2 = self.conv4_2(conv3)
        ori2 = self.ori2(conv4_2)
        seg2 = self.seg2(conv4_2)

        conv4_3 = self.conv4_3(conv3)
        ori3 = self.ori3(conv4_3)
        seg3 = self.seg3(conv4_3)

        ori_out = torch.sigmoid(ori1 + ori2 + ori3)
        seg_out = torch.sigmoid(seg1 + seg2 + seg3)
        seg_out = F.interpolate(seg_out, scale_factor=8, mode="bilinear", align_corners=True)

        # enhance part
        enh_real = self.enh_img_real(input)
        enh_imag = self.enh_img_imag(input)
        ori_peak = orientation_highest_peak(ori_out)
        ori_peak = select_max_orientation(ori_peak)
        ori_up = F.interpolate(ori_peak, scale_factor=8, mode="nearest")
        enh_real = (enh_real * ori_up).sum(1, keepdim=True)
        enh_imag = (enh_imag * ori_up).sum(1, keepdim=True)
        phase_img = torch.atan2(enh_imag, enh_real)
        seg_soft = F.softsign(seg_out)

        if self.requires_grad:
            return {
                "real": enh_real,
                "imag": enh_imag,
                "seg": seg_out,
                "phase": phase_img,
                "soft": seg_soft,
            }
        else:
            return {
                "real": enh_real.detach(),
                "imag": enh_imag.detach(),
                "seg": seg_out.detach(),
                "phase": phase_img.detach(),
                "soft": seg_soft.detach(),
            }


class GRIDNET4(nn.Module):
    """FingerNet based votenet

    Parameters:
        [None]
    Returns:
        [None]
    """

    def __init__(
        self,
        num_in=1,
        num_pose_2d=[1, 1, 1],
        num_layers=[64, 128, 256, 512],
        img_ppi=500,
        middle_shape=[512, 512],
        with_tv=False,
        with_enh=False,
        bin_type="arcsin",
        activate="sigmoid",
        pretrained=False,
    ) -> None:
        super().__init__()
        self.num_center = num_pose_2d[:2]
        self.img_ppi = img_ppi
        self.with_tv = with_tv
        self.middle_shape = middle_shape
        self.with_enh = with_enh
        self.bin_type = bin_type
        self.activate = activate

        self.preprocess_enh = EnhanceNet(requires_grad=False) if self.with_enh else nn.Identity()
        self.preprocess_tv = FastCartoonTexture(sigma=2.5 * img_ppi / 500) if self.with_tv else nn.Identity()
        self.input_layer = nn.Sequential(
            NormalizeModule(m0=0, var0=1),
            FingerprintCompose(win_size=np.rint(8 * img_ppi / 500).astype(int)),
        )

        block = resnet.BasicBlock
        base_model = resnet._resnet("resnet18", block, [2, 2, 2, 2], pretrained, True, num_layers=num_layers, num_in=3)
        base_layers = list(base_model.children())
        self.layer0 = nn.Sequential(*base_layers[:3])  # size=(N, num_layers[0], x.H/2, x.W/2)
        self.layer1 = nn.Sequential(*base_layers[3:5])  # size=(N, num_layers[0], x.H/4, x.W/4)
        self.layer2 = base_layers[5]  # size=(N, num_layers[1], x.H/8, x.W/8)
        self.layer3 = base_layers[6]  # size=(N, num_layers[2], x.H/16, x.W/16)
        self.layer4 = base_layers[7]  # size=(N, num_layers[3], x.H/32, x.W/32)
        # center, grid, segmentation
        num_up = 3
        self.decoder = DecoderSkip2(num_layers[-1], num_layers=num_layers[-2 : -2 - num_up : -1], expansion=block.expansion)
        self.pixels_out = nn.Conv2d(num_layers[-1 - num_up] * block.expansion, sum(self.num_center) * 2 + 2, 1)

    def get_prediction(self, input):
        processed_enh = self.preprocess_enh(input)
        processed_enh = processed_enh["phase"] if self.with_enh else processed_enh
        processed_tv = self.preprocess_tv(input)

        processed = self.input_layer(processed_tv)

        # encoder
        layer0 = self.layer0(processed)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        decoder = self.decoder((layer4, layer3, layer2, layer1))
        pixels_out = torch.split(self.pixels_out(decoder), (1, *self.num_center, *self.num_center, 1), dim=1)
        # center, grid, and segmentation
        out_seg = torch.sigmoid(pixels_out[0])
        out_center = pixels_out[1:3]
        out_grid = pixels_out[3:5]
        out_att = torch.sigmoid(pixels_out[-1]) * out_seg.detach()

        att_c = out_seg
        att_d = out_att

        out_center_2d, out_theta_2d, out_exp = dense_hough_voting4(
            out_center,
            out_grid,
            att_c,
            att_d,
            input.size(-2),
            input.size(-1),
            self.img_ppi,
            self.middle_shape,
            bin_type=self.bin_type,
            activate=self.activate,
        )
        out_pose_2d = torch.cat([*out_center_2d, out_theta_2d], dim=1)

        return {"pose_2d": out_pose_2d, "seg": out_seg}

    def forward(self, input):
        processed_enh = self.preprocess_enh(input)
        processed_enh = processed_enh["phase"] if self.with_enh else processed_enh
        processed_tv = self.preprocess_tv(input)

        processed = self.input_layer(processed_tv)

        # encoder
        layer0 = self.layer0(processed)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        decoder = self.decoder((layer4, layer3, layer2, layer1))
        pixels_out = torch.split(self.pixels_out(decoder), (1, *self.num_center, *self.num_center, 1), dim=1)
        # center, grid, and segmentation
        out_seg = torch.sigmoid(pixels_out[0])
        out_center = pixels_out[1:3]
        out_grid = pixels_out[3:5]
        out_att = torch.sigmoid(pixels_out[-1]) * out_seg.detach()

        att_c = out_seg
        att_d = out_att

        out_center_2d, out_theta_2d, out_exp = dense_hough_voting4(
            out_center,
            out_grid,
            att_c,
            att_d,
            input.size(-2),
            input.size(-1),
            self.img_ppi,
            self.middle_shape,
            bin_type=self.bin_type,
            activate=self.activate,
        )
        out_pose_2d = torch.cat([*out_center_2d, out_theta_2d], dim=1)

        return {
            "center": out_center,
            "grid": out_grid,
            "pose_2d": out_pose_2d,
            "seg": out_seg,
            "img_sup": [processed_tv, *out_exp[:2]],
            "seg_sup": [out_att, *out_exp[2:4]],
        }
