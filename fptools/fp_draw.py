"""
This file (fp_draw.py) is designed for:
    functions for draw
Copyright (c) 2021, Yongjie Duan. All rights reserved.
"""
import os
import sys

# os.chdir(sys.path[0])
import os.path as osp
import numpy as np
from glob import glob
from scipy import ndimage as sndi
import matplotlib
import matplotlib.pylab as plt
import matplotlib.cm as cm
from matplotlib.patches import ConnectionPatch
import matplotlib.ticker as ticker

from .uni_io import mkdir


def draw_pose(ax, pose, length=100, color="blue"):
    x, y, theta = pose
    start = (x, y)
    end = (x - length * np.sin(theta * np.pi / 180.0), y - length * np.cos(theta * np.pi / 180.0))
    ax.plot(start[0], start[1], marker="o", color=color)
    ax.arrow(start[0], start[1], end[0] - start[0], end[1] - start[1], width=2, fc=color, ec=color)


def draw_img_with_pose(
    img, pose, save_path, scale=1, cmap="gray", vmin=None, vmax=None, mask=None, length=100, color="blue", text_label=None
):
    # plot
    fig = plt.figure()
    ax = fig.add_subplot(111)

    if scale != 1:
        img = sndi.zoom(img, scale, order=1, cval=img[0].max())
        pose[:2] = pose[:2] * scale

    ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
    draw_pose(ax, pose, length=length, color=color)

    if text_label is not None:
        plt.text(10, 10, text_label, size=10, color="g")

    # ax.set_xlim(0, img.shape[1])
    # ax.set_ylim(img.shape[0], 0)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def draw_orientation(ax, ori, mask=None, factor=8, stride=32, color="lime", linewidth=1.5):
    """draw orientation filed

    Parameters:
        [None]
    Returns:
        [None]
    """
    ori = ori * np.pi / 180
    for ii in range(stride // factor // 2, ori.shape[0], stride // factor):
        for jj in range(stride // factor // 2, ori.shape[1], stride // factor):
            if mask is not None and mask[ii, jj] == 0:
                continue
            x, y, o, r = jj, ii, ori[ii, jj], stride * 0.8
            ax.plot(
                [x * factor - 0.5 * r * np.cos(o), x * factor + 0.5 * r * np.cos(o)],
                [y * factor - 0.5 * r * np.sin(o), y * factor + 0.5 * r * np.sin(o)],
                "-",
                color=color,
                linewidth=linewidth,
            )


def draw_img_with_orientation(
    img, ori, save_path, factor=8, stride=16, cmap="gray", vmin=None, vmax=None, mask=None, color="lime", dpi=100
):
    # plot
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
    draw_orientation(ax, ori, mask=mask, factor=factor, stride=stride, color=color, linewidth=dpi / 50)

    ax.set_xlim(0, img.shape[1])
    ax.set_ylim(img.shape[0], 0)
    ax.set_axis_off()
    fig.tight_layout()
    fig.set_size_inches(img.shape[1] * 1.0 / dpi, img.shape[0] * 1.0 / dpi)
    fig.savefig(save_path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)


def draw_minutiae(ax, mnt_lst, marker="o", size=12, arrow_length=20, color="red", linewidth=0.6, x_offset=0, y_offset=0):
    for mnt in mnt_lst:
        try:
            x, y, ori = mnt[:3]
            x += x_offset
            y += y_offset
            dx, dy = arrow_length * np.cos(ori * np.pi / 180), arrow_length * np.sin(ori * np.pi / 180)
            ax.scatter(x, y, marker=marker, s=size, facecolors="none", edgecolor=color, linewidths=linewidth)
            ax.plot([x, x + dx], [y, y + dy], "-", color=color, linewidth=linewidth)
        except:
            x, y = mnt[:2]
            ax.scatter(x, y, marker=marker, s=size, facecolors="none", edgecolor=color, linewidths=linewidth)


def draw_minutia_on_finger(
    img, mnt_lst, save_path, cmap="gray", marker="o", vmin=None, vmax=None, color="red", linewidth=1.2
):
    # plot
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
    draw_minutiae(
        ax,
        mnt_lst,
        marker=marker,
        size=np.rint(0.09 * min(img.shape[0], img.shape[1])),
        arrow_length=np.rint(0.03 * min(img.shape[0], img.shape[1])),
        color=color,
        linewidth=linewidth,
    )

    ax.set_xlim(0, img.shape[1])
    ax.set_ylim(img.shape[0], 0)
    ax.set_axis_off()
    fig.tight_layout()
    mkdir(osp.dirname(save_path))
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def draw_minutiae_pair(
    ax,
    img1,
    img2,
    mnts1,
    mnts2,
    pair_sim=None,
    cmin=None,
    cmax=None,
    cmap="gray",
    vmin=None,
    vmax=None,
    markercolor="brown",
    linecolor="royalblue",
    linewidth=0.6,
):
    img_shape1 = np.array(img1.shape[:2])
    img_shape2 = np.array(img2.shape[:2])
    img_height = max(img_shape1[0], img_shape2[0])
    img1 = np.pad(img1, ((0, img_height - img_shape1[0])), mode="edge")
    img2 = np.pad(img2, ((0, img_height - img_shape2[0])), mode="edge")
    img = np.concatenate((img1, img2), axis=1)
    ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)

    arrow_length = np.rint(0.02 * min(img.shape))
    draw_minutiae(ax, mnts1, linewidth=0.6, arrow_length=arrow_length, color=markercolor)
    draw_minutiae(ax, mnts2, linewidth=0.6, arrow_length=arrow_length, color=markercolor, x_offset=img1.shape[1])

    if pair_sim is not None:
        cmin = cmin if cmin is not None else min(pair_sim)
        cmax = cmax if cmax is not None else max(pair_sim)
        norm = matplotlib.colors.Normalize(vmin=cmin, vmax=cmax, clip=True)
        mapper = cm.ScalarMappable(norm=norm, cmap=cm.get_cmap("plasma"))
        cb = plt.colorbar(mapper, ax=ax, fraction=0.025, aspect=25, orientation="vertical")
        cb.ax.tick_params(length=0)
        cb.ax.tick_params(labelsize=8)
        cb.set_ticks([np.round(cmin, 2), np.round(cmax, 2)])
        cb.update_ticks()

    for ii in range(len(mnts1)):
        if pair_sim is not None:
            lc = mapper.to_rgba(pair_sim[ii])
        else:
            lc = linecolor
        ax.plot(
            [mnts1[ii, 0], mnts2[ii, 0] + img1.shape[1]],
            [mnts1[ii, 1], mnts2[ii, 1]],
            "-",
            color=lc,
            linewidth=linewidth,
        )
    # ax.set_xlim(0, img.shape[1])
    # ax.set_ylim(img.shape[0], 0)
    ax.set_axis_off()


def draw_minutiae_pair_on_finger(
    img1,
    img2,
    mnts1,
    mnts2,
    save_path,
    pair_sim=None,
    cmin=None,
    cmax=None,
    cmap="gray",
    vmin=None,
    vmax=None,
    markercolor="brown",
    linecolor="steelblue",
    linewidth=0.6,
    text_label=None,
    dpi=180,
):
    # plot
    fig = plt.figure(dpi=dpi)
    ax = fig.add_subplot(111)
    draw_minutiae_pair(
        ax,
        img1,
        img2,
        mnts1,
        mnts2,
        pair_sim,
        cmin,
        cmax,
        cmap,
        vmin,
        vmax,
        markercolor=markercolor,
        linecolor=linecolor,
        linewidth=linewidth,
    )
    if text_label is not None:
        plt.text(10, 10, text_label, size=10, color="g")
    fig.tight_layout()

    if not osp.isdir(osp.dirname(save_path)):
        os.makedirs(osp.dirname(save_path))
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def draw_minutiae_pair_only(
    ax, mnts1, mnts2, cmap="gray", vmin=None, vmax=None, markercolor="red", linecolor="green", linewidth=1.5
):
    for ii in range(len(mnts1)):
        ax.scatter(
            mnts1[ii, 0],
            mnts1[ii, 1],
            marker="s",
            s=5,
            facecolors="none",
            edgecolor=markercolor,
            linewidths=linewidth,
        )
        ax.scatter(
            mnts2[ii, 0] + 1000,
            mnts2[ii, 1],
            marker="s",
            s=5,
            facecolors="none",
            edgecolor=markercolor,
            linewidths=linewidth,
        )
        ax.plot(
            [mnts1[ii, 0], mnts2[ii, 0] + 1000],
            [mnts1[ii, 1], mnts2[ii, 1]],
            "-",
            color=linecolor,
            markersize=3,
            markerfacecolor="none",
        )
    # ax.set_xlim(0, img.shape[1])
    # ax.set_ylim(img.shape[0], 0)
    ax.set_axis_on()


def draw_minutiae_pair_only_on_finger(
    mnts1,
    mnts2,
    save_path,
    cmap="gray",
    vmin=None,
    vmax=None,
    markercolor="red",
    linecolor="green",
    linewidth=1.5,
    text_label=None,
):
    # plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    draw_minutiae_pair_only(
        ax,
        mnts1,
        mnts2,
        cmap,
        vmin,
        vmax,
        markercolor=markercolor,
        linecolor=linecolor,
        linewidth=linewidth,
    )
    if text_label is not None:
        plt.text(10, 10, text_label, size=10, color="g")
    fig.tight_layout()

    if not osp.isdir(osp.dirname(save_path)):
        os.makedirs(osp.dirname(save_path))
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    prefix = ""
