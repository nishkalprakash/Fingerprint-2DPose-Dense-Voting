"""
This file (trainer_i2m.py) is designed for:
    trainer for resnet (single frame based)
Copyright (c) 2021, Yongjie Duan. All rights reserved.
"""
import argparse
import logging
import os

import torch
import torch.nn as nn
from tqdm import tqdm

from models.dataloader import MyDataset
from models.losses import *
from models.model_zoo import *
from trainer import Trainer
from utils.logger import averageScalar
from utils.metrics import *
from utils.misc import load_model
from utils.visualize import draw_on_image


class MyTrainer(Trainer):
    def __init__(self, cfg_path, gpus) -> None:
        super().__init__(cfg_path, gpus)

        # model
        try:
            model = eval(self.exp_name.upper())(
                num_in=self.num_in,
                num_pose_2d=self.num_pose_2d,
                num_layers=self.num_layers,
                img_ppi=self.img_ppi,
                middle_shape=self.middle_shape,
                with_tv=self.with_tv,
                with_enh=self.with_enh,
                bin_type=self.bin_type,
                activate=self.activate,
                pretrained=self.pretrained,
            )
            if self.with_enh:
                load_model(model.preprocess_enh, "./logs/fingernet/model.pth", by_name=True)
            if "ori" in self.loss_attrs:
                ori_model = GridOri(self.seg_zoom, requires_grad=False)
                # ori_model = FingerNetOri(self.seg_zoom, requires_grad=False)
                # load_model(ori_model, "./logs/fingernet/model.pth", by_name=True)

            # old version, before 2022/05/22
            # if "gridnet" in self.exp_name:
            #     load_model(model, "./logs/fingernet/model.pth", by_name=True)
        except Exception as ex:
            raise ValueError(ex)
        self.make_model(model)
        if "ori" in self.loss_attrs:
            self.ori_model = nn.DataParallel(ori_model, device_ids=self.gpus)
            self.ori_model.to(self.main_dev)
            self.ori_model.eval()

        # dataset
        trainset = MyDataset(
            self.prefix,
            f"./datasets/train_{self.data_name}.pkl",
            img_ppi=self.img_ppi,
            ranges_2d=self.ranges_2d,
            middle_shape=self.middle_shape,
            with_bg=self.with_bg,
            seg_zoom=self.seg_zoom,
            do_aug=self.with_aug,
            repeat=self.repeat,
        )
        valset = MyDataset(
            self.prefix,
            f"./datasets/valid_{self.data_name}.pkl",
            img_ppi=self.img_ppi,
            ranges_2d=self.ranges_2d,
            middle_shape=self.middle_shape,
            with_bg=self.with_bg,
            seg_zoom=self.seg_zoom,
            seed=2022,
        )
        self.make_dataset(trainset, valset)

    def train(self, epoch):
        self.model.train()

        loss_scalar = averageScalar()
        loss_units_scalar = {k: averageScalar() for k in self.losses.keys()}
        metric_units_scalar = {k: averageScalar() for k in self.metrics.keys()}

        with tqdm(total=self.n_train, desc=f"Epo {epoch}/{self.num_epochs}") as pbar:
            for iterx, item in enumerate(self.trainloader):
                img = item["img"].to(self.main_dev)
                seg = item["seg"].to(self.main_dev)
                pose_2d = item["pose_2d"].to(self.main_dev)
                pose_3d = item["pose_3d"].to(self.main_dev)
                seg_flags = item["seg_flags"].to(self.main_dev)
                pose_flags = item["pose_flags"].to(self.main_dev)
                name = item["name"]

                ### wrapped in anomaly detecting section
                # with torch.autograd.detect_anomaly():
                output = self.model(img)

                B = img.size(0)
                loss = 0
                for k, v in self.losses.items():
                    if k == "offset":
                        mask = seg_flags.view(-1, 1, 1, 1) * seg + (1 - seg_flags.view(-1, 1, 1, 1)) * output["seg"].detach()
                        loss_k = v[0] * v[1](
                            output["center"], pose_2d[:, :2], img_H=img.size(-2), img_W=img.size(-1), mask=mask
                        )
                    elif k == "grid":
                        mask = seg_flags.view(-1, 1, 1, 1) * seg + (1 - seg_flags.view(-1, 1, 1, 1)) * output["seg"].detach()
                        loss_k = v[0] * v[1](
                            output["grid"], pose_2d[:, :2], pose_2d[:, 2:], img_H=img.size(-2), img_W=img.size(-1), mask=mask
                        )
                    elif k == "theta":
                        loss_k = v[0] * v[1](output["theta"], pose_2d[:, 2:], deg=True)
                    elif k == "pose":
                        if pose_flags.sum() > 0:
                            loss_k = v[0] * v[1](
                                [x[pose_flags > 0] for x in output["pose"]], pose_3d[pose_flags > 0], deg=True
                            )
                        else:
                            loss_k = 0
                    # elif k == "pml":
                    #     loss_k = v[0] * v[1](output["metric_pose"], pose)
                    elif k == "ori":
                        mask = seg_flags.view(-1, 1, 1, 1) * seg + (1 - seg_flags.view(-1, 1, 1, 1)) * output["seg"].detach()
                        sincos = self.ori_model(item["img_ori"].to(self.main_dev))
                        loss_k = v[0] * (
                            v[1](output["ori"][0], sincos[0], mask=mask) + v[1](output["ori"][1], sincos[1], mask=mask)
                        )
                    elif k == "seg":
                        if seg_flags.sum() > 0:
                            pred = output["seg"][seg_flags > 0]
                            target = seg[seg_flags > 0]
                            loss_k = v[0] * (v[1](pred, target) + v[1](1 - pred, 1 - target)) / 2
                        else:
                            loss_k = 0
                    elif k == "bce":
                        if seg_flags.sum() > 0:
                            pred = output["seg"][seg_flags > 0]
                            target = seg[seg_flags > 0]
                            loss_k = v[0] * (v[1](pred, target) + v[1](1 - pred, 1 - target)) / 2
                        else:
                            loss_k = 0
                    # elif k == "edge":
                    #     loss_k = v[0] * v[1](output["seg"], seg)
                    elif k == "mse":
                        loss_k = v[0] * (
                            v[1](output["pose_2d"][:, :2], pose_2d[:, :2], is_angle=False)
                            # + v[1](output["pose_2d"][:, 2:], pose_2d[:, 2:], is_angle=True, deg=True)
                            + v[1](output["pose_2d"][:, 2:], pose_2d[:, 2:], is_angle=False)  # after 2022/06/24
                        )
                    elif k == "conf":
                        conf_center = torch.exp(
                            -((output["pose_2d"][:, :2] - pose_2d[:, :2]) ** 2).sum(dim=1) / (2 * 20 ** 2)
                        ).detach()
                        conf_theta = torch.exp(
                            -normalize_angle(output["pose_2d"][:, 2] - pose_2d[:, 2]) ** 2 / (2 * 10 ** 2)
                        ).detach()
                        loss_k = v[0] * (v[1](output["conf"][0], conf_center) + v[1](output["conf"][1], conf_theta))
                    else:
                        raise ValueError(f"uncupported loss component {k}")

                    if torch.isnan(loss_k):
                        print(f"loss {k} has nan value")
                        raise ValueError()

                    loss = loss + loss_k
                    loss_units_scalar[k].update(loss_k.item(), B)
                loss_scalar.update(loss.item(), B)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                ### wrapped in anomaly detecting section

                for k, v in self.metrics.items():
                    if k == "center":
                        metric_k = v(output["pose_2d"][:, :2].detach().cpu().numpy(), pose_2d[:, :2].cpu().numpy())
                    elif k == "theta":
                        metric_k = v(output["pose_2d"][:, 2].detach().cpu().numpy(), pose_2d[:, 2].cpu().numpy())
                    elif k == "seg":
                        mask = (seg_flags > 0).cpu().numpy()
                        metric_k = v(output["seg"].detach().cpu().numpy()[mask], seg.cpu().numpy()[mask])
                    elif k == "pose":
                        mask = (pose_flags > 0).cpu().numpy()
                        metric_k = v(output["pose_3d"].detach().cpu().numpy()[mask], pose_3d.cpu().numpy()[mask])
                    else:
                        raise ValueError()
                    metric_units_scalar[k].update(metric_k.item(), B)

                pbar.update(B)
                pbar.set_postfix(
                    {
                        **{f"l_{k}": f"{v.avg:.2f}" for k, v in loss_units_scalar.items()},
                        **{f"m_{k}": f"{v.avg:.1f}" for k, v in metric_units_scalar.items()},
                    }
                )

                # logging for losses and intermedia visualizations
                if not self.debug and iterx % 50 == 0:
                    # scalar
                    for k, v in loss_units_scalar.items():
                        self.tb_writer.add_scalar(f"train_loss_{k}", v.val, epoch * len(self.trainloader) + iterx)
                    for k, v in metric_units_scalar.items():
                        self.tb_writer.add_scalar(f"train_metric_{k}", v.val, epoch * len(self.trainloader) + iterx)

                    self.tb_writer.add_figure(
                        "train_figure",
                        draw_on_image(
                            name,
                            [img, img, *output["img_sup"], *sincos]
                            if "ori" in self.loss_attrs
                            else [img, img, *output["img_sup"]],
                            [pose_2d, output["pose_2d"]],
                            [pose_3d, output["pose_3d"]] if "pose_3d" in output else [],
                            [seg, output["seg"], *output["seg_sup"]],
                            appendix=[conf_center, conf_theta, *output["conf"]] if "conf" in output else None,
                        ),
                        epoch * len(self.trainloader) + iterx,
                    )

            pbar.close()

        return loss_scalar.avg, {k: v.avg for k, v in metric_units_scalar.items()}

    def evaluate(self, epoch, context, dataloader):
        if context == "test":
            print("testing====")
            best_ckp = torch.load(os.path.join(self.ckp_dir, "best.pth.tar"), map_location=lambda storage, _: storage)
            self.model.module.load_state_dict(best_ckp["model"])

        self.model.eval()

        loss_scalar = averageScalar()
        loss_units_scalar = {k: averageScalar() for k in self.losses.keys()}
        metric_units_scalar = {k: averageScalar() for k in self.metrics.keys()}

        for iterx, item in enumerate(dataloader):
            img = item["img"].to(self.main_dev)
            seg = item["seg"].to(self.main_dev)
            pose_2d = item["pose_2d"].to(self.main_dev)
            pose_3d = item["pose_3d"].to(self.main_dev)
            seg_flags = item["seg_flags"].to(self.main_dev)
            pose_flags = item["pose_flags"].to(self.main_dev)
            name = item["name"]

            with torch.no_grad():
                output = self.model(img)

            B = img.size(0)
            loss = 0
            for k, v in self.losses.items():
                if k == "offset":
                    mask = seg_flags.view(-1, 1, 1, 1) * seg + (1 - seg_flags.view(-1, 1, 1, 1)) * output["seg"].detach()
                    loss_k = v[0] * v[1](output["center"], pose_2d[:, :2], img_H=img.size(-2), img_W=img.size(-1), mask=mask)
                elif k == "grid":
                    mask = seg_flags.view(-1, 1, 1, 1) * seg + (1 - seg_flags.view(-1, 1, 1, 1)) * output["seg"].detach()
                    loss_k = v[0] * v[1](
                        output["grid"], pose_2d[:, :2], pose_2d[:, 2:], img_H=img.size(-2), img_W=img.size(-1), mask=mask
                    )
                elif k == "theta":
                    loss_k = v[0] * v[1](output["theta"], pose_2d[:, 2:], deg=True)
                elif k == "pose":
                    if pose_flags.sum() > 0:
                        loss_k = v[0] * v[1]([x[pose_flags > 0] for x in output["pose"]], pose_3d[pose_flags > 0], deg=True)
                    else:
                        loss_k = 0
                # elif k == "pml":
                #     loss_k = v[0] * v[1](output["metric_pose"], pose)
                elif k == "ori":
                    mask = seg_flags.view(-1, 1, 1, 1) * seg + (1 - seg_flags.view(-1, 1, 1, 1)) * output["seg"].detach()
                    sincos = self.ori_model(item["img_ori"].to(self.main_dev))
                    loss_k = v[0] * (
                        v[1](output["ori"][0], sincos[0], mask=mask) + v[1](output["ori"][1], sincos[1], mask=mask)
                    )
                elif k == "seg":
                    if seg_flags.sum() > 0:
                        pred = output["seg"][seg_flags > 0]
                        target = seg[seg_flags > 0]
                        loss_k = v[0] * (v[1](pred, target) + v[1](1 - pred, 1 - target)) / 2
                    else:
                        loss_k = 0
                elif k == "bce":
                    if seg_flags.sum() > 0:
                        pred = output["seg"][seg_flags > 0]
                        target = seg[seg_flags > 0]
                        loss_k = v[0] * (v[1](pred, target) + v[1](1 - pred, 1 - target)) / 2
                    else:
                        loss_k = 0
                # elif k == "edge":
                #     loss_k = v[0] * v[1](output["seg"], seg)
                elif k == "mse":
                    loss_k = v[0] * (
                        v[1](output["pose_2d"][:, :2], pose_2d[:, :2], is_angle=False)
                        # + v[1](output["pose_2d"][:, 2:], pose_2d[:, 2:], is_angle=True, deg=True)
                        + v[1](output["pose_2d"][:, 2:], pose_2d[:, 2:], is_angle=False)  # after 2022/06/24
                    )
                elif k == "conf":
                    conf_center = torch.exp(
                        -((output["pose_2d"][:, :2] - pose_2d[:, :2]) ** 2).sum(dim=1) / (2 * 20 ** 2)
                    ).detach()
                    conf_theta = torch.exp(
                        -normalize_angle(output["pose_2d"][:, 2] - pose_2d[:, 2]) ** 2 / (2 * 10 ** 2)
                    ).detach()
                    loss_k = v[0] * (v[1](output["conf"][0], conf_center) + v[1](output["conf"][1], conf_theta))
                else:
                    raise ValueError(f"uncupported loss component {k}")

                loss = loss + loss_k
                loss_units_scalar[k].update(loss_k, B)
            loss_scalar.update(loss, B)

            for k, v in self.metrics.items():
                if k == "center":
                    metric_k = v(output["pose_2d"][:, :2].detach().cpu().numpy(), pose_2d[:, :2].cpu().numpy())
                elif k == "theta":
                    metric_k = v(output["pose_2d"][:, 2].detach().cpu().numpy(), pose_2d[:, 2].cpu().numpy())
                elif k == "seg":
                    mask = (seg_flags > 0).cpu().numpy()
                    metric_k = v(output["seg"].detach().cpu().numpy()[mask], seg.cpu().numpy()[mask])
                elif k == "pose":
                    mask = (pose_flags > 0).cpu().numpy()
                    metric_k = v(output["pose_3d"].detach().cpu().numpy()[mask], pose_3d.cpu().numpy()[mask])
                else:
                    raise ValueError()
                metric_units_scalar[k].update(metric_k.item(), B)

        if not self.debug:
            # scalar
            for k, v in loss_units_scalar.items():
                self.tb_writer.add_scalar(f"valid_loss_{k}", v.avg, epoch)
            for k, v in metric_units_scalar.items():
                self.tb_writer.add_scalar(f"valid_metric_{k}", v.avg, epoch)

            self.tb_writer.add_figure(
                "valid_figure",
                draw_on_image(
                    name,
                    [img, img, *output["img_sup"], *sincos] if "ori" in self.loss_attrs else [img, img, *output["img_sup"]],
                    [pose_2d, output["pose_2d"]],
                    [pose_3d, output["pose_3d"]] if "pose_3d" in output else [],
                    [seg, output["seg"], *output["seg_sup"]],
                    appendix=[conf_center, conf_theta, *output["conf"]] if "conf" in output else None,
                ),
                epoch,
            )

        return loss_scalar.avg, {k: v.avg for k, v in metric_units_scalar.items()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Trainer for resnet-like (direct predict)")
    parser.add_argument("--yaml", "-y", default="gridnet_ablation", type=str)
    parser.add_argument("--gpus", "-g", default=[5], type=int, nargs="+")
    args = parser.parse_args()

    yaml_path = f"./train_{args.yaml}.yaml"
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")
    logging.info(f"loading training profile from {yaml_path}")

    t = MyTrainer(yaml_path, args.gpus)
    t.run()
