"""
This file (trainer.py) is designed for:
    base class of trainer
Copyright (c) 2022, Yongjie Duan. All rights reserved.
"""
import logging
import os
import os.path as osp
import random
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import yaml
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from fptools import uni_io
from models.losses import *
from models.model_zoo import *
from utils.logger import Logger
from utils.metrics import *
from utils.misc import save_model


class Trainer:
    def __init__(self, cfg_path, gpus) -> None:
        params = self.load_config_file(cfg_path)
        self.update_attrs(params)
        self.set_seed()
        params.update({"seed": self.seed, "time_string": self.time_string, "gpus": gpus})
        params["ckp_dir"] = osp.join(params["ckp_dir"], params["exp_name"], self.time_string)
        self.update_attrs(params)

        if self.debug:
            self.batch_size = 2

        print(yaml.dump(params, allow_unicode=True, default_flow_style=False))

        logging.info(f"Current ckechpoint: {self.ckp_dir}")
        uni_io.mkdir(self.ckp_dir)
        if not osp.exists(osp.join(self.ckp_dir, "configs.yaml")):
            with open(osp.join(self.ckp_dir, "configs.yaml"), "w") as fp:
                yaml.safe_dump(params, fp)

        # tensorboard logger if not debug
        self.tb_writer = SummaryWriter(self.ckp_dir) if not self.debug else None

        self.txt_logger = Logger(os.path.join(self.ckp_dir, "log.txt"), resume=self.resume_ckp is not None)
        metric_names = [f"{phase}{key.capitalize()}" for key in self.metric_keys for phase in ["Train", "Valid"]]
        self.txt_logger.set_names(["lr", "TrainLoss", "ValidLoss", *metric_names])

    def load_config_file(self, config_file):
        return yaml.safe_load(open(config_file, "r"))

    def set_seed(self):
        if "seed" not in self.__dict__:
            seed = hash(time.time()) % 10000
            self.seed = seed
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)

        if "time_string" not in self.__dict__:
            if self.resume_ckp is None:
                # time_string = np.random.choice(list(string.digits + string.ascii_letters), size=(10,))
                time_string = time.strftime("%Y%m%d%H%M%S", time.localtime(time.time()))
                self.time_string = "".join(time_string)
            else:
                self.time_string = osp.basename(self.resume_ckp)

    def update_attrs(self, kwargs):
        self.__dict__.update(kwargs)

    def make_dataset(self, trainset, valset):
        self.n_train = len(trainset)
        self.n_val = len(valset)
        logging.info(f"Len of train: {self.n_train}")
        logging.info(f"Len of val: {self.n_val}")

        workers = 8 if not self.debug else 0
        self.trainloader = DataLoader(
            dataset=trainset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=True,
        )
        self.valloader = DataLoader(
            dataset=valset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=True,
        )

    def make_model(self, model):
        self.model = model

        if self.pretrained:
            logging.info("Loading pretrained parameters")

        logging.info(
            "=> Total params: {:.2f}M".format(
                sum(p.numel() for p in filter(lambda p: p.requires_grad, self.model.parameters())) / 1.0e6
            )
        )

        training_parameters = (
            self.model.parameters()
            if not self.pretrained
            else [
                {"params": get_finetune_parameters(self.model), "lr": self.lr / 10},
                {"params": get_other_parameters(self.model), "lr": self.lr},
            ]
        )

        self.optimizer = optim.AdamW(training_parameters, lr=self.lr, weight_decay=0.01)
        self.lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="min", factor=0.1, patience=10)

        if self.resume_ckp is None:
            logging.info("Training from scratch")
            self.start_epoch = 0
        else:
            logging.info(f"Resuming existing trained model: {self.resume_ckp}")

            name_lst = glob(osp.join("./", self.resume_ckp, "e_*.pth.tar"))
            name_lst.sort()
            ckp = torch.load(name_lst[-1], map_location=lambda storage, loc: storage)
            self.model.load_state_dict(ckp["model"])

            self.start_epoch = ckp["epoch"] + 1
            try:
                self.optimizer.load_state_dict(ckp["optim"])
                self.lr_scheduler.load_state_dict(ckp["schedule"])
            except Exception as ex:
                print("Cannot load state dict of optimizer or learning rate scheduler", ex)

        self.model = nn.DataParallel(self.model, device_ids=self.gpus)
        cudnn.benchmark = True

        self.main_dev = torch.device(f"cuda:{self.gpus[0]}")
        self.model.to(self.main_dev)

        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.main_dev)

        self.apply_loss_metrics()

    def apply_loss_metrics(self):
        self.losses = {}
        for k, v in self.loss_attrs.items():
            if k == "offset":
                self.losses[k] = [
                    v[0],
                    CenterLoss(self.num_pose_2d[:2], self.img_ppi, self.middle_shape, self.bin_type, self.activate).to(
                        self.main_dev
                    ),
                ]
            elif k == "grid":
                self.losses[k] = [
                    v[0],
                    GridLoss(self.num_pose_2d[:2], self.img_ppi, self.middle_shape, self.bin_type, self.activate).to(
                        self.main_dev
                    ),
                ]
            elif k == "center":
                self.losses[k] = [v[0], MSELoss().to(self.main_dev)]
            elif k == "theta":
                self.losses[k] = [v[0], ThetaLoss(self.num_pose_2d[2], ranges=self.ranges_2d).to(self.main_dev)]
            elif k == "mse":
                self.losses[k] = [v[0], MSELoss().to(self.main_dev)]
            elif k == "seg":
                self.losses[k] = [v[0], DiceLoss().to(self.main_dev)]
            elif k == "bce":
                self.losses[k] = [v[0], ImgBCELoss(w_pos=10, w_neg=0.5).to(self.main_dev)]
            else:
                raise ValueError(f"Unsupport loss type: {k}")

        self.metrics = {}
        for k in self.metric_keys:
            if k == "center":
                self.metrics[k] = center_metric
            elif k == "theta":
                self.metrics[k] = angle_metric
            elif k == "pose":
                self.metrics[k] = angle_metric
            elif k == "seg":
                self.metrics[k] = iou_metric
            else:
                raise ValueError(f"Unsupport metric type: {k}")

    def run(self):
        start_epoch = self.start_epoch
        best_error = 100000

        for epoch in range(start_epoch, start_epoch + self.num_epochs):
            lr = self.optimizer.param_groups[-1]["lr"]
            logging.info(f"Current learning rate: {lr:.5f}")

            train_loss, tr_metrics = self.train(epoch)
            metric_str = " ".join([f"{k}: {v:.3f}" for k, v in tr_metrics.items()])
            logging.info(f"Train average loss: {train_loss:.3f}, average metric: {metric_str}")

            valid_loss, vl_metrics = self.evaluate(epoch, "validation", self.valloader)
            metric_str = " ".join([f"{k}: {v:.3f}" for k, v in vl_metrics.items()])
            logging.info(f"Validation average loss: {valid_loss:.3f}, average metric: {metric_str}")

            if epoch % self.save_intervals == 0:
                pth_name = f"e_{epoch:04d}.pth.tar"
                save_model(self.model, self.optimizer, self.lr_scheduler, epoch, os.path.join(self.ckp_dir, pth_name))
            if valid_loss < best_error:
                save_model(self.model, self.optimizer, self.lr_scheduler, epoch, os.path.join(self.ckp_dir, "best.pth.tar"))
                best_error = valid_loss
                logging.info(f"Save best checkpoint {epoch}")

            metric_values = [metric[key] for key in self.metric_keys for metric in [tr_metrics, vl_metrics]]
            self.txt_logger.append([lr, train_loss, valid_loss, *metric_values])

            self.lr_scheduler.step(valid_loss)
            if lr < 1e-2 * self.lr:
                break

    def train(self):
        pass

    def evaluate(self):
        pass


pretrained_layers = ["layer1", "layer2", "layer3", "layer4"]
yin_pretrained_layers = ["conv1", "conv2", "conv3", "conv4_1", "conv4_2", "conv4_3", "ori1", "ori2", "ori3"]
# yin_pretrained_layers = []

# if using resnet
def get_finetune_parameters(model):
    b = [
        (name, module)
        for name, module in model.named_children()
        if name in pretrained_layers or name in yin_pretrained_layers
    ]
    for (_, module) in b:
        for name, param in module.named_parameters(recurse=True):
            if param.requires_grad:
                yield param


def get_other_parameters(model):
    b = [
        (name, module)
        for name, module in model.named_children()
        if name not in pretrained_layers and name not in yin_pretrained_layers
    ]
    for (_, module) in b:
        for name, param in module.named_parameters(recurse=True):
            if param.requires_grad:
                yield param
