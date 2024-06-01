"""
This file (logger.py) is designed for:
    logger
Copyright (c) 2021, Yongjie Duan. All rights reserved.
"""
# A simple torch style logger
# (C) Wei YANG 2017
from __future__ import absolute_import
import matplotlib.pyplot as plt
import os
import sys
import numpy as np
import time

__all__ = ["averageScalar", "averageArray", "Logger", "LoggerMonitor", "savefig"]


class Profile(object):
    def __enter__(self):
        self.begin = time.time()
        return self

    def __exit__(self, type, value, traceback):
        self.period = time.time() - self.begin

    def __call__(self):
        return self.period


class averageArray:
    def __init__(self, dim=0, MIN=1000, MAX=-1000):
        self.dim = dim
        self.sum = np.zeros(dim)
        self.avg = np.zeros(dim)
        self.min = np.full(shape=(self.dim), fill_value=MIN)
        self.max = np.full(shape=(self.dim), fill_value=MAX)
        self.count = 0
        self.meta = []
        self.metaarray = np.empty((1, dim))

    def update(self, batch_data, need_abs=True):
        """value is a batch of data"""
        assert isinstance(batch_data, np.ndarray) and batch_data.shape[1] == self.dim
        self.meta.append(batch_data)
        self.metaarray = np.concatenate(self.meta, axis=0)
        batches = batch_data.shape[0]
        if need_abs:
            self.sum += np.abs(batch_data).sum(axis=0)
        else:
            self.sum += batch_data.sum(axis=0)
        self.count += batches
        self.avg = self.sum * 1.0 / self.count
        self.val = batch_data
        self.min = np.where(batch_data.min(axis=0) < self.min, batch_data.min(axis=0), self.min)
        self.max = np.where(batch_data.max(axis=0) > self.max, batch_data.max(axis=0), self.max)
        self.std = np.std(self.metaarray, axis=0)


class averageScalar:
    def __init__(self):
        self.count = 0
        self.sum = 0
        self.avg = 0

    def update(self, value, bb):
        self.count += bb
        self.sum += value * bb
        self.avg = self.sum / self.count
        self.val = value


def savefig(fname, dpi=None):
    dpi = 150 if dpi == None else dpi
    plt.savefig(fname, dpi=dpi)
    plt.close()


def plot_overlap(logger, names=None):
    names = logger.names if names == None else names
    numbers = logger.numbers
    for _, name in enumerate(names):
        x = np.arange(len(numbers[name]))
        plt.plot(x, np.asarray(numbers[name]))
    return [logger.title + "(" + name + ")" for name in names]


class Logger(object):
    """Save training process to log file with simple plot function."""

    def __init__(self, fpath, title=None, resume=False):
        self.file = None
        self.resume = resume
        self.title = "" if title == None else title
        if fpath is not None:
            if resume:
                self.file = open(fpath, "r")
                name = self.file.readline()
                self.names = name.rstrip().split("\t")
                self.numbers = {}
                for _, name in enumerate(self.names):
                    self.numbers[name] = []

                for numbers in self.file:
                    numbers = numbers.rstrip().split("\t")
                    for i in range(0, len(numbers)):
                        self.numbers[self.names[i]].append(numbers[i])
                self.file.close()
                self.file = open(fpath, "a")
            else:
                self.file = open(fpath, "w")

    def set_names(self, names):
        if self.resume:
            pass
        # initialize numbers as empty list
        self.numbers = {}
        self.names = names
        for _, name in enumerate(self.names):
            self.file.write(name)
            self.file.write("\t")
            self.numbers[name] = []
        self.file.write("\n")
        self.file.flush()

    def append(self, numbers):
        assert len(self.names) == len(numbers), "Numbers do not match names"
        for index, num in enumerate(numbers):
            if isinstance(num, str):
                self.file.write(num)
            else:
                self.file.write("{0:.6f}".format(num))
            self.file.write("\t")
            self.numbers[self.names[index]].append(num)
        self.file.write("\n")
        self.file.flush()

    def plot(self, names=None):
        plt.figure()
        names = self.names if names == None else names
        numbers = self.numbers
        for _, name in enumerate(names):
            x = np.arange(len(numbers[name]))
            plt.plot(x, np.asarray(numbers[name]))
        plt.legend([self.title + "(" + name + ")" for name in names])
        plt.grid(True)

    def close(self):
        if self.file is not None:
            self.file.close()


class LoggerMonitor(object):
    """Load and visualize multiple logs."""

    def __init__(self, paths):
        """paths is a distionary with {name:filepath} pair"""
        self.loggers = []
        for title, path in paths.items():
            logger = Logger(path, title=title, resume=True)
            self.loggers.append(logger)

    def plot(self, names=None):
        plt.figure()
        plt.subplot(121)
        legend_text = []
        for logger in self.loggers:
            legend_text += plot_overlap(logger, names)
        plt.legend(legend_text, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
        plt.grid(True)


if __name__ == "__main__":
    # # Example
    # logger = Logger('test.txt')
    # logger.set_names(['Train loss', 'Valid loss','Test loss'])

    # length = 100
    # t = np.arange(length)
    # train_loss = np.exp(-t / 10.0) + np.random.rand(length) * 0.1
    # valid_loss = np.exp(-t / 10.0) + np.random.rand(length) * 0.1
    # test_loss = np.exp(-t / 10.0) + np.random.rand(length) * 0.1

    # for i in range(0, length):
    #     logger.append([train_loss[i], valid_loss[i], test_loss[i]])
    # logger.plot()

    # Example: logger monitor
    paths = {
        "resadvnet20": "/home/wyang/code/pytorch-classification/checkpoint/cifar10/resadvnet20/log.txt",
        "resadvnet32": "/home/wyang/code/pytorch-classification/checkpoint/cifar10/resadvnet32/log.txt",
        "resadvnet44": "/home/wyang/code/pytorch-classification/checkpoint/cifar10/resadvnet44/log.txt",
    }

    field = ["Valid Acc."]

    monitor = LoggerMonitor(paths)
    monitor.plot(names=field)
    savefig("test.eps")
