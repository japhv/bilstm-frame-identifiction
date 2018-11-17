"""
NSynth classification using PyTorch

Authors: Japheth Adhavan, Jason St. George
"""

import torch
import utils
import models
import data
import os

import logging

import argparse
import numpy as np

import pytorch_nsynth

import matplotlib as mpl
if os.environ.get('DISPLAY', '') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')


logging.basicConfig(level=logging.INFO)


def train():
    pass


def test():
    pass


def validate():
    pass


def main(args):
    gpu_idx = utils.get_free_gpu()
    device = torch.device("cuda:{}".format(gpu_idx))
    logging.info("Using device cuda:{}".format(gpu_idx))

    train_loader = data.get_data_loader("train", batch_size=32, shuffle=True)

    print("Size of train loader: {}".format(len(train_loader)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='NSynth classifier')
    parser.add_argument('--test', action='store_true', default=False,
                        help='disables training, loads model')
    main(parser.parse_args())