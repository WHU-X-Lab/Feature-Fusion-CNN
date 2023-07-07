#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Configuration of Googlenet
"""

# configuration of alexnet
weight_path = "AlexNet1.pth"  # path to the weigths
alexnet_path = 'Alexnet5.pth'  # path to the net
N_FEATURES = 2

# params of training
MAX_EPOCH = 100
BATCH_SIZE = 48
LR = 0.00001
device = 'cuda:0'

# configuration of decission tree
MAX_DEPTH = 5
tree_path = 'dt_alex.pkl'

# data path
## the path of training data
train_dir = r"function_test_20"
valid_dir = r"function_test_20_val"
# train_dir = "origin_diff"
# valid_dir = "valid_diff"
## valid_dir=pathlib.Path("valid")
weight_decay_f = 0.00001
milestones = [50,70,90,110]
## the path of data for prediction
pred_dir = 'predict'