# -- coding:utf-8
"""Configuration of Diffnet
"""

# configuration of different
weight_path = r'save_model/different\best_model.pth\\'  # path to the weigths
alexnet_path = r'save_model/different\best_model.pth\\'  # path to the net
N_FEATURES = 2

# params of training
MAX_EPOCH = 150
BATCH_SIZE = 16

LR = 0.00001
device = 'cuda:0'

# configuration of decission tree
MAX_DEPTH = 5
tree_path = 'dt.pkl'

# data path
## the path of training data
train_dir = "origin_diff"
valid_dir = "valid_diff"
# train_dir = "Rect"
# valid_dir = "valid_Rect"
## valid_dir=pathlib.Path("valid")

# SGD参数
weight_decay = 0.00005
# milestones = [30,50,70]
gamma = 0.1
weight_decay_f = 0.00001
milestones = [50,70,90,110]
## the path of data for prediction
pred_dir = 'valid_diff'
