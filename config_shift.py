# -- coding:utf-8
"""Configuration of Shiftnet
"""

# configuration of shifted
weight_path = r'save_model/shift\best_model.pth\\'  # path to the weigths
alexnet_path = r'save_model/shift\best_model.pth\\'  # path to the net
N_FEATURES = 2

# params of training
MAX_EPOCH = 20
BATCH_SIZE = 50

LR = 0.00001
device = 'cuda:0'

# configuration of decission tree
MAX_DEPTH = 5
tree_path = 'dt.pkl'

# data path
# the path of training data
# train_dir = "train_stn"
train_dir = "origin_shift"
valid_dir = "valid_shift"
# valid_dir=pathlib.Path("valid")

# SGD参数
weight_decay = 0.00005
# milestones = [7,28,70,150]
gamma = 0.1
weight_decay_f = 0.00001 # l2正则化参数
milestones = [50,70,90,110]
## the path of data for prediction
pred_dir = 'valid_shift'
