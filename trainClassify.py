# -- coding:utf-8
# Feature Fusion
import os
import random
import string
from collections import defaultdict

import torchmetrics
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.utils import compute_class_weight
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader, TensorDataset, WeightedRandomSampler
import torch
from torchvision import transforms
from PIL import Image
from ShiftNet import ShiftNet
from DifferenceNet import DifferenceNet
from GoogleNet import GoogLeNet
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

# 固定随机数种子
seed = 123
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

N_FEATURES = 2
device = 'cuda:0'
# 定义图像转换
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    # train_shift
    # transforms.Normalize(mean=[0.89731014, 0.9391688, 0.9396487], std=[0.18013711, 0.09479259, 0.12887895])
    # origin_diff
    # transforms.Normalize(mean=[0.9251727, 0.95890087, 0.9619809], std=[0.14847293, 0.07731944, 0.101800375])
])


class DiffImgDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.imgs = os.listdir(img_dir)
        self.imgs_dict = {}
        for img in self.imgs:
            id_num = int(img.split('groupid')[1].split('diff')[0])
            label_num = int(img.split('diff')[1].split('x')[0])
            if id_num not in self.imgs_dict:
                self.imgs_dict[id_num] = []
            self.imgs_dict[id_num].append((img, label_num))

    def __len__(self):
        return len(self.imgs_dict)

    def __getitem__(self, idx):
        id_num = list(self.imgs_dict.keys())[idx]
        imgs_list = self.imgs_dict[id_num]
        if len(imgs_list) > 4:
            imgs_list = sorted(imgs_list, key=lambda x: x[1], reverse=True)[:4]
        elif len(imgs_list) < 4:
            while len(imgs_list) < 4:
                # imgs_list.append(("null", 0))  # 方法一：补0
                imgs_list.append(random.choice(imgs_list)) # 方法二：随机复制
        imgs = []
        for img_name, label in imgs_list:
            # 补0
            # if img_name is not "null":
            #     img_path = os.path.join(self.img_dir, img_name)
            #     img = Image.open(img_path).convert('RGB')
            #     if self.transform:
            #         img = self.transform(img)
            # elif img_name is "null":
            #     img = torch.zeros([3, 224, 224])
            # 随机复制
            img_path = os.path.join(self.img_dir, img_name)
            img = Image.open(img_path).convert('RGB')
            if self.transform:
                img = self.transform(img)

            imgs.append(img)
        return torch.stack(imgs), id_num


shiftimg_dir = 'train_shift'


class ShiftImgDataset(Dataset):
    def __init__(self, data_path=shiftimg_dir, transform=None):
        self.data_path = data_path
        self.transform = transform
        self.imgs = os.listdir(data_path)
        self.imgs_dict = {}
        for img in self.imgs:
            id_num = int(img.split('groupid')[1].split('ex')[0])
            label_num = int(img.split('ex')[1].split('.')[0])
            if id_num not in self.imgs_dict:
                self.imgs_dict[id_num] = []
            self.imgs_dict[id_num].append((img, label_num))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        id = list(self.imgs_dict.keys())[index]
        imgs_list = self.imgs_dict[id]
        for img_name, label in imgs_list:
            img_path = os.path.join(self.data_path, img_name)
            img = Image.open(img_path).convert('RGB')
            if self.transform:
                img = self.transform(img)
        return img, id, label


def extract_features(model, dataloader):
    features_dict = {}
    model.eval()
    with torch.no_grad():
        for data in dataloader:
            imgs, id_nums = data
            batch_size = imgs.size(0)
            imgs = imgs.view(-1, 3, 224, 224)
            features = model(imgs)
            features = features.view(batch_size, -1)
            for i in range(batch_size):
                features_dict[id_nums[i].item()] = (features[i].cpu().numpy())  # , labels[i].item())
    return features_dict


# shiftimg_dir = 'train_shift'
diffimg_dir = 'train_diff'
difffeatures_dir = 'diff_features' # （1）随机复制
# difffeatures_dir = 'diff_features0'  # （2）缺失维度补0
shiftfeatures_dir = 'shift_features'
if not os.path.exists(difffeatures_dir):
    os.makedirs(difffeatures_dir)

diffdataset = DiffImgDataset(img_dir=diffimg_dir, transform=transform)
diffimgloader = DataLoader(dataset=diffdataset, batch_size=64)
shiftdataset = ShiftImgDataset(transform=transform)
shiftimgloader = DataLoader(dataset=shiftdataset, batch_size=64)

# 加载预训练模型
# 载入模型参数
# DiffNet = DifferenceNet(num_classes=2, init_weights=True)
DiffNet = GoogLeNet(num_classes=2)
DiffNet.load_state_dict(torch.load(r'save_model\diff_Goo\last_model.pth'))
# DiffNet.cuda()
ShiftNet = ShiftNet(num_classes=2, init_weights=True)
ShiftNet.load_state_dict(torch.load(r'save_model\shift\last_model.pth'))

# features_diff = extract_features(model=DiffNet, dataloader=diffimgloader)
# for id_num in tqdm(features_diff):
#     feature_path = os.path.join(difffeatures_dir, f'id{id_num}.npy')
#     np.save(feature_path, features_diff[id_num])

# features_shift = extract_features(model=ShiftNet, dataloader=shiftimgloader)
# for id_num in tqdm(features_shift):
#     feature_path = os.path.join(shiftfeatures_dir, f'id{id_num}ex{features_shift[id_num][1]}.npy')
#     np.save(feature_path, features_shift[id_num][0])


# 遍历文件夹中的所有文件
fusion_features_dict = {}
# 准备标签值
labels_dict = {}
for file_name in tqdm(os.listdir(shiftfeatures_dir)):
    # 检查文件是否为 .npy 文件
    if file_name.endswith('.npy'):
        # 提取 id
        id = file_name.split('id')[1].split('ex')[0]
        label = int(file_name.split('ex')[1].split('.')[0])
        if label == 0:
            label = 0
        elif label != 0:
            label = 1
        # 读取特征
        diff_filename = f'id{id}.npy'
        shift_feature = np.load(os.path.join(shiftfeatures_dir, file_name))
        if diff_filename in os.listdir(difffeatures_dir):
            diff_feature = np.load(os.path.join(difffeatures_dir, diff_filename))
            # 融合特征
            fusion_features_dict[id] = np.concatenate((shift_feature, diff_feature),
                                                      axis=0)  # shape: (num_samples, 1024+4608)
            labels_dict[id] = label  # shape: (num_samples,)

        # if id in features_shift:
        #     fusion_features_dict[id] = (
        #     np.concatenate((features_shift[id][0], difffeature), axis=0), features_shift[id][1])
        # else:
        #     fusion_features_dict[id] = (difffeature, features_shift[id][1])

# 准备 SVM 输入数据
X = []
y = []
for id in tqdm(fusion_features_dict):
    # print("X", fusion_features_dict[id])
    # print("y", labels_dict[id])
    X.append(fusion_features_dict[id])
    y.append(labels_dict[id])  # 根据您的数据设置标签
#
print('GoogLeNet+Diff/AlexNet+Shift: (1)随机复制✔（2）补0')
# # 分割数据
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
# # 计算类别权重
# class_weight = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
# cw = dict(enumerate(class_weight))
# # writer = SummaryWriter()
# # 训练 SVM 模型并输出结果
# clf = svm.SVC(class_weight=cw)
# clf.fit(X_train, y_train)
# y_train_pred = clf.predict(X_train)
# y_test_pred = clf.predict(X_test)
#
# # 计算分类评价指标
# target_names = ['class 0', 'class 1']
# print(classification_report(y_train, y_train_pred, target_names=target_names,digits=4))
# print(classification_report(y_test, y_test_pred, target_names=target_names,digits=4))

# NN
# 按照一定比例分割训练集和验证集
# 创建数据集
# dataset = TensorDataset(torch.tensor(X), torch.tensor(y))

# 按照一定比例分割训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
y_train_array = np.stack(y_train)
X_train_array = np.stack(X_train)
X_val_array = np.stack(X_val)
y_val_array = np.stack(y_val)
# train_size = int(0.8 * len(dataset))
# val_size = len(dataset) - train_size
# train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
# 获取训练集和验证集的特征和标签
class_sample_counts = [3418, 422]
weights = 1. / torch.tensor(class_sample_counts, dtype=torch.float)
samples_weights = weights[y_train]
sampler = WeightedRandomSampler(weights=samples_weights, num_samples=len(samples_weights), replacement=True)
train_dataset = TensorDataset(torch.tensor(X_train_array), torch.tensor(y_train_array))
val_dataset = TensorDataset(torch.tensor(X_val_array), torch.tensor(y_val_array))
# 创建 DataLoader
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
val_loader = DataLoader(val_dataset, batch_size=batch_size)


# 定义模型
class NNet(nn.Module):
    def __init__(self):
        super(NNet, self).__init__()
        self.fc1 = nn.Linear(8704, 544)
        # self.fc2 = nn.Linear(4352,544)
        self.fc2 = nn.Linear(544, 2)
        self.dropout = nn.Dropout(p=0.4)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        # x = self.dropout(x)
        # x = torch.relu(self.fc2(x))
        x = self.fc2(x)
        return x


model = NNet().to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())
# optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9, weight_decay=0.00001)
# lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50,70,90,110], gamma=0.1)
s = f"NNet,{difffeatures_dir},{shiftfeatures_dir},batch{batch_size}"
writer = SummaryWriter(comment=s)
# 训练模型
num_epochs = 150
for epoch in range(num_epochs):
    print(f"epoch{epoch + 1}\n-------------------")
    # 训练阶段
    test_recall = torchmetrics.Recall(average='none', num_classes=N_FEATURES).to(device)
    test_precision = torchmetrics.Precision(average='none', num_classes=N_FEATURES).to(device)
    test_F1 = torchmetrics.F1Score(num_classes=N_FEATURES, average='none').to(device)
    model.train()
    loss, current, n = 0, 0, 0
    for i, (features, labels) in enumerate(train_loader):
        # 前向传播
        features, labels = features.to(device), labels.long().to(device)
        outputs = model(features)
        cur_loss = criterion(outputs, labels)
        # torch.max返回每行最大的概率和最大概率的索引,由于批次是16，所以返回16个概率和索引
        _, pred = torch.max(outputs.data, 1)

        test_F1(outputs.argmax(1), labels)
        test_recall(outputs.argmax(1), labels)
        test_precision(outputs.argmax(1), labels)

        # 反向传播和优化
        optimizer.zero_grad()
        cur_loss.backward()
        optimizer.step()
        # 取出loss值和精度值
        loss += cur_loss.item()
        current += (pred == labels).sum().item()
        n += labels.size(0)
        # print(f"train loss: {rate * 100:.1f}%,{cur_loss:.3f}")
    total_recall = test_recall.compute()
    total_precision = test_precision.compute()
    total_F1 = test_F1.compute()
    # total_acc = test_acc.compute()
    print(f"train_loss' : {(loss / n):.3f}  train_acc : {(current / n):.3f}")
    print("recall of every test dataset class: ", total_recall)
    print("precision of every test dataset class: ", total_precision)
    print("F1 of every test dataset class: ", total_F1)
    writer.add_scalar('Train/Loss', loss / n, epoch)
    writer.add_scalar('Train/Acc', current / n, epoch)
    writer.add_scalar('Train/Recall', total_recall[1].item(), epoch)
    writer.add_scalar('Train/Precision', total_precision[1].item(), epoch)
    writer.add_scalar('Train/F1', total_F1[1].item(), epoch)
    test_precision.reset()
    test_recall.reset()
    test_F1.reset()
    # 验证阶段
    model.eval()
    valloss, valcurrent, n = 0.0, 0.0, 0
    val_recall = torchmetrics.Recall(average='none', num_classes=N_FEATURES).to(device)
    val_precision = torchmetrics.Precision(average='none', num_classes=N_FEATURES).to(device)
    val_F1 = torchmetrics.F1Score(average='none', num_classes=N_FEATURES).to(device)
    with torch.no_grad():
        correct = 0
        total = 0
        for features, labels in val_loader:
            features, labels = features.to(device), labels.long().to(device)
            outputs = model(features)
            val_F1(outputs.argmax(1), labels)
            val_recall(outputs.argmax(1), labels)
            val_precision(outputs.argmax(1), labels)
            val_loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            valloss += val_loss.item()
            correct += (predicted == labels).sum().item()

    print('Epoch [{}/{}], Validation Accuracy: {:.3f}%'
          .format(epoch + 1, num_epochs, 100 * correct / total))
    valtotal_recall = val_recall.compute()
    valtotal_precision = val_precision.compute()
    valtotal_F1 = val_F1.compute()
    print(f"valid_loss' : {(valloss / total):.3f}")
    print("recall of every test dataset class: ", valtotal_recall)
    print("precision of every test dataset class: ", valtotal_precision)
    print("F1 of every test dataset class: ", valtotal_F1)
    writer.add_scalar('Valid/Loss', valloss / total, epoch)
    writer.add_scalar('Valid/Acc', current / total, epoch)
    writer.add_scalar('Valid/Recall', valtotal_recall[1].item(), epoch)
    writer.add_scalar('Valid/Precision', valtotal_precision[1].item(), epoch)
    writer.add_scalar('Valid/F1', valtotal_F1[1].item(), epoch)
    test_precision.reset()
    test_recall.reset()
    test_F1.reset()
    # lr_scheduler.step()