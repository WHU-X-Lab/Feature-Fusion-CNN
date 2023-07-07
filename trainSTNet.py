# -- coding:utf-8

import time

import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics
from torch.optim import lr_scheduler
from torch.utils.data import WeightedRandomSampler, DataLoader
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from config_stn import *
from dataset import BuildingDataset
# from CUBdataset import CUB
import pandas as pd
import torch.nn.functional as F
from STNLeNet import STNLeNet
import os
from torchvision import transforms
import numpy as np
# import transformsCUB
import torchvision
from matplotlib.patches import Rectangle

plt.ion()
# 固定随机数种子
seed = 123
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# 鸟类数据集
path = 'CUB_200_2011'
TRAIN_MEAN = [0.48560741861744905, 0.49941626449353244, 0.43237713785804116]
TRAIN_STD = [0.2321024260764962, 0.22770540015765814, 0.2665100547329813]
TEST_MEAN = [0.4862169586881995, 0.4998156522834164, 0.4311430419332438]
TEST_STD = [0.23264268069040475, 0.22781080253662814, 0.26667253517177186]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(90),
    transforms.ToTensor(),
    # train_stn
    # transforms.Normalize(mean=[0.89734197, 0.9392724, 0.9398877], std=[0.17993543, 0.094597824, 0.12841283])
    # Area
    # transforms.Normalize(mean=[0.9328658, 0.96335715, 0.96645814], std=[0.13745053, 0.07245668, 0.0951552])
    # function_test_20_old
    # transforms.Normalize(mean=[0.9318218, 0.9599232, 0.96266234], std=[0.13674922, 0.07559318, 0.09820192])
    # function_test_20
    transforms.Normalize(mean=[0.933881, 0.957784, 0.9582756], std=[0.13598508, 0.07826422, 0.10344314])
    # CUB-200-2011鸟类数据集
    # transformsCUB.ToCVImage(),
    # transformsCUB.RandomResizedCrop(224),
    # transformsCUB.RandomHorizontalFlip(),
    # transformsCUB.ToTensor(),
    # transformsCUB.Normalize(TRAIN_MEAN, TRAIN_STD)
])
transform_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(90),
    transforms.ToTensor(),
    # test_rect_train
    # transforms.Normalize(mean = [0.70381516, 0.8888911, 0.92238843], std =  [0.40284097, 0.11763619, 0.15052465])
    # test_area_train
    # transforms.Normalize(mean=[0.84431416, 0.9468392, 0.96895444], std = [0.2916492, 0.08882934, 0.098634705])
    # valid_stn
    # transforms.Normalize(mean=[0.8980922, 0.9372042, 0.93724376], std=[0.18020877, 0.09663033, 0.13077204])
    # transforms.Normalize(mean = [0.8904361, 0.9363585, 0.94014686], std = [0.165799, 0.09002642, 0.12338792])
    # compareValid
    # transforms.Normalize(mean = [0.84808147, 0.9096524, 0.91053045], std = [0.20253241, 0.10373465, 0.14877456])
    # valid_Area
    # transforms.Normalize(mean=[0.9347815, 0.96474624, 0.9671215], std=[0.13583878, 0.07196212, 0.09568332])
    # function_test_20
    transforms.Normalize(mean=[0.92881185, 0.95924205, 0.96284324], std=[0.13795583, 0.07532157, 0.09843103])

    # CUB-200-2011鸟类数据集
    # transformsCUB.ToCVImage(),
    # transformsCUB.Resize(224),
    # transformsCUB.ToTensor(),
    # transformsCUB.Normalize(TEST_MEAN, TEST_STD)
])

df = pd.DataFrame(columns=['loss', 'accuracy'])


# 定义训练函数
def train(dataloader, model, loss_fn, optimizer, epoch):
    loss, current, n = 0.0, 0.0, 0
    test_recall = torchmetrics.Recall(average='none', num_classes=N_FEATURES).to(device)
    test_precision = torchmetrics.Precision(average='none', num_classes=N_FEATURES).to(device)
    test_F1 = torchmetrics.F1Score(num_classes=N_FEATURES, average='none').to(device)
    model.train()
    # enumerate返回为数据和标签还有批次
    for batch, (X, y) in enumerate(dataloader):
        # 前向传播
        X, y = X.to(device), y.to(device)
        output = model(X)
        # output, output2, output1 = model(X)
        cur_loss0 = loss_fn(output, y)
        # cur_loss1 = loss_fn(output1, y)
        # cur_loss2 = loss_fn(output2, y)

        # torch.max返回每行最大的概率和最大概率的索引,由于批次是16，所以返回16个概率和索引
        _, pred = torch.max(output, axis=1)

        # 计算每批次的准确率， output.shape[0]为该批次的多少
        cur_acc = torch.sum(y == pred) / output.shape[0]
        cur_loss = cur_loss0  # + cur_loss1 * 0.3 + cur_loss2 * 0.3
        # test_acc(output.argmax(1), y)
        test_F1(output.argmax(1), y)
        test_recall(output.argmax(1), y)
        test_precision(output.argmax(1), y)
        # 反向传播
        optimizer.zero_grad()
        cur_loss.backward()
        optimizer.step()
        # 取出loss值和精度值
        loss += cur_loss.item()
        current += cur_acc.item()
        n = n + 1
        rate = (batch + 1) / train_num
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


# 定义验证函数
def val(dataloader, model, loss_fn, epoch):
    # 将模型转为验证模式
    model.eval()
    loss, current, n = 0.0, 0.0, 0
    test_recall = torchmetrics.Recall(average='none', num_classes=N_FEATURES).to(device)
    test_precision = torchmetrics.Precision(average='none', num_classes=N_FEATURES).to(device)
    test_F1 = torchmetrics.F1Score(average='none', num_classes=N_FEATURES).to(device)
    # 非训练，推理期用到（测试时模型参数不用更新， 所以no_grad）
    # print(torch.no_grad)
    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            output = model(X)
            # output1 = output.squeeze(-1)
            cur_loss = loss_fn(output, y)
            test_F1(output.argmax(1), y)
            test_recall(output.argmax(1), y)
            test_precision(output.argmax(1), y)
            _, pred = torch.max(output, axis=1)
            cur_acc = torch.sum(y == pred) / output.shape[0]

            loss += cur_loss.item()
            current += cur_acc.item()
            n = n + 1
    total_recall = test_recall.compute()
    total_precision = test_precision.compute()
    total_F1 = test_F1.compute()
    print(f"valid_loss' : {(loss / n):.3f}  valid_acc : {(current / n):.3f}")
    print("recall of every test dataset class: ", total_recall)
    print("precision of every test dataset class: ", total_precision)
    print("F1 of every test dataset class: ", total_F1)
    writer.add_scalar('Valid/Loss', loss / n, epoch)
    writer.add_scalar('Valid/Acc', current / n, epoch)
    writer.add_scalar('Valid/Recall', total_recall[1].item(), epoch)
    writer.add_scalar('Valid/Precision', total_precision[1].item(), epoch)
    writer.add_scalar('Valid/F1', total_F1[1].item(), epoch)
    test_precision.reset()
    test_recall.reset()
    test_F1.reset()
    df.loc[epoch] = {'loss': loss / n, 'accuracy': current / n}
    return current / n


def convert_image_np(inp):
    """Convert a Tensor to numpy image."""
    inp = inp.numpy().transpose((1, 2, 0))  # 将颜色通道放到后面
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean  # 用公式"(x-mean)/std"，将每个元素分布到(-1,1)，也就是标准化
    inp = np.clip(inp, 0, 1)  # # clip这个函数将将数组中的元素限制在a_min, a_max之间，大于a_max的就使得它等于 a_max，小于a_min,的就使得它等于a_min
    return inp


def attention(feature_map):
    # Flatten feature map spatial dimensions
    batch_size, num_channels, height, width = feature_map.size()
    feature_map_flat = feature_map.view(batch_size, num_channels, height * width)

    # Compute self-attention using matrix multiplication
    query = torch.matmul(feature_map_flat.transpose(1, 2), feature_map_flat)  # [batch_size, height*width, height*width]
    key = torch.matmul(feature_map_flat, feature_map_flat.transpose(1, 2))  # [batch_size, num_channels, num_channels]
    value = feature_map_flat  # [batch_size, num_channels, height*width]

    # Compute attention weights using softmax
    attention_weights = torch.softmax(query, dim=-1)  # [batch_size, height*width, height*width]

    # Compute attention centers using weighted average
    attention_centers = torch.bmm(value, attention_weights)  # [batch_size, num_channels, height*width]
    attention_centers = attention_centers.view(batch_size, num_channels, height, width)

    return attention_centers
    # attention_map = torch.sum(image, dim=1).to(device)
    # attention_map = attention_map / torch.max(attention_map)
    # x = torch.sum(attention_map, dim=2).to(device)
    # y = torch.sum(attention_map, dim=1).to(device)
    # x_c = torch.sum(x * torch.arange(x.size(1)).to(device)).to(device) / (torch.sum(x, dim=1)).to(device)
    # y_c = torch.sum(y * torch.arange(y.size(1)).to(device)).to(device) / (torch.sum(y, dim=1)).to(device)
    # return x_c, y_c


# We want to visualize the output of the spatial transformers layer
# after the training, we visualize a batch of input images and
# the corresponding transformed batch using STN.

def visualize_stn(model):
    with torch.no_grad():
        # Get a batch of training data
        data = next(iter(train_loader))[0]
        model.eval()
        # 使用STN模型得到变换后的图像和变换矩阵
        transformed_data1, transformed_data2 = model.stn(data.to(device))
        # transformed_data1, theta = model.stn(data.to(device))
        # transformed_data1, theta = model.stn(transformed_data0.to(device))
        batch_size, _, h, w = transformed_data1.shape
        # 计算图像注意力中心
        x_center, y_center = attention(transformed_data1.to(device))
        x_center2, y_center2 = attention(transformed_data2.to(device))
        # attention_corners1,attention_corners2, batch_size = get_attention_box(model, device, data)
        # 可视化注意力区域边界框（显示整个批次）
        # 创建matplotlib的Figure和Axes对象
        fig, axs = plt.subplots(nrows=batch_size // 4 + (batch_size % 4 != 0),
                                ncols=4,
                                figsize=(15, (batch_size // 4 + (batch_size % 4 != 0)) * 3))
        for i in range(batch_size):
            ax = axs[i // 4, i % 4]
            ax.imshow(convert_image_np(data[i]))
            x_c = x_center[i].item()
            y_c = y_center[i].item()
            x_c2 = x_center2[i].item()
            y_c2 = y_center2[i].item()
            # x0 = x_c-50
            # y0 = y_c-50
            # x1 = x_c+50
            # y1 = y_c+50
            x0 = max(0, x_c - 50)
            y0 = max(0, y_c - 50)
            x1 = min(w - 1, x_c + 50)
            y1 = min(h - 1, y_c + 50)
            x2_0 = max(0, x_c2 - 50)
            y2_0 = max(0, y_c2 - 50)
            x2_1 = min(w - 1, x_c2 + 50)
            y2_1 = min(h - 1, y_c2 + 50)
            rect = plt.Rectangle((x0, y0), x1 - x0 + 1, y1 - y0 + 1,
                                 fill=False,
                                 edgecolor='red',
                                 linewidth=3.5)
            rect2 = plt.Rectangle((x2_0, y2_0), x2_1 - x2_0 + 1, y2_1 - y2_0 + 1,
                                  fill=False,
                                  edgecolor='green',
                                  linewidth=3.5)
            ax.add_patch(rect)
            ax.add_patch(rect2)
            #
            # x_min2 = attention_corners2[i, :, 0].min()
            # x_max2 = attention_corners2[i, :, 0].max()
            # y_min2 = attention_corners2[i, :, 1].min()
            # y_max2 = attention_corners2[i, :, 1].max()
            # rect2 = plt.Rectangle((x_min2, y_min2), x_max2 - x_min2 + 1, y_max2 - y_min2 + 1,
            #                      fill=False,
            #                      edgecolor='green',
            #                      linewidth=3.5)
            # ax.add_patch(rect2)

        # -------以下为原可视化代码--------
        # input_tensor = data.cpu()
        # #tensor1 = model.stn(data)
        # transformed_input_tensor = model.stn(data).cpu()
        #
        # in_grid = convert_image_np(
        #     torchvision.utils.make_grid(input_tensor))
        #
        # out_grid = convert_image_np(
        #     torchvision.utils.make_grid(transformed_input_tensor))
        #
        # # Plot the results side-by-side
        # f, axarr = plt.subplots(1, 2)
        # axarr[0].imshow(in_grid)
        # axarr[0].set_title('Dataset Images')
        #
        # axarr[1].imshow(out_grid)
        # axarr[1].set_title('Transformed Images')


if __name__ == '__main__':
    s = f"STNLeNet,{train_dir},{valid_dir},batch{BATCH_SIZE},lr{LR},wd{weight_decay}"
    writer = SummaryWriter(comment=s)
    # build MyDataset
    # class_sample_counts = [33288,4128] #compareTrain
    # class_sample_counts = [3257, 381]  # train_stn
    # class_sample_counts = [33444, 4128]  # function_test_20_old
    class_sample_counts = [8376, 4128]  # function_test_20
    # class_sample_counts = [3118, 281]  # Area or Rect
    weights = 1. / torch.tensor(class_sample_counts, dtype=torch.float)
    # 建筑数据集读取
    # 这个 get_classes_for_all_imgs是关键
    train_data = BuildingDataset(data_dir=train_dir, transform=transform)
    train_targets = train_data.get_classes_for_all_imgs()
    samples_weights = weights[train_targets]
    sampler = WeightedRandomSampler(weights=samples_weights, num_samples=len(samples_weights), replacement=True)

    valid_data = BuildingDataset(data_dir=valid_dir, transform=transform_val)

    # build DataLoader
    train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=4,
                              pin_memory=True, sampler=sampler)
    valid_loader = DataLoader(dataset=valid_data, batch_size=BATCH_SIZE, num_workers=4, pin_memory=True,
                              shuffle=True)
    # CUB鸟类数据集读取
    # train_dataset = CUB(
    #     path,
    #     train=True,
    #     transform=transform,
    #     target_transform=None
    # )
    # # print(len(train_dataset))
    # train_loader = DataLoader(
    #     train_dataset,
    #     batch_size=BATCH_SIZE,
    #     num_workers=4,
    #     shuffle=True
    # )
    #
    # test_dataset = CUB(
    #     path,
    #     train=False,
    #     transform=transform_val,
    #     target_transform=None
    # )
    #
    # valid_loader = DataLoader(
    #     test_dataset,
    #     batch_size=BATCH_SIZE,
    #     num_workers=4,
    #     shuffle=True
    # )

    # AlexNet model and training
    net = STNLeNet(num_classes=N_FEATURES)
    # net = STNet(num_classes=N_FEATURES, init_weights=True)
    # net = STNGoogLeNet(num_classes=N_FEATURES)
    # net = GoogLeNet(num_classes=N_FEATURES)
    # 模拟输入数据，进行网络可视化
    # input_data = Variable(torch.rand(16, 3, 224, 224))
    # with writer:
    #     writer.add_graph(net, (input_data,))pythp
    # 模型进入GPU
    # if torch.cuda.device_count() > 1:
    #     print("Use", torch.cuda.device_count(), 'gpus')
    #     net = nn.DataParallel(net)

    # net.load_state_dict(torch.load('./save_model/stn/last_model.pth'), False)
    # net.to(device)
    model = net.to(device)

    # 定义损失函数（交叉熵损失）
    loss_fn = nn.CrossEntropyLoss()
    # loss_fn = focal_loss(alpha= [1.12196,9.20306],gamma=2,num_classes=2)
    # loss_fn = nn.BCEWithLogitsLoss()

    # 定义优化器,SGD,
    # optimizer = optim.Adam(net.parameters(), lr=LR,weight_decay=weight_decay)
    optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=weight_decay_f)

    # 学习率按数组自定义变化
    lr_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

    # 开始训练
    epoch = MAX_EPOCH
    min_acc = 0
    train_num = len(train_loader)
    for t in range(epoch):
        start = time.time()
        print(f"epoch{t + 1}\n-------------------")
        train(train_loader, net, loss_fn, optimizer, t)
        a = val(valid_loader, net, loss_fn, t)
        lr_scheduler.step()
        print("目前学习率:", optimizer.param_groups[0]['lr'])
        # 保存最好的模型权重文件
        if a > min_acc:
            folder = 'save_model'
            if not os.path.exists(folder):
                os.mkdir('save_model')
            min_acc = a
            print('save best model', )
            torch.save(net.state_dict(), "save_model/stn/best_model.pth")
        # 保存最后的权重文件
        torch.save(net.state_dict(), "save_model/stn/every_model.pth")
        if t == epoch - 1:
            torch.save(net.state_dict(), "save_model/stn/last_model.pth")
        finish = time.time()
        time_elapsed = finish - start
        print('本次训练耗时 {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
    # visualize_stn(model)
    # plt.ioff()
    # plt.show()
    print(f'** Finished Training **')
    df.to_csv('runs/train.txt', index=True, sep=';')
