import torch.nn as nn
import torch
import torch.nn.functional as F


# 定义STN网络结构
class STN(nn.Module):
    def __init__(self):
        super(STN, self).__init__()
        # 定义 STN 的局部化网络
        # Spatial transformer localization-network
        self.theta_fixed = 0
        self.localization = nn.Sequential(
            nn.AvgPool2d(2),  # 下采样 3*224*224 -》 3*112*112
            nn.Conv2d(3, 8, kernel_size=7),  # 8*216*218 -》 8*106*106
            nn.MaxPool2d(2, stride=2),  # 8*109*109 -》 8*53*53
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),  # 10*105*105 -》 10*49*49
            nn.MaxPool2d(2, stride=2),  # 10 * 52 * 52 -》 10*24*24
            nn.ReLU(True),
            nn.Conv2d(10, 12, kernel_size=3),  # 12 * 50 * 50 -》 12*22*22
            nn.MaxPool2d(2, stride=2),  # 12*25*25 -》 12*11*11
            nn.ReLU(True)
        )

        # 定义回归器
        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(12 * 11 * 11, 32),  # 按局部网络计算结果修改
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        # 初始化回归器的权重和偏置
        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, x):
        # Localisation net
        xs = self.localization(x)  # 卷积
        xs = xs.view(-1, 12 * 11 * 11)  # resize Tensor维度重构，按局部网络计算结果修改
        # xs = self.dropout(xs)
        theta = self.fc_loc(xs)  # 全连接（6个参数）
        theta = theta.view(-1, 2, 3)
        # print("修改前的theta：", theta)
        theta[:, 0:1, 1:2] = self.theta_fixed  # 只位移和缩放，不旋转
        theta[:, 1:, 0:1] = self.theta_fixed
        # print("修改后的theta:", theta)
        # Grid generator
        grid = F.affine_grid(theta, x.size())
        # Sampler
        x = F.grid_sample(x, grid)
        return x, theta


# 4.定义googlenet网络
# aux_logits=True：使用辅助分类器。
# init_weights=False：初始化权重。
# self.aux_logits = aux_logits->把是否使用辅助分类器传入到类变量当中。
# ceil_mode=True->代表卷积后参数向上取整
class STNGoogLeNet(nn.Module):
    def __init__(self, num_classes=1000, aux_logits=True, init_weights=False):
        super(STNGoogLeNet, self).__init__()
        self.aux_logits = aux_logits

        self.conv1 = BasicConv2d(6, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.conv2 = BasicConv2d(64, 64, kernel_size=1)
        self.conv3 = BasicConv2d(64, 192, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        # 按照inception结构（该层输入大小[上层输出大小],1x1,3x3reduce,3x3,5x5reduce,5x5）来写
        self.inception3a = Inception(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = Inception(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception4a = Inception(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = Inception(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = Inception(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128)

        if self.aux_logits:
            self.aux1 = InceptionAux(512, num_classes)
            self.aux2 = InceptionAux(528, num_classes)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.4)
        self.relu = nn.ReLU(True)
        self.fc = nn.Linear(1024, num_classes)
        self.stn1 = STN()
        self.stn2 = STN()
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x1, theta1 = self.stn1(x)
        x2, theta2 = self.stn2(x)
        x = torch.cat((x1, x2), dim=1)
        # N x 3 x 224 x 224 -》 N x 6 x 224 x 224(拼接两个，两倍）
        x = self.conv1(x)
        # N x 64 x 112 x 112
        x = self.maxpool1(x)
        # N x 64 x 56 x 56
        x = self.conv2(x)
        # N x 64 x 56 x 56
        x = self.conv3(x)
        # N x 192 x 56 x 56
        x = self.maxpool2(x)

        # N x 192 x 28 x 28
        x = self.inception3a(x)
        # N x 256 x 28 x 28
        x = self.inception3b(x)
        # N x 480 x 28 x 28
        x = self.maxpool3(x)
        # N x 480 x 14 x 14
        x = self.inception4a(x)
        # N x 512 x 14 x 14
        # training=true/false看当前处于哪种模式,aux是否用到辅助分类器
        if self.training and self.aux_logits:  # eval model lose this layer
            aux1 = self.aux1(x)

        x = self.inception4b(x)
        # N x 512 x 14 x 14
        x = self.inception4c(x)
        # N x 512 x 14 x 14
        x = self.inception4d(x)
        # N x 528 x 14 x 14
        if self.training and self.aux_logits:  # eval model lose this layer
            aux2 = self.aux2(x)

        x = self.inception4e(x)
        # N x 832 x 14 x 14
        x = self.maxpool4(x)
        # N x 832 x 7 x 7
        x = self.inception5a(x)
        # N x 832 x 7 x 7
        x = self.inception5b(x)
        # N x 1024 x 7 x 7

        x = self.avgpool(x)
        # N x 1024 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 1024
        x = self.dropout(x)
        x = self.fc(x)
        # N x 1000 (num_classes)
        if self.training and self.aux_logits:  # eval model lose this layer
            return x, aux2, aux1  # 主分支输出值，辅助分类器输出值

        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


# 2.定义Inception结构
# 一共有4个branch,最后用torch,cat()合并成一个矩阵，1代表深度
class Inception(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
        super(Inception, self).__init__()

        self.branch1 = BasicConv2d(in_channels, ch1x1, kernel_size=1)

        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, ch3x3red, kernel_size=1),
            BasicConv2d(ch3x3red, ch3x3, kernel_size=3, padding=1)  # 保证输出大小等于输入大小
        )

        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, ch5x5red, kernel_size=1),
            BasicConv2d(ch5x5red, ch5x5, kernel_size=5, padding=2)  # 保证输出大小等于输入大小
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(in_channels, pool_proj, kernel_size=1)
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        outputs = [branch1, branch2, branch3, branch4]
        return torch.cat(outputs, 1)


# 3.定义分类器模块
# 1）nn.AvgPool2d(kernel_size=5, stride=3)：平均池化下采样-》核大小为5x5，步长为3。
# 2）BasicConv2d()：卷积激活
# 3）nn.Linear(2048, 1024)、nn.Linear(1024, num_classes);经过两个全连接层得到分类的一维向量。
# 4）torch.flatten(x, 1)：从深度方向对特征矩阵进行推平处理，从3维降到2维。
# 5）模块前向传播总流程：输入特征矩阵x->平均池化AvgPool2d->卷积BasicConv2d->降维推平flatten->随机失活Dropout->激活Relu->随机失活Dropout->全连接fc->得到分类。

class InceptionAux(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(InceptionAux, self).__init__()
        self.averagePool = nn.AvgPool2d(kernel_size=5, stride=3)
        self.conv = BasicConv2d(in_channels, 128, kernel_size=1)  # output[batch, 128, 4, 4]

        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        # aux1: N x 512 x 14 x 14, aux2: N x 528 x 14 x 14
        x = self.averagePool(x)
        # aux1: N x 512 x 4 x 4, aux2: N x 528 x 4 x 4
        x = self.conv(x)
        # N x 128 x 4 x 4
        x = torch.flatten(x, 1)
        x = F.dropout(x, 0.5, training=self.training)
        # N x 2048
        x = F.relu(self.fc1(x), inplace=True)
        x = F.dropout(x, 0.5, training=self.training)
        # N x 1024
        x = self.fc2(x)
        # N x num_classes
        return x


# 1.首先定义一个基本卷积模块包含一个卷积层和一个Relu激活层和一个正向传播函数。
# in_channels->输入特征矩阵的深度
# out_channels->输出特征矩阵的深度。其中 self.conv = nn.Conv2d()中的out_channels也代表卷积核个数
# **kwargs 允许你将不定长度的键值对, 作为参数传递给一个函数。 如果你想要在一个函数里处理带名字的参数, 你应该使用**kwargs
class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x
