import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch
import numpy as np
affine_par = True # 定义一个全局变量affine_par，用于控制BatchNorm层中是否有可学习的仿射参数

# 定义一个函数outS，用于计算输出尺寸
def outS(i):
    i = int(i) # 将输入转换为整数
    i = (i+1)/2 # 对输入加1后除以2
    i = int(np.ceil((i+1)/2.0)) # 再次对结果加1后除以2，并向上取整
    i = (i+1)/2 # 最后再次对结果加1后除以2
    return i # 返回最终计算结果

# 定义一个函数conv3x3，用于创建3x3的卷积层，包含padding
def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False) # 返回一个卷积层对象

# 定义一个基础块BasicBlock，用于构建ResNet中的基础模块
class BasicBlock(nn.Module):
    expansion = 1 # 基础块的扩展系数为1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride) # 创建第一个卷积层
        self.bn1 = nn.BatchNorm2d(planes, affine = affine_par) # 创建第一个BatchNorm层
        self.relu = nn.ReLU(inplace=True) # 创建ReLU激活函数
        self.conv2 = conv3x3(planes, planes) # 创建第二个卷积层
        self.bn2 = nn.BatchNorm2d(planes, affine = affine_par) # 创建第二个BatchNorm层
        self.downsample = downsample # 定义下采样操作
        self.stride = stride # 保存步长信息

    def forward(self, x):
        residual = x # 保存输入作为残差连接

        out = self.conv1(x) # 通过第一个卷积层
        out = self.bn1(out) # 通过第一个BatchNorm层
        out = self.relu(out) # 应用ReLU激活函数

        out = self.conv2(out) # 通过第二个卷积层
        out = self.bn2(out) # 通过第二个BatchNorm层

        if self.downsample is not None: # 如果存在下采样操作
            residual = self.downsample(x) # 对输入进行下采样

        out += residual # 将残差连接加到当前输出上
        out = self.relu(out) # 再次应用ReLU激活函数

        return out # 返回最终输出

# 定义一个瓶颈块Bottleneck，用于构建ResNet中的瓶颈模块
class Bottleneck(nn.Module):
    expansion = 4 # 瓶颈块的扩展系数为4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False) # change # 创建1x1卷积层
        self.bn1 = nn.BatchNorm2d(planes,affine = affine_par) # 创建BatchNorm层
        for i in self.bn1.parameters():
            i.requires_grad = False # 设置BatchNorm层参数不需要梯度

        padding = dilation # 根据膨胀率设置padding
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, # change # 创建3x3卷积层
                               padding=padding, bias=False, dilation = dilation) # 设置膨胀率
        self.bn2 = nn.BatchNorm2d(planes,affine = affine_par) # 创建第二个BatchNorm层
        for i in self.bn2.parameters():
            i.requires_grad = False # 设置BatchNorm层参数不需要梯度
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False) # 创建1x1卷积层
        self.bn3 = nn.BatchNorm2d(planes * 4, affine = affine_par) # 创建第三个BatchNorm层
        for i in self.bn3.parameters():
            i.requires_grad = False # 设置BatchNorm层参数不需要梯度
        self.relu = nn.ReLU(inplace=True) # 创建ReLU激活函数
        self.downsample = downsample # 定义下采样操作
        self.stride = stride # 保存步长信息


    def forward(self, x):
        residual = x # 保存输入作为残差连接

        out = self.conv1(x)  # 通过第一个卷积层
        out = self.bn1(out)  # 通过第一个BatchNorm层
        out = self.relu(out)  # 应用ReLU激活函数

        out = self.conv2(out)  # 通过第二个卷积层
        out = self.bn2(out)  # 通过第二个BatchNorm层
        out = self.relu(out)  # 再次应用ReLU激活函数

        out = self.conv3(out)  # 通过第三个卷积层
        out = self.bn3(out)  # 通过第三个BatchNorm层

        if self.downsample is not None:  # 如果存在下采样操作
            residual = self.downsample(x)  # 对输入进行下采样

        out += residual  # 将残差连接加到当前输出上
        out = self.relu(out)  # 再次应用ReLU激活函数

        return out  # 返回最终输出

    # 定义一个分类模块Classifier_Module，用于构建分类层
class Classifier_Module(nn.Module):

    def __init__(self, dilation_series, padding_series, num_classes):
        super(Classifier_Module, self).__init__()
        self.conv2d_list = nn.ModuleList() # 创建一个模块列表
        for dilation, padding in zip(dilation_series, padding_series): # 遍历膨胀率和padding值
            self.conv2d_list.append(nn.Conv2d(2048, num_classes, kernel_size=3, stride=1, padding=padding, dilation=dilation, bias = True)) # 添加卷积层到模块列表

        for m in self.conv2d_list: # 初始化卷积层权重
            m.weight.data.normal_(0, 0.01) # 使用正态分布初始化权重

    def forward(self, x):
        out = self.conv2d_list[0](x) # 通过第一个卷积层
        for i in range(len(self.conv2d_list)-1): # 遍历剩余的卷积层
            out += self.conv2d_list[i+1](x) # 将每个卷积层的输出相加
            return out # 返回最终的输出


# 定义一个ResNet网络
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes):
        self.inplanes = 64 # 设置初始的通道数为64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, # 创建第一个卷积层
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, affine = affine_par) # 创建第一个BatchNorm层
        for i in self.bn1.parameters():
            i.requires_grad = False # 设置BatchNorm层参数不需要梯度
        self.relu = nn.ReLU(inplace=True) # 创建ReLU激活函数
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True) # change # 创建最大池化层
        self.layer1 = self._make_layer(block, 64, layers[0]) # 创建第一个层级
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2) # 创建第二个层级
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2) # 创建第三个层级
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4) # 创建第四个层级
        self.layer5 = self._make_pred_layer(Classifier_Module, [6,12,18,24],[6,12,18,24],num_classes) # 创建分类层

        for m in self.modules(): # 初始化网络参数
            if isinstance(m, nn.Conv2d): # 如果是卷积层
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels # 计算权重数量
                m.weight.data.normal_(0, 0.01) # 使用正态分布初始化权重
            elif isinstance(m, nn.BatchNorm2d): # 如果是BatchNorm层
                m.weight.data.fill_(1) # 将权重设置为1
                m.bias.data.zero_() # 将偏置设置为0
        #        for i in m.parameters():
        #            i.requires_grad = False

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None # 初始化下采样操作为None
        if stride != 1 or self.inplanes != planes * block.expansion or dilation == 2 or dilation == 4: # 如果需要下采样
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, # 创建1x1卷积层用于下采样
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion,affine = affine_par)) # 创建BatchNorm层
        for i in downsample._modules['1'].parameters():
            i.requires_grad = False # 设置BatchNorm层参数不需要梯度
        layers = [] # 创建层列表
        layers.append(block(self.inplanes, planes, stride,dilation=dilation, downsample=downsample)) # 添加第一个块
        self.inplanes = planes * block.expansion # 更新通道数
        for i in range(1, blocks): # 添加剩余的块
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers) # 返回一个由所有块组成的顺序容器
    def _make_pred_layer(self,block, dilation_series, padding_series,num_classes):
        return block(dilation_series,padding_series,num_classes) # 创建分类层

    def forward(self, x):
        x = self.conv1(x)  # 通过第一个卷积层
        x = self.bn1(x)  # 通过第一个BatchNorm层
        x = self.relu(x)  # 应用ReLU激活函数
        x = self.maxpool(x)  # 通过最大池化层
        x = self.layer1(x)  # 通过第一个层级
        x = self.layer2(x)  # 通过第二个层级
        x = self.layer3(x)  # 通过第三个层级
        x = self.layer4(x)  # 通过第四个层级
        x = self.layer5(x)  # 通过分类层

        return x # 返回最终输出

    def get_1x_lr_params_NOscale(self):
        """
        This generator returns all the parameters of the net except for 
        the last classification layer. Note that for each batchnorm layer, 
        requires_grad is set to False in deeplab_resnet.py, therefore this function does not return 
        any batchnorm parameter
        这个生成器返回网络中除了最后分类层之外的所有参数。
        注意，对于每个BatchNorm层，requires_grad已经在deeplab_resnet.py中设置为False，
        因此这个函数不会返回任何BatchNorm层的参数。
        """
        b = []

        b.append(self.conv1)
        b.append(self.bn1)
        b.append(self.layer1)
        b.append(self.layer2)
        b.append(self.layer3)
        b.append(self.layer4)

        for i in range(len(b)):  # 遍历所有层
            for j in b[i].modules():  # 遍历层中的所有模块
                jj = 0
                for k in j.parameters():  # 遍历模块的所有参数
                    jj += 1
                    if k.requires_grad:  # 如果参数需要梯度
                        yield k  # 返回参数

    def get_10x_lr_params(self):
        """
        This generator returns all the parameters for the last layer of the net,
        which does the classification of pixel into classes
        这个生成器返回网络最后一层的所有参数，
        这一层负责将像素分类到不同的类别。
        """
        b = []
        b.append(self.layer5.parameters())

        for j in range(len(b)):
            for i in b[j]:
                yield i
            


    def optim_parameters(self, args):
        return [{'params': self.get_1x_lr_params_NOscale(), 'lr': args.learning_rate}, # 返回除了最后一层之外的所有参数和它们的学习率
                {'params': self.get_10x_lr_params(), 'lr': 10*args.learning_rate}]  # 返回最后一层的参数和它们的学习率

# 定义一个函数Res_Deeplab，用于创建一个ResNet-Deeplab模型
def Res_Deeplab(num_classes=21):
    model = ResNet(Bottleneck,[3, 4, 23, 3], num_classes) # 创建ResNet模型
    return model # 返回模型对象

