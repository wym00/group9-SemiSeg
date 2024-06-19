import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable

# 定义一个二维交叉熵损失函数类
class CrossEntropy2d(nn.Module):

    def __init__(self, size_average=True, ignore_label=255):
        super(CrossEntropy2d, self).__init__()
        self.size_average = size_average  # 是否对损失进行平均
        self.ignore_label = ignore_label  # 忽略的标签值

    def forward(self, predict, target, weight=None):
        """
            前向传播函数
        参数:
            predict:(n, c, h, w) - 预测值，n个样本，c个类别，h高，w宽
            target:(n, h, w) - 目标值，n个样本，h高，w宽
            weight (Tensor, optional): 手动为每个类别设置的权重，如果提供，必须是一个大小为"nclasses"的张量
        """
        assert not target.requires_grad  # 确保目标值不需要梯度
        assert predict.dim() == 4  # 确保预测值是4维
        assert target.dim() == 3  # 确保目标值是3维
        assert predict.size(0) == target.size(0), "{0} vs {1} ".format(predict.size(0), target.size(0))  # 确保批次大小一致
        assert predict.size(2) == target.size(1), "{0} vs {1} ".format(predict.size(2), target.size(1))  # 确保高度一致
        assert predict.size(3) == target.size(2), "{0} vs {1} ".format(predict.size(3), target.size(3))  # 确保宽度一致
        n, c, h, w = predict.size()  # 获取预测值的维度信息
        target_mask = (target >= 0) * (target != self.ignore_label)  # 创建掩码，忽略指定标签
        target = target[target_mask]  # 应用掩码，过滤掉忽略的标签
        if not target.data.dim():  # 如果目标值为空，返回零张量
            return Variable(torch.zeros(1))
        predict = predict.transpose(1, 2).transpose(2, 3).contiguous()  # 调整预测值的维度顺序
        predict = predict[target_mask.view(n, h, w, 1).repeat(1, 1, 1, c)].view(-1, c)  # 应用掩码并重塑预测值
        loss = F.cross_entropy(predict, target, weight=weight, size_average=self.size_average)  # 计算交叉熵损失
        return loss  # 返回损失值


# 定义一个二维二分类交叉熵损失函数类
class BCEWithLogitsLoss2d(nn.Module):

    def __init__(self, size_average=True, ignore_label=255):
        super(BCEWithLogitsLoss2d, self).__init__()
        self.size_average = size_average  # 是否对损失进行平均
        self.ignore_label = ignore_label  # 忽略的标签值

    def forward(self, predict, target, weight=None):
        """
            前向传播函数
        参数:
            predict:(n, 1, h, w) - 预测值，n个样本，1个类别，h高，w宽
            target:(n, 1, h, w) - 目标值，n个样本，1个类别，h高，w宽
            weight (Tensor, optional): 手动为每个类别设置的权重，如果提供，必须是一个大小为"nclasses"的张量
        """
        assert not target.requires_grad  # 确保目标值不需要梯度
        assert predict.dim() == 4  # 确保预测值是4维
        assert target.dim() == 4  # 确保目标值是4维
        assert predict.size(0) == target.size(0), "{0} vs {1} ".format(predict.size(0), target.size(0))  # 确保批次大小一致
        assert predict.size(2) == target.size(2), "{0} vs {1} ".format(predict.size(2), target.size(2))  # 确保高度一致
        assert predict.size(3) == target.size(3), "{0} vs {1} ".format(predict.size(3), target.size(3))  # 确保宽度一致
        n, c, h, w = predict.size()  # 获取预测值的维度信息
        target_mask = (target >= 0) * (target != self.ignore_label)  # 创建掩码，忽略指定标签
        target = target[target_mask]  # 应用掩码，过滤掉忽略的标签
        if not target.data.dim():  # 如果目标值为空，返回零张量
            return Variable(torch.zeros(1))
        predict = predict[target_mask]  # 应用掩码到预测值
        loss = F.binary_cross_entropy_with_logits(predict, target, weight=weight,
                                                  size_average=self.size_average)  # 计算二分类交叉熵损失
        return loss  # 返回损失值
