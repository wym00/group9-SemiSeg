import os
import os.path as osp
import numpy as np
import random
import matplotlib.pyplot as plt
import collections
import torch
import torchvision
import cv2
from torch.utils import data
from PIL import Image

# 定义了三个用于处理VOC数据集的类，分别是训练数据集类（VOCDataSet）、
# 带有Ground Truth的训练数据集类（VOCGTDataSet）和测试数据集类（VOCDataTestSet）。
# 每个类都实现了数据加载、预处理（如缩放、裁剪、镜像翻转）和数据返回等功能。

# VOC数据集类
class VOCDataSet(data.Dataset):
    def __init__(self, root, list_path, max_iters=None, crop_size=(321, 321), mean=(128, 128, 128), scale=True, mirror=True, ignore_label=255):
        self.root = root  # 数据集根目录
        self.list_path = list_path  # 数据集列表文件路径
        self.crop_h, self.crop_w = crop_size  # 裁剪尺寸
        self.scale = scale  # 是否进行缩放
        self.ignore_label = ignore_label  # 忽略标签
        self.mean = mean  # 图像均值
        self.is_mirror = mirror  # 是否进行镜像翻转
        self.img_ids = [i_id.strip() for i_id in open(list_path)]  # 读取图像ID
        if not max_iters==None:
	        self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids))) # 如果指定最大迭代次数，则重复图像ID
        self.files = [] # 存储图像和标签文件路径
        # for split in ["train", "trainval", "val"]:
        for name in self.img_ids:
            img_file = osp.join(self.root, "JPEGImages/%s.jpg" % name) # 图像文件路径
            label_file = osp.join(self.root, "SegmentationClassAug/%s.png" % name) # 标签文件路径
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": name
            })

    def __len__(self):
        return len(self.files) # 返回文件数量

    def generate_scale_label(self, image, label):
        f_scale = 0.5 + random.randint(0, 11) / 10.0 # 生成缩放因子
        image = cv2.resize(image, None, fx=f_scale, fy=f_scale, interpolation = cv2.INTER_LINEAR) # 缩放图像
        label = cv2.resize(label, None, fx=f_scale, fy=f_scale, interpolation = cv2.INTER_NEAREST) # 缩放标签
        return image, label

    def __getitem__(self, index):
        datafiles = self.files[index]  # 获取索引对应的文件
        image = cv2.imread(datafiles["img"], cv2.IMREAD_COLOR)  # 读取图像
        label = cv2.imread(datafiles["label"], cv2.IMREAD_GRAYSCALE)  # 读取标签
        size = image.shape  # 获取图像尺寸
        name = datafiles["name"]  # 获取文件名
        if self.scale:
            image, label = self.generate_scale_label(image, label)  # 如果需要缩放，则缩放图像和标签
        image = np.asarray(image, np.float32)  # 转换为浮点型数组
        image -= self.mean  # 减去均值
        img_h, img_w = label.shape  # 获取标签尺寸
        pad_h = max(self.crop_h - img_h, 0)  # 计算需要填充的高度
        pad_w = max(self.crop_w - img_w, 0)  # 计算需要填充的宽度
        if pad_h > 0 or pad_w > 0:
            img_pad = cv2.copyMakeBorder(image, 0, pad_h, 0,
                pad_w, cv2.BORDER_CONSTANT,
                value=(0.0, 0.0, 0.0)) # 填充图像
            label_pad = cv2.copyMakeBorder(label, 0, pad_h, 0,
                pad_w, cv2.BORDER_CONSTANT,
                value=(self.ignore_label,)) # 填充标签
        else:
            img_pad, label_pad = image, label

        img_h, img_w = label_pad.shape
        h_off = random.randint(0, img_h - self.crop_h)  # 随机裁剪高度偏移
        w_off = random.randint(0, img_w - self.crop_w)  # 随机裁剪宽度偏移
        image = np.asarray(img_pad[h_off: h_off + self.crop_h, w_off: w_off + self.crop_w], np.float32)  # 裁剪图像
        label = np.asarray(label_pad[h_off: h_off + self.crop_h, w_off: w_off + self.crop_w], np.float32)  # 裁剪标签
        image = image[:, :, ::-1]  # 将RGB图像转换为BGR
        image = image.transpose((2, 0, 1))  # 调整图像维度
        if self.is_mirror:
            flip = np.random.choice(2) * 2 - 1  # 随机选择是否翻转
            image = image[:, :, ::flip]  # 翻转图像
            label = label[:, ::flip]  # 翻转标签

        return image.copy(), label.copy(), np.array(size), name  # 返回图像、标签、尺寸和文件名

# VOCGT数据集类
class VOCGTDataSet(data.Dataset):
    def __init__(self, root, list_path, max_iters=None, crop_size=(321, 321), mean=(128, 128, 128), scale=True, mirror=True, ignore_label=255):
        self.root = root
        self.list_path = list_path
        self.crop_size = crop_size
        self.crop_h, self.crop_w = crop_size
        self.scale = scale
        self.ignore_label = ignore_label
        self.mean = mean
        self.is_mirror = mirror
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        if not max_iters==None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []
        for name in self.img_ids:
            img_file = osp.join(self.root, "JPEGImages/%s.jpg" % name)
            label_file = osp.join(self.root, "SegmentationClassAug/%s.png" % name)
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": name
            })

    def __len__(self):
        return len(self.files)

    def generate_scale_label(self, image, label):
        f_scale = 0.5 + random.randint(0, 11) / 10.0
        image = cv2.resize(image, None, fx=f_scale, fy=f_scale, interpolation = cv2.INTER_LINEAR)
        label = cv2.resize(label, None, fx=f_scale, fy=f_scale, interpolation = cv2.INTER_NEAREST)
        return image, label

    def __getitem__(self, index):
        datafiles = self.files[index]
        image = cv2.imread(datafiles["img"], cv2.IMREAD_COLOR)
        label = cv2.imread(datafiles["label"], cv2.IMREAD_GRAYSCALE)
        size = image.shape
        name = datafiles["name"]

        attempt = 0
        while attempt < 10 :
            if self.scale:
                image, label = self.generate_scale_label(image, label)

            img_h, img_w = label.shape
            pad_h = max(self.crop_h - img_h, 0)
            pad_w = max(self.crop_w - img_w, 0)
            if pad_h > 0 or pad_w > 0:
                attempt += 1
                continue
            else:
                break

        if attempt == 10 :
            image = cv2.resize(image, self.crop_size, interpolation = cv2.INTER_LINEAR)
            label = cv2.resize(label, self.crop_size, interpolation = cv2.INTER_NEAREST)


        image = np.asarray(image, np.float32)
        image -= self.mean

        img_h, img_w = label.shape
        h_off = random.randint(0, img_h - self.crop_h)
        w_off = random.randint(0, img_w - self.crop_w)
        image = np.asarray(image[h_off : h_off+self.crop_h, w_off : w_off+self.crop_w], np.float32)
        label = np.asarray(label[h_off : h_off+self.crop_h, w_off : w_off+self.crop_w], np.float32)
        image = image[:, :, ::-1]  # change to BGR
        image = image.transpose((2, 0, 1))
        if self.is_mirror:
            flip = np.random.choice(2) * 2 - 1
            image = image[:, :, ::flip]
            label = label[:, ::flip]

        return image.copy(), label.copy(), np.array(size), name

# VOC测试数据集类
class VOCDataTestSet(data.Dataset):
    def __init__(self, root, list_path, crop_size=(505, 505), mean=(128, 128, 128)):
        self.root = root
        self.list_path = list_path
        self.crop_h, self.crop_w = crop_size
        self.mean = mean
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        self.files = []
        # for split in ["train", "trainval", "val"]:
        for name in self.img_ids:
            img_file = osp.join(self.root, "JPEGImages/%s.jpg" % name)
            self.files.append({
                "img": img_file
            })

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]
        image = cv2.imread(datafiles["img"], cv2.IMREAD_COLOR)
        size = image.shape
        name = osp.splitext(osp.basename(datafiles["img"]))[0]
        image = np.asarray(image, np.float32)
        image -= self.mean

        img_h, img_w, _ = image.shape
        pad_h = max(self.crop_h - img_h, 0)
        pad_w = max(self.crop_w - img_w, 0)
        if pad_h > 0 or pad_w > 0:
            image = cv2.copyMakeBorder(image, 0, pad_h, 0,
                pad_w, cv2.BORDER_CONSTANT,
                value=(0.0, 0.0, 0.0))
        image = image.transpose((2, 0, 1))
        return image, name, size


if __name__ == '__main__':
    dst = VOCDataSet("./data", is_transform=True)
    trainloader = data.DataLoader(dst, batch_size=4)
    for i, data in enumerate(trainloader):
        imgs, labels = data
        if i == 0:
            img = torchvision.utils.make_grid(imgs).numpy()
            img = np.transpose(img, (1, 2, 0))
            img = img[:, :, ::-1]
            plt.imshow(img)
            plt.show()
