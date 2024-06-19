import argparse
import cv2
import torch
import torch.nn as nn
from torch.utils import data, model_zoo
import numpy as np
import pickle
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import scipy.misc
import torch.backends.cudnn as cudnn
import sys
import os
import os.path as osp
import pickle
from packaging import version

from model.deeplab import Res_Deeplab
from model.discriminator import FCDiscriminator
from utils.loss import CrossEntropy2d, BCEWithLogitsLoss2d
from dataset.voc_dataset import VOCDataSet, VOCGTDataSet



import matplotlib.pyplot as plt
import random
import timeit
start = timeit.default_timer() # 开始计时

IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32) # 图像均值

MODEL = 'DeepLab' # 模型名称
BATCH_SIZE = 10 # 批处理大小
ITER_SIZE = 1 # 迭代次数
NUM_WORKERS = 4 # 工作线程数
DATA_DIRECTORY = './dataset/VOC2012' # 数据集目录
DATA_LIST_PATH = './dataset/voc_list/train_aug.txt' # 数据集列表路径
IGNORE_LABEL = 255 # 忽略标签
INPUT_SIZE = '321,321' # 输入尺寸
LEARNING_RATE = 2.5e-4 # 学习率
MOMENTUM = 0.9 # 动量
NUM_CLASSES = 21 # 类别数
NUM_STEPS = 20000 # 步数
POWER = 0.9 # 幂
RANDOM_SEED = 1234 # 随机种子
RESTORE_FROM = 'http://vllab1.ucmerced.edu/~whung/adv-semi-seg/resnet101COCO-41f33a49.pth'  # 恢复模型参数的路径
SAVE_NUM_IMAGES = 2 # 保存的图像数量
SAVE_PRED_EVERY = 5000 # 每隔多少步保存预测结果
SNAPSHOT_DIR = './snapshots/' # 快照目录
WEIGHT_DECAY = 0.0005 # 权重衰减

LEARNING_RATE_D = 1e-4 # 判别器的学习率
LAMBDA_ADV_PRED = 0.1 # 对抗训练的lambda

PARTIAL_DATA=0.5 # 部分数据

SEMI_START=5000 # 半监督学习开始的步数
LAMBDA_SEMI=0.1 # 半监督学习的lambda
MASK_T=0.2 # 半监督学习的掩码阈值

LAMBDA_SEMI_ADV=0.001 # 半监督对抗训练的lambda
SEMI_START_ADV=0 # 半监督对抗训练开始的步数
D_REMAIN=True # 是否训练D与未标记数据


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--model", type=str, default=MODEL,
                        help="available options : DeepLab/DRN")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--iter-size", type=int, default=ITER_SIZE,
                        help="Accumulate gradients for ITER_SIZE iterations.")
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS,
                        help="number of workers for multithread dataloading.")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the PASCAL VOC dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--partial-data", type=float, default=PARTIAL_DATA,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--partial-id", type=str, default=None,
                        help="restore partial id list")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of images.")
    parser.add_argument("--is-training", action="store_true",
                        help="Whether to updates the running means and variances during the training.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--learning-rate-D", type=float, default=LEARNING_RATE_D,
                        help="Base learning rate for discriminator.")
    parser.add_argument("--lambda-adv-pred", type=float, default=LAMBDA_ADV_PRED,
                        help="lambda_adv for adversarial training.")
    parser.add_argument("--lambda-semi", type=float, default=LAMBDA_SEMI,
                        help="lambda_semi for adversarial training.")
    parser.add_argument("--lambda-semi-adv", type=float, default=LAMBDA_SEMI_ADV,
                        help="lambda_semi for adversarial training.")
    parser.add_argument("--mask-T", type=float, default=MASK_T,
                        help="mask T for semi adversarial training.")
    parser.add_argument("--semi-start", type=int, default=SEMI_START,
                        help="start semi learning after # iterations")
    parser.add_argument("--semi-start-adv", type=int, default=SEMI_START_ADV,
                        help="start semi learning after # iterations")
    parser.add_argument("--D-remain", type=bool, default=D_REMAIN,
                        help="Whether to train D with unlabeled data")
    parser.add_argument("--momentum", type=float, default=MOMENTUM,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--not-restore-last", action="store_true",
                        help="Whether to not restore last (FC) layers.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS,
                        help="Number of training steps.")
    parser.add_argument("--power", type=float, default=POWER,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--random-mirror", action="store_true",
                        help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--random-scale", action="store_true",
                        help="Whether to randomly scale the inputs during the training.")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED,
                        help="Random seed to have reproducible results.")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--restore-from-D", type=str, default=None,
                        help="Where restore model parameters from.")
    parser.add_argument("--save-num-images", type=int, default=SAVE_NUM_IMAGES,
                        help="How many images to save.")
    parser.add_argument("--save-pred-every", type=int, default=SAVE_PRED_EVERY,
                        help="Save summaries and checkpoint every often.")
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument("--gpu", type=int, default=0,
                        help="choose gpu device.")
    return parser.parse_args()

args = get_arguments() # 获取参数

def loss_calc(pred, label, gpu):
    """
    This function returns cross entropy loss for semantic segmentation
    这个函数返回语义分割的交叉熵损失
    """
    # out shape batch_size x channels x h x w -> batch_size x channels x h x w
    # label shape h x w x 1 x batch_size  -> batch_size x 1 x h x w
    label = Variable(label.long()).cuda(gpu) # 将标签转换为Variable并移到GPU上
    criterion = CrossEntropy2d().cuda(gpu) # 定义交叉熵损失函数并移到GPU上

    return criterion(pred, label) # 返回损失值


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr*((1-float(iter)/max_iter)**(power)) # 计算学习率


def adjust_learning_rate(optimizer, i_iter):
    lr = lr_poly(args.learning_rate, i_iter, args.num_steps, args.power) # 计算学习率
    optimizer.param_groups[0]['lr'] = lr # 设置优化器的学习率
    if len(optimizer.param_groups) > 1 :
        optimizer.param_groups[1]['lr'] = lr * 10 # 如果优化器的参数组大于1，则设置第二个参数组的学习率为10倍

def adjust_learning_rate_D(optimizer, i_iter):
    lr = lr_poly(args.learning_rate_D, i_iter, args.num_steps, args.power) # 计算判别器的学习率
    optimizer.param_groups[0]['lr'] = lr # 设置优化器的学习率
    if len(optimizer.param_groups) > 1 :
        optimizer.param_groups[1]['lr'] = lr * 10 # 如果优化器的参数组大于1，则设置第二个参数组的学习率为10倍

def one_hot(label):
    label = label.numpy() # 将标签转换为numpy数组
    one_hot = np.zeros((label.shape[0], args.num_classes, label.shape[1], label.shape[2]), dtype=label.dtype) # 创建一个全零的one_hot数组
    for i in range(args.num_classes):
        one_hot[:,i,...] = (label==i) # 将标签转换为one_hot编码
    # 处理忽略标签
    #handle ignore labels
    return torch.FloatTensor(one_hot) # 返回one_hot编码的标签

def make_D_label(label, ignore_mask):
    ignore_mask = np.expand_dims(ignore_mask, axis=1) # 在第1维度上扩展忽略掩码
    D_label = np.ones(ignore_mask.shape)*label # 创建一个全为label的数组
    D_label[ignore_mask] = 255 # 将忽略掩码处的标签设置为255
    D_label = Variable(torch.FloatTensor(D_label)).cuda(args.gpu) # 将标签转换为Variable并移到GPU上

    return D_label # 返回标签


def main():

    h, w = map(int, args.input_size.split(',')) # 将输入尺寸转换为整数
    input_size = (h, w) # 输入尺寸

    cudnn.enabled = True # 开启cudnn
    gpu = args.gpu # GPU设备

    # create network
    # 创建网络
    model = Res_Deeplab(num_classes=args.num_classes) # 创建模型

    # load pretrained parameters
    # 加载预训练参数
    if args.restore_from[:4] == 'http' :
        saved_state_dict = model_zoo.load_url(args.restore_from) # 从url加载模型参数
    else:
        saved_state_dict = torch.load(args.restore_from) # 从本地加载模型参数

    # only copy the params that exist in current model (caffe-like)
    # 只复制存在于当前模型中的参数（类似于caffe）
    new_params = model.state_dict().copy()  # 复制模型参数
    for name, param in new_params.items():
        print(name)
        if name in saved_state_dict and param.size() == saved_state_dict[name].size():
            new_params[name].copy_(saved_state_dict[name])  # 复制参数
            print('copy {}'.format(name))
    model.load_state_dict(new_params) # 加载模型参数


    model.train() # 设置模型为训练模式
    model.cuda(args.gpu) # 将模型移到GPU上

    cudnn.benchmark = True # 开启cudnn的benchmark模式

    # init D
    # 初始化D
    model_D = FCDiscriminator(num_classes=args.num_classes) # 创建判别器模型
    if args.restore_from_D is not None:
        model_D.load_state_dict(torch.load(args.restore_from_D)) # 加载判别器模型参数
    model_D.train() # 设置判别器模型为训练模式
    model_D.cuda(args.gpu) # 将判别器模型移到GPU上


    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir) # 如果快照目录不存在，则创建目录


    train_dataset = VOCDataSet(args.data_dir, args.data_list, crop_size=input_size,
                    scale=args.random_scale, mirror=args.random_mirror, mean=IMG_MEAN) # 创建训练数据集

    train_dataset_size = len(train_dataset) # 训练数据集的大小

    train_gt_dataset = VOCGTDataSet(args.data_dir, args.data_list, crop_size=input_size,
                       scale=args.random_scale, mirror=args.random_mirror, mean=IMG_MEAN) # 创建训练标签数据集

    if args.partial_data is None:
        trainloader = data.DataLoader(train_dataset,
                        batch_size=args.batch_size, shuffle=True, num_workers=5, pin_memory=True) # 创建训练数据加载器

        trainloader_gt = data.DataLoader(train_gt_dataset,
                        batch_size=args.batch_size, shuffle=True, num_workers=5, pin_memory=True) # 创建训练标签数据加载器
    else:
        #sample partial data
        # 采样部分数据
        partial_size = int(args.partial_data * train_dataset_size) # 部分数据的大小

        if args.partial_id is not None:
            train_ids = pickle.load(open(args.partial_id)) # 加载训练id
            print('loading train ids from {}'.format(args.partial_id))
        else:
            train_ids = range(train_dataset_size) # 创建训练id
            np.random.shuffle(train_ids) # 打乱训练id

        pickle.dump(train_ids, open(osp.join(args.snapshot_dir, 'train_id.pkl'), 'wb')) # 保存训练id

        train_sampler = data.sampler.SubsetRandomSampler(train_ids[:partial_size]) # 创建训练采样器
        train_remain_sampler = data.sampler.SubsetRandomSampler(train_ids[partial_size:]) # 创建剩余训练采样器
        train_gt_sampler = data.sampler.SubsetRandomSampler(train_ids[:partial_size]) # 创建训练标签采样器

        trainloader = data.DataLoader(train_dataset,
                        batch_size=args.batch_size, sampler=train_sampler, num_workers=3, pin_memory=True) # 创建训练数据加载器
        trainloader_remain = data.DataLoader(train_dataset,
                        batch_size=args.batch_size, sampler=train_remain_sampler, num_workers=3, pin_memory=True) # 创建剩余训练数据加载器
        trainloader_gt = data.DataLoader(train_gt_dataset,
                        batch_size=args.batch_size, sampler=train_gt_sampler, num_workers=3, pin_memory=True) # 创建训练标签数据加载器

        trainloader_remain_iter = enumerate(trainloader_remain) # 创建剩余训练数据迭代器


    trainloader_iter = enumerate(trainloader) # 创建训练数据迭代器
    trainloader_gt_iter = enumerate(trainloader_gt) # 创建训练标签数据迭代器


    # implement model.optim_parameters(args) to handle different models' lr setting
    # 实现model.optim_parameters(args)来处理不同模型的学习率设置

    # optimizer for segmentation network
    # 分割网络的优化器
    optimizer = optim.SGD(model.optim_parameters(args),
                lr=args.learning_rate, momentum=args.momentum,weight_decay=args.weight_decay) # 创建优化器
    optimizer.zero_grad() # 清空优化器的梯度

    # optimizer for discriminator network
    optimizer_D = optim.Adam(model_D.parameters(), lr=args.learning_rate_D, betas=(0.9,0.99)) # 创建优化器
    optimizer_D.zero_grad() # 清空优化器的梯度

    # loss/ bilinear upsampling
    # 损失/ 双线性上采样
    bce_loss = BCEWithLogitsLoss2d() # 创建二元交叉熵损失函数
    interp = nn.Upsample(size=(input_size[1], input_size[0]), mode='bilinear') # 创建双线性上采样层

    if version.parse(torch.__version__) >= version.parse('0.4.0'):
        interp = nn.Upsample(size=(input_size[1], input_size[0]), mode='bilinear', align_corners=True) # 创建双线性上采样层
    else:
        interp = nn.Upsample(size=(input_size[1], input_size[0]), mode='bilinear') # 创建双线性上采样层


    # labels for adversarial training
    # 对抗训练的标签
    pred_label = 0 # 预测标签
    gt_label = 1 # 真实标签


    for i_iter in range(args.num_steps): # 对训练步骤进行循环
        # 初始化各种损失值
        loss_seg_value = 0
        loss_adv_pred_value = 0
        loss_D_value = 0
        loss_semi_value = 0
        loss_semi_adv_value = 0

        optimizer.zero_grad() # 清空优化器的梯度
        adjust_learning_rate(optimizer, i_iter) # 调整学习率
        optimizer_D.zero_grad() # 清空优化器D的梯度
        adjust_learning_rate_D(optimizer_D, i_iter) # 调整优化器D的学习率

        for sub_i in range(args.iter_size): # 对每个小批量进行循环

            # train G

            # don't accumulate grads in D
            # 训练生成器G

            # 不在D中累积梯度
            for param in model_D.parameters():
                param.requires_grad = False

            # do semi first
            # 首先进行半监督学习
            if (args.lambda_semi > 0 or args.lambda_semi_adv > 0 ) and i_iter >= args.semi_start_adv :
                try:
                    _, batch = trainloader_remain_iter.next()
                except:
                    trainloader_remain_iter = enumerate(trainloader_remain)
                    _, batch = trainloader_remain_iter.next()

                # only access to img
                # 只访问图像
                images, _, _, _ = batch
                images = Variable(images).cuda(args.gpu)

                # 预测
                pred = interp(model(images))
                pred_remain = pred.detach()

                # 计算损失
                D_out = interp(model_D(F.softmax(pred)))
                D_out_sigmoid = F.sigmoid(D_out).data.cpu().numpy().squeeze(axis=1)

                ignore_mask_remain = np.zeros(D_out_sigmoid.shape).astype(np.bool)

                loss_semi_adv = args.lambda_semi_adv * bce_loss(D_out, make_D_label(gt_label, ignore_mask_remain))
                loss_semi_adv = loss_semi_adv/args.iter_size

                #loss_semi_adv.backward()
                loss_semi_adv_value += loss_semi_adv.data.cpu().numpy()[0]/args.lambda_semi_adv

                if args.lambda_semi <= 0 or i_iter < args.semi_start:
                    loss_semi_adv.backward()
                    loss_semi_value = 0
                else:
                    # produce ignore mask
                    # 生成忽略掩码
                    semi_ignore_mask = (D_out_sigmoid < args.mask_T)

                    semi_gt = pred.data.cpu().numpy().argmax(axis=1)
                    semi_gt[semi_ignore_mask] = 255

                    semi_ratio = 1.0 - float(semi_ignore_mask.sum())/semi_ignore_mask.size
                    print('semi ratio: {:.4f}'.format(semi_ratio))

                    if semi_ratio == 0.0:
                        loss_semi_value += 0
                    else:
                        semi_gt = torch.FloatTensor(semi_gt)

                        loss_semi = args.lambda_semi * loss_calc(pred, semi_gt, args.gpu)
                        loss_semi = loss_semi/args.iter_size
                        loss_semi_value += loss_semi.data.cpu().numpy()[0]/args.lambda_semi
                        loss_semi += loss_semi_adv
                        loss_semi.backward()

            else:
                loss_semi = None
                loss_semi_adv = None

            # train with source
            # 使用源数据进行训练
            try:
                _, batch = trainloader_iter.next()
            except:
                trainloader_iter = enumerate(trainloader)
                _, batch = trainloader_iter.next()

            images, labels, _, _ = batch
            images = Variable(images).cuda(args.gpu)
            ignore_mask = (labels.numpy() == 255)
            pred = interp(model(images))

            loss_seg = loss_calc(pred, labels, args.gpu)

            D_out = interp(model_D(F.softmax(pred)))

            loss_adv_pred = bce_loss(D_out, make_D_label(gt_label, ignore_mask))

            loss = loss_seg + args.lambda_adv_pred * loss_adv_pred

            # proper normalization
            # 正确的归一化
            loss = loss/args.iter_size
            loss.backward()
            loss_seg_value += loss_seg.data.cpu().numpy()[0]/args.iter_size
            loss_adv_pred_value += loss_adv_pred.data.cpu().numpy()[0]/args.iter_size

            # 训练判别器D

            # 恢复requires_grad
            for param in model_D.parameters():
                param.requires_grad = True

            # train with pred
            # 使用预测值进行训练
            pred = pred.detach()

            if args.D_remain:
                pred = torch.cat((pred, pred_remain), 0)
                ignore_mask = np.concatenate((ignore_mask,ignore_mask_remain), axis = 0)

            D_out = interp(model_D(F.softmax(pred)))
            loss_D = bce_loss(D_out, make_D_label(pred_label, ignore_mask))
            loss_D = loss_D/args.iter_size/2
            loss_D.backward()
            loss_D_value += loss_D.data.cpu().numpy()[0]


            # train with gt
            # get gt labels
            # 使用真实标签进行训练
            try:
                _, batch = trainloader_gt_iter.next()
            except:
                trainloader_gt_iter = enumerate(trainloader_gt)
                _, batch = trainloader_gt_iter.next()

            _, labels_gt, _, _ = batch
            D_gt_v = Variable(one_hot(labels_gt)).cuda(args.gpu)
            ignore_mask_gt = (labels_gt.numpy() == 255)

            D_out = interp(model_D(D_gt_v))
            loss_D = bce_loss(D_out, make_D_label(gt_label, ignore_mask_gt))
            loss_D = loss_D/args.iter_size/2
            loss_D.backward()
            loss_D_value += loss_D.data.cpu().numpy()[0]



        optimizer.step()# 更新优化器的参数
        optimizer_D.step() # 更新优化器D的参数

        print('exp = {}'.format(args.snapshot_dir))
        print('iter = {0:8d}/{1:8d}, loss_seg = {2:.3f}, loss_adv_p = {3:.3f}, loss_D = {4:.3f}, loss_semi = {5:.3f}, loss_semi_adv = {6:.3f}'.format(i_iter, args.num_steps, loss_seg_value, loss_adv_pred_value, loss_D_value, loss_semi_value, loss_semi_adv_value))

        if i_iter >= args.num_steps-1:
            print('save model ...')
            torch.save(model.state_dict(),osp.join(args.snapshot_dir, 'VOC_'+str(args.num_steps)+'.pth'))
            torch.save(model_D.state_dict(),osp.join(args.snapshot_dir, 'VOC_'+str(args.num_steps)+'_D.pth'))
            break

        if i_iter % args.save_pred_every == 0 and i_iter!=0:
            print('taking snapshot ...')
            torch.save(model.state_dict(),osp.join(args.snapshot_dir, 'VOC_'+str(i_iter)+'.pth'))
            torch.save(model_D.state_dict(),osp.join(args.snapshot_dir, 'VOC_'+str(i_iter)+'_D.pth'))

    end = timeit.default_timer()
    print(end - start, 'seconds') # 打印训练所用时间


if __name__ == '__main__':
    main()
