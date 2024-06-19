import os, sys
import numpy as np
import cv2
from multiprocessing import Pool 
import copyreg
import types
import argparse

# 定义一个用于序列化的方法
def _pickle_method(m):
    if m.im_self is None:
        return getattr, (m.im_class, m.im_func.func_name)
    else:
        return getattr, (m.im_self, m.im_func.func_name)

copyreg.pickle(types.MethodType, _pickle_method)

# 定义一个混淆矩阵类
class ConfusionMatrix(object):

    def __init__(self, nclass, classes=None):
        self.nclass = nclass  # 类别数量
        self.classes = classes  # 类别名称
        self.M = np.zeros((nclass, nclass))  # 初始化混淆矩阵

        # 添加预测结果到混淆矩阵
        assert(np.max(pred) <= self.nclass)
        assert(len(gt) == len(pred))
        for i in range(len(gt)):
            if not gt[i] == 255: # 忽略标签为255的像素
                self.M[gt[i], pred[i]] += 1.0

    # 添加其他矩阵
    def addM(self, matrix):
        assert(matrix.shape == self.M.shape)
        self.M += matrix

    def __str__(self):
        pass

    # 计算召回率
    def recall(self):
        recall = 0.0
        for i in range(self.nclass):
            recall += self.M[i, i] / np.sum(self.M[:, i])

        return recall/self.nclass

    # 计算精度
    def accuracy(self):
        accuracy = 0.0
        for i in range(self.nclass):
            accuracy += self.M[i, i] / np.sum(self.M[i, :])

        return accuracy/self.nclass

    # 计算jaccard指数
    def jaccard(self):
        jaccard = 0.0
        jaccard_perclass = []
        for i in range(self.nclass):
            if not self.M[i, i] == 0:
                jaccard_perclass.append(self.M[i, i] / (np.sum(self.M[i, :]) + np.sum(self.M[:, i]) - self.M[i, i]))

        return np.sum(jaccard_perclass)/len(jaccard_perclass), jaccard_perclass, self.M

    # 生成混淆矩阵
    def generateM(self, item):
        gt, pred = item
        m = np.zeros((self.nclass, self.nclass))
        assert(len(gt) == len(pred))
        for i in range(len(gt)):
            if gt[i] < self.nclass: #and pred[i] < self.nclass: # 忽略超出类别范围的标签
                m[gt[i], pred[i]] += 1.0
        return m


if __name__ == '__main__':
    args = parse_args()

    m_list = []
    data_list = []
    test_ids = [i.strip() for i in open(args.test_ids) if not i.strip() == '']
    for index, img_id in enumerate(test_ids):
        if index % 100 == 0:
            print('%d processd'%(index))
        pred_img_path = os.path.join(args.pred_dir, img_id+'.png')
        gt_img_path = os.path.join(args.gt_dir, img_id+'.png')
        pred = cv2.imread(pred_img_path, cv2.IMREAD_GRAYSCALE)  # 读取预测结果图像
        gt = cv2.imread(gt_img_path, cv2.IMREAD_GRAYSCALE)  # 读取标签图像
        data_list.append([gt.flatten(), pred.flatten()])  # 展平并添加到数据列

    ConfM = ConfusionMatrix(args.class_num)  # 创建混淆矩阵实例
    f = ConfM.generateM
    pool = Pool()  # 创建进程池
    m_list = pool.map(f, data_list)  # 使用多进程计算混淆矩阵
    pool.close() 
    pool.join() 
    
    for m in m_list:
        ConfM.addM(m) # 合并所有的混淆矩阵

    aveJ, j_list, M = ConfM.jaccard()  # 计算平均jaccard指数
    with open(args.save_path, 'w') as f:  # 将结果保存到文件
        f.write('meanIOU: ' + str(aveJ) + '\n')
        f.write(str(j_list)+'\n')
        f.write(str(M)+'\n')
