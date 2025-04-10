import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import time
import random
import matplotlib.pyplot as plt
from scipy.io import savemat
from sklearn import metrics

def plot_matrix(y_true, y_pred, labels_name, title=None, thresh=0.8, axis_labels=None):

    # 利用sklearn中的函数生成混淆矩阵并归一化
    cm = metrics.confusion_matrix(y_true, y_pred, labels=labels_name, sample_weight=None)  # 生成混淆矩阵
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # 归一化

    plt.figure(figsize=(10, 8))

    # 画图，如果希望改变颜色风格，可以改变此部分的cmap=pl.get_cmap('Blues')处
    plt.imshow(cm, interpolation='nearest', cmap=plt.get_cmap('Blues'))
    # plt.colorbar()  # 绘制图例

    # 图像标题
    if title is not None:
        plt.title(title)

    # 绘制坐标

    num_local = np.array(range(len(labels_name)))
    if axis_labels is None:
        axis_labels = labels_name
    plt.xticks(num_local, axis_labels, fontsize=25, fontfamily="Times New Roman")  # 将标签印在x轴坐标上， 并倾斜45度
    plt.yticks(num_local, axis_labels, fontsize=25, fontfamily="Times New Roman")  # 将标签印在y轴坐标上
    plt.ylabel('True label', labelpad=10, fontsize=30, fontfamily="Times New Roman")
    plt.xlabel('Predicted label', labelpad=10, fontsize=30, fontfamily="Times New Roman")

    font = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 12,
             }

    # 将百分比打印在相应的格子内，大于thresh的用白字，小于的用黑字
    for i in range(np.shape(cm)[0]):
        for j in range(np.shape(cm)[1]):
            a = cm[i][j] * 100
            if a > 0:
                plt.text(j, i, format(a, '.2f') + '%',
                         fontdict=font,
                         ha="center", va="center",
                         color="white" if cm[i][j] > thresh else "black")  # 如果要更改颜色风格，需要同时更改此行
            else:
                plt.text(j, i, 0,
                         fontdict=font,
                         ha="center", va="center",
                         color="white" if cm[i][j] > thresh else "black")  # 如果要更改颜色风格，需要同时更改此行

true = pd.read_csv('./result/true_3800.csv')
pre = pd.read_csv('./result/pre_2800.csv')
true = true.iloc[:, 1]
pre = pre.iloc[:, 1]
print(true.shape)
print(pre.shape)

# 根据真实标签与预测标签绘制混淆矩阵
plot_matrix(true, pre, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
plt.savefig("./image/Confusion Matrix 3800.png")
plt.show()