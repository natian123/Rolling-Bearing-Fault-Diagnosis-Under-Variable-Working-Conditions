import itertools

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms
from sklearn.decomposition import PCA
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd
from HITpreprocess import DataSet,TYPE


# 指定运算设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ----------------- 加载数据集 ------------------------
# 加载训练集
dataset_train = DataSet(dataset_path='D:\\pyxiangmu\\natian\\data\\轴承2800', type=TYPE.TRAIN )
source_dl = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=128, pin_memory=True, shuffle=True, num_workers=0)
# 加载验证集
dataset_eval = DataSet(dataset_path='D:\\pyxiangmu\\natian\\data\\轴承2800', type=TYPE.EVAL )
target_dl = torch.utils.data.DataLoader(dataset=dataset_eval, batch_size=128, pin_memory=True, shuffle=False, num_workers=0)

dataset_test = DataSet(dataset_path='D:\\pyxiangmu\\natian\\data\\轴承2800', type=TYPE.TEST )
test_dl = torch.utils.data.DataLoader(dataset=dataset_test, batch_size=128, pin_memory=True, shuffle=False, num_workers=0)

# 定义残差网络
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 用于调整维度的卷积层（如果输入和输出通道数不匹配）
        self.adjust_dimensions = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        # 如果输入和输出通道数不匹配，应用维度调整
        if x.size(1) != out.size(1):
            residual = self.adjust_dimensions(x)

        out += residual
        out = self.relu(out)
        return out

# 创建残差网络特征提取器
class FeatureExtractor(nn.Module):
    def __init__(self, num_blocks):
        super(FeatureExtractor, self).__init__()
        self.in_channels = 1
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU()
        self.layer1 = self.make_layer(32, num_blocks)

    def make_layer(self, out_channels, num_blocks):
        layers = []
        for _ in range(num_blocks):
            self.in_channels = out_channels
            layers.append(ResidualBlock(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        # 在这里添加形状调整，将输入从 (batch_size, channels, height, width) 调整为 (batch_size, 1, height, width)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.layer1(out)
        return out

class feature_extractor(nn.Module):
    def __init__(self):
        super(feature_extractor, self).__init__()
        # 定义特征提取器
        self.feature_extractor = FeatureExtractor(num_blocks=2)
    def forward(self, input):
        features = self.feature_extractor(input)
        return features

class WGAN_G(nn.Module):
    def __init__(self, input_size):
        super(WGAN_G, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return x

# 定义判别器
class WGAN_D(nn.Module):
    def __init__(self, input_size=32):
        super(WGAN_D, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

def compute_w_div(real_samples, real_out, fake_samples, fake_out):
    # 定义参数
    k = 2
    p = 6
    # 计算真实空间的梯度
    # weight = torch.full((real_samples.size(0),), 1, device=device)
    real_out = real_out.view(real_out.shape[0], -1)
    fake_out = fake_out.view(fake_out.shape[0], -1)
    real_grad = autograd.grad(outputs=real_out,
                              inputs=real_samples,
                              grad_outputs=torch.ones_like(real_out),
                              create_graph=True,
                              retain_graph=True,
                              only_inputs=True, allow_unused=True)[0]
    if real_grad is not None:
        # L2范数,将真实梯度向量展平为一行，然后计算每一行的元素的平方和，也就是计算真实梯度的范数。
        real_grad_norm = real_grad.view(real_grad.size(0), -1).pow(2).sum(1)
    else:
        real_grad_norm = 0
    # 计算模拟空间的梯度
    fake_grad = autograd.grad(outputs=fake_out,
                              inputs=fake_samples,
                              grad_outputs=torch.ones_like(fake_out),
                              create_graph=True,
                              retain_graph=True,
                              only_inputs=True, allow_unused=True)[0]
    if fake_grad is not None:
        # L2范数
        fake_grad_norm = fake_grad.view(fake_grad.size(0), -1).pow(2).sum(1)
    else:
        fake_grad_norm = 0
    # 计算W散度距离
    div_gp = torch.mean(real_grad_norm ** (p / 2) + fake_grad_norm ** (p / 2)) * k / 2
    return div_gp

# LMMD
def gaussian_kernel(x1, x2, sigma):
    return torch.exp(-torch.norm(x1 - x2) ** 2 / (2 * sigma ** 2))
def lmmd_loss(source_features, target_features, alpha=0.2, sigma=1.0, num_local_regions=10):
    n = source_features.size(0)
    m = target_features.size(0)
    n_local = n // num_local_regions
    m_local = m // num_local_regions
    lmmd = 0.0
    for i in range(num_local_regions):
        for j in range(num_local_regions):
            source_local = source_features[i * n_local: (i + 1) * n_local]
            target_local = target_features[j * m_local: (j + 1) * m_local]
            Kx = torch.zeros((n_local, n_local))
            Ky = torch.zeros((m_local, m_local))
            for p in range(n_local):
                for q in range(n_local):
                    Kx[p, q] = gaussian_kernel(source_local[p], source_local[q], sigma)
            for p in range(m_local):
                for q in range(m_local):
                    Ky[p, q] = gaussian_kernel(target_local[p], target_local[q], sigma)
            Kxy = torch.zeros((n_local, m_local))
            for p in range(n_local):
                for q in range(m_local):
                    Kxy[p, q] = gaussian_kernel(source_local[p], target_local[q], sigma)

            if n_local != 0 and m_local != 0:
                lmmd_local = (1.0 / (n_local * (n_local - 1))) * Kx.sum() - (2.0 / (n_local * m_local)) * Kxy.sum() + (
                            1.0 / (m_local * (m_local - 1))) * Ky.sum()

            lmmd += lmmd_local
    lmmd /= (num_local_regions * num_local_regions)
    return lmmd


class Domain_adaptation(nn.Module):
    def __init__(self):
        super(Domain_adaptation, self).__init__()

        # 定义领域鉴别器
        self.discriminator = nn.Sequential(
            nn.Linear(32 * 32 * 32, 128),  # 更新输入维度
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Sigmoid()
        )
        # 定义故障分类器
        self.fault_classifier = nn.Sequential(
            nn.Linear(32 * 32 * 32, 128),  # 更新输入维度
            #nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(128, 32),
            #nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(32, 3),  # 更新输出类别数量
        )
        # L2正则化项
        l2_lambda = 0.01  # 超参数，用于控制正则化强度
        self.l2_lambda = l2_lambda

    def forward(self, source_features, target_features, label):
        # 域适应
        mmd_loss = lmmd_loss(source_features, target_features)
        # 利用鉴别器预测目标域故障类型
        source_feature = source_features.view(source_features.size(0), -1)
        target_feature = target_features.view(target_features.size(0), -1)
        domain_features = torch.cat((source_feature, target_feature), 0)
        domain_discriminator_predictions = self.discriminator(domain_features)
        domain_classifier_predictions = self.fault_classifier(target_feature)
        discriminator_loss = F.binary_cross_entropy_with_logits(domain_discriminator_predictions,
                                                                torch.zeros_like(domain_discriminator_predictions))
        class_loss = F.cross_entropy(domain_classifier_predictions, label)

        # 计算L2正则化损失
        l2_loss = 0.0
        for param in self.fault_classifier.parameters():
            l2_loss += torch.sum(param ** 2)

        # 计算总损失
        classifier_loss = class_loss + self.l2_lambda * l2_loss
        loss = discriminator_loss + classifier_loss
        return domain_classifier_predictions, classifier_loss, discriminator_loss, loss, mmd_loss

def plot_confusion_matrix(confusion, classes):
            plt.imshow(confusion, interpolation='nearest', cmap=plt.cm.Blues)
            plt.title("Confusion Matrix")
            plt.colorbar()
            tick_marks = np.arange(len(classes))
            plt.xticks(tick_marks, classes, rotation=45)
            plt.yticks(tick_marks, classes)

            fmt = 'd'
            thresh = confusion.max() / 2.0
            for i, j in itertools.product(range(confusion.shape[0]), range(confusion.shape[1])):
                plt.text(j, i, format(confusion[i, j], fmt), horizontalalignment="center",
                         color="white" if confusion[i, j] > thresh else "black")

            plt.tight_layout()
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.show()

# 实例化模型
if __name__ == '__main__':
    z_dimension = 10

# 创建网络模型
z_dimension = 10
Domain = Domain_adaptation()
Domain = Domain.to(device).train()
D = WGAN_D().to(device).train()
feature_extractor = feature_extractor()
G = WGAN_G(z_dimension).to(device)

params = sum([param.nelement() for param in feature_extractor.parameters()])
print("Number of parameter: %.2fM" % (params/1e6))

params = sum([param.nelement() for param in Domain.parameters()])
print("Number of parameter: %.2fM" % (params/1e6))

params = sum([param.nelement() for param in D.parameters()])
print("Number of parameter: %.2fM" % (params/1e6))

params = sum([param.nelement() for param in G.parameters()])
print("Number of parameter: %.2fM" % (params/1e6))

