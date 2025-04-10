# 只有残差网络和普通对抗网络
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from model7 import *

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
C_optimizer = optim.Adam(Class.parameters(), lr=0.01)
D_optimizer = optim.Adam(Discr.parameters(), lr=0.01)
d_optimizer = optim.Adam(D.parameters(), lr=0.001)
g_s_optimizer = optim.Adam(G_S.parameters(), lr=0.001)
g_t_optimizer = optim.Adam(G_T.parameters(), lr=0.001)
best_acc = 0.0
# 记录训练的次数
total_train__step = 0
# 记录测试的次数
total_test_step = 0
# 训练的轮数
num_epochs = 50
z_dimension = 10


# 约束函数
def my_func(q):
    x = -10 * q
    y = torch.exp(x)
    z = 2 / (1 + y) - 1
    return z


q = torch.tensor([0.2])
a = my_func(q).to(device)
r = torch.tensor([0.2])
b = my_func(r).to(device)

# 添加tensorboard
# writer = SummaryWriter("../logs_train")
for epoch in range(num_epochs):
    print("-----第{}轮训练开始-----".format(epoch + 1))
    running_loss = 0.0
    # 开始训练步骤

    for source_data, target_data in zip(source_dl, target_dl):
        source = source_data['input']  # 获取输入张量
        source = source.to(torch.float32)
        s_label = source_data['label']
        s_label = s_label.to(device)
        target = target_data['input']  # 获取目标张量
        target = target.to(torch.float32)
        t_label = target_data['label']
        t_label = t_label.to(device)
        target_features = feature(target).to(device)
        source_features = feature(source).to(device)

        num_img = source_features.size(0)
        for jj in range(2):
            d_optimizer.zero_grad()
            # 处理源域数据
            real_source = source_features.clone().detach().to(device).requires_grad_(True)
            real_out_source = D(real_source).to(device).requires_grad_()
            z_source = torch.randn((num_img, z_dimension), device=device)
            fake_img_source = G_S(z_source).clone().detach().to(device).requires_grad_(True)
            fake_out_source = D(fake_img_source).to(device).requires_grad_()
            # 处理目标域数据
            real_target = target_features.clone().detach().to(device).requires_grad_(True)
            real_out_target = D(real_target).to(device).requires_grad_()
            z_target = torch.randn((num_img, z_dimension), device=device)
            fake_img_target = G_T(z_target).clone().detach().to(device).requires_grad_(True)
            fake_out_target = D(fake_img_target).to(device).requires_grad_()
            # 计算梯度惩罚
            gradient_penalty_div_source = compute_w_div(real_source, real_out_source, fake_img_source,
                                                        fake_out_source)
            gradient_penalty_div_target = compute_w_div(real_target, real_out_target, fake_img_target,
                                                        fake_out_target)
            # 计算判别器损失
            d_loss = -torch.mean(real_out_source) + torch.mean(fake_out_source) + gradient_penalty_div_source
            d_loss += -torch.mean(real_out_target) + torch.mean(fake_out_target) + gradient_penalty_div_target
            d_loss.backward(retain_graph=True)
            d_optimizer.step()
            # for name, param in D.named_parameters():
            # if param.grad is not None:
            # print(f'Discriminator Gradient - {name}: {param.grad.norm().item()}')
        for jj in range(1):
            g_s_optimizer.zero_grad()
            z_s = torch.randn(num_img, z_dimension).to(device)
            fake_img_s = G_S(z_s)
            fake_out_s = D(fake_img_s)
            g_s_loss = -torch.mean(fake_out_s)
            g_s_loss.backward()
            g_s_optimizer.step()
        # for name, param in G_S.named_parameters():
        # if param.grad is not None:
        #  print(f'Generator Gradient - {name}: {param.grad.norm().item()}')

        for jj in range(1):
            g_t_optimizer.zero_grad()
            z_t = torch.randn(num_img, z_dimension).to(device)
            fake_img_t = G_T(z_t)
            fake_out_t = D(fake_img_t)
            g_t_loss = -torch.mean(fake_out_t)
            g_t_loss.backward()
            g_t_optimizer.step()
        # for name, param in G_T.named_parameters():
        # if param.grad is not None:
        #  print(f'Generator Gradient - {name}: {param.grad.norm().item()}')

        domain_discriminator_predictions, discriminator_loss = Discr(source_features, target_features)
        lmd = lmmd_loss(source_features, target_features)
        predictions, class_loss = Class(source_features, s_label)

        c = 1 - (discriminator_loss.mean() + lmd.mean()) / (discriminator_loss.mean() + lmd.mean() + d_loss.mean())
        total_loss = (1 - c) * (lmd + a * discriminator_loss) + c * b * d_loss + class_loss
        # 优化器模型
        C_optimizer.zero_grad()
        D_optimizer.zero_grad()
        total_loss.backward()
        C_optimizer.step()
        D_optimizer.step()
        running_loss += total_loss.item()
        total_train__step = total_train__step + 1
        if total_train__step % 10 == 0: print(
            "训练次数{},total_loss:{},running_loss:{}, class_loss:{}, discriminator_loss:{}, lmd:{}".format(
                total_train__step, total_loss.item(), running_loss, class_loss, discriminator_loss, lmd))
        print(
            'Epoch [{}/{}], d_loss: {:.6f}, g_s_loss: {:.6f} , g_t_loss: {:.6f} , D real S: {:.6f}, D real T: {:.6f}, D fake S: {:.6f}, D fake T: {:.6f}'
            .format(epoch, num_epochs, d_loss.data, g_s_loss.data, g_t_loss.data,
                    real_out_source.data.mean(), real_out_target.data.mean(), fake_out_s.data.mean(),
                    fake_out_t.data.mean()))

    total_valid_loss = 0.0
    total_valid_steps = 0.0
    num_correct = 0.0
    num_total = 0.0
    with torch.no_grad():
        for source_data, target_data in zip(source_eval_dl, target_eval_dl):
            source = source_data['input']  # 获取输入张量
            source = source.to(torch.float32)
            s_label = source_data['label']
            s_label = s_label.to(device)
            target = target_data['input']  # 获取目标张量
            target = target.to(torch.float32)
            t_label = target_data['label']
            t_label = t_label.to(device)
            target_features = feature(target).to(device)
            source_features = feature(source).to(device)

            num_img = source_features.size(0)
            for jj in range(1):
                d_optimizer.zero_grad()
                # 处理源域数据
                real_source = source_features.clone().detach().to(device).requires_grad_(True)
                real_out_source = D(real_source).to(device).requires_grad_()
                z_source = torch.randn((num_img, z_dimension), device=device)
                fake_img_source = G_S(z_source).clone().detach().to(device).requires_grad_(True)
                fake_out_source = D(fake_img_source).to(device).requires_grad_()
                # 处理目标域数据
                real_target = target_features.clone().detach().to(device).requires_grad_(True)
                real_out_target = D(real_target).to(device).requires_grad_()
                z_target = torch.randn((num_img, z_dimension), device=device)
                fake_img_target = G_T(z_target).clone().detach().to(device).requires_grad_(True)
                fake_out_target = D(fake_img_target).to(device).requires_grad_()
                # 计算梯度惩罚
                # gradient_penalty_div_source = compute_w_div(real_source, real_out_source, fake_img_source,
                # fake_out_source)
                # gradient_penalty_div_target = compute_w_div(real_target, real_out_target, fake_img_target,
                # fake_out_target)
                # 计算判别器损失
                d_loss = -torch.mean(real_out_source) + torch.mean(fake_out_source)
                d_loss += -torch.mean(real_out_target) + torch.mean(fake_out_target)
                # d_loss.backward()
                d_optimizer.step()
                # for name, param in D.named_parameters():
                # if param.grad is not None:
                # print(f'Discriminator Gradient - {name}: {param.grad.norm().item()}')
            for jj in range(1):
                g_s_optimizer.zero_grad()
                z_s = torch.randn(num_img, z_dimension).to(device)
                fake_img_s = G_S(z_s)
                fake_out_s = D(fake_img_s)
                g_s_loss = -torch.mean(fake_out_s)
                # g_s_loss.backward()
                g_s_optimizer.step()
            # for name, param in G_S.named_parameters():
            # if param.grad is not None:
            #  print(f'Generator Gradient - {name}: {param.grad.norm().item()}')

            for jj in range(1):
                g_t_optimizer.zero_grad()
                z_t = torch.randn(num_img, z_dimension).to(device)
                fake_img_t = G_T(z_t)
                fake_out_t = D(fake_img_t)
                g_t_loss = -torch.mean(fake_out_t)
                # g_t_loss.backward()
                g_t_optimizer.step()

            domain_discriminator_predictions, discriminator_loss = Discr(source_features, target_features)
            lmd = lmmd_loss(source_features, target_features)
            predictions, class_loss = Class(target_features, t_label)

            c = 1 - (discriminator_loss.mean() + lmd.mean()) / (
                    discriminator_loss.mean() + lmd.mean() + d_loss.mean())
            total_loss = (1 - c) * (lmd + a * discriminator_loss) + c * b * d_loss + class_loss
            predicted = torch.argmax(predictions, dim=1)
            num_total += t_label.size(0)
            num_correct += (predicted == t_label.to(device)).sum().item()
            accuracy = num_correct / num_total * 100
            print("验证集准确率: {}".format(accuracy))
