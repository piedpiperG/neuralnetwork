import time

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
import torch.optim as optim
from model import RNN, BiRNN
from utils import random_training_example, category_to_tensor, input_to_tensor, input_to_tensor_reverse, \
    create_training_samples, target_to_tensor, letter_to_tensor
from static import n_categories, n_letters, all_letters, device

n_hidden = 128


# 前向训练函数
def train():
    rnn = RNN(n_letters, n_hidden, n_letters).to(device)

    loss = nn.NLLLoss()
    total_loss = 0
    all_losses = []
    epoch_num = 100000
    lr = 0.0005
    epoch_start_time = time.time()

    for epoch in range(epoch_num):
        rnn.zero_grad()
        train_loss = 0
        hidden = rnn.init_hidden()
        category_tensor, input_tensor, target_tensor = random_training_example()
        target_tensor.unsqueeze_(-1)

        for i in range(input_tensor.size()[0]):
            output, hidden = rnn(category_tensor, input_tensor[i], hidden)
            train_loss += loss(output, target_tensor[i])  # 每次的loss都需要计算
        train_loss.backward()

        for p in rnn.parameters():
            p.data.add_(p.grad.data, alpha=-lr)

        total_loss += train_loss.item() / input_tensor.size()[0]  # 该名字的平均损失

        if epoch % 5000 == 0:
            print('[%05d/%03d%%] %2.2f sec(s) Loss: %.4f' %
                  (epoch, epoch / epoch_num * 100, time.time() - epoch_start_time,
                   train_loss.item() / input_tensor.size()[0]))
            epoch_start_time = time.time()

        if (epoch + 1) % 500 == 0:
            all_losses.append(total_loss / 500)
            total_loss = 0

    torch.save(rnn.state_dict(), '../model/rnn_params.pkl')  # 保存模型的数据

    plt.figure()
    plt.plot(all_losses)
    plt.show()


# 前向预测函数
def predict(category, start_letter):
    rnn = RNN(n_letters, n_hidden, n_letters).to(device)
    rnn.load_state_dict(torch.load('../model/rnn_params.pkl'))  # 加载模型训练所得到的参数
    max_length = 20  # 名字的最大长度
    with torch.no_grad():
        category_tensor = category_to_tensor(category)
        input = input_to_tensor(start_letter)
        hidden = rnn.init_hidden()

        output_name = start_letter
        top5_each_step = []  # 用于存储每步的前5个字符及其概率

        for i in range(max_length):
            output, hidden = rnn(category_tensor, input[0], hidden)
            top_v, top_i = output.topk(5)  # 选出最大的值，返回其value和index

            top5_each_step.append((top_i, top_v))

            top_i = top_i[0][0].item()
            if top_i == n_letters - 1:  # n-letters-1是EOS
                break
            else:
                letter = all_letters[top_i]
                output_name += letter
            input = input_to_tensor(letter)  # 更新input，继续循环迭代
    return output_name, top5_each_step


# 反向训练函数
def train_reverse():
    rnn = RNN(n_letters, n_hidden, n_letters).to(device)

    loss = nn.NLLLoss()
    total_loss = 0
    all_losses = []
    epoch_num = 100000
    lr = 0.0005
    epoch_start_time = time.time()

    for epoch in range(epoch_num):
        rnn.zero_grad()
        train_loss = 0
        hidden = rnn.init_hidden()

        category_tensor, input_tensor, target_tensor = random_training_example(-1)
        target_tensor.unsqueeze_(-1)

        for i in range(input_tensor.size()[0]):
            output, hidden = rnn(category_tensor, input_tensor[i], hidden)
            train_loss += loss(output, target_tensor[i])  # 计算每次的损失

        train_loss.backward()

        for p in rnn.parameters():
            p.data.add_(p.grad.data, alpha=-lr)

        total_loss += train_loss.item() / input_tensor.size()[0]  # 计算该名字的平均损失

        if epoch % 5000 == 0:
            print('[%05d/%03d%%] %2.2f sec(s) Loss: %.4f' %
                  (epoch, epoch / epoch_num * 100, time.time() - epoch_start_time,
                   train_loss.item() / input_tensor.size()[0]))
            epoch_start_time = time.time()

        if (epoch + 1) % 500 == 0:
            all_losses.append(total_loss / 500)
            total_loss = 0

    torch.save(rnn.state_dict(), '../model/rnn_params_reverse.pkl')  # 保存反向训练的模型数据

    plt.figure()
    plt.plot(all_losses)
    plt.show()


# 反向预测函数
def predict_reverse(category, end_letters):
    rnn = RNN(n_letters, n_hidden, n_letters).to(device)
    rnn.load_state_dict(torch.load('../model/rnn_params_reverse.pkl'))  # 加载反向训练的模型
    max_length = 20

    with torch.no_grad():
        category_tensor = category_to_tensor(category)
        input = input_to_tensor_reverse(end_letters)
        hidden = rnn.init_hidden()

        output_name = end_letters[::-1]
        top5_each_step = []  # 用于存储每步的前5个字符及其概率

        for i in range(max_length - len(end_letters)):
            output, hidden = rnn(category_tensor, input[0], hidden)
            topv, topi = output.topk(5)
            top5_each_step.append((topi, topv))

            topi = topi[0][0]
            if topi == n_letters - 1:
                break
            else:
                letter = all_letters[topi]
                output_name += letter
            input = input_to_tensor_reverse(letter)

        return output_name[::-1], top5_each_step


# 部分训练函数
def train_partial():
    # 初始化模型、损失函数和优化器
    model = BiRNN(n_letters, 128, n_letters).to(device)
    loss = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.005)

    # 训练参数
    total_loss = 0
    all_losses = []
    n_iters = 100000
    print_every = 5000
    plot_every = 1000
    current_loss = 0

    # 记录训练开始时间
    epoch_start_time = time.time()

    for epoch in range(1, n_iters + 1):
        train_loss = 0
        line = random_training_example(-2)
        samples = create_training_samples(line)

        for partial_name, full_name in samples:
            input_tensor = input_to_tensor(partial_name).to(device)
            input_tensor = input_tensor.type(torch.LongTensor).to(device)  # 类型转换为 LongTensor 并移至设备
            target_tensor = target_to_tensor(full_name).to(device)
            target_tensor = target_tensor.to(device)  # 确保target_tensor已经是 LongTensor
            # target_tensor = target_tensor.squeeze()  # 将多余的维度去除，保留 1 维
            target_tensor = target_tensor.unsqueeze(-1)

            # 单次训练迭代
            model.train()
            optimizer.zero_grad()
            output = model(input_tensor)
            # 使用 torch.squeeze() 去除中间的维度
            output = output[:, 0, :]

            print(f'output:{output.size()}')
            print(f'target:{target_tensor.size()}')
            for i in range(input_tensor.size()[0]):
                print(target_tensor[i].size())
                train_loss += loss(output, target_tensor[i])

            train_loss.backward()
            optimizer.step()
            total_loss += train_loss

        if epoch % print_every == 0:
            print(
                f'[{iter}/{n_iters}] {time.time() - epoch_start_time:.2f} sec(s) Loss: {total_loss / print_every:.4f}')
            total_loss = 0

        if epoch % plot_every == 0:
            all_losses.append(current_loss / plot_every)
            current_loss = 0
    # 保存模型
    torch.save(model.state_dict(), '../model/rnn_params_partial.pkl')

    # 绘制损失曲线
    plt.figure()
    plt.plot(all_losses)
    plt.show()


# 部分预测函数
def predict_partial(category, start_letters):
    max_length = 20
    # 加载双向RNN模型
    model = BiRNN(n_letters, 128, n_letters).to(device)
    model.load_state_dict(torch.load('../model/rnn_params_partial.pkl'))
    model.eval()

    with torch.no_grad():
        category_tensor = category_to_tensor(category).to(device)
        input_tensor = input_to_tensor(start_letters).to(device)

        output_name = start_letters
        top5_each_step = []  # 存储每步的前5个字符及其概率

        for i in range(max_length):
            output = model(input_tensor)
            topv, topi = output[-1].topk(5)  # 选择最大的5个值

            top5 = [(all_letters[idx], prob.item()) for idx, prob in zip(topi[0], topv[0])]
            top5_each_step.append(top5)

            topi = topi[0][0]  # 选择概率最高的字符
            if topi == n_letters - 1:  # EOS标记
                break
            else:
                letter = all_letters[topi]
                output_name += letter
                input_tensor = torch.cat((input_tensor, letter_to_tensor(letter).to(device)), 0)

        return output_name, top5_each_step


def plot_predictions(top5_each_step):
    # 设置图表大小和标题
    plt.figure(figsize=(22, 12))
    plt.title("Top 5 Predictions at Each Step")

    # 遍历每个时间步
    for step, (topi, topv) in enumerate(top5_each_step):
        # 获取前5个字符及其概率
        characters = []
        for i in topi[0]:
            if i.item() == 58:
                characters.append('EOF')
            else:
                characters.append(all_letters[i])
        # characters = [all_letters[i] for i in topi[0]]
        values = topv[0].exp()  # 将LogSoftmax转换回概率

        # 创建条形图
        plt.subplot(len(top5_each_step), 1, step + 1)
        bars = plt.bar(characters, values.cpu(), align='center', alpha=0.7)
        plt.ylim(0, 1)  # 设置y轴范围
        plt.xticks(characters)
        plt.ylabel('Probability')

        # 为每个条形图添加数值标签
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.01, round(yval, 2), ha='center', va='bottom')

    # 调整布局
    plt.tight_layout()
    plt.show()
