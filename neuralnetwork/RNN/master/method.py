import time

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from model import RNN
from utils import random_training_example, category_to_tensor, input_to_tensor
from static import n_categories, n_letters, all_letters, device

n_hidden = 128


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

    torch.save(rnn.state_dict(), 'rnn_params.pkl')  # 保存模型的数据

    plt.figure()
    plt.plot(all_losses)
    plt.show()


def predict(category, start_letter):
    rnn = RNN(n_letters, n_hidden, n_letters).to(device)
    rnn.load_state_dict(torch.load('rnn_params.pkl'))  # 加载模型训练所得到的参数
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
