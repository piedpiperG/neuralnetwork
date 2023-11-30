import os
import glob
import random
import string
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# 1.准备工作

# 创建存储各个类别的名字的字典
all_kinds_names = {}
all_kinds = []
all_files = glob.glob('data/*.txt')
for f in all_files:
    kind = os.path.splitext(os.path.basename(f))[0]
    all_kinds.append(kind)
    one_kind_names = open(f, encoding='utf-8').read().strip().split('\n')
    all_kinds_names[kind] = one_kind_names

# 名字类别数
num_of_all_kinds = len(all_kinds)

# 所有字符数=特殊字符数+大小写英文字符数+EOS结束标记
all_letters = string.ascii_letters + " .,;'-"
num_of_all_letters = len(all_letters) + 1


# 随机获取(类别, 该类别的名字)对儿,并将对儿转换为所需要的(类别, 输入, 目标)格式张量
def random_train():
    kind = random.choice(all_kinds)
    name = random.choice(all_kinds_names[kind])

    # 类别张量
    kind_tensor = torch.zeros(1, num_of_all_kinds)
    kind_tensor[0][all_kinds.index(kind)] = 1

    # 输入名字张量
    input_name_tensor = torch.zeros(len(name), 1, num_of_all_letters)
    for i in range(len(name)):
        letter = name[i]
        letter_index = all_letters.find(letter)
        if letter_index != -1:
            input_name_tensor[i][0][letter_index] = 1
        else:
            input_name_tensor[i][0][num_of_all_letters - 1] = 1  # 处理找不到字符的情况

    # 目标名字张量
    letter_indexes = []
    for j in range(1, len(name)):
        letter_index = all_letters.find(name[j])
        if letter_index != -1:
            letter_indexes.append(letter_index)
        else:
            letter_indexes.append(num_of_all_letters - 1)  # 处理找不到字符的情况
    letter_indexes.append(num_of_all_letters - 1)  # EOS标记
    target_name_tensor = torch.LongTensor(letter_indexes)

    return kind_tensor, input_name_tensor, target_name_tensor



# 2.构造神经网络

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(num_of_all_kinds + input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(num_of_all_kinds + input_size + hidden_size, output_size)
        self.o2o = nn.Linear(hidden_size + output_size, output_size)
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, category, input, hidden):
        input_combined = torch.cat((category, input, hidden), 1)
        hidden = self.i2h(input_combined)
        output = self.i2o(input_combined)
        output_combined = torch.cat((hidden, output), 1)
        output = self.o2o(output_combined)
        output = self.dropout(output)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)


# 3.训练神经网络

rnn = RNN(num_of_all_letters, 128, num_of_all_letters)
losses = 0
L_loss = []
criterion = nn.NLLLoss()
learning_rate = 0.0005
# optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)


def train(kind_tensor, input_name_tensor, target_name_tensor):
    hidden = rnn.initHidden()
    rnn.zero_grad()
    target_name_tensor.unsqueeze_(-1)
    loss = 0
    for i in range(input_name_tensor.size(0)):
        output, hidden = rnn(kind_tensor, input_name_tensor[i], hidden)
        loss += criterion(output, target_name_tensor[i])
    # loss = torch.tensor(float(loss), requires_grad=True)
    loss.backward()
    for j in rnn.parameters():
        j.data.add_(j.grad.data, alpha=-learning_rate)  # 修改弃用的方法
    # optimizer.step()
    return output, loss.item()/input_name_tensor.size(0)


for i in range(1, 100001):
    output, loss = train(*random_train())
    losses += loss
    if(i % 5000 == 0):
        print('\n Waiting... {}%\tloss:{}'.format((i/1000), round(loss, 5)))
    if(i % 500 == 0):
        L_loss.append(round(losses/500, 5))
        losses = 0

# 4.损失数据作图反应神经网络学习情况(打印在代码末段执行

plt.figure()
plt.plot(L_loss)

# 5.预测名字

print('\n\n名字的类数及类别：{}\t{}\n'.format(num_of_all_kinds, all_kinds))


def predict(kind, first='A'):
    with torch.no_grad():
        kind_tensor = torch.zeros(1, num_of_all_kinds)
        kind_tensor[0][all_kinds.index(kind)] = 1
        input = torch.zeros(len(first), 1, num_of_all_letters)
        input[0][0][all_letters.find(first[0])] = 1
        hidden = rnn.initHidden()
        predict_name = first
        for i in range(7):
            output, hidden = rnn(kind_tensor, input[0], hidden)
            tv, ti = output.topk(1)
            # if(i == 0):
            #     print('\ntv:{}\nti:{}'.format(tv, ti))
            t = ti[0][0]
            if(t == num_of_all_letters - 1):
                break
            else:
                predict_name += all_letters[t]
            input = torch.zeros(len(first), 1, num_of_all_letters)
            input[0][0][all_letters.find(first[0])] = 1
        return predict_name


# 6.测试及打印结果

# 预测男孩的名字,首字母为L
first_letter = 'G'
which_kind = 'male'
print('预测首字母为{}的{}名字'.format(first_letter, which_kind))
print('\n 男生名字为：{}\n'.format(predict(which_kind, first_letter)))
# 预测女孩的名字,首字母为R
first_letter = 'M'
which_kind = 'female'
print('预测首字母为{}的{}名字'.format(first_letter, which_kind))
print('\n 女生名字为：{}\n'.format(predict(which_kind, first_letter)))

plt.show()
