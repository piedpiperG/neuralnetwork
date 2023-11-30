import torch
import torch.nn as nn
from static import n_categories, n_letters, all_letters, device


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(n_categories + input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(n_categories + input_size + hidden_size, output_size)
        self.o2o = nn.Linear(hidden_size + output_size, output_size)
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.LogSoftmax(dim=1)  # dim=1表示对第1维度的数据进行logsoftmax操作

    def forward(self, category, input, hidden):
        input_tmp = torch.cat((category, input, hidden), 1)
        hidden = self.i2h(input_tmp)
        output = self.i2o(input_tmp)
        output_tmp = torch.cat((hidden, output), 1)
        output = self.o2o(output_tmp)
        output = self.dropout(output)
        output = self.softmax(output)
        return output, hidden

    def init_hidden(self):  # 隐藏层初始化0操作
        return torch.zeros(1, self.hidden_size).to(device)
