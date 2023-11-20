import numpy as np


def load_word_vectors(file_name):
    word_vectors = {}
    with open(file_name, 'r', encoding='utf-8') as file:
        for line in file:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            word_vectors[word] = vector
    return word_vectors


def create_embedding_matrix(word_vectors):
    words = list(word_vectors.keys())
    embedding_matrix = np.zeros((len(words), len(word_vectors[words[0]])))
    word_to_index = {word: i for i, word in enumerate(words)}
    for word, i in word_to_index.items():
        embedding_matrix[i] = word_vectors[word]
    return embedding_matrix, word_to_index


# Load word vectors
word_vectors_20NG = load_word_vectors('word_vectors_20NG.txt')
# word_vectors_MR = load_word_vectors('word_vectors_MR.txt')

# Create embedding matrices
embedding_matrix_20NG, word_to_index_20NG = create_embedding_matrix(word_vectors_20NG)
# embedding_matrix_MR, word_to_index_MR = create_embedding_matrix(word_vectors_MR)

import torch
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split


def load_20ng_dataset(file_path):
    documents = []
    labels = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split('\t')  # 使用制表符分割每行
            if len(parts) > 1:  # 确保行有足够的部分
                document_path = parts[0]
                label = parts[-1]  # 假设最后一个部分是类别标签
                documents.append(document_path)
                labels.append(label)
    return documents, labels


# 加载数据集
file_path = '20ng.txt'  # 根据实际路径调整
documents, labels = load_20ng_dataset(file_path)

# 构建边的索引
edge_index = []
for doc in documents:
    for i in range(len(doc) - 1):
        src = word_to_index_20NG.get(doc[i])
        dest = word_to_index_20NG.get(doc[i + 1])
        if src is not None and dest is not None:
            edge_index.append((src, dest))

edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

# 划分训练集和测试集
train_indices, test_indices = train_test_split(range(len(labels)), test_size=0.2)

# 创建掩码
train_mask = torch.zeros(len(labels), dtype=torch.bool)
test_mask = torch.zeros(len(labels), dtype=torch.bool)

train_mask[train_indices] = True
test_mask[test_indices] = True

# 创建节点特征矩阵
x = torch.tensor(embedding_matrix_20NG, dtype=torch.float)

# 创建标签
y = torch.tensor(labels, dtype=torch.long)

# 构建图数据
data = Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask, test_mask=test_mask)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCN(nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, num_classes)

    def forward(self, x, edge_index):
        # First layer
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        # Second layer
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN(num_features=300, hidden_channels=16, num_classes=num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

model.train()
for epoch in range(200):
    optimizer.zero_grad()
    out = model(data.x.to(device), data.edge_index.to(device))
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask].to(device))
    loss.backward()
    optimizer.step()

model.eval()
pred = model(data.x.to(device), data.edge_index.to(device)).max(1)[1]
correct = pred[data.test_mask].eq(data.y[data.test_mask].to(device)).sum().item()
accuracy = correct / data.test_mask.sum().item()
print('Accuracy:', accuracy)
