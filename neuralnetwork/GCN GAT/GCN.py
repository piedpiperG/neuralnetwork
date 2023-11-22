import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import pandas as pd
import torch
from torch_geometric.data import Data


# 加载词向量
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


# 加载数据集
def load_dataset(file_name):
    documents = []
    labels = []
    with open(file_name, 'r', encoding='utf-8') as file:
        for line in file:
            split_line = line.split("\t")
            labels.append(split_line[0])
            documents.append(split_line[1].split())
    return documents, labels


# 构建图卷积神经网络
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

    # 加载数据集


# 数据预处理
word_vectors_MR = load_word_vectors('word_vectors_MR.txt')
embedding_matrix_MR, word_to_index_MR = create_embedding_matrix(word_vectors_MR)
