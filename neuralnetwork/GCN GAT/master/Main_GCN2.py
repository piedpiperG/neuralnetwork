import re
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import defaultdict
from scipy.sparse import csr_matrix
import torch
from torch_geometric.data import Data
from Model_GCN import GCN
import torch.optim as optim
import torch.nn.functional as F
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# 文本清洗
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\b\w{1,2}\b', '', text)  # 去除短词
    text = re.sub(r'[^a-z\s]', '', text)  # 去除非字母字符
    lemmatizer = WordNetLemmatizer()
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stopwords.words('english')]
    return ' '.join(words)


# 加载数据集
def load_dataset(file_path):
    with open(file_path, 'r', encoding='latin1') as file:
        data = file.readlines()
    return [clean_text(line.strip().split('\t')[1]) for line in data]


# 加载词向量
def load_word_vectors(filename):
    word_vecs = {}
    with open(filename, 'r') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vec = np.asarray(values[1:], dtype='float32')
            word_vecs[word] = vec
    return word_vecs


# 构建共现矩阵
def build_cooccurrence_matrix(texts, vocab, window_size=4):
    cooccurrence_counts = defaultdict(int)
    for text in texts:
        words = text.split()
        for i, word in enumerate(words):
            if word in vocab:
                start = max(0, i - window_size)
                end = min(len(words), i + window_size + 1)
                for j in range(start, end):
                    if i != j and words[j] in vocab:
                        pair = tuple(sorted([word, words[j]]))
                        cooccurrence_counts[pair] += 1
    return cooccurrence_counts


# 构建邻接矩阵
def build_adjacency_matrix(vocab, cooccurrence_counts):
    word_to_id = {word: i for i, word in enumerate(vocab)}
    row, col, data = [], [], []
    for (word1, word2), count in cooccurrence_counts.items():
        row.append(word_to_id[word1])
        col.append(word_to_id[word2])
        data.append(count)  # 或者可以用其他方式来定义边的权重
    return csr_matrix((data, (row, col)), shape=(len(vocab), len(vocab)))


# 构建特征矩阵2
def build_feature_matrix(texts):
    vectorizer = TfidfVectorizer(max_features=1000)  # 取最重要的1000个特征
    X = vectorizer.fit_transform(texts)
    return X.toarray()


# 构建邻接矩阵2
def build_adjacency_matrix2(feature_matrix):
    cosine_sim = cosine_similarity(feature_matrix)
    adjacency_matrix = (cosine_sim > 0.5).astype(int)  # 举例：相似度大于0.5则认为有连接
    np.fill_diagonal(adjacency_matrix, 0)  # 对角线元素设为0，避免自环
    return csr_matrix(adjacency_matrix)  # 使用稀疏矩阵表示


# 获取节点特征
def get_feature_matrix(vocab, word_vecs):
    features = np.zeros((len(vocab), len(next(iter(word_vecs.values())))))
    word_to_id = {word: i for i, word in enumerate(vocab)}
    for word, vec in word_vecs.items():
        if word in word_to_id:
            features[word_to_id[word]] = vec
    return features


# 加载标签数据
def load_labels(file_path):
    label_map = {'trade': 0, 'grain': 1, 'ship': 2, 'acq': 3, 'earn': 4, 'interest': 5, 'money-fx': 6,
                 'crude': 7}  # 定义类别映射
    labels = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            label_name = line.strip().split()[1]
            labels.append(label_map[label_name])
    return labels


# 训练
def train():
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss


# 测试
def test():
    model.eval()
    logits, accs = model(data), []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs


if __name__ == '__main__':
    # 加载文档数据
    texts = load_dataset('r8-train.txt')

    # 构建特征矩阵
    feature_matrix = build_feature_matrix(texts)

    # 构建邻接矩阵
    adjacency_matrix = build_adjacency_matrix2(feature_matrix)

    # 转换邻接矩阵为 COO 格式的边索引
    edge_index = torch.tensor(np.array(adjacency_matrix.nonzero()), dtype=torch.long)

    # 加载标签并转换为 Tensor
    labels = torch.tensor(load_labels('r8-train-label.txt'), dtype=torch.long)

    # 构建数据对象
    data = Data(x=torch.tensor(feature_matrix, dtype=torch.float), edge_index=edge_index, y=labels)
    # 假设 N 是文档的总数
    N = len(texts)
    print(N)

    # 生成随机索引来划分训练、验证和测试集
    indices = torch.randperm(N)

    # 计算训练、验证和测试集的大小
    train_size = int(0.7 * N)
    val_size = int(0.15 * N)

    # 创建掩码
    train_mask = torch.zeros(N, dtype=torch.bool)
    val_mask = torch.zeros(N, dtype=torch.bool)
    test_mask = torch.zeros(N, dtype=torch.bool)

    # 分配掩码
    train_mask[indices[:train_size]] = True
    val_mask[indices[train_size:train_size + val_size]] = True
    test_mask[indices[train_size + val_size:]] = True

    # 将掩码添加到数据对象中
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    # 初始化模型
    num_features = feature_matrix.shape[1]  # 特征维度
    num_classes = len(set(labels.numpy()))  # 类别数
    print(num_classes)
    model = GCN(num_features=num_features, num_classes=num_classes)

    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # 训练和测试循环保持不变
    for epoch in range(200):
        loss = train()
        torch.cuda.empty_cache()  # 清理 GPU 缓存
        train_acc, val_acc, test_acc = test()
        print(
            f'Epoch: {epoch + 1:03d}, Loss: {loss:.4f}, Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}')
