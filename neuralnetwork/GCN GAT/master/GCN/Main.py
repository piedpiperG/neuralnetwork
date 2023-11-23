import pickle
import re
import numpy as np
from matplotlib import pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import defaultdict
from scipy.sparse import csr_matrix
import torch
from torch_geometric.data import Data
from Model import GCN, GAT, TextCNN, TextLSTM
import torch.optim as optim
import torch.nn.functional as F
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
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


# 构建特征矩阵
def build_feature_matrix(texts):
    vectorizer = TfidfVectorizer(max_features=1000)  # 取最重要的1000个特征
    X = vectorizer.fit_transform(texts)
    return X.toarray()


# 构建邻接矩阵
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


def load_labels(file_path):
    # 读取所有标签
    with open(file_path, 'r', encoding='utf-8') as file:
        labels = [line.strip().split()[1] for line in file]

    # 创建一个从标签名到整数的映射
    unique_labels = sorted(set(labels))
    label_map = {label: i for i, label in enumerate(unique_labels)}

    # 将标签转换为对应的整数
    labels = [label_map[label] for label in labels]
    return labels, label_map


def tokenize_and_pad_texts(texts, max_len=500):
    # 创建并训练分词器
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)

    # 将文本转换为整数序列
    sequences = tokenizer.texts_to_sequences(texts)

    # 填充序列以获得统一的长度
    padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post')

    return padded_sequences, tokenizer.word_index


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


def train_cnn():
    model.train()
    optimizer.zero_grad()
    out = model(texts_tensor)  # 使用处理过的文本序列
    loss = F.cross_entropy(out[train_mask], labels_tensor[train_mask])
    loss.backward()
    optimizer.step()
    return loss


def test_cnn():
    model.eval()
    logits = model(texts_tensor)  # 使用处理过的文本序列
    accs = []
    for mask in [train_mask, val_mask, test_mask]:
        pred = logits[mask].max(1)[1]
        acc = pred.eq(labels_tensor[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs


def plot_training_results(train_losses, train_accuracies, val_accuracies, test_accuracies):
    # 绘制损失图
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.title("Training Loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    # 绘制准确率图
    plt.figure(figsize=(10, 5))
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.plot(test_accuracies, label='Test Accuracy')
    plt.title("Accuracy per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # 加载文档数据
    # texts = load_dataset('data/r52.txt')
    FILE_NAME = 'r52'
    CLEANED_TEXTS_FILE = f'data/{FILE_NAME}-clean.pkl'  # 定义保存清理过的文本数据的文件名
    # 加载或保存清理后的文本数据
    if os.path.exists(CLEANED_TEXTS_FILE):
        with open(CLEANED_TEXTS_FILE, 'rb') as f:
            texts = pickle.load(f)
    else:
        # 加载文档数据
        texts = load_dataset(f'data/{FILE_NAME}.txt')
        # 进行文本清洗
        texts = [clean_text(text) for text in texts]
        # 保存清理过的文本数据到文件
        with open(CLEANED_TEXTS_FILE, 'wb') as f:
            pickle.dump(texts, f)
    # 划分数据集
    # 假设 N 是文档的总数
    N = len(texts)
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

    # GCN,GAT构建
    # 构建特征矩阵
    feature_matrix = build_feature_matrix(texts)
    # 构建邻接矩阵
    adjacency_matrix = build_adjacency_matrix2(feature_matrix)
    # 转换邻接矩阵为 COO 格式的边索引
    edge_index = torch.tensor(np.array(adjacency_matrix.nonzero()), dtype=torch.long)
    # 加载标签并转换为 Tensor
    labels, label_map = load_labels(f'data/{FILE_NAME}-label.txt')
    labels = torch.tensor(labels, dtype=torch.long)
    # 构建数据对象
    data = Data(x=torch.tensor(feature_matrix, dtype=torch.float), edge_index=edge_index, y=labels)

    # 将掩码添加到数据对象中
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    # 初始化模型
    num_features = feature_matrix.shape[1]  # 特征维度
    num_classes = len(set(labels.numpy()))  # 类别数

    # 选择模型
    # GCN 图卷积网络
    model = GCN(num_features=num_features, num_classes=num_classes)
    # GAT 注意力
    # model = GAT(num_features=num_features, num_classes=num_classes)

    # CNN 卷积网络
    # 设定最大序列长度
    MAX_SEQUENCE_LENGTH = 500
    # 获取整数序列和词汇表
    texts_int_seq, word_index = tokenize_and_pad_texts(texts, MAX_SEQUENCE_LENGTH)
    VOCAB_SIZE = len(word_index) + 1  # 加1因为索引0通常是填充符号
    EMBEDDING_DIM = 100  # 可以根据需要调整
    FILTER_SIZES = [3, 4, 5]  # 卷积核的大小
    NUM_FILTERS = 100  # 每种大小的卷积核的数量
    NUM_CLASSES = len(set(labels))  # 类别数
    # 获取整数序列和词汇表
    texts_tensor = torch.tensor(texts_int_seq, dtype=torch.long)  # 转换为 Tensor
    # 加载标签
    labels_tensor = torch.tensor(labels, dtype=torch.long)

    # CNN模型加载
    # model = TextCNN(VOCAB_SIZE, EMBEDDING_DIM, NUM_CLASSES, FILTER_SIZES, NUM_FILTERS)
    # LSTM模型加载
    HIDDEN_DIM = 128  # 隐藏层维度，可以调整
    # model = TextLSTM(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, NUM_CLASSES)

    # 用于保存损失和准确率的列表
    train_losses = []
    train_accuracies = []
    val_accuracies = []
    test_accuracies = []
    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=0.1)

    # GCN和GAT的训练和测试循环
    for epoch in range(200):
        loss = train()
        torch.cuda.empty_cache()  # 清理 GPU 缓存
        train_acc, val_acc, test_acc = test()

        # 保存数据
        train_losses.append(loss.item())
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        test_accuracies.append(test_acc)

        print(
            f'Epoch: {epoch + 1:03d}, Loss: {loss:.4f}, Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}')

    # # CNN的训练和测试循环
    # for epoch in range(200):
    #     loss = train_cnn()
    #     torch.cuda.empty_cache()  # 清理 GPU 缓存
    #     train_acc, val_acc, test_acc = test_cnn()

    #      # 保存数据
    #     train_losses.append(loss.item())
    #     train_accuracies.append(train_acc)
    #     val_accuracies.append(val_acc)
    #     test_accuracies.append(test_acc)

    #     print(
    #         f'Epoch: {epoch + 1:03d}, Loss: {loss:.4f}, Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}')
    plot_training_results(train_losses, train_accuracies, val_accuracies, test_accuracies)
    torch.save(model, 'model/model_GCN.pth')
