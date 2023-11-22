import re
import os
from gensim.models import Word2Vec
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# NLTK资源下载
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')


def clean_text(text):
    text = text.lower()
    text = re.sub(r'\b\w{1,2}\b', '', text)  # 去除短词
    text = re.sub(r'[^a-z\s]', '', text)  # 去除非字母字符
    lemmatizer = WordNetLemmatizer()
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stopwords.words('english')]
    return ' '.join(words)


def load_and_clean_dataset(file_path):
    with open(file_path, 'r', encoding='latin1') as file:
        data = file.readlines()
    return [clean_text(line.strip()) for line in data]


# 数据集
# datasets = ['20NG.txt', 'R8.txt', 'R52.txt', 'Ohsumed.txt', 'MR.txt']
datasets = ['r8-test.txt']
dataset_paths = [os.path.join('', dataset) for dataset in datasets]

# 可以在corpus/* 路径下添加数据集文件
# 本代码使用的数据集文件来自于作业描述
# https://github.com/yao8839836/text_gcn/tree/master/data
cleaned_datasets = {dataset: load_and_clean_dataset(path) for dataset, path in zip(datasets, dataset_paths)}


# 下面word2vec的参数可自行调节
def train_word2vec(texts, vector_size=300, window=5, min_count=5):
    tokenized_texts = [text.split() for text in texts]
    model = Word2Vec(sentences=tokenized_texts, vector_size=vector_size, window=window, min_count=min_count, workers=4)
    return model


# 分别为每个数据集训练word2vec
models = {dataset: train_word2vec(texts) for dataset, texts in cleaned_datasets.items()}


def save_word_vectors(model, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        for word in model.wv.index_to_key:
            vector = model.wv[word]
            vector_str = ' '.join([str(x) for x in vector])
            f.write(f'{word} {vector_str}\n')


# 保存词向量
for dataset, model in models.items():
    save_word_vectors(model, f'word_vectors_{dataset}')
