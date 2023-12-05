import json
import re
import os
from gensim.models import Word2Vec
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# NLTK资源下载
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('omw-1.4')


def clean_text(text):
    text = text.lower()
    # text = re.sub(r'\b\w{1,2}\b', '', text)  # 去除短词
    text = re.sub(r'[^a-z\s]', '', text)  # 去除非字母字符
    lemmatizer = WordNetLemmatizer()
    words = text.split()
    # words = [lemmatizer.lemmatize(word) for word in words if word not in stopwords.words('english')]
    return ' '.join(words)


def load_and_clean_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return {key: clean_text(value) for key, value in data.items()}


def train_word2vec(texts, vector_size=300, window=5, min_count=5):
    tokenized_texts = [text.split() for text in texts]
    model = Word2Vec(sentences=tokenized_texts, vector_size=vector_size, window=window, min_count=min_count, workers=4)
    return model


def save_word_vectors(model, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        for word in model.wv.index_to_key:
            vector = model.wv[word]
            vector_str = ' '.join([str(x) for x in vector])
            f.write(f'{word} {vector_str}\n')


# JSON文件路径
json_file_path = 'train_captions.json'  # 请替换为您的JSON文件路径

# 从JSON文件加载并清洗数据
cleaned_data = load_and_clean_json(json_file_path)

# 训练Word2Vec模型
model = train_word2vec(list(cleaned_data.values()))

# 保存词向量
save_word_vectors(model, f'{json_file_path}_word_vectors.txt')
