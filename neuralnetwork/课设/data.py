import string

from torch.utils.data import Dataset
from PIL import Image
import json
import os
import numpy as np


def load_word_vectors(filename):
    word_vectors = {}
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            word_vectors[word] = vector
    return word_vectors


# 加载词向量
word_vectors = load_word_vectors('word2vec/test_captions.json_word_vectors.txt')  # 替换为您的词向量文件路径


def text_to_vectors(text, word_vectors):
    # 转换为小写并移除标点
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    vectors = [word_vectors.get(word, np.zeros(len(next(iter(word_vectors.values()))))) for word in words]
    return np.array(vectors)


class ImageTextDataset(Dataset):
    def __init__(self, json_file_path, image_folder_path, word_vectors):
        """
            Args:
               json_file_path (string): JSON文件的路径，包含图片名称和对应的描述。
               image_folder_path (string): 包含图片的文件夹路径。
        """
        with open(json_file_path, 'r', encoding='utf-8') as file:
            self.descriptions = json.load(file)
        self.image_folder_path = image_folder_path
        self.word_vectors = word_vectors

    def __len__(self):
        return len(self.descriptions)

    def __getitem__(self, idx):
        image_name, description = list(self.descriptions.items())[idx]
        image_path = os.path.join(self.image_folder_path, image_name)
        image = Image.open(image_path)
        image = np.array(image)  # 将图像转换为NumPy数组
        description_vectors = text_to_vectors(description, self.word_vectors)

        return {'image': image, 'description_vectors': description_vectors, 'description': description}


# 使用示例
json_file_path = 'word2vec/test_captions.json'  # 替换为您的JSON文件路径
image_folder_path = 'E:\杂项下载\课设数据集\images'  # 替换为您的图片文件夹路径
dataset = ImageTextDataset(json_file_path, image_folder_path, word_vectors)

sample = dataset[0]
print(sample['image'].shape, sample['description_vectors'], sample['description'])  # 输出第一个样本的图像形状和描述
