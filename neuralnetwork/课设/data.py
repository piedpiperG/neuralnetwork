import string
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import json
import os
import numpy as np


class ImageTextDataset(Dataset):
    def __init__(self, json_file_path, image_folder_path, max_seq_length):
        """
        Args:
            json_file_path (string): JSON文件的路径，包含图片名称和对应的描述。
            image_folder_path (string): 包含图片的文件夹路径。
            max_seq_length (int): 最大序列长度。
        """
        with open(json_file_path, 'r', encoding='utf-8') as file:
            self.descriptions = json.load(file)
        self.image_folder_path = image_folder_path
        self.max_seq_length = max_seq_length
        self.word_to_index = self.create_vocab(self.descriptions)

    def preprocess_text(self, text):
        """
        对文本进行预处理，将句号与单词分离。
        """
        # 使用空格替换句号，确保句号被视为独立的单词
        text = text.replace('.', ' . ')
        return text.lower().translate(str.maketrans('', '', string.punctuation.replace('.', '')))

    def create_vocab(self, descriptions):
        word_freq = {}
        for _, description in descriptions.items():
            preprocessed_text = self.preprocess_text(description)
            words = preprocessed_text.split()
            for word in words:
                if word not in word_freq:
                    word_freq[word] = 1
                else:
                    word_freq[word] += 1

        vocab = {'<pad>': 0, '<sos>': 1, '<eos>': 2, '<unk>': 3, '.': 4}  # 将句号添加为特殊符号
        idx = 5  # 从索引5开始添加其他单词
        for word in sorted(word_freq, key=word_freq.get, reverse=True):
            if word not in vocab:  # 防止重复添加
                vocab[word] = idx
                idx += 1

        return vocab

    def text_to_indices(self, text):
        preprocessed_text = self.preprocess_text(text)
        words = preprocessed_text.split()
        indices = [self.word_to_index['<sos>']] + \
                  [self.word_to_index.get(word, self.word_to_index['<unk>']) for word in words] + \
                  [self.word_to_index['<eos>']]
        indices += [self.word_to_index['<pad>']] * (self.max_seq_length - len(indices))
        return indices[:self.max_seq_length]

    def __len__(self):
        return len(self.descriptions)

    def __getitem__(self, idx):
        image_name, description = list(self.descriptions.items())[idx]
        image_path = os.path.join(self.image_folder_path, image_name)

        image = Image.open(image_path)
        image = image.resize((750, 1101))
        image = np.array(image)
        if image.shape[2] == 4:
            image = image[..., :3]
        image = torch.from_numpy(image).permute(2, 0, 1)

        indices = self.text_to_indices(description)
        indices_tensor = torch.tensor(indices, dtype=torch.long)

        return {'image': image, 'indices': indices_tensor, 'description': description}


def get_max_seq_length(json_file_path):
    """
    计算给定 JSON 文件中最长文本描述的长度。

    Args:
        json_file_path (string): JSON文件的路径。

    Returns:
        int: 最大序列长度。
    """
    with open(json_file_path, 'r', encoding='utf-8') as file:
        descriptions = json.load(file)

    max_length = 0
    for _, description in descriptions.items():
        # 移除标点符号，并以空格分割成单词
        words = description.lower().translate(str.maketrans('', '', string.punctuation)).split()
        # 计算长度（加2是因为<sos>和<eos>标记）
        length = len(words) + 2
        if length > max_length:
            max_length = length

    return max_length


# 使用示例
max_seq_length = max(get_max_seq_length('word2vec/train_captions.json'),
                     get_max_seq_length('word2vec/test_captions.json'))
json_file_path = 'word2vec/test_captions.json'  # JSON文件路径
image_folder_path = 'E:\杂项下载\课设数据集\images'  # 图片文件夹路径
dataset = ImageTextDataset(json_file_path, image_folder_path, max_seq_length)
dataloader = DataLoader(dataset, batch_size=4)

for batch in dataloader:
    print(batch['image'], batch['indices'])
