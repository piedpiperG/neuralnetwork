import glob
import os
import random
import torch
import unicodedata

from static import n_categories, n_letters, all_letters, device


def find_files(path):
    """
    :param path:文件路径
    :return: 文件列表地址
    """
    return glob.glob(path)  # glob模块提供了一个函数用于从目录通配符搜索中生成文件列表:


# 处理国际化文本数据时常见的一步，特别是当你处理包含特殊字符或重音符号的语言（如法语、德语、西班牙语等）时
def unicode_to_ascii(str):
    """
    :param str:名字
    :return:返回均采用NFD编码方式的名字
    """
    return ''.join(
        c for c in unicodedata.normalize('NFD', str)  # 对文字采用相同的编码方式
        if unicodedata.category(c) != 'Mn' and c in all_letters
    )


def read_lines(files_list):
    """
    读取每个文件的内容
    :param files_list:文件所在地址列表
    :return:{国家：名字列表}
    """
    category_lines = {}
    all_categories = []
    for file in files_list:
        # os.path.splitext:分割路径，返回路径名和文件扩展名的元组
        # os.path.basename:返回文件名
        category = os.path.splitext(os.path.basename(file))[0]
        line = [unicode_to_ascii(line) for line in open(file)]
        all_categories.append(category)
        category_lines[category] = line
    return all_categories, category_lines


# 随机选择一个类别
def random_choice(obj):
    return obj[random.randint(0, len(obj) - 1)]


def category_to_tensor(category):
    """
    将类别转换成张量
    :param category:类别
    :return:张量
    """
    li = all_categories.index(category)
    tensor = torch.zeros(1, n_categories)
    tensor[0][li] = 1
    return tensor.to(device)


def input_to_tensor(input):
    """
    将输入进行one-hot编码
    :param word: 单词
    :return: 张量
    """
    tensor = torch.zeros(len(input), 1, n_letters)
    for i, letter in enumerate(input):
        tensor[i][0][all_letters.find(letter)] = 1
    return tensor.to(device)


def target_to_tensor(input):
    """
    对目标输出进行one-hot编码，即为从第二个字母开始至结束字母的索引，以及EOS的索引
    :param input:单词
    :return:张量
    """
    letter_indexes = [all_letters.find(input[i]) for i in range(1, len(input))]
    letter_indexes.append(n_letters - 1)  # 最后一位的索引是EOS
    return torch.LongTensor(letter_indexes).to(device)


def random_training_example():
    category = random_choice(all_categories)
    input = random_choice(category_lines[category])
    category_tensor = category_to_tensor(category)
    input_tensor = input_to_tensor(input)
    target_tensor = target_to_tensor(input)
    return category_tensor, input_tensor, target_tensor


files_list = find_files('../data/*.txt')
all_categories, category_lines = read_lines(files_list)

# 将txt文件中的类别提取出来,返回一个列表
# print(all_categories)
# 构建字典，每一个类别对应一个名字列表
# print(category_lines)
