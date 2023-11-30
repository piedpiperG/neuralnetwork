import string

import torch

all_letters = string.ascii_letters + " .,;'-"  # string.ascii_letters的作用是生成所有的英文字母
n_letters = len(all_letters) + 1  # 多加的1是指EOS
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
n_categories = 3
