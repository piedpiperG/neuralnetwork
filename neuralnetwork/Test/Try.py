import torch
import torch.nn as nn

# 定义转置卷积层
transposed_conv = nn.ConvTranspose2d(in_channels=3, out_channels=64, kernel_size=4, stride=2, padding=1)

# 随机生成输入数据
input_data = torch.randn(1, 3, 32, 32)  # (batch_size, in_channels, height, width)

# 进行转置卷积操作
output_data = transposed_conv(input_data)

# 输出结果的形状
print(output_data.shape)
