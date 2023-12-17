import torch
import torch.nn as nn

mat1 = torch.rand(1, 3)
mat2 = torch.rand(2, 1)
print(f'mat1:{mat1}, mat2:{mat2}')

print('----------------直接相减----------------')
print(f'mat1-mat2={mat1 - mat2}')

print('----------------原地操作inplace----------------')
# 调整张量形状以使它们兼容
# 扩展mat1和mat2，使它们可以广播到相同形状
mat1_expanded = mat1.expand(2, 3)
mat2_expanded = mat2.expand(2, 3)
# 尝试进行原地操作，但由于原地操作不能改变形状，我们使用扩展后的张量
mat1_inplace_adjusted = mat1_expanded.clone()
mat1_inplace_adjusted.sub_(mat2_expanded)
print(f'mat1-mat2={mat1_inplace_adjusted}')

# print(mat2[0].expand(1, 3).contiguous())
print('----------------使用torch.nn----------------')
# 创建一个线性层，权重为单位矩阵，偏置设置为mat2[0]
linear = nn.Linear(3, 3, bias=True)
linear.weight.data = torch.eye(3)
linear.bias.data = -mat2[0].expand(1, 3).contiguous()
# 创建一个线性层，权重为单位矩阵，偏置设置为mat2[1]
linear2 = nn.Linear(3, 3, bias=True)
linear2.weight.data = torch.eye(3)
linear2.bias.data = -mat2[1].expand(1, 3).contiguous()
# 应用线性层
mat1_expanded = mat1.expand(2, 3)  # 扩展 mat1 以匹配 mat2 的形状
result_1 = linear(mat1_expanded[0].view(1, -1))
result_2 = linear2(mat1_expanded[1].view(1, -1))
result = torch.cat((result_1, result_2), dim=0)
print(f'mat1 - mat2 = {result}')
