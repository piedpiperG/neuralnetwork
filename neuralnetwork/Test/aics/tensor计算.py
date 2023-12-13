import torch

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

print('----------------使用torch.nn----------------')



