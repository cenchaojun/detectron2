import torch
a = torch.tensor([1, 2, 3, 4])
print(a)
b = torch.tensor([4, 5, 6])
print(b)
x, y = torch.meshgrid(a, b)
print(x)
print(y)

# 结果显示：
# tensor([1, 2, 3, 4])
# tensor([4, 5, 6])
# tensor([[1, 1, 1],
#         [2, 2, 2],
#         [3, 3, 3],
#         [4, 4, 4]])
# tensor([[4, 5, 6],
#         [4, 5, 6],
#         [4, 5, 6],
#         [4, 5, 6]])