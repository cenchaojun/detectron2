import torch

# 以win大小为2来举例
coords_h = torch.arange(2)
print(coords_h)
coords_w = torch.arange(2)
print(coords_w)

# 生成坐标
coords = torch.meshgrid([coords_h, coords_w])
print(coords)

# 对两个张量沿指定维度进行堆叠，返回的Tensor会多一个维。
coords = torch.stack(coords)
print(coords)

# 进行展开成一个2维向量操作。
coords_flatten = torch.flatten(coords, 1)
print(coords_flatten.squeeze(dim=1))

# 利用pytorch的广播机制，分别在第一维，第二维，插入一个维度，进行广播相减，得到 2, wh*ww, wh*ww的张量
relative_coords_first = coords_flatten[:, :, None]  # 2, wh*ww, 1
print(relative_coords_first)
print(relative_coords_first.shape)

relative_coords_second = coords_flatten[:, None, :] # 2, 1, wh*ww
print(relative_coords_second)
print(relative_coords_second.shape)

# 进行相减得到一个最终的相对位置坐标
relative_coords = relative_coords_first - relative_coords_second # 最终得到 2, wh*ww, wh*ww 形状的张量
print(relative_coords)
print(relative_coords.shape)

# 因为采取的相减策略得到相对位置信息，所以需要将索引由负数改为0，加上一个偏移量，让其从0开始。
relative_coords = relative_coords.permute(1, 2, 0).contiguous() # Wh*Ww, Wh*Ww, 2
relative_coords[:, :, 0] += 2 - 1
relative_coords[:, :, 1] += 2 - 1
print(relative_coords)

# 因为在二维度上，（2，1）和（1，2）的坐标信息是不一样的，但是要展开成1维，就没有办法区分
# 所以在高或者宽上乘上一个win的大小，用于区分宽和高
relative_coords[:, :, 0] *= 2 * 2 - 1
print(relative_coords)

# 最后在一维度上进行求和，展开成一维坐标，之后注册为一个不参与网络学习的变量
relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww

print(relative_position_index)

# print(coords.size)