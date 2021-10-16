import torch

coords_h = torch.arange(2)
coords_w = torch.arange(2)
coords = torch.meshgrid([coords_h, coords_w])
print(coords)

coords = torch.stack(coords)
coords_flatten = torch.flatten(coords, 1)
print(coords_flatten )

relative_coords_first = coords_flatten[:, :, None]  # 2, wh*ww, 1
print(relative_coords_first)
print(relative_coords_first.shape)
relative_coords_second = coords_flatten[:, None, :] # 2, 1, wh*ww
print(relative_coords_second)
print(relative_coords_second.shape)
relative_coords = relative_coords_first - relative_coords_second # 最终得到 2, wh*ww, wh*ww 形状的张量
print(relative_coords)
print(relative_coords.shape)

relative_coords = relative_coords.permute(1, 2, 0).contiguous() # Wh*Ww, Wh*Ww, 2
relative_coords[:, :, 0] += 2 - 1
relative_coords[:, :, 1] += 2 - 1

print(relative_coords)

relative_coords[:, :, 0] *= 2 * 2 - 1

print(relative_coords)

relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww

print(relative_position_index)

# print(coords.size)