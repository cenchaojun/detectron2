
import torch

H = 8
W = 8
window_size = 2
shift_size = 1
# 生成全零张量
img_mask = torch.zeros((1, H, W, 1))
print(img_mask)
# 按区域划分mask
h_slices = (slice(0, -window_size),
            slice(-window_size, -shift_size),
            slice(-shift_size, None))
w_slices = (slice(0, -window_size),
            slice(-window_size, -shift_size),
            slice(-shift_size, None))
cnt = 0
for h in h_slices:
    for w in w_slices:
        img_mask[:, h, w, :] = cnt
        cnt += 1
print(img_mask.squeeze(dim=3))