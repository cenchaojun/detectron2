import torch
import numpy as np


torch.set_printoptions(threshold=np.inf)

def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    # 调用view之前需要先调用contiguous()，否则容易报错，contiguous是对tensor进行深拷贝
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

Hp = 8
Wp = 8
window_size = 4
shift_size = 2
img_mask = torch.zeros(1, Hp, Wp, 1)  # 1 Hp Wp 1
print(img_mask)

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
        print(img_mask.squeeze(dim=3))
        cnt += 1
# 不同的窗口进行分开
mask_windows = window_partition(img_mask, window_size)  # nW, window_size, window_size, 1
print(mask_windows.squeeze(dim=3))
print("===================================================================================")
# 将其每个窗口拉伸到一一个维度
mask_windows = mask_windows.view(-1, window_size * window_size)
print(mask_windows)
print("msk_windows===================================================================================")
print(mask_windows.unsqueeze(1))
print("mask_windows.unsqueeze(1)===================================================================================")
print(mask_windows.unsqueeze(2))
print("mask_windows.unsqueeze(2)===================================================================================")

#
attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
print(attn_mask)
print("===================================================================================")
# 将不等于0的点全部mask为-100，
attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
print(attn_mask)




# print(mask_windows.squeeze(dim=3))
