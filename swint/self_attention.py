import torch


# 定义输入
x = [
  [1, 0, 1, 0], # Input 1
  [0, 2, 0, 2], # Input 2
  [1, 1, 1, 1]  # Input 3
 ]
x = torch.tensor(x, dtype=torch.float32)

# 初始化qkv的权重
w_key = [
  [0, 0, 1],
  [1, 1, 0],
  [0, 1, 0],
  [1, 1, 0]
]
w_query = [
  [1, 0, 1],
  [1, 0, 0],
  [0, 0, 1],
  [0, 1, 1]
]
w_value = [
  [0, 2, 0],
  [0, 3, 0],
  [1, 0, 3],
  [1, 1, 0]
]
w_key = torch.tensor(w_key, dtype=torch.float32)
w_query = torch.tensor(w_query, dtype=torch.float32)
w_value = torch.tensor(w_value, dtype=torch.float32)


keys = x @ w_key
querys = x @ w_query
values = x @ w_value

print(keys)


print(querys)


print(values)

# 计算注意力得分
attn_scores = querys @ keys.T


from torch.nn.functional import softmax

# 计算softmax后的注意力得分
attn_scores_softmax = softmax(attn_scores, dim=-1)

# # For readability, approximate the above as follows
# attn_scores_softmax = [
#   [0.0, 0.5, 0.5],
#   [0.0, 1.0, 0.0],
#   [0.0, 0.9, 0.1]
# ]
# attn_scores_softmax = torch.tensor(attn_scores_softmax)



weighted_values = values[:,None] * attn_scores_softmax.T[:,:,None]

outputs = weighted_values.sum(dim=0)