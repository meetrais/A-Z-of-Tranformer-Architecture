import torch

inputs = torch.tensor(
    [[0.43, 0.15, 0.89], # Your
     [0.55, 0.87, 0.66], # journey 
     [0.57, 0.85, 0.64], # starts 
     [0.22, 0.58, 0.33], # with
     [0.77, 0.25, 0.10], # one
     [0.05, 0.80, 0.55]] # step
)

query = inputs[1]
print("query: ", query)

attn_scores_2 = torch.empty(inputs.shape[0])
for i, x_i in enumerate(inputs):
    attn_scores_2[i] = torch.dot(x_i, query)
print("attn_scores_2: ", attn_scores_2)

attn_weights_2_tmp = attn_scores_2 / attn_scores_2.sum()
print("Attention weights:", attn_weights_2_tmp)
print("Sum:", attn_weights_2_tmp.sum())

print("\n################################################################\n")

def softmax_naive(x):
    return torch.exp(x) / torch.exp(x).sum(dim=0)

attn_weights_2_naive = softmax_naive(attn_scores_2)
print("Attention weights:", attn_weights_2_naive)
print("Sum:", attn_weights_2_naive.sum())

print("\n################################################################\n")

attn_weights_2 = torch.softmax(attn_scores_2, dim=0)
print("Attention weights:", attn_weights_2)
print("Sum:", attn_weights_2.sum())

print("\n################################################################\n")

query = inputs[1]
print("query: ", query)

context_vec_2 = torch.zeros(query.shape)
for i,x_i in enumerate(inputs):
    context_vec_2 += attn_weights_2[i]*x_i

print("context_vec_2: ",context_vec_2)

print("\n################################################################\n")

attn_scores = torch.empty(6, 6)
for i, x_i in enumerate(inputs):
    for j, x_j in enumerate(inputs):
        attn_scores[i, j] = torch.dot(x_i, x_j)
print(attn_scores)

print("\n################################################################\n")

attn_scores = inputs @ inputs.T
print(attn_scores)

print("\n################################################################\n")

attn_weights = torch.softmax(attn_scores, dim=-1)
print(attn_weights)

print("\n################################################################\n")

row_2_sum = sum([0.1385, 0.2379, 0.2333, 0.1240, 0.1082, 0.1581])
print("Row 2 sum:", row_2_sum)
print("All row sums:", attn_weights.sum(dim=-1))

print("\n################################################################\n")

all_context_vecs = attn_weights @ inputs
print(all_context_vecs)
print("\nPrevious 2nd context vector:", context_vec_2)

print("\n################################################################\n")

x_2 = inputs[1]
d_in = inputs.shape[1]
d_out = 2

torch.manual_seed(123)
W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_key = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)

query_2 = x_2 @ W_query
key_2 = x_2 @ W_key
value_2 = x_2 @ W_value

print("x_2.shape: ", x_2.shape)
print("W_query.shape: ", W_query.shape)
print("x_2: ", x_2)
print("W_query: ", W_query)
print("query_2: ", query_2)

print("\n################################################################\n")

keys = inputs @ W_key
values = inputs @ W_value
print("keys.shape:", keys.shape)
print("keys: ", keys)
print("values.shape:", values.shape)
print("values: ", values)

print("\n################################################################\n")





