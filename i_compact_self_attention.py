import torch
import torch.nn as nn

class SelfAttention_v1(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))

    def forward(self, x):
        keys = x @ self.W_key
        queries = x @ self.W_query
        values = x @ self.W_value
        attn_scores = queries @ keys.T # omega
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1
        )
        context_vec = attn_weights @ values
        return context_vec

if __name__ == "__main__":
    inputs = torch.tensor(
        [[0.43, 0.15, 0.89], # Your
        [0.55, 0.87, 0.66], # journey 
        [0.57, 0.85, 0.64], # starts 
        [0.22, 0.58, 0.33], # with
        [0.77, 0.25, 0.10], # one
        [0.05, 0.80, 0.55]] # step
    )

    d_in = inputs.shape[1]
    d_out = 2

    torch.manual_seed(123)
    sa_v1 = SelfAttention_v1(d_in, d_out)
    print(sa_v1(inputs))