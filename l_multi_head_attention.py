import torch
import k_casual_self_attention as csa
import torch.nn as nn

class MultiHeadAttentionWrapper(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        self.heads = nn.ModuleList(
            [csa.CausalAttention(
                d_in, d_out, context_length, dropout, qkv_bias
            )
            for _ in range(num_heads)]
        )

    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)

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

    batch = torch.stack((inputs, inputs), dim=0)
    print("\nbatch.shape: ", batch.shape)
    print("\n################################################################\n")

    torch.manual_seed(123)
    context_length = batch.shape[1] # This is the number of tokens
    d_in, d_out = 3, 2
    mha = MultiHeadAttentionWrapper(
        d_in, d_out, context_length, 0.0, num_heads=2
    )
    context_vecs = mha(batch)

    print(context_vecs)
    print("\ncontext_vecs.shape:", context_vecs.shape)
    print("\n################################################################\n")

