import torch.nn as nn
import m_efficient_multi_head_attention as mha
import p_layernorm_gelu_feedforward as lgf

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = mha.MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"])
        self.ff = lgf.FeedForward(cfg)
        self.norm1 = lgf.LayerNorm(cfg["emb_dim"])
        self.norm2 = lgf.LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        return x
