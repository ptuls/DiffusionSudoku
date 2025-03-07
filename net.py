import torch
import torch.nn as nn
from einops import rearrange


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2), nn.GELU(), nn.Linear(dim * mult * 2, dim)
        )

    def forward(self, x):
        return self.net(x) + x


class Attention(nn.Module):
    def __init__(self, dim, heads=8):
        super().__init__()
        self.heads = heads
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim)

    def forward(self, x):
        q, k, v = map(
            lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads),
            self.to_qkv(x).chunk(3, dim=-1),
        )
        attn_out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        out = rearrange(attn_out, "b h n d -> b n (h d)", h=self.heads)
        return self.to_out(out)


class TransformerBlock(nn.Module):
    def __init__(self, head_dim, heads=8):
        super().__init__()
        dim = head_dim * heads
        self.attn = Attention(dim, heads)
        self.ff = FeedForward(dim)
        self.attn_norm = nn.LayerNorm(dim)
        self.ff_norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ff(self.ff_norm(x))
        return x


class Transformer(nn.Module):
    def __init__(self, head_dim=64, heads=8, num_classes=10, depth=12, seq_len=81):
        super().__init__()
        dim = head_dim * heads
        self.embed = nn.Embedding(num_classes, dim)
        self.pos_emb = nn.Embedding(seq_len, dim)
        self.layers = nn.ModuleList([TransformerBlock(head_dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.to_logits = nn.Linear(dim, num_classes)

    def forward(self, x):
        x = self.embed(x)
        pos_idx = torch.arange(x.shape[1], device=x.device)
        pos_embs = self.pos_emb(pos_idx)
        x = x + pos_embs
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.to_logits(x)
