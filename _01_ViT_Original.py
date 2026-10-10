import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, patch_size=16, emb_size=768, img_size=224):
        super().__init__()
        self.patch_size = patch_size
        self.emb_size = emb_size
        self.img_size = img_size
        self.num_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim=768, num_heads=12):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def _attention(self, q, k, v):
        attn = torch.matmul(q, k.transpose(-2, -1))
        attn = attn / math.sqrt(k.size(-1))
        attn = F.softmax(attn, dim=-1)
        output = torch.matmul(attn, v)
        return output

    def forward(self, x):
        B, H, C = x.size()
        q = self.q_proj(x).view(B, H, self.num_heads, C // self.num_heads).transpose(1, 2)
        k = self.k_proj(x).view(B, H, self.num_heads, C // self.num_heads).transpose(1, 2)
        v = self.v_proj(x).view(B, H, self.num_heads, C // self.num_heads).transpose(1, 2)
        x = self._attention(q, k, v)
        x = self.out_proj(x.transpose(1, 2).contiguous().view(B, H, C))
        return x

class MLPBlock(nn.Module):
    def __init__(self, embed_dim=768, mlp_dim=3072):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, embed_dim)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim=768, num_heads=12, mlp_dim=3072):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim, eps=1e-12)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim, eps=1e-12)
        self.mlp = MLPBlock(embed_dim, mlp_dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class ViT(nn.Module):

    def __init__(self, in_channels=3, patch_size=16, emb_size=768, img_size=224, num_heads=12, mlp_dim=3072, num_layers=12, num_classes=10):
        super().__init__()
        self.patch_embedding = PatchEmbedding(in_channels, patch_size, emb_size, img_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_size))
        self.pos_embedding = nn.Parameter(torch.zeros(1, 1 + self.patch_embedding.num_patches, emb_size))
        self.transformer_blocks = nn.ModuleList([TransformerBlock(emb_size, num_heads, mlp_dim) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(emb_size, eps=1e-12)
        self.head = nn.Linear(emb_size, num_classes)

    def forward(self, x, labels=None):
        B = x.size(0)
        x = self.patch_embedding(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embedding
        for block in self.transformer_blocks:
            x = block(x)
        x = self.norm(x)
        cls_output = x[:, 0]
        logits = self.head(cls_output)
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
            return {"loss": loss,"logits": logits}
        return {"logits": logits}
