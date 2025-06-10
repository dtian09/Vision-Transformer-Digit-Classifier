'''
Vision Transformer (ViT) class
'''
import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=28, patch_size=7, in_channels=1, emb_dim=64):
        '''
        img_size: size of each dimension of the image (e.g. 28 for 28x28)
        patch_size: size of each dimension of the patch (e.g. 7 for 7x7)
        in_channels: number of channels in the image (1 for grayscale, 3 for RGB)
        emb_dim: dimension of the embedding (e.g. 64 for 64-dimensional embedding)
        '''
        super().__init__()
        self.patch_size = patch_size
        self.patch_dim = patch_size * patch_size * in_channels
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Linear(self.patch_dim, emb_dim)

    def forward(self, x):
        '''
        x: input image (B, C, H, W)
        B: batch size (number of images in the batch)
        C: number of channels (1 for grayscale, 3 for RGB)
        H: height of the image
        W: width of the image
        '''
        B, C, H, W = x.shape
        x = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)#B x C x 4 x 4 x 7 x 7
        x = x.contiguous().view(B, C, -1, self.patch_size, self.patch_size)#flatten the patches into a single dimension B x C x 16 x 7 x 7
        x = x.permute(0, 2, 1, 3, 4).reshape(B, self.n_patches, -1)#B x 16 x 49
        x = self.proj(x)#B x 16 x emb_dim
        return x

class TransformerEncoderLayer(nn.Module):
    def __init__(self, emb_dim=64, n_heads=4, mlp_dim=256):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=emb_dim, num_heads=n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(emb_dim)
        self.norm2 = nn.LayerNorm(emb_dim)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, emb_dim)
        )

    def forward(self, x):
        x = x + self.attn(x, x, x, need_weights=False)[0]
        x = self.norm1(x)
        x = x + self.mlp(x)
        x = self.norm2(x)
        return x

class ViT(nn.Module):
    def __init__(self, img_size=28, patch_size=7, emb_dim=64, n_heads=4, mlp_dim=256, n_layers=6, n_classes=10):
        super().__init__()
        self.patch_size = patch_size
        self.patch_embed = PatchEmbedding(img_size, patch_size, 1, emb_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + 16, emb_dim))  # 16 patches + 1 cls
        self.transformer = nn.Sequential(*[
            TransformerEncoderLayer(emb_dim, n_heads, mlp_dim)
            for _ in range(n_layers)
        ])
        self.head = nn.Linear(emb_dim, n_classes)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)  # B x 16 x emb_dim
        cls_token = self.cls_token.expand(B, -1, -1)  # B x 1 x emb_dim
        x = torch.cat([cls_token, x], dim=1)  # B x 17 x emb_dim
        x = x + self.pos_embed  # Add position info
        x = self.transformer(x)
        return self.head(x[:, 0])  # Classify using [CLS] token
