"""
CvT-13 backbone: Microsoft official implementation if available,
otherwise fallback to a simplified custom CvT.
"""

import torch
import torch.nn as nn
from functools import partial
from code.args import get_args
from types import SimpleNamespace


try:
    from code.models.cvt_ms.cls_cvt import ConvolutionalVisionTransformer
    _has_ms = True
except ImportError:
    _has_ms = False


from code.utils.logger import get_logger

#args = get_args()
#logger = get_logger(args.run_dir, name="train")


class ConvEmbedding(nn.Module):
    def __init__(self, in_channels=3, embed_dim=64, kernel_size=7, stride=4, padding=2):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size, stride, padding)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, (H, W)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x


class CvT(nn.Module):
    def __init__(self, img_size=224, in_channels=3, num_classes=2, embed_dim=64, depth=6, num_heads=4):
        super().__init__()
        self.embed = ConvEmbedding(in_channels, embed_dim)
        self.blocks = nn.ModuleList([TransformerBlock(embed_dim, num_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        self.num_features = embed_dim

    def forward(self, x):
        x, _ = self.embed(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        x = x.mean(1)
        return x


class MicrosoftCvTBackbone(nn.Module):
    """
    Wrapper around Microsoft CvT-13 implementation (without CvTConfig).
    """
    def __init__(self, pretrained=True):
        super().__init__()
        if not _has_ms:
            raise ImportError("Microsoft CvT implementation not found (cvt_ms.cls_cvt).")

        # Directly define configuration values (instead of CvTConfig)
        spec = {
            "NUM_STAGES": 3,
            "PATCH_SIZE": [7, 3, 3],
            "PATCH_STRIDE": [4, 2, 2],
            "PATCH_PADDING": [2, 1, 1],
            "DIM_EMBED": [64, 192, 384],
            "NUM_HEADS": [1, 3, 6],
            "DEPTH": [1, 2, 10],
            "MLP_RATIO": [4.0, 4.0, 4.0],
            "ATTN_DROP_RATE": [0.0, 0.0, 0.0],
            "DROP_RATE": [0.0, 0.0, 0.0],
            "DROP_PATH_RATE": [0.0, 0.0, 0.1],
            "QKV_BIAS": [True, True, True],
            "CLS_TOKEN": [False, False, True],
            "QKV_PROJ_METHOD": ["dw_bn", "dw_bn", "dw_bn"],
            "KERNEL_QKV": [3, 3, 3],
            "PADDING_KV": [1, 1, 1],
            "STRIDE_KV": [2, 2, 2],
            "PADDING_Q": [1, 1, 1],
            "STRIDE_Q": [1, 1, 1],
            "INIT": "trunc_norm"
        }

        self.model = ConvolutionalVisionTransformer(
            in_chans=3,
            num_classes=2,
            act_layer=nn.GELU,
            norm_layer=partial(nn.LayerNorm, eps=1e-5),
            init=spec["INIT"],
            spec=SimpleNamespace(**spec)  # Convert dict to object with attributes
        )
        self.num_features = 384

        if pretrained:
            try:
                self.model.init_weights(pretrained='', pretrained_layers=['*'], verbose=True)
            except Exception:
                pass

    def forward(self, x):
        return self.model.forward_features(x)


def cvt_13(pretrained=False, **kwargs):
    if _has_ms:
        return MicrosoftCvTBackbone(pretrained=pretrained)
    return CvT(embed_dim=64, depth=6, num_heads=4, **kwargs)