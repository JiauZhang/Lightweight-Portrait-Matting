import torch
import torch.nn as nn
from .swin_transformer_v2 import SwinTransformerV2, window_reverse

class SwinTransformerV2Encoder(SwinTransformerV2):
    def __init__(self, pretrained_backbone=False):
        # /configs/swinv2/swinv2_tiny_patch4_window16_256.yaml
        super().__init__(
            img_size=512, patch_size=4, drop_path_rate=0.2,
            embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 16, 12, 24],
            window_size=16,
        )
        self.patch_size = 4
        self.window_size = 16
        self.down_scale = self.patch_size * 2 ** int(self.num_layers - 1)
        if pretrained_backbone:
            self.load_state_dict(torch.hub.load_state_dict_from_url(
                'https://github.com/SwinTransformer/storage/releases/download/v2.0.0/swinv2_tiny_patch4_window16_256.pth'
            ))

        del self.norm
        del self.avgpool
        del self.head

    def forward(self, x):
        h, w = x.shape[2:]
        H, W = h // self.down_scale, w // self.down_scale
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)

        x = window_reverse(x, self.window_size, H, W)
        return x

class LiPM(nn.Module):
    def __init__(self, pretrained_backbone=False):
        super().__init__()
        self.encoder = SwinTransformerV2Encoder(pretrained_backbone)

    def forward(self, x):
        ...