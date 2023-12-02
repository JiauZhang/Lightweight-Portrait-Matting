import torch
import torch.nn as nn
from .swin_transformer_v2 import SwinTransformerV2, window_reverse

class SwinTransformerV2Encoder(SwinTransformerV2):
    def __init__(self, img_size=512):
        # /configs/swinv2/swinv2_tiny_patch4_window16_256.yaml
        super().__init__(
            img_size=img_size, patch_size=4, drop_path_rate=0.2,
            embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 16, 12, 24],
            window_size=16,
        )
        self.patch_size = 4
        self.window_size = 16
        self.down_scale = self.patch_size * 2 ** int(self.num_layers - 1)

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

def upsample(x):
    n, h, w, c = x.shape
    assert c % 4 == 0
    x0, x1, x2, x3 = torch.chunk(x, 4, dim=-1)
    x = torch.empty((n, 2*h, 2*w, c//4))
    x[:, 0::2, 0::2, :] = x0
    x[:, 1::2, 0::2, :] = x1
    x[:, 0::2, 1::2, :] = x2
    x[:, 1::2, 1::2, :] = x3
    return x

class Decoder(nn.Module):
    def __init__(self, in_chans):
        super().__init__()
        self.proj1 = nn.Linear(in_chans, in_chans)
        self.proj2 = nn.Linear(in_chans//4, in_chans//4)
        self.proj3 = nn.Linear(in_chans//16, in_chans//16)
        self.proj4 = nn.Linear(in_chans//64, 32)
        self.proj5 = nn.Linear(8, 16)

    def forward(self, x):
        x = self.proj1(x)
        x = upsample(x)
        x = self.proj2(x)
        x = upsample(x)
        x = self.proj3(x)
        x = upsample(x)
        x = self.proj4(x)
        x = upsample(x)
        x = self.proj5(x)
        x = upsample(x)
        return x

class LiPM(nn.Module):
    def __init__(self):
        super().__init__()
        self.img_size = 512
        self.encoder = SwinTransformerV2Encoder(img_size=self.img_size)
        self.decoder = Decoder(self.encoder.num_features)

    def forward(self, x):
        hid = self.encoder(x) # n, h, w, c
        hid = self.decoder(hid)
        hid = hid.permute(0, 3, 1, 2)
        return hid