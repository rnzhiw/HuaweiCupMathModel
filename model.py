import torch
import torch.nn as nn
import math
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model = nn.Sequential(
            nn.LayerNorm([729]),
            nn.Dropout(0.1),
            nn.Linear(729, 512),
            nn.ReLU(inplace=True),
    
            nn.LayerNorm([512]),
            nn.Dropout(0.1),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
    
            nn.LayerNorm([128]),
            nn.Dropout(0.1),
            nn.Linear(128, 5)
        )

        
    def forward(self, x):
        y = self.model(x)
        return y
    
    

class Attention(nn.Module):
    def __init__(self, dim, ratio=4, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1, linear=False):
        super().__init__()

        self.scale = qk_scale or 1 ** -0.5
        self.ratio = ratio
        self.q = nn.Linear(dim, dim//ratio, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim*2//ratio, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim//ratio, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        B, C = x.shape
        q = self.q(x).unsqueeze(-1) # [1776, 729]
        kv = self.kv(x).reshape(B, C//self.ratio, -1, 1).permute(2, 0, 1, 3) # [1776, 1458]
        k, v = kv[0], kv[1] # [1776, 729, 1]
        # print(q.shape, k.shape, v.shape)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        o = (attn @ v).squeeze(-1)
        # print(x.shape)
        o = self.proj(o)
        o = self.proj_drop(o)

        return o + x
