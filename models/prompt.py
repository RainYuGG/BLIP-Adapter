import logging
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from typing import Optional, Tuple, Type, Union

from timm.models.vision_transformer import PatchEmbed
from timm.models.layers import trunc_normal_



class PromptGenerator(nn.Module):
    """Prompt Generator (Adapter)"""
    def __init__(
        self,
        img_size: Union[int, tuple[int]] = 224,
        patch_size: Union[int, tuple[int]] = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        scale_factor: int = 32,
        prompt_type: str = "half"
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.scale_factor = scale_factor
        self.depth = depth

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        self.embed_helper = nn.Linear(embed_dim, embed_dim//scale_factor)
        
        for i in range(self.depth):
            lightweight_mlp = nn.Sequential(
                nn.Linear(self.embed_dim//self.scale_factor, self.embed_dim//self.scale_factor),
                nn.GELU()
            )
            setattr(self, 'lightweight_mlp_{}'.format(str(i)), lightweight_mlp)

        self.shared_mlp = nn.Linear(self.embed_dim//self.scale_factor, self.embed_dim)

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

    def get_prompt(self, embedding_feature):
        prompts = []
        for i in range(self.depth):
            lightweight_mlp = getattr(self, 'lightweight_mlp_{}'.format(str(i)))
            # prompt = proj_prompt(prompt)
            prompt = lightweight_mlp(embedding_feature)
            prompts.append(self.shared_mlp(prompt))
        return prompts

    # def forward(self, x: torch.Tensor) -> torch.Tensor:
    #     return x