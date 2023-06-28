import logging
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from typing import Optional, Tuple, Type, Union

from timm.models.vision_transformer import PatchEmbed
from timm.models.layers import trunc_normal_
import torchvision.transforms as transforms


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
        adapter_type: str = "vit"
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.scale_factor = scale_factor
        self.depth = depth
        self.adapter_type = adapter_type
        print("adapter_type: ", self.adapter_type)
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        
        self.handcrafted_embed_helper = nn.Linear(self.embed_dim, self.embed_dim//self.scale_factor)
        self.embedding_generator = nn.Linear(self.embed_dim, self.embed_dim//self.scale_factor)

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

    def get_prompt(self, handcrafted_feature, embedding_feature):
        prompts = []
        for i in range(self.depth):
            lightweight_mlp = getattr(self, 'lightweight_mlp_{}'.format(str(i)))
            # prompt = proj_prompt(prompt)
            prompt = lightweight_mlp(handcrafted_feature + embedding_feature)
            prompts.append(self.shared_mlp(prompt))
        return prompts

    def rgb2gray(self, x):
        x = transforms.Grayscale(num_output_channels=3)(x)
        return x
    
    def fft(self, x, rate):
        # the smaller rate, the smoother; the larger rate, the darker
        # rate = 4, 8, 16, 32
        mask = torch.zeros(x.shape).to(x.device)
        w, h = x.shape[-2:]
        line = int((w * h * rate) ** .5 // 2)
        mask[:, :, w//2-line:w//2+line, h//2-line:h//2+line] = 1

        fft = torch.fft.fftshift(torch.fft.fft2(x, norm="forward"))
        # mask[fft.float() > self.freq_nums] = 1
        # high pass: 1-mask, low pass: mask
        fft = fft * (1 - mask)
        # fft = fft * mask
        fr = fft.real
        fi = fft.imag

        fft_hires = torch.fft.ifftshift(torch.complex(fr, fi))
        inv = torch.fft.ifft2(fft_hires, norm="forward").real

        inv = torch.abs(inv)

        return inv
    
    def init_embeddings(self, x):
        return self.embedding_generator(x)
    
    def init_handcrafted(self, x):
        if self.adapter_type == "vit_grayscale":
            x = self.rgb2gray(x)
        if self.adapter_type == "vit_fft":
            x = self.fft(x, rate=0.25)
        return self.patch_embed(x)

    # def forward(self, x: torch.Tensor) -> torch.Tensor:
    #     return x