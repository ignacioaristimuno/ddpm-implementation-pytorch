"""
Implementation of a simple Denoising Diffusion Probabilistic Model based on the
blog 'The Annotated Diffusion' by Hugging Face and adapted so the code is more
begginer-friendly for those looking to understande Diffusion Models.

https://huggingface.co/blog/annotated-diffusion
"""

from functools import partial
import math
import torch
from torch import nn


class SinusoidalPositionEmbeddings(nn.Module):
    """
    The authors of the DDPM paper employ Sinusoidal Position Embeddings to encode t,
    inspired by the Transformer's architecture (Vaswani et al., 2017). This makes the
    neural network aware at which particular time step it is operating for every image
    in the batch, which relates to the noise level that is still present in the image.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ConvBlock(nn.Module):
    """
    Basic Convolutional Block inspired in the paper Wide Residual Networks (Zagoruyko
    et al., 2017). https://arxiv.org/pdf/1605.07146
    """

    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)  # Similar to BatchNorm
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if scale_shift is not None:
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x


class ResNetBlock(nn.Module):
    """
    ResNet Block inspired in the original ResNet paper 'Deep Residual Learning for
    Image Recognition', (Kaiming He et al., 2015). https://arxiv.org/abs/1512.03385
    """

    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out))
            if time_emb_dim is not None
            else None
        )
        self.block_1 = ConvBlock(dim, dim_out, groups=groups)
        self.block_2 = ConvBlock(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        h = self.block_1(x)

        if (self.mlp is not None) and (time_emb is not None):
            time_emb = self.mlp(time_emb)
            time_emb = time_emb.unsqueeze(-1).unsqueeze(-1)
            h = time_emb + h

        h = self.block_2(h)
        return h + self.res_conv(x)  # Residual in ResNet block (not U-Net)


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.GroupNorm(1, dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


class Attention(nn.Module):
    """
    Multi-head attention module as proposed in 'Attention Is All You Need'
    (Vaswani et al., 2017). https://arxiv.org/abs/1706.03762
    """

    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = [t.view(b, self.heads, -1, h * w) for t in qkv]
        q = q * self.scale
        sim = torch.einsum("bhid,bhjd->bhij", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)
        out = torch.einsum("bhij,bhjd->bhid", attn, v)
        out = out.view(b, -1, h, w)
        out = self.to_out(out)
        return out


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            nn.GroupNorm(1, dim),
        )

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = [t.view(b, self.heads, -1, h * w) for t in qkv]
        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)
        q = q * self.scale
        context = torch.einsum("bhd,bhe->bhde", k, v)
        out = torch.einsum("bhde,bhdn->bhen", context, q)
        out = out.view(b, -1, h, w)
        out = self.to_out(out)
        return out


class UNet(nn.Module):
    def __init__(
        self,
        image_size,  # image_size
        dim_mults=(1, 2, 4, 8),
        channels=3,
        with_time_emb=True,
        resnet_block_groups=8,
    ):
        super().__init__()
        init_dim = image_size // 3 * 2
        self.init_conv = nn.Conv2d(channels, init_dim, 7, padding=3)

        dims = [init_dim, *map(lambda m: image_size * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        resnet_block = partial(ResNetBlock, groups=resnet_block_groups)

        # Time embeddings
        if with_time_emb:
            time_dim = image_size * 4
            self.time_mlp = nn.Sequential(
                SinusoidalPositionEmbeddings(image_size),
                nn.Linear(image_size, time_dim),
                nn.GELU(),
                nn.Linear(time_dim, time_dim),
            )
        else:
            time_dim = None
            self.time_mlp = None

        # Downsample layers
        num_resolutions = len(in_out)
        self.downs = nn.ModuleList([])

        for k, (dim_in, dim_out) in enumerate(in_out):
            is_last = k >= (num_resolutions - 1)
            self.downs.append(
                nn.ModuleList(
                    [
                        resnet_block(dim_in, dim_out, time_emb_dim=time_dim),
                        resnet_block(dim_out, dim_out, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                        nn.Identity()
                        if is_last
                        else nn.Conv2d(dim_out, dim_out, 4, 2, 1),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = resnet_block(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = resnet_block(mid_dim, mid_dim, time_emb_dim=time_dim)

        # Upsample layers
        self.ups = nn.ModuleList([])
        for k, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = k >= (num_resolutions - 1)

            self.ups.append(
                nn.ModuleList(
                    [
                        resnet_block(dim_out * 2, dim_in, time_emb_dim=time_dim),
                        resnet_block(dim_in, dim_in, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                        nn.Identity()
                        if is_last
                        else nn.ConvTranspose2d(dim_in, dim_in, 4, 2, 1),
                    ]
                )
            )

        self.final_conv = nn.Sequential(
            resnet_block(image_size, image_size), nn.Conv2d(image_size, channels, 1)
        )

    def forward(self, x, time):
        x = self.init_conv(x)
        t = self.time_mlp(time) if (self.time_mlp is not None) else None
        h = []

        # Downsample
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            x = block2(x, t)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        # Bottleneck
        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        # Upsample
        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)
            x = block2(x, t)
            x = attn(x)
            x = upsample(x)

        return self.final_conv(x)
