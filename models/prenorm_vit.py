import einops
import torch
from kappamodules.transformer import PrenormBlock
from kappamodules.vit import VitPatchEmbed, VitPosEmbed2d, VitClassTokens
from torch import nn


class PrenormVit(nn.Module):
    def __init__(
            self,
            patch_size,
            dim,
            depth,
            num_heads,
            layerscale=None,
            input_shape=(3, 224, 224),
            mlp_hidden_dim=None,
            drop_path_rate=0.,
            drop_path_decay=True,
            num_cls_tokens=1,
            use_last_norm=True,
            eps=1e-6,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.dim = dim
        self.depth = depth
        self.num_heads = num_heads
        self.layerscale = layerscale
        self.input_shape = input_shape
        self.drop_path_rate = drop_path_rate
        self.drop_path_decay = drop_path_decay
        self.use_last_norm = use_last_norm
        self.eps = eps

        # initialize patch_embed
        self.patch_embed = VitPatchEmbed(
            dim=dim,
            num_channels=input_shape[0],
            resolution=input_shape[1:],
            patch_size=patch_size,
        )

        # pos embed
        self.pos_embed = VitPosEmbed2d(seqlens=self.patch_embed.seqlens, dim=dim)

        # 0, 1 or more cls tokens
        self.cls_tokens = VitClassTokens(dim=dim, num_tokens=num_cls_tokens)

        # stochastic depth
        if drop_path_decay and drop_path_rate > 0.:
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        else:
            dpr = [drop_path_rate] * depth

        # blocks
        self.blocks = nn.ModuleList([
            PrenormBlock(
                dim=dim,
                num_heads=num_heads,
                mlp_hidden_dim=mlp_hidden_dim,
                layerscale=layerscale,
                norm_ctor=nn.LayerNorm,
                drop_path=dpr[i],
                eps=eps,
            )
            for i in range(depth)
        ])
        if use_last_norm:
            self.norm = nn.LayerNorm(dim, eps=eps)
        else:
            self.norm = nn.Identity()

        self.output_shape = (self.patch_embed.num_patches + self.cls_tokens.num_tokens, dim)

    def forward(self, x):
        # embed patches
        x = self.patch_embed(x)
        # add pos_embed
        x = self.pos_embed(x)
        # flatten to 1d
        x = einops.rearrange(x, "b ... d -> b (...) d")
        # add cls token
        x = self.cls_tokens(x)
        # apply blocks
        for blk in self.blocks:
            x = blk(x)
        # last norm
        x = self.norm(x)
        return x
