from functools import partial

import torch

from models.prenorm_vit import PrenormVit

dependencies = ["torch", "kappamodules", "einops"]

VIT_CONFIGS = dict(
    b8=dict(patch_size=8, dim=768, depth=12, num_heads=12),
    b16=dict(patch_size=16, dim=768, depth=12, num_heads=12),
    l16=dict(patch_size=16, dim=1024, depth=24, num_heads=16),
    h14=dict(patch_size=14, dim=1280, depth=32, num_heads=16),
    twob14=dict(patch_size=14, dim=2560, depth=24, num_heads=32),
)

URL_CONFIS = {
    "mae_b16": dict(
        ctor=PrenormVit,
        ctor_kwargs=VIT_CONFIGS["b16"],
        url="https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth",
        preprocess="mae",
    ),
    "mae_l16": dict(
        ctor=PrenormVit,
        ctor_kwargs=VIT_CONFIGS["l16"],
        url="https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_large.pth",
        preprocess="mae",
    ),
    "mae_h14": dict(
        ctor=PrenormVit,
        ctor_kwargs=VIT_CONFIGS["h14"],
        url="https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_huge.pth",
        preprocess="mae",
    ),
    "mae_twob14": dict(
        ctor=PrenormVit,
        ctor_kwargs=VIT_CONFIGS["twob14"],
        url="https://dl.fbaipublicfiles.com/maws/pretrain/mae_in1k/vit_2b14.pt",
        preprocess="mae",
    ),
}

TORCHHUB_CONFIGS = {
    # MIM-Refiner
    "mae_l16_refined": dict(repo_or_dir="ml-jku/MIM-Refiner", model="mae_refined_l16"),
    "mae_h14_refined": dict(repo_or_dir="ml-jku/MIM-Refiner", model="mae_refined_h14"),
    "mae_twob14_refined": dict(repo_or_dir="ml-jku/MIM-Refiner", model="mae_refined_twob14"),
    "d2v2_l16_refined": dict(repo_or_dir="ml-jku/MIM-Refiner", model="d2v2_refined_l16"),
    "d2v2_h14_refined": dict(repo_or_dir="ml-jku/MIM-Refiner", model="d2v2_refined_h14"),
    # DINOv2
    "dinov2_vits14": dict(repo_or_dir="facebookresearch/dinov2", model="dinov2_vits14"),
    "dinov2_vitb14": dict(repo_or_dir="facebookresearch/dinov2", model="dinov2_vitb14"),
    "dinov2_vitl14": dict(repo_or_dir="facebookresearch/dinov2", model="dinov2_vitl14"),
    "dinov2_vitg14": dict(repo_or_dir="facebookresearch/dinov2", model="dinov2_vitg14"),
    # DINOv2 with registers
    "dinov2_vits14_reg": dict(repo_or_dir="facebookresearch/dinov2", model="dinov2_vits14_reg"),
    "dinov2_vitb14_reg": dict(repo_or_dir="facebookresearch/dinov2", model="dinov2_vitb14_reg"),
    "dinov2_vitl14_reg": dict(repo_or_dir="facebookresearch/dinov2", model="dinov2_vitl14_reg"),
    "dinov2_vitg14_reg": dict(repo_or_dir="facebookresearch/dinov2", model="dinov2_vitg14_reg"),
}


def load_from_url(ctor, ctor_kwargs, url, preprocess, **kwargs):
    model = ctor(**ctor_kwargs, **kwargs)
    sd = torch.hub.load_state_dict_from_url(url, map_location="cpu")
    if preprocess == "mae":
        sd = sd["model"]
        # MAE uses flat patch_embed with 0s for CLS token, i.e. shape=(1, 197, dim)
        # convert to kappamodules format (retain spatial dimensions and remove CLS) -> (1, 14, 14, dim)
        assert "pos_embed" in sd
        pos_embed = sd.pop("pos_embed")
        assert torch.all(pos_embed[:, 0] == 0)
        pos_embed = pos_embed[:, 1:]
        sd["pos_embed.embed"] = pos_embed.reshape(*model.pos_embed.embed.shape)
        # kappamodules has different key for CLS token
        sd["cls_tokens.tokens"] = sd.pop("cls_token")
    else:
        raise NotImplementedError

    model.load_state_dict(sd)
    return model


for name, config in URL_CONFIS.items():
    globals()[name] = partial(load_from_url, **config)

for name, config in TORCHHUB_CONFIGS.items():
    globals()[name] = partial(torch.hub.load, **config)
