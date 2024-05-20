from functools import partial

import torch

from models.prenorm_vit import PrenormVit
from models.postnorm_vit import PostnormVit

dependencies = ["torch", "kappamodules", "einops"]

VIT_CONFIGS = dict(
    s16=dict(patch_size=16, dim=384, depth=12, num_heads=6),
    s8=dict(patch_size=8, dim=384, depth=12, num_heads=6),
    b16=dict(patch_size=16, dim=768, depth=12, num_heads=12),
    b8=dict(patch_size=8, dim=768, depth=12, num_heads=12),
    l16=dict(patch_size=16, dim=1024, depth=24, num_heads=16),
    h16=dict(patch_size=16, dim=1280, depth=32, num_heads=16),
    h14=dict(patch_size=14, dim=1280, depth=32, num_heads=16),
    twob14=dict(patch_size=14, dim=2560, depth=24, num_heads=32),
)

URL_CONFIS = {
    # MAE
    "in1k_mae_b16": dict(
        ctor=PrenormVit,
        ctor_kwargs=VIT_CONFIGS["b16"],
        url="https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth",
        file_name="in1k_mae_b16",
        preprocess="mae",
    ),
    "in1k_mae_l16": dict(
        ctor=PrenormVit,
        ctor_kwargs=VIT_CONFIGS["l16"],
        url="https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_large.pth",
        file_name="in1k_mae_l16",
        preprocess="mae",
    ),
    "in1k_mae_h14": dict(
        ctor=PrenormVit,
        ctor_kwargs=VIT_CONFIGS["h14"],
        url="https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_huge.pth",
        file_name="in1k_mae_h14",
        preprocess="mae",
    ),
    "in1k_mae_twob14": dict(
        ctor=PrenormVit,
        ctor_kwargs=VIT_CONFIGS["twob14"],
        url="https://dl.fbaipublicfiles.com/maws/pretrain/mae_in1k/vit_2b14.pt",
        file_name="in1k_mae_twob14",
        preprocess="maws",
    ),
    # dBOT
    "in1k_dbot_b16": dict(
        ctor=PrenormVit,
        ctor_kwargs=VIT_CONFIGS["b16"],
        url="https://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/mmodal/dbot/84.5_dbot_base_pre.pth",
        file_name="in1k_dbot_b16",
        preprocess="dbot",
    ),
    "in1k_dbot_l16": dict(
        ctor=PrenormVit,
        ctor_kwargs=VIT_CONFIGS["l16"],
        url="https://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/mmodal/dbot/86.6_dbot_large_pre.pth",
        file_name="in1k_dbot_l16",
        preprocess="dbot",
    ),
    "in1k_dbot_h14": dict(
        ctor=PrenormVit,
        ctor_kwargs=VIT_CONFIGS["h14"],
        url="https://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/mmodal/dbot/87.4_dbot_huge_pre.pth",
        file_name="in1k_dbot_h14",
        preprocess="dbot",
    ),
    # MUGS
    "in1k_mugs_s16": dict(
        ctor=PrenormVit,
        ctor_kwargs=VIT_CONFIGS["s16"],
        url="https://huggingface.co/zhoupans/Mugs/resolve/main/pretrained%20models/vit_small_800ep/vit_small_backbone_800ep.pth",
        file_name="in1k_mugs_s16",
        preprocess="mugs",
    ),
    "in1k_mugs_b16": dict(
        ctor=PrenormVit,
        ctor_kwargs=VIT_CONFIGS["b16"],
        url="https://huggingface.co/zhoupans/Mugs/resolve/main/pretrained%20models/vit_base_400ep/vit_base_backbone_400ep.pth",
        file_name="in1k_mugs_b16",
        preprocess="mugs",
    ),
    "in1k_mugs_l16": dict(
        ctor=PrenormVit,
        ctor_kwargs=VIT_CONFIGS["l16"],
        url="https://huggingface.co/zhoupans/Mugs/resolve/main/pretrained%20models/vit_large_250ep/vit_large_backbone_250ep.pth",
        file_name="in1k_mugs_l16",
        preprocess="mugs",
    ),
    # DINO
    "in1k_dino_s16": dict(
        ctor=PrenormVit,
        ctor_kwargs=VIT_CONFIGS["s16"],
        url="https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth",
        file_name="in1k_dino_s16.pt",
        preprocess="dino",
    ),
    "in1k_dino_s8": dict(
        ctor=PrenormVit,
        ctor_kwargs=VIT_CONFIGS["s8"],
        url="https://dl.fbaipublicfiles.com/dino/dino_deitsmall8_pretrain/dino_deitsmall8_pretrain.pth",
        file_name="in1k_dino_s8.pt",
        preprocess="dino",
    ),
    "in1k_dino_b16": dict(
        ctor=PrenormVit,
        ctor_kwargs=VIT_CONFIGS["b16"],
        url="https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth",
        file_name="in1k_dino_b16.pt",
        preprocess="dino",
    ),
    "in1k_dino_b8": dict(
        ctor=PrenormVit,
        ctor_kwargs=VIT_CONFIGS["b8"],
        url="https://dl.fbaipublicfiles.com/dino/dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth",
        file_name="in1k_dino_b8.pt",
        preprocess="dino",
    ),
    # iBOT
    "in1k_ibot_s16": dict(
        ctor=PrenormVit,
        ctor_kwargs=VIT_CONFIGS["s16"],
        url="https://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/archive/2022/ibot/vits_16/checkpoint_teacher.pth",
        file_name="in1k_ibot_s16_teacher.pth",
        preprocess="ibot",
    ),
    "in1k_ibot_b16": dict(
        ctor=PrenormVit,
        ctor_kwargs=VIT_CONFIGS["b16"],
        url="https://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/archive/2022/ibot/vitb_16_rand_mask/checkpoint_teacher.pth",
        file_name="in1k_ibot_b16_teacher.pth",
        preprocess="ibot",
    ),
    "in1k_ibot_l16": dict(
        ctor=PrenormVit,
        ctor_kwargs=VIT_CONFIGS["l16"],
        url="https://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/archive/2022/ibot/vitl_16_rand_mask/checkpoint_teacher.pth",
        file_name="in1k_ibot_l16_teacher.pth",
        preprocess="ibot",
    ),
    # data2vec 2.0
    "in1k_d2v2_b16": dict(
        ctor=PostnormVit,
        ctor_kwargs=VIT_CONFIGS["b16"],
        url="https://dl.fbaipublicfiles.com/fairseq/data2vec2/base_imagenet.pt",
        file_name="in1k_d2v2_b16.pt",
        preprocess="d2v2",
    ),
    "in1k_d2v2_l16": dict(
        ctor=PostnormVit,
        ctor_kwargs=VIT_CONFIGS["l16"],
        url="https://dl.fbaipublicfiles.com/fairseq/data2vec2/large_imagenet.pt",
        file_name="in1k_d2v2_l16.pt",
        preprocess="d2v2",
    ),
    "in1k_d2v2_h14": dict(
        ctor=PostnormVit,
        ctor_kwargs=VIT_CONFIGS["h14"],
        url="https://dl.fbaipublicfiles.com/fairseq/data2vec2/huge_imagenet.pt",
        file_name="in1k_d2v2_h14.pt",
        preprocess="d2v2",
    ),
    # CrossMAE
    "in1k_crossmae_s16": dict(
        ctor=PrenormVit,
        ctor_kwargs=dict(**VIT_CONFIGS["s16"], use_last_norm=False),
        url="https://huggingface.co/longlian/CrossMAE/resolve/main/vits-mr0.75-kmr0.75-dd12/imagenet-mae-cross-vits-pretrain-wfm-mr0.75-kmr0.75-dd12-ep800-ui.pth",
        file_name="in1k_crossmae_s16",
        preprocess="crossmae",
    ),
    "in1k_crossmae_b16": dict(
        ctor=PrenormVit,
        ctor_kwargs=dict(**VIT_CONFIGS["b16"], use_last_norm=False),
        url="https://huggingface.co/longlian/CrossMAE/resolve/main/vitb-mr0.75-kmr0.75-dd12/imagenet-mae-cross-vitb-pretrain-wfm-mr0.75-kmr0.75-dd12-ep800-ui.pth",
        file_name="in1k_crossmae_b16",
        preprocess="crossmae",
    ),
    "in1k_crossmae_b16res448": dict(
        ctor=PrenormVit,
        ctor_kwargs=dict(**VIT_CONFIGS["b16"], use_last_norm=False, input_shape=(3, 448, 448)),
        url="https://huggingface.co/longlian/CrossMAE/resolve/main/vitb-mr0.75-kmr0.75-dd12-448-400/imagenet-mae-cross-vitb-pretrain-wfm-mr0.75-kmr0.25-dd12-ep400-ui-res-448.pth",
        file_name="in1k_crossmae_b16res448",
        preprocess="crossmae",
    ),
    "in1k_crossmae_l16": dict(
        ctor=PrenormVit,
        ctor_kwargs=dict(**VIT_CONFIGS["l16"], use_last_norm=False),
        url="https://huggingface.co/longlian/CrossMAE/resolve/main/vitl-mr0.75-kmr0.75-dd12/imagenet-mae-cross-vitl-pretrain-wfm-mr0.75-kmr0.75-dd12-ep800-ui.pth",
        file_name="in1k_crossmae_l16",
        preprocess="crossmae",
    ),
    # MAE-CT
    "in1k_maectaug_b16": dict(
        ctor=PrenormVit,
        ctor_kwargs=VIT_CONFIGS["b16"],
        url="https://ml.jku.at/research/maect/download/maectaug_base16.th",
        file_name="in1k_maectaug_b16",
        preprocess="maect",
    ),
    "in1k_maectaug_l16": dict(
        ctor=PrenormVit,
        ctor_kwargs=VIT_CONFIGS["l16"],
        url="https://ml.jku.at/research/maect/download/maectaug_large16.th",
        file_name="in1k_maectaug_l16",
        preprocess="maect",
    ),
    "in1k_maectaug_h16": dict(
        ctor=PrenormVit,
        ctor_kwargs=VIT_CONFIGS["h16"],
        url="https://ml.jku.at/research/maect/download/maectaug_huge16.th",
        file_name="in1k_maectaug_h16",
        preprocess="maect",
    ),
    "in1k_maectaug_h14": dict(
        ctor=PrenormVit,
        ctor_kwargs=VIT_CONFIGS["h14"],
        url="https://ml.jku.at/research/maect/download/maectaug_huge14.th",
        file_name="in1k_maectaug_h14",
        preprocess="maect",
    ),
}

TORCHHUB_CONFIGS = {
    # MIM-Refiner
    "in1k_mae_l16_refined": dict(repo_or_dir="ml-jku/MIM-Refiner", model="mae_refined_l16"),
    "in1k_mae_h14_refined": dict(repo_or_dir="ml-jku/MIM-Refiner", model="mae_refined_h14"),
    "in1k_mae_twob14_refined": dict(repo_or_dir="ml-jku/MIM-Refiner", model="mae_refined_twob14"),
    "in1k_d2v2_l16_refined": dict(repo_or_dir="ml-jku/MIM-Refiner", model="d2v2_refined_l16"),
    "in1k_d2v2_h14_refined": dict(repo_or_dir="ml-jku/MIM-Refiner", model="d2v2_refined_h14"),
    # DINOv2
    "lvd142m_dinov2_vits14": dict(repo_or_dir="facebookresearch/dinov2", model="dinov2_vits14"),
    "lvd142m_dinov2_vitb14": dict(repo_or_dir="facebookresearch/dinov2", model="dinov2_vitb14"),
    "lvd142m_dinov2_vitl14": dict(repo_or_dir="facebookresearch/dinov2", model="dinov2_vitl14"),
    "lvd142m_dinov2_vitg14": dict(repo_or_dir="facebookresearch/dinov2", model="dinov2_vitg14"),
    # DINOv2 with registers
    "lvd142m_dinov2_vits14_reg": dict(repo_or_dir="facebookresearch/dinov2", model="dinov2_vits14_reg"),
    "lvd142m_dinov2_vitb14_reg": dict(repo_or_dir="facebookresearch/dinov2", model="dinov2_vitb14_reg"),
    "lvd142m_dinov2_vitl14_reg": dict(repo_or_dir="facebookresearch/dinov2", model="dinov2_vitl14_reg"),
    "lvd142m_dinov2_vitg14_reg": dict(repo_or_dir="facebookresearch/dinov2", model="dinov2_vitg14_reg"),
}


def load_from_url(ctor, ctor_kwargs, url, preprocess, file_name=None, **kwargs):
    model = ctor(**ctor_kwargs, **kwargs)
    sd = torch.hub.load_state_dict_from_url(url, map_location="cpu", file_name=file_name)
    if preprocess in ["mae", "maws", "crossmae", "dbot"]:
        if preprocess in ["mae", "crossmae", "dbot"]:
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
        # remove decoder
        if preprocess in ["crossmae", "dbot"]:
            sd = {
                key: value for key, value in sd.items()
                if "decoder" not in key and "dec_" not in key and "wfm" not in key and "mask_token" not in key
            }
    elif preprocess == "mugs":
        sd = sd["state_dict"]
        # Mugs uses flat patch_embed with pos_embed for CLS token != 0, i.e. shape=(1, 197, dim)
        # convert to kappamodules format (retain spatial dimensions and include pos_embed into CLS) -> (1, 14, 14, dim)
        assert "pos_embed" in sd
        assert "cls_token" in sd
        pos_embed = sd.pop("pos_embed")
        cls_token = sd.pop("cls_token") + pos_embed[:, :1]
        pos_embed = pos_embed[:, 1:]
        sd["pos_embed.embed"] = pos_embed.reshape(*model.pos_embed.embed.shape)
        # kappamodules has different key for CLS token
        sd["cls_tokens.tokens"] = cls_token
        # remove relation_blocks
        sd = {key: value for key, value in sd.items() if not key.startswith("relation_blocks")}
    elif preprocess == "dino":
        # DINO uses flat patch_embed with pos_embed for CLS token != 0, i.e. shape=(1, 197, dim)
        # convert to kappamodules format (retain spatial dimensions and include pos_embed into CLS) -> (1, 14, 14, dim)
        assert "pos_embed" in sd
        assert "cls_token" in sd
        pos_embed = sd.pop("pos_embed")
        cls_token = sd.pop("cls_token") + pos_embed[:, :1]
        pos_embed = pos_embed[:, 1:]
        sd["pos_embed.embed"] = pos_embed.reshape(*model.pos_embed.embed.shape)
        # kappamodules has different key for CLS token
        sd["cls_tokens.tokens"] = cls_token
    elif preprocess == "ibot":
        sd = sd["state_dict"]
        # iBOT uses flat patch_embed with pos_embed for CLS token != 0, i.e. shape=(1, 197, dim)
        # convert to kappamodules format (retain spatial dimensions and include pos_embed into CLS) -> (1, 14, 14, dim)
        assert "pos_embed" in sd
        assert "cls_token" in sd
        pos_embed = sd.pop("pos_embed")
        cls_token = sd.pop("cls_token") + pos_embed[:, :1]
        pos_embed = pos_embed[:, 1:]
        sd["pos_embed.embed"] = pos_embed.reshape(*model.pos_embed.embed.shape)
        # kappamodules has different key for CLS token
        sd["cls_tokens.tokens"] = cls_token
    elif preprocess == "d2v2":
        sd = sd["model"]
        sd["patch_embed.proj.weight"] = sd.pop("modality_encoders.IMAGE.local_encoder.proj.weight")
        sd["patch_embed.proj.bias"] = sd.pop("modality_encoders.IMAGE.local_encoder.proj.bias")
        sd["cls_tokens.tokens"] = sd.pop("modality_encoders.IMAGE.extra_tokens")
        pos_embed = sd.pop("modality_encoders.IMAGE.fixed_positional_encoder.positions")
        sd["pos_embed.embed"] = pos_embed.reshape(*model.pos_embed.embed.shape)
        sd["embed_norm.weight"] = sd.pop("modality_encoders.IMAGE.context_encoder.norm.weight")
        sd["embed_norm.bias"] = sd.pop("modality_encoders.IMAGE.context_encoder.norm.bias")
        sd = {k: v for k, v in sd.items() if "decoder" not in k}
        sd.pop("_ema", None)
    elif preprocess == "maect":
        # MAE-CT uses flat pos_embed no entry for the CLS token, i.e. shape=(1, 196, dim)
        # convert to kappamodules format (retain spatial dimensions and remove CLS) -> (1, 14, 14, dim)
        assert "pos_embed" in sd
        pos_embed = sd.pop("pos_embed")
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
