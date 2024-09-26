# Ready-to-use pre-trained self-supervised learning models loaded via torchhub

This repository provides an interface to load models from publicly available checkpoints.


ImageNet-1K pre-trained:

- [MAE](https://github.com/facebookresearch/mae#fine-tuning-with-pre-trained-checkpoints)
    - ViT-B/16 `model = torch.hub.load("BenediktAlkin/torchhub-ssl", "in1k_mae_b16")`
    - ViT-L/16 `model = torch.hub.load("BenediktAlkin/torchhub-ssl", "in1k_mae_l16")`
    - ViT-H/14 `model = torch.hub.load("BenediktAlkin/torchhub-ssl", "in1k_mae_h14")`
    - ViT-2B/14 `model = torch.hub.load("BenediktAlkin/torchhub-ssl", "in1k_mae_twob14")`
- [MAE refined (MIM-Refiner)](https://github.com/ml-jku/MIM-Refiner#pre-trained-models)
    - ViT-L/16 `model = torch.hub.load("BenediktAlkin/torchhub-ssl", "in1k_mae_l16_refined")`
    - ViT-H/14 `model = torch.hub.load("BenediktAlkin/torchhub-ssl", "in1k_mae_h14_refined")`
    - ViT-2B/14 `model = torch.hub.load("BenediktAlkin/torchhub-ssl", "in1k_mae_twob14_refined")`
- data2vec 2.0
    - ViT-B/16 `model = torch.hub.load("BenediktAlkin/torchhub-ssl", "in1k_d2v2_b16")`
    - ViT-L/16 `model = torch.hub.load("BenediktAlkin/torchhub-ssl", "in1k_d2v2_l16")`
    - ViT-H/14 `model = torch.hub.load("BenediktAlkin/torchhub-ssl", "in1k_d2v2_h14")`
- [data2vec 2.0 refined (MIM-Refiner)](https://github.com/ml-jku/MIM-Refiner#pre-trained-models)
    - ViT-L/16 `model = torch.hub.load("BenediktAlkin/torchhub-ssl", "in1k_d2v2_l16_refined")`
    - ViT-H/14 `model = torch.hub.load("BenediktAlkin/torchhub-ssl", "in1k_d2v2_h14_refined")`
- [dBOT](https://github.com/liuxingbin/dbot?tab=readme-ov-file#pre-trained-and-fine-tuned-models) (student = teacher architecture)
    - ViT-B/16 `model = torch.hub.load("BenediktAlkin/torchhub-ssl", "in1k_dbot_b16")`
    - ViT-L/16 `model = torch.hub.load("BenediktAlkin/torchhub-ssl", "in1k_dbot_l16")`
    - ViT-H/14 `model = torch.hub.load("BenediktAlkin/torchhub-ssl", "in1k_dbot_h14")`
- [dBOT refined (MIM-Refiner)](https://github.com/ml-jku/MIM-Refiner#pre-trained-models)
    - ViT-L/16 `model = torch.hub.load("BenediktAlkin/torchhub-ssl", "in1k_dbot_l16_refined")`
    - ViT-H/14 `model = torch.hub.load("BenediktAlkin/torchhub-ssl", "in1k_dbot_h14_refined")`
- DINO
    - ViT-S/16 `model = torch.hub.load("BenediktAlkin/torchhub-ssl", "in1k_dino_s16")`
    - ViT-S/8 `model = torch.hub.load("BenediktAlkin/torchhub-ssl", "in1k_dino_s8")`
    - ViT-B/16 `model = torch.hub.load("BenediktAlkin/torchhub-ssl", "in1k_dino_b16")`
    - ViT-B/8 `model = torch.hub.load("BenediktAlkin/torchhub-ssl", "in1k_dino_b8")`
- iBOT
    - ViT-S/16 `model = torch.hub.load("BenediktAlkin/torchhub-ssl", "in1k_ibot_s16")`
    - ViT-B/16 `model = torch.hub.load("BenediktAlkin/torchhub-ssl", "in1k_ibot_b16")`
    - ViT-L/16 `model = torch.hub.load("BenediktAlkin/torchhub-ssl", "in1k_ibot_l16")`
- [Mugs](https://github.com/sail-sg/mugs#pretrained-models-on-imagenet-1k)
    - ViT-S/16 `model = torch.hub.load("BenediktAlkin/torchhub-ssl", "in1k_mugs_s16")`
    - ViT-B/16 `model = torch.hub.load("BenediktAlkin/torchhub-ssl", "in1k_mugs_b16")`
    - ViT-L/16 `model = torch.hub.load("BenediktAlkin/torchhub-ssl", "in1k_mugs_l16")`
- [CrossMAE](https://github.com/TonyLianLong/CrossMAE?tab=readme-ov-file#models)
    - ViT-S/16 `model = torch.hub.load("BenediktAlkin/torchhub-ssl", "in1k_crossmae_s16")`
    - ViT-B/16 `model = torch.hub.load("BenediktAlkin/torchhub-ssl", "in1k_crossmae_b16")`
    - ViT-L/16 `model = torch.hub.load("BenediktAlkin/torchhub-ssl", "in1k_crossmae_l16")`
- [MAE-CT](https://github.com/ml-jku/MAE-CT?tab=readme-ov-file#mae-ctaug)
    - ViT-B/16 `model = torch.hub.load("BenediktAlkin/torchhub-ssl", "in1k_maectaug_b16")`
    - ViT-L/16 `model = torch.hub.load("BenediktAlkin/torchhub-ssl", "in1k_maectaug_l16")`
    - ViT-H/16 `model = torch.hub.load("BenediktAlkin/torchhub-ssl", "in1k_maectaug_h16")`
    - ViT-H/14 `model = torch.hub.load("BenediktAlkin/torchhub-ssl", "in1k_maectaug_h14")`
- [I-JEPA](https://github.com/facebookresearch/ijepa?tab=readme-ov-file#pretrained-models)
    - ViT-H/14 `model = torch.hub.load("BenediktAlkin/torchhub-ssl", "in1k_ijepa_h14")`
    - ViT-H/16_448 `model = torch.hub.load("BenediktAlkin/torchhub-ssl", "in1k_ijepa_h16res448")`

# ImageNet-21K pre-trained

- [I-JEPA](https://github.com/facebookresearch/ijepa?tab=readme-ov-file#pretrained-models)
    - ViT-H/14 `model = torch.hub.load("BenediktAlkin/torchhub-ssl", "in21k_ijepa_h14")`
    - ViT-g/16 `model = torch.hub.load("BenediktAlkin/torchhub-ssl", "in21k_ijepa_g16")`

# LVD-142M pre-trained (dataset from DINOv2)

- [DINOv2](https://github.com/facebookresearch/dinov2)
    - ViT-S/14 `model = torch.hub.load("BenediktAlkin/torchhub-ssl", "lvd142M_dinov2_vits14")`
    - ViT-B/14 `model = torch.hub.load("BenediktAlkin/torchhub-ssl", "lvd142M_dinov2_vitb14")`
    - ViT-L/14 `model = torch.hub.load("BenediktAlkin/torchhub-ssl", "lvd142M_dinov2_vitl14")`
    - ViT-g/14 `model = torch.hub.load("BenediktAlkin/torchhub-ssl", "lvd142M_dinov2_vitg14")`
- [DINOv2 with registers](https://github.com/facebookresearch/dinov2)
    - ViT-S/14 `model = torch.hub.load("BenediktAlkin/torchhub-ssl", "lvd142M_dinov2_vits14_reg")`
    - ViT-B/14 `model = torch.hub.load("BenediktAlkin/torchhub-ssl", "lvd142M_dinov2_vitb14_reg")`
    - ViT-L/14 `model = torch.hub.load("BenediktAlkin/torchhub-ssl", "lvd142M_dinov2_vitl14_reg")`
    - ViT-g/14 `model = torch.hub.load("BenediktAlkin/torchhub-ssl", "lvd142M_dinov2_vitg14_reg")`


# Infos

Old models that have public implementations without [FlashAttention](https://arxiv.org/abs/2205.14135)
are ported to an implementation that uses FlashAttention in the form of
[`scaled_dot_product_attention`](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) 
(requires `torch >= 2.0`).


# Sources

Sources for the checkpoints are from the corresponding repositories:

- [MAE](https://github.com/facebookresearch/mae#fine-tuning-with-pre-trained-checkpoints)
- [MIM-Refiner](https://github.com/ml-jku/MIM-Refiner#pre-trained-models)
- [MAWS](https://github.com/facebookresearch/maws)
- [data2vec 2.0](https://github.com/facebookresearch/fairseq/tree/main/examples/data2vec)
- [DINOv2](https://github.com/facebookresearch/dinov2)
- [Mugs](https://github.com/sail-sg/mugs#pretrained-models-on-imagenet-1k)
- [DINO](https://github.com/facebookresearch/dino#pretrained-models)
- [iBOT](https://github.com/bytedance/ibot#pre-trained-models)
- [CrossMAE](https://github.com/TonyLianLong/CrossMAE?tab=readme-ov-file#models)