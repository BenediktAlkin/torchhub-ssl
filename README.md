# Ready-to-use pre-trained self-supervised learning models loaded via torchhub

This repository provides an interface to load models from publicly available checkpoints.

ImageNet-1K pre-trained:

- [MAE](https://github.com/facebookresearch/mae#fine-tuning-with-pre-trained-checkpoints)
    - ViT-B/16 `model = torch.hub.load("BenediktAlkin/torchhub-ssl", "mae_b16")`
    - ViT-L/16 `model = torch.hub.load("BenediktAlkin/torchhub-ssl", "mae_l16")`
    - ViT-H/14 `model = torch.hub.load("BenediktAlkin/torchhub-ssl", "mae_h14")`
    - ViT-2B/14 `model = torch.hub.load("BenediktAlkin/torchhub-ssl", "mae_twob14")`
- [MAE refined (MIM-Refiner)](https://github.com/ml-jku/MIM-Refiner#pre-trained-models)
    - ViT-L/16 `model = torch.hub.load("BenediktAlkin/torchhub-ssl", "mae_l16_refined")`
    - ViT-H/14 `model = torch.hub.load("BenediktAlkin/torchhub-ssl", "mae_h14_refined")`
    - ViT-2B/14 `model = torch.hub.load("BenediktAlkin/torchhub-ssl", "mae_twob14_refined")`
- [data2vec 2.0 refined (MIM-Refiner)](https://github.com/ml-jku/MIM-Refiner#pre-trained-models)
    - ViT-L/16 `model = torch.hub.load("BenediktAlkin/torchhub-ssl", "d2v2_l16_refined")`
    - ViT-H/14 `model = torch.hub.load("BenediktAlkin/torchhub-ssl", "d2v2_h14_refined")`
- MoCo-v3 (TODO)
- DINO (TODO)
- iBOT (TODO)
- Mugs (TODO)

# ImageNet-21K

- iBOT (TODO)
- I-JEPA (TODO)

# LVD-142M pre-trained (dataset from DINOv2)

- [DINOv2](https://github.com/facebookresearch/dinov2)
    - ViT-S/14 `model = torch.hub.load("BenediktAlkin/torchhub-ssl", "dinov2_vits14")`
    - ViT-B/14 `model = torch.hub.load("BenediktAlkin/torchhub-ssl", "dinov2_vitb14")`
    - ViT-L/14 `model = torch.hub.load("BenediktAlkin/torchhub-ssl", "dinov2_vitl14")`
    - ViT-g/14 `model = torch.hub.load("BenediktAlkin/torchhub-ssl", "dinov2_vitg14")`
- [DINOv2 with registers](https://github.com/facebookresearch/dinov2)
    - ViT-S/14 `model = torch.hub.load("BenediktAlkin/torchhub-ssl", "dinov2_vits14_reg")`
    - ViT-B/14 `model = torch.hub.load("BenediktAlkin/torchhub-ssl", "dinov2_vitb14_reg")`
    - ViT-L/14 `model = torch.hub.load("BenediktAlkin/torchhub-ssl", "dinov2_vitl14_reg")`
    - ViT-g/14 `model = torch.hub.load("BenediktAlkin/torchhub-ssl", "dinov2_vitg14_reg")`

# IG-3B pre-trained

- [MAE](https://github.com/facebookresearch/maws)
    - TODO

Sources for the checkpoints are from the corresponding repositories:

- [MAE](https://github.com/facebookresearch/mae#fine-tuning-with-pre-trained-checkpoints)
- [MIM-Refiner](https://github.com/ml-jku/MIM-Refiner#pre-trained-models)
- [MAWS](https://github.com/facebookresearch/maws)
- [data2vec 2.0](https://github.com/facebookresearch/fairseq/tree/main/examples/data2vec)
- [DINOv2](https://github.com/facebookresearch/dinov2)