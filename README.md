# P<sup>2</sup>SAM
> Official PyTorch implementation of the following paper:\
> Part-aware Personalized Segment Anything Model for Patient-Specific Segmentation\
> Chenhui Zhao and Liyue Shen\
> University of Michigan\
> [![arXiv](https://img.shields.io/badge/arXiv%20paper-2403.05433-b31b1b.svg)](https://arxiv.org/abs/2403.05433)&nbsp;

## Overview
<p align="center"><img src="https://github.com/Zch0414/p2sam/blob/master/figures/intro.jpg" width=96% height=96% class="center"></p>

We propose P<sup>2</sup>SAM, a training-free method that adapts a promptable segmentation model to new patients using only one-shot patient-specific data. 
P<sup>2</sup>SAM incorporates a **part-aware prompt mechanism** and a **distribution-based retrieval approach** to enhance generalization across:

- **Tasks:** P<sup>2</sup>SAM enhances generalization across different patient-specific segmentation tasks.
- **Models:** P<sup>2</sup>SAM can be plugged into various promptable segmentation models, such as SAM, fine-tuned SAM, and SAM 2.
- **Domains:** P<sup>2</sup>SAM performs effectively in both medical and natural image domains.

## Updates
- **(2025-01)** Release P<sup>2</sup>SAM's SAM fine-tuning code and fine-tuned models.
- **(2025-01)** Release P<sup>2</sup>SAM code for adaptive nsclc segmentation.
- **(2025-01)** Release P<sup>2</sup>SAM code for endoscopy video segmentation.
- **(2024-07)** Release P<sup>2</sup>SAM code for personalized segmentation.

## Getting Started

Create an Python environment with:
```
conda env create -f environment.yaml
```

Prepare datasets with:
- [Datasets](data/README.md)

Prepare pre-trained and fine-tuned models with:
- [Models](pretrained_weights/README.md)

Fine-tune SAM on custom datasets with:
- [Fine-tuning](fine_tuning/README.md)

Test P<sup>2</sup>SAM on out-of-distribution datasets with:
- [Testing](testing/README.md)

## Acknowledgement
This repository is built using the 
[DEiT](https://github.com/facebookresearch/deit/tree/main), 
[SAM](https://github.com/facebookresearch/segment-anything), 
[PerSAM](https://github.com/ZrrSkywalker/Personalize-SAM) 
repositories.

## Citation
If you find this repository helpful, please consider citing:
```
@article{zhao2024part,
  title={Part-aware Personalized Segment Anything Model for Patient-Specific Segmentation},
  author={Zhao, Chenhui and Shen, Liyue},
  journal={arXiv preprint arXiv:2403.05433},
  year={2024}
}
```
