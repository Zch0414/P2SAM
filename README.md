# [Part-aware Personalized Segment Anything Model for Patient-Specific Segmentation](https://arxiv.org/abs/2403.05433)

Official PyTorch implementation of $P^{2}SAM$, from the following paper:

[Part-aware Personalized Segment Anything Model for Patient-Specific Segmentation](https://arxiv.org/abs/2403.05433).\
Chenhui Zhao and Liyue Shen\
University of Michigan\
[[`arXiv`](https://arxiv.org/abs/2403.05433)]

--- 

<p align="center">
<img src="https://github.com/Zch0414/p2sam/blob/master/figures/method_1.jpg" width=100% height=100% 
class="center">
</p>

<p align="center">
<img src="https://github.com/Zch0414/p2sam/blob/master/figures/method_2.jpg" width=100% height=100% 
class="center">
</p>

We propose $P^{2}SAM$, a method to adapt a segmentation model to any new patients relying only on one-shot patient-specific data. $P^{2}SAM$ comprises a novel part-aware prompt mechanism and distribution-based retrieval approach to filter outlier prompts. These two components effectively mitigate ambiguity and enhance the robust generalization capacity.

## Todo list
- [x] PerSeg Demo Code
- [ ] Fine-tuned Model and Patient-Specific Segmentation Code on 4D-Lung Dataset  
- [ ] Fine-tuned Model and Patient-Specific Segmentation Code on CVC-ClinicDB Dataset
- [ ] SAM Fine-tuning Code

<!-- ✅ ⬜️  -->

## Results and Fine-tuned Models (coming soon)
### Qualitative Result on 4D-Lung and CVC-ClinicDB Datasets

<p align="center">
<img src="https://github.com/Zch0414/p2sam/blob/master/figures/result_1.jpg" width=100% height=100% 
class="center">
</p>

### Qualitative Result on PerSeg Dataset

<p align="center">
<img src="https://github.com/Zch0414/p2sam/blob/master/figures/result_2.jpg" width=100% height=100% 
class="center">
</p>

### ImageNet-1K trained models

| name | resolution |acc@1 | #params | FLOPs | model |
|:---:|:---:|:---:|:---:| :---:|:---:|
| ConvNeXt-T | 224x224 | 82.1 | 28M | 4.5G | [model](https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth) |
| ConvNeXt-S | 224x224 | 83.1 | 50M | 8.7G | [model](https://dl.fbaipublicfiles.com/convnext/convnext_small_1k_224_ema.pth) |
| ConvNeXt-B | 224x224 | 83.8 | 89M | 15.4G | [model](https://dl.fbaipublicfiles.com/convnext/convnext_base_1k_224_ema.pth) |
| ConvNeXt-B | 384x384 | 85.1 | 89M | 45.0G | [model](https://dl.fbaipublicfiles.com/convnext/convnext_base_1k_384.pth) |
| ConvNeXt-L | 224x224 | 84.3 | 198M | 34.4G | [model](https://dl.fbaipublicfiles.com/convnext/convnext_large_1k_224_ema.pth) |
| ConvNeXt-L | 384x384 | 85.5 | 198M | 101.0G | [model](https://dl.fbaipublicfiles.com/convnext/convnext_large_1k_384.pth) |


## Installation
Please check [INSTALL.md](INSTALL.md) for installation instructions. 

## Evaluation
We give an example evaluation command for a ImageNet-22K pre-trained, then ImageNet-1K fine-tuned ConvNeXt-B:

Single-GPU
```
python main.py --model convnext_base --eval true \
--resume https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_1k_224.pth \
--input_size 224 --drop_path 0.2 \
--data_path /path/to/imagenet-1k
```
Multi-GPU
```
python -m torch.distributed.launch --nproc_per_node=8 main.py \
--model convnext_base --eval true \
--resume https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_1k_224.pth \
--input_size 224 --drop_path 0.2 \
--data_path /path/to/imagenet-1k
```

This should give 
```
* Acc@1 85.820 Acc@5 97.868 loss 0.563
```

- For evaluating other model variants, change `--model`, `--resume`, `--input_size` accordingly. You can get the url to pre-trained models from the tables above. 
- Setting model-specific `--drop_path` is not strictly required in evaluation, as the `DropPath` module in timm behaves the same during evaluation; but it is required in training. See [TRAINING.md](TRAINING.md) or our paper for the values used for different models.

## Training
See [TRAINING.md](TRAINING.md) for training and fine-tuning instructions.

## Acknowledgement
This repository is built using the [timm](https://github.com/rwightman/pytorch-image-models) library, [DeiT](https://github.com/facebookresearch/deit) and [BEiT](https://github.com/microsoft/unilm/tree/master/beit) repositories.

## License
This project is released under the MIT license. Please see the [LICENSE](LICENSE) file for more information.

## Citation
If you find this repository helpful, please consider citing:
```
@Article{liu2022convnet,
  author  = {Zhuang Liu and Hanzi Mao and Chao-Yuan Wu and Christoph Feichtenhofer and Trevor Darrell and Saining Xie},
  title   = {A ConvNet for the 2020s},
  journal = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year    = {2022},
}
```
