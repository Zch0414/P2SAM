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

### 4D-Lung

| name | direct transfer | $P^{2}SAM$ | # tuned params | model |
|:---:|:---:|:---:|:---:|:---:|
| SAM-B | 58.18 | 66.68 | 93.8M | coming soon |
| SAM-L | 61.11 | 67.23 | 312.5M | coming soon |
| SAM-B tuned with LoRA | 56.10 | 64.38 | 5.5M | coming soon |
| SAM-L tuned with LoRA | 57.83 | 67.00 | 5.9M | coming soon |

### CVC-ClinicDB

| name | direct transfer | $P^{2}SAM$ | # tuned params | model |
|:---:|:---:|:---:|:---:|:---:|
| SAM-B | 84.62 | 86.40 | 93.8M | coming soon |
| SAM-L | 86.68 | 88.76 | 312.5M | coming soon |
| SAM-B tuned with LoRA | 77.20 | 81.16 | 5.5M | coming soon |
| SAM-L tuned with LoRA | 80.03 | 82.60 | 5.9M | coming soon |

### PerSeg

| name | PerSAM | PerSAM-F | $P^{2}SAM$ |
|:---:|:---:|:---:|:---:|
| SAM-B | 64.0 | 87.2 | 90.0 |
| SAM-L | 86.6 | 92.2 | 95.6 |
| SAM-H | 89.3 | 95.3 | 95.7 |

## Evaluation
We give an example evaluation command for PerSeg:

```
python main.py --model convnext_base --eval true \
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

## Fine-tuning on Medical Datasets
Coming soon.

## Acknowledgement
This repository is built using the [PerSAM](https://github.com/ZrrSkywalker/Personalize-SAM) repositories.

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
