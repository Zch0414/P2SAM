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

We propose $P^{2}SAM$, a training-free method to adapt the segmentation model to any new patients relying only on one-shot patient-specific data. $P^{2}SAM$ comprises a novel part-aware prompt mechanism and a distribution-based retrieval approach to filter outlier prompts. These two components effectively mitigate ambiguity and enhance the robust generalization capacity.

## Todo list
- [x] Personalized Segmentation code on PerSeg Dataset
- [ ] Fine-tuned Model on NSCLC-Radiomics Dataset
- [ ] Patient-Specific Segmentation Code on 4D-Lung Dataset
- [ ] Fine-tuned Model on Kvasir-SEG Dataset
- [ ] Patient-Specific Segmentation Code on CVC-ClinicDB Dataset
- [ ] SAM Fine-tuning Code

<!-- ✅ ⬜️  -->

## Results and Fine-tuned Models (coming soon)

### Fine-tuned on NSCLC-Radiomics and Tested on 4D-Lung

| name | direct transfer | $P^{2}SAM$ | # tuned params | model |
|:---:|:---:|:---:|:---:|:---:|
| SAM-B | 58.18 | 66.68 | 93.8M | coming soon |
| SAM-L | 61.11 | 67.23 | 312.5M | coming soon |
| SAM-B tuned with LoRA | 56.10 | 64.38 | 5.5M | coming soon |
| SAM-L tuned with LoRA | 57.83 | 67.00 | 5.9M | coming soon |

### Fine-tuned on Kvasir-SEG and Tested on CVC-ClinicDB

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

## Getting Started
### Personalized Segmentation on PerSeg
See [PerSAM](https://github.com/ZrrSkywalker/Personalize-SAM) to prepare the PerSeg dataset and SAM model weight. We give an example command for personalized segmentation on PerSeg:

First
```
python p2sam_perseg.py --data '/data/perseg' --outdir '/p2sam_perseg' \
--ckpt '/segment_anything_model/sam_vit_h.pth' --sam-type 'vit_h'\
--min-num-pos 1 --max-num-pos 5 \
```
Then
```
python eval_miou_perseg.py --pred-path '/p2sam_perseg' --gt-path '/data/perseg/Annotations' \
```

This should give 
```
* mIoU 95.6
```

- For evaluating other model variants, change `--ckpt`, and `--sam-type` accordingly.
- Setting `--vis` for visualization.

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
