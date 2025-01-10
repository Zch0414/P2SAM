# Preparing Models for P<sup>2</sup>SAM

Download the model weights of 
[SAM](https://github.com/facebookresearch/segment-anything), 
[MedSAM](https://github.com/bowang-lab/MedSAM), 
and [P<sup>2</sup>SAM's fine-tuned SAM](https://drive.google.com/drive/folders/19cQOpn2HvxNvgo5hf56Irjd3_d2y8qOS). Organize them as follows:

```
pretrained_weights/
  sam_vit_b.pth
  sam_vit_l.pth
  sam_vit_h.pth
  medsam_vit_b.pth
  endoscopy_full_base/checkpoint.pth
  endoscopy_full_large/checkpoint.pth
  endoscopy_lora_base/checkpoint.pth
  endoscopy_lora_large/checkpoint.pth
  nsclc_full_base/checkpoint.pth
  nsclc_full_large/checkpoint.pth
  nsclc_lora_base/checkpoint.pth
  nsclc_lora_large/checkpoint.pth
```
