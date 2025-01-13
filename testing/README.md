# Test P<sup>2</sup>SAM

Test P<sup>2</sup>SAM for 
adaptive radiation segmentation on the 4D-Lung dataset, 
endoscopy video segmentation on the CVC-ClinicDB dataset, 
peronalized segmentation on the PerSeg datasets.

## Adaptive Radiation Segmentation on 4D-Lung

**Direct-transfer Baseline**

```
python /direct_transfer/direct_transfer_4d_lung.py\
  --data 'data/lung_pro/4d_lung_multi_visits'\
  --outdir 'results/direct_transfer/4d_lung'\
  --ckpt 'pretrained_weights/nsclc_full_large/checkpoint.pth'\
  --sam-type 'vit_l'\
  --encoder-type 'timm'
```

**P<sup>2</sup>SAM**

```
python /p2sam/p2sam_4d_lung.py\
  --data 'data/lung_pro/4d_lung_multi_visits'\
  --outdir 'results/p2sam/4d_lung'\
  --ckpt 'pretrained_weights/nsclc_full_large/checkpoint.pth'\
  --sam-type 'vit_l'\
  --encoder-type 'timm'\
  --max-num-pos 3\
  --min-num-pos 1\
  --max-num-neg 1\
  --min-num-neg 1
```

## Endoscopy Video Segmentation on CVC-ClinicDB

**Direct-transfer Baseline**

```
python /direct_transfer/direct_transfer_cvc_clinicdb.py\
  --data 'data/endoscopy_pro/cvc_clinicdb'\
  --outdir 'results/direct_transfer/cvc_clinicdb'\
  --ckpt 'pretrained_weights/endoscopy_full_large/checkpoint.pth'\
  --sam-type 'vit_l'\
  --encoder-type 'timm'
```

**P<sup>2</sup>SAM**

```
python /p2sam/p2sam_cvc_clinicdb.py\
  --data 'data/endoscopy_pro/cvc_clinicdb'\
  --outdir 'results/p2sam/cvc_clinicdb'\
  --ckpt 'pretrained_weights/endoscopy_full_large/checkpoint.pth'\
  --sam-type 'vit_l'\
  --encoder-type 'timm'\
  --max-num-pos 5\
  --min-num-pos 1\
  --max-num-neg 1\
  --min-num-neg 1
```

## Personalized Segmentation on PerSeg

```
python /p2sam/p2sam_perseg.py\
  --data 'data/perseg'\
  --outdir 'results/p2sam/perseg'\
  --ckpt 'pretrained_weights/sam_vit_h.pth'\
  --sam-type 'vit_h'\
  -- guided-attn \
  --max-num-pos 5\
  --min-num-pos 1
```
