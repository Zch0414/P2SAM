# Fine-tune SAM on Custom Datasets

We provide an implementation to fine-tune SAM on custom datasets, adhering closely to its iterative training strategy. The code is built up on the [DEiT](https://github.com/facebookresearch/deit/tree/main) repository.

## Fine-tune on the NSCLC-Radiomics Dataset

**Base w/o LoRA**

```
python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py\
    --online-record \
    --name '${run_name}'\
    --batch-size 4\      
    --epochs 36\
    --lr 1e-4\
    --sam-type 'vit_b'\
    --encoder-type 'timm'\
    --pretrained-weight 'pretrained_weights/sam_vit_b.pth'\
    --opt adamW\
    --weight-decay 0.01\
    --warmup-epochs 4\
    --cooldown-epochs 0\
    --dataset 'nsclc-radiomics'\
    --data-dir 'data/lung_pro/nsclc_radiomics/'\
    --output-dir '${output_dir}'\
    --seed 42 
```

**Base w/ LoRA**

```
python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py\
    --online-record \
    --name '${run_name}'\
    --batch-size 4\      
    --epochs 36\
    --lr 1e-4\
    --sam-type 'vit_b'\
    --encoder-type 'timm'\
    --pretrained-weight 'pretrained_weights/sam_vit_b.pth'\
    --lora \
    --lora-rank 1\
    --opt adamW\
    --weight-decay 0.01\
    --warmup-epochs 4\
    --cooldown-epochs 0\
    --dataset 'nsclc-radiomics'\
    --data-dir 'data/lung_pro/nsclc_radiomics/'\
    --output-dir '${output_dir}'\
    --seed 42 
```

**Large w/o LoRA**

```
python -m torch.distributed.launch --nproc_per_node=2 --use_env main.py\
    --online-record \
    --name '${run_name}'\
    --batch-size 2\      
    --epochs 36\
    --lr 1e-4\
    --sam-type 'vit_l'\
    --encoder-type 'timm'\
    --pretrained-weight 'pretrained_weights/sam_vit_l.pth'\
    --opt adamW\
    --weight-decay 0.01\
    --warmup-epochs 4\
    --cooldown-epochs 0\
    --dataset 'nsclc-radiomics'\
    --data-dir 'data/lung_pro/nsclc_radiomics/'\
    --output-dir '${output_dir}'\
    --seed 42 
```

**Large w/ LoRA**

```
python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py\
    --online-record \
    --name '${run_name}'\
    --batch-size 4\      
    --epochs 36\
    --lr 1e-4\
    --sam-type 'vit_l'\
    --encoder-type 'timm'\
    --pretrained-weight 'pretrained_weights/sam_vit_l.pth'\
    --lora \
    --lora-rank 1\
    --opt adamW\
    --weight-decay 0.01\
    --warmup-epochs 4\
    --cooldown-epochs 0\
    --dataset 'nsclc-radiomics'\
    --data-dir 'data/lung_pro/nsclc_radiomics/'\
    --output-dir '${output_dir}'\
    --seed 42 
```

## Fine-tune on the Kvasir-SEG Dataset

**Base w/o LoRA**

```
python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py\
    --online-record \
    --name '${run_name}'\
    --batch-size 4\      
    --epochs 100\
    --lr 1e-4\
    --sam-type 'vit_b'\
    --encoder-type 'timm'\
    --pretrained-weight 'pretrained_weights/sam_vit_b.pth'\
    --opt adamW\
    --weight-decay 0.05\
    --warmup-epochs 10\
    --cooldown-epochs 0\
    --sched-on-updates \
    --dataset 'kvasir-seg'\
    --data-dir 'data/endoscopy_pro/kvasir_seg/'\
    --output-dir '${output_dir}'\
    --seed 42 
```

**Base w/ LoRA**

```
python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py\
    --online-record \
    --name '${run_name}'\
    --batch-size 4\      
    --epochs 100\
    --lr 1e-4\
    --sam-type 'vit_b'\
    --encoder-type 'timm'\
    --pretrained-weight 'pretrained_weights/sam_vit_b.pth'\
    --lora \
    --lora-rank 1\
    --opt adamW\
    --weight-decay 0.05\
    --warmup-epochs 10\
    --cooldown-epochs 0\
    --sched-on-updates \
    --dataset 'kvasir-seg'
    --data-dir 'data/endoscopy_pro/kvasir_seg/'\
    --output-dir '${output_dir}'\
    --seed 42 
```

**Large w/o LoRA**

```
python -m torch.distributed.launch --nproc_per_node=2 --use_env main.py\
    --online-record \
    --name '${run_name}'\
    --batch-size 2\      
    --epochs 100\
    --lr 1e-4\
    --sam-type 'vit_l'\
    --encoder-type 'timm'\
    --pretrained-weight 'pretrained_weights/sam_vit_l.pth'\
    --opt adamW\
    --weight-decay 0.05\
    --warmup-epochs 10\
    --cooldown-epochs 0\
    --sched-on-updates \
    --dataset 'kvasir-seg'\
    --data-dir 'data/endoscopy_pro/kvasir_seg/'\
    --output-dir '${output_dir}'\
    --seed 42
```

**Large w/ LoRA**

```
python -m torch.distributed.launch --nproc_per_node=2 --use_env main.py\
    --online-record \
    --name '${run_name}'\
    --batch-size 2\      
    --epochs 100\
    --lr 1e-4\
    --sam-type 'vit_l'\
    --encoder-type 'timm'\
    --pretrained-weight 'pretrained_weights/sam_vit_l.pth'\
    --lora \
    --lora-rank 1\
    --opt adamW\
    --weight-decay 0.05\
    --warmup-epochs 10\
    --cooldown-epochs 0\
    --sched-on-updates \
    --dataset 'kvasir-seg'\
    --data-dir 'data/endoscopy_pro/kvasir_seg/'\
    --output-dir '${output_dir}'\
    --seed 42 
```

## Note

These hyperparameters have not been carefully tuned. However, the implementation should achieve reasonable results: ~90% Dice score on the Kvasir-SEG dataset and ~65% Dice score on the NSCLC-Radiomics dataset without any human-provided prompts.
