import os
import time
import json
import argparse
import datetime
import random
import wandb
import numpy as np
import pickle5 as pickle

import torch
import torch.backends.cudnn as cudnn

from pathlib import Path

from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler

from datasets.kvasir_seg import build_dataset as build_endoscopy
from datasets.nsclc_radiomics import build_dataset as build_nsclc
from engine import train_one_epoch, evaluate

import utils
from model import create_model


def get_args_parser():
    parser = argparse.ArgumentParser('training and evaluation script', add_help=False)
    # Wandb
    parser.add_argument('--online-record', action='store_true')
    parser.add_argument('--name', default='', type=str)
    
    # Basic
    parser.add_argument('--batch-size', default=4, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--unscale-lr', action='store_true')

    # Model
    parser.add_argument('--sam-type', default='vit_b', type=str)
    parser.add_argument('--encoder-type', default='timm', type=str)
    parser.add_argument('--pretrained-weight', default='pretrained_weights/sam_vit_b.pth', type=str)

    ## Parameter-Efficient Fine-Tuning
    parser.add_argument('--lora', action='store_true')
    parser.add_argument('--lora-rank', default=1, type=int)
    parser.add_argument('--freeze-image-encoder', action='store_true')
    parser.add_argument('--freeze-sparse-prompt', action='store_true')
    parser.add_argument('--freeze-dense-prompt', action='store_true')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamW', type=str, metavar='OPTIMIZER',
                        help='Optimizer ("sgd", "adamW")')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='weight decay (default: 0.0 for SGD, 0.05 for adamW)')
    
    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler ("poly", "cosine")')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate (default: 1e-4)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-6)')
    
    parser.add_argument('--warmup-epochs', type=int, default=10, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=5, metavar='N',
                    help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=5, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1 for cosine, 0.9 for poly)')
    parser.add_argument('--sched-on-updates', action='store_false',
                        help='False for step on epochs. True for step on iterations.')
    parser.set_defaults(sched_on_updates=True)

    # Dataset parameters
    parser.add_argument('--dataset', default='kvasir-seg', type=str)
    parser.add_argument('--dataset-dir', default='data/endoscopy_pro/kvasir-seg/')
    parser.add_argument('--output-dir', default='',
                        help='path where to save, empty for no saving')
    
    # Resume && Evaluation
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--test', action='store_true', help='Perform test evaluation')

    # Device
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--dist-eval', action='store_true', default=False, help='Enabling distributed evaluation')
    parser.add_argument('--num-workers', default=8, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)

    # Distributed training parameters
    parser.add_argument('--distributed', action='store_true', default=False, help='Enabling distributed training')
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    # Will not affact any
    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR') 
    return parser


def main(args):
    utils.init_distributed_mode(args)
    print(args)
    if args.online_record:
        wandb.wandb.init(project='p2sam-training', name=args.name, config=args)
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) 
    cudnn.benchmark = False
    cudnn.deterministic = True

    if args.dataset == 'kvasir-seg':
        dataset_train, dataset_val, dataset_ts = build_endoscopy(args.pretrained_weight, root=args.dataset_dir)
    if args.dataset == 'nsclc-radiomics':
        dataset_train, dataset_val, dataset_ts = build_nsclc(args.pretrained_weight, root=args.dataset_dir)
    if args.test:
        args.eval = True
        dataset_val = dataset_ts
    print(f"Train dataset length: {len(dataset_train)}")
    print(f"Val dataset length: {len(dataset_val)}")

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=1,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    print(f"Creating model...")
    model = create_model(
        sam_type=args.sam_type, checkpoint=args.pretrained_weight, encoder_type=args.encoder_type, 
        lora=args.lora, r=args.lora_rank, enable_lora=[True, True, True], 
        freeze_image_encoder=args.freeze_image_encoder,
        freeze_sparse_prompt=args.freeze_sparse_prompt,
        freeze_dense_prompt=args.freeze_dense_prompt)
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    n_parameters_total = sum(p.numel() for p in model_without_ddp.parameters())
    print('number of total params: %.2f MB' % (n_parameters_total / 1024 / 1024))
    n_parameters_tuning = sum(p.numel() for p in model_without_ddp.parameters() if p.requires_grad)
    print('number of tuning params: %.2f MB' % (n_parameters_tuning / 1024 / 1024))
    
    if not args.unscale_lr:
        linear_scaled_lr = args.lr * args.batch_size * utils.get_world_size() / 256.0
        args.lr = linear_scaled_lr

    optimizer = create_optimizer(args, model_without_ddp)
    loss_scaler = NativeScaler()

    lr_scheduler, num_epochs = create_scheduler(args, optimizer, updates_per_epoch=len(data_loader_train))
    args.epochs = num_epochs

    output_dir = Path(args.output_dir)
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint) # [hard code] for medsam, use checkpoint, otherwise, use checkpoint['model'].
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            args.start_epoch = checkpoint['epoch'] + 1
            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])
    if args.eval:
        test_stats = evaluate(data_loader_val, model, device, args)
        if args.online_record:
            wandb.log(test_stats)
        return

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    best_result = checkpoint['result'] if args.resume else 0.0
    print('Current best result:', best_result)
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model, 
            data_loader_train, optimizer, lr_scheduler if args.sched_on_updates else None, 
            device, epoch, loss_scaler,
        )

        if not args.sched_on_updates:
            lr_scheduler.step(epoch)

        test_stats = evaluate(data_loader_val, model, device, args)

        if args.online_record:
            wandb.log({'train_stats': train_stats, 'test_stats': test_stats})

        curr_result = test_stats['no_prompt_dice']
        if args.output_dir and curr_result >= best_result:
            best_result = curr_result
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'scaler': loss_scaler.state_dict(),
                    'args': args,
                    'result': best_result,
                }, checkpoint_path)
            print(f'Model Saved! Best Result: {best_result}')
        
        
        log_stats = {
            **{f"train_{k}": v for k, v in train_stats.items()},
            **{f"val_{k}": v for k, v in test_stats.items()},
            "epoch": epoch,
            "num_updates": (epoch + 1) * len(data_loader_train),
        }
        
        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
