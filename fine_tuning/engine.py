"""
Train and eval functions used in main.py
"""
import math
from typing import Iterable

import torch
import torch.nn.functional as F

from torch.nn import MSELoss
from torchvision.ops import sigmoid_focal_loss as focal_loss
from monai.losses import DiceLoss

from monai.metrics import DiceMetric

import utils


def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer, lr_scheduler,
                    device: torch.device, epoch: int, loss_scaler, set_training_mode=True):
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 100

    model.train(set_training_mode)
    num_updates = epoch * len(data_loader)
    
    dice_loss = DiceLoss(reduction='mean', sigmoid=True) 
    mse_loss = MSELoss(reduction='mean')
    losses = dict()

    for batch in metric_logger.log_every(data_loader, print_freq, header):
        imgs = batch["image"].float().to(device) # [B, 3, H, W]
        mask_gts = batch["label"] # [B, N, H, W]
        mask_gts = mask_gts - (mask_gts.max()+mask_gts.min()) / 2.0
        mask_gts = (mask_gts > 0.0).long().to(device)

        # first iteration
        prompt_idx = torch.randint(3, (1,))[0].item()
        if  prompt_idx == 0 :
            point_coords, point_values = None, None
            boxes = None
        elif prompt_idx == 1:
            mask_preds = torch.zeros_like(mask_gts, device=device)
            point_coords, point_values = utils.batched_mask_to_point(mask_gts, mask_preds, num_points=1)
            boxes = None
        elif prompt_idx == 2:
            point_coords, point_values = None, None
            boxes = utils.batched_mask_to_box(mask_gts)
            boxes = utils.add_noise_to_boxes(boxes)
        
        with torch.cuda.amp.autocast():
            image_embeddings, low_res_masks, iou_predictions = model(
                imgs, iteration=0,
                point_coords=point_coords, point_values=point_values, boxes=boxes, masks=None,
            ) # [B, C, H/16, W/16], [B, N, 1, H/4, H/4], [B, N, 1]

            mask_preds = F.interpolate(low_res_masks.squeeze(2), (1024, 1024), mode="bilinear", align_corners=False) # [B, N, H, W]
            iou_predictions = iou_predictions.flatten(0, 1) # [B*N, 1]
            iou_groundtruthes = utils.calculate_miou(mask_preds, mask_gts).flatten(0, 1) # [B*N, 1]
            
            loss = dice_loss(mask_preds.flatten(0, 1).unsqueeze(1), mask_gts.flatten(0, 1).unsqueeze(1)) \
                   + 20.0 * focal_loss(mask_preds.flatten(0, 1).unsqueeze(1), mask_gts.flatten(0, 1).unsqueeze(1).float(), reduction='mean') \
                   + mse_loss(iou_predictions, iou_groundtruthes)
        
        optimizer.zero_grad()
        if loss_scaler is not None:
            loss_scaler(
                loss,
                optimizer,
                parameters=model.parameters(),
            )
        else:
            loss.backward()
            optimizer.step()

        losses[f'iter_{0:02d}_loss_value'] = (loss.item(), 1)
        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()))
        
        #following iterations
        image_embeddings = image_embeddings.detach()
        
        low_res_masks = low_res_masks.detach()
        mask_preds = (mask_preds.detach() > 0.0).float()
        for i in range(1, 11):
            if i in [torch.randint(1, 10, (1,)).item(), 10]:
                point_coords, point_values = None, None
            else:
                num_points = torch.randint(1, 10, (1,)).item()
                point_coords, point_values = utils.batched_mask_to_point(mask_gts, mask_preds, num_points=num_points)
                
            with torch.cuda.amp.autocast():
                _, low_res_masks, iou_predictions = model(
                    image_embeddings, iteration=i,
                    point_coords=point_coords, point_values=point_values, boxes=None, masks=low_res_masks,
                ) # [B, C, H/16, W/16], [B, N, 1, H/4, H/4], [B, N, 1]
                
                mask_preds = F.interpolate(low_res_masks.squeeze(2), (1024, 1024), mode="bilinear", align_corners=False) # [B, N, H, W]
                iou_predictions = iou_predictions.flatten(0, 1) # [B*N, 1]
                iou_groundtruthes = utils.calculate_miou(mask_preds, mask_gts).flatten(0, 1) # [B*N, 1]

                loss = dice_loss(mask_preds.flatten(0, 1).unsqueeze(1), mask_gts.flatten(0, 1).unsqueeze(1)) \
                    + 20.0 * focal_loss(mask_preds.flatten(0, 1).unsqueeze(1), mask_gts.flatten(0, 1).unsqueeze(1).float(), reduction='mean') \
                    + mse_loss(iou_predictions, iou_groundtruthes)      
                  
            optimizer.zero_grad()
            if loss_scaler is not None:
                loss_scaler(
                    loss,
                    optimizer,
                    parameters=model.parameters(),
                )
            else:
                loss.backward()
                optimizer.step()

            losses[f'iter_{i:02d}_loss_value'] = (loss.item(), 1)
            if not math.isfinite(loss.item()):
                print("Loss is {}, stopping training".format(loss.item()))

            low_res_masks = low_res_masks.detach()
            mask_preds = (mask_preds.detach() > 0.0).float()

        if lr_scheduler is not None:
            num_updates += 1
            lr_scheduler.step_update(num_updates=num_updates)

        torch.cuda.synchronize()
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        for k, v in losses.items():
            metric_logger.update(**{f"{k}": v[0], "n": v[1]})

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, args):
    acc_func = DiceMetric(include_background=True, get_not_nans=True)
    scores = dict()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    for batch in metric_logger.log_every(data_loader, 50, header):
        imgs = batch["image"].float().to(device)
        mask_gts = batch["label"] # [B, N, H, W]
        mask_gts = mask_gts - (mask_gts.max()+mask_gts.min()) / 2.0
        mask_gts = (mask_gts > 0.0).long().to(device)

        # no prompt
        point_coords, point_values = None, None
        boxes = None
        masks = None
        with torch.cuda.amp.autocast():
            image_embeddings, low_res_masks, _ = model(
                imgs, iteration=0, 
                point_coords=point_coords, point_values=point_values, boxes=boxes, masks=masks,
            ) # [B, C, H/16, W/16], [B, N, 1, H/4, H/4], [B, N, 1]
        
        mask_preds = F.interpolate(low_res_masks.squeeze(2), (1024, 1024), mode="bilinear", align_corners=False) # [B, N, H, W]
        mask_preds = (mask_preds > 0.0).float()
        
        acc_func.reset()
        acc_func(mask_preds, mask_gts)
        dice, not_nans = acc_func.aggregate()
        scores['no_prompt_dice'] = (dice, not_nans)

        if args.eval and args.output_dir:
            name = batch['name'][0]
            utils.show_result(
                point_coords, point_values, boxes, 
                mask_preds[0], mask_gts[0], imgs[0], batch['resize_size'][0],
                title='{}_{}_{:.4f}'.format(name, "no prompt", dice.item()), 
                output_dir=args.output_dir,
            )

        # mask prompt
        point_coords, point_values = None, None
        boxes = None
        masks = low_res_masks.detach()
        with torch.cuda.amp.autocast():
            _, low_res_masks, _ = model(
                image_embeddings, iteration=1, 
                point_coords=point_coords, point_values=point_values, boxes=boxes, masks=masks,
            ) # [B, C, H/16, W/16], [B, N, 1, H/4, H/4], [B, N, 1]

        mask_preds = F.interpolate(low_res_masks.squeeze(2), (1024, 1024), mode="bilinear", align_corners=False) # [B, N, H, W]
        mask_preds = (mask_preds > 0.0).float()
        
        acc_func.reset()
        acc_func(mask_preds, mask_gts)
        dice, not_nans = acc_func.aggregate()
        scores['mask_prompt_dice'] = (dice, not_nans)

        if args.eval and args.output_dir:
            name = batch['name'][0]
            utils.show_result(
                point_coords, point_values, boxes, 
                mask_preds[0], mask_gts[0], imgs[0], batch['resize_size'][0],
                title='{}_{}_{:.4f}'.format(name, "mask prompt", dice.item()), 
                output_dir=args.output_dir,
            )

        # point prompt
        mask_preds = torch.zeros_like(mask_gts, device=device)
        point_coords, point_values = utils.batched_mask_to_point(mask_gts, mask_preds, num_points=1)
        boxes = None
        masks = None
        with torch.cuda.amp.autocast():
            image_embeddings, low_res_masks, _ = model(
                imgs, iteration=0, 
                point_coords=point_coords, point_values=point_values, boxes=boxes, masks=masks,
            ) # [B, C, H/16, W/16], [B, N, 1, H/4, H/4], [B, N, 1]
        
        mask_preds = F.interpolate(low_res_masks.squeeze(2), (1024, 1024), mode="bilinear", align_corners=False) # [B, N, H, W]
        mask_preds = (mask_preds > 0.0).float()
        
        acc_func.reset()
        acc_func(mask_preds, mask_gts)
        dice, not_nans = acc_func.aggregate()
        scores['point_prompt_dice'] = (dice, not_nans)

        if args.eval and args.output_dir:
            name = batch['name'][0]
            utils.show_result(
                point_coords, point_values, boxes, 
                mask_preds[0], mask_gts[0], imgs[0], batch['resize_size'][0],
                title='{}_{}_{:.4f}'.format(name, "point prompt", dice.item()), 
                output_dir=args.output_dir,
            )

        # box prompt
        point_coords, point_values = None, None
        boxes = utils.batched_mask_to_box(mask_gts)
        masks = None
        with torch.cuda.amp.autocast():
            _, low_res_masks, _ = model(
                image_embeddings, iteration=1, 
                point_coords=point_coords, point_values=point_values, boxes=boxes, masks=masks,
            ) # [B, C, H/16, W/16], [B, N, 1, H/4, H/4], [B, N, 1]

        mask_preds = F.interpolate(low_res_masks.squeeze(2), (1024, 1024), mode="bilinear", align_corners=False) # [B, N, H, W]
        mask_preds = (mask_preds > 0.0).float()
        
        acc_func.reset()
        acc_func(mask_preds, mask_gts)
        dice, not_nans = acc_func.aggregate()
        scores['box_prompt_dice'] = (dice, not_nans)

        if args.eval and args.output_dir:
            name = batch['name'][0]
            utils.show_result(
                point_coords, point_values, boxes, 
                mask_preds[0], mask_gts[0], imgs[0], batch['resize_size'][0],
                title='{}_{}_{:.4f}'.format(name, "box prompt", dice.item()), 
                output_dir=args.output_dir,
            )

        for k, v in scores.items():
            metric_logger.update(**{f"{k}": v[0], "n": v[1]})

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}