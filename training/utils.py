"""
Misc functions, including distributed helpers.

Mostly copy-paste from torchvision references.
"""
import os
import time
from collections import defaultdict, deque
import datetime

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import torch
import torch.distributed as dist


def batched_mask_to_point(gt_masks, pred_masks, num_points):
    """
    Randomly sample num_points points with non-zero values for each mask and their respective values.

    Masks should be in the format [B, N, H, W].

    Returns two tensors:
    1) A tensor of shape [B, N, m, 2] with the x, y coordinates of the sampled points.
    2) A tensor of shape [B, N, m] with the values of the sampled points.
    """
    masks = gt_masks - pred_masks

    b, n, _, _ = masks.shape
    masks = masks.flatten(0, 1)

    # Get the indices and values of all non-zero values in the mask
    mask_indices, y, x = torch.nonzero(masks, as_tuple=True)
    values = masks[mask_indices, y, x]

    # Compute prefix sum to find intervals
    mask_counts = (masks != 0).sum(dim=[1,2])
    prefix_sum = torch.cat([torch.tensor([0], device=masks.device), mask_counts.cumsum(dim=0)[:-1]])

    # Sample values within each mask's interval
    sampled_indices = []
    for start, count in zip(prefix_sum, mask_counts):
        if count == 0:
            sampled_indices.append(torch.tensor([0]))
        else:
            indices = 1 + start.clone().cpu() + torch.randperm(count.clone().cpu())[0: min(num_points, count.clone().cpu())]
            sampled_indices.append(indices)

    # Expand each sample to num_points samples, filling with 0 if necessary
    sampled_indices = [torch.cat([idx, torch.full((num_points - len(idx),), 0, dtype=torch.long)], dim=0) for idx in sampled_indices]
    sampled_indices = torch.stack(sampled_indices)

    x = torch.cat([torch.tensor([0], device=masks.device), x], dim=0)
    y = torch.cat([torch.tensor([0], device=masks.device), y], dim=0)
    values = torch.cat([torch.tensor([0], device=masks.device), values], dim=0)

    # Get the sampled coordinates and values
    sampled_coords = torch.stack([x[sampled_indices], y[sampled_indices]], dim=2)
    sampled_values = values[sampled_indices]

    sampled_coords = sampled_coords.reshape(b, n, num_points, 2)
    sampled_values = sampled_values.reshape(b, n, num_points)

    # switch 0 to -1, -1 to 0
    sampled_values[sampled_values != 1] -= 1
    sampled_values[sampled_values == -2] += 2

    return sampled_coords, sampled_values


def batched_mask_to_neg_point(gt_masks, pred_masks, num_points):
    """
    Randomly sample num_points negative points with non-zero values for each mask and their respective values.

    Masks should be in the format [B, N, H, W].

    Returns two tensors:
    1) A tensor of shape [B, N, m, 2] with the x, y coordinates of the sampled points.
    2) A tensor of shape [B, N, m] with the values of the sampled points.
    """
    masks = pred_masks - gt_masks
    masks[masks <= 0] = 0

    b, n, _, _ = masks.shape
    masks = masks.flatten(0, 1)

    # Get the indices and values of all non-zero values in the mask
    mask_indices, y, x = torch.nonzero(masks, as_tuple=True)
    values = masks[mask_indices, y, x]

    # Compute prefix sum to find intervals
    mask_counts = (masks != 0).sum(dim=[1,2])
    prefix_sum = torch.cat([torch.tensor([0], device=masks.device), mask_counts.cumsum(dim=0)[:-1]])

    # Sample values within each mask's interval
    sampled_indices = []
    for start, count in zip(prefix_sum, mask_counts):
        if count == 0:
            sampled_indices.append(torch.tensor([0]))
        else:
            sampled_indices.append(1 + torch.randint(start, start + count, (min(num_points, count),)))

    # Expand each sample to num_points samples, filling with 0 if necessary
    sampled_indices = [torch.cat([idx, torch.full((num_points - len(idx),), 0, dtype=torch.long)], dim=0) for idx in sampled_indices]
    sampled_indices = torch.stack(sampled_indices)

    x = torch.cat([torch.tensor([0], device=masks.device), x], dim=0)
    y = torch.cat([torch.tensor([0], device=masks.device), y], dim=0)
    values = torch.cat([torch.tensor([0], device=masks.device), values], dim=0)

    # Get the sampled coordinates and values
    sampled_coords = torch.stack([x[sampled_indices], y[sampled_indices]], dim=2)
    sampled_values = values[sampled_indices]

    sampled_coords = sampled_coords.reshape(b, n, num_points, 2)
    sampled_values = sampled_values.reshape(b, n, num_points)

    # switch 1 to 0, 0 to -1
    sampled_values -= 1

    return sampled_coords, sampled_values


def batched_mask_to_pos_point(gt_masks, pred_masks, num_points):
    """
    Randomly sample num_points positive points with non-zero values for each mask and their respective values.

    Masks should be in the format [B, N, H, W].

    Returns two tensors:
    1) A tensor of shape [B, N, m, 2] with the x, y coordinates of the sampled points.
    2) A tensor of shape [B, N, m] with the values of the sampled points.
    """
    masks = gt_masks - pred_masks
    masks[masks <= 0] = 0

    b, n, _, _ = masks.shape
    masks = masks.flatten(0, 1)

    # Get the indices and values of all non-zero values in the mask
    mask_indices, y, x = torch.nonzero(masks, as_tuple=True)
    values = masks[mask_indices, y, x]

    # Compute prefix sum to find intervals
    mask_counts = (masks != 0).sum(dim=[1,2])
    prefix_sum = torch.cat([torch.tensor([0], device=masks.device), mask_counts.cumsum(dim=0)[:-1]])

    # Sample values within each mask's interval
    sampled_indices = []
    for start, count in zip(prefix_sum, mask_counts):
        if count == 0:
            sampled_indices.append(torch.tensor([0]))
        else:
            sampled_indices.append(1 + torch.randint(start, start + count, (min(num_points, count),)))

    # Expand each sample to num_points samples, filling with 0 if necessary
    sampled_indices = [torch.cat([idx, torch.full((num_points - len(idx),), 0, dtype=torch.long)], dim=0) for idx in sampled_indices]
    sampled_indices = torch.stack(sampled_indices)

    x = torch.cat([torch.tensor([0], device=masks.device), x], dim=0)
    y = torch.cat([torch.tensor([0], device=masks.device), y], dim=0)
    values = torch.cat([torch.tensor([0], device=masks.device), values], dim=0)

    # Get the sampled coordinates and values
    sampled_coords = torch.stack([x[sampled_indices], y[sampled_indices]], dim=2)
    sampled_values = values[sampled_indices]

    sampled_coords = sampled_coords.reshape(b, n, num_points, 2)
    sampled_values = sampled_values.reshape(b, n, num_points)

    # switch 0 to -1
    sampled_values[sampled_values != 1] -= 1

    return sampled_coords, sampled_values

def batched_mask_to_box(masks: torch.Tensor) -> torch.Tensor:
    """
    Calculates boxes in XYXY format around masks. Return [0,0,0,0] for
    an empty mask. For input shape C1xC2x...xHxW, the output shape is C1xC2x...x4.
    The mask should dtype of bool.
    """
    masks = masks.bool()

    # torch.max below raises an error on empty inputs, just skip in this case
    if torch.numel(masks) == 0:
        return torch.zeros(*masks.shape[:-2], 4, device=masks.device)

    # Normalize shape to CxHxW
    shape = masks.shape
    h, w = shape[-2:]
    if len(shape) > 2:
        masks = masks.flatten(0, -3)
    else:
        masks = masks.unsqueeze(0)

    # Get top and bottom edges
    in_height, _ = torch.max(masks, dim=-1)
    in_height_coords = in_height * torch.arange(h, device=in_height.device)[None, :]
    bottom_edges, _ = torch.max(in_height_coords, dim=-1)
    in_height_coords = in_height_coords + h * (~in_height)
    top_edges, _ = torch.min(in_height_coords, dim=-1)

    # Get left and right edges
    in_width, _ = torch.max(masks, dim=-2)
    in_width_coords = in_width * torch.arange(w, device=in_width.device)[None, :]
    right_edges, _ = torch.max(in_width_coords, dim=-1)
    in_width_coords = in_width_coords + w * (~in_width)
    left_edges, _ = torch.min(in_width_coords, dim=-1)

    # If the mask is empty the right edge will be to the left of the left edge.
    # Replace these boxes with [0, 0, 0, 0]
    empty_filter = (right_edges < left_edges) | (bottom_edges < top_edges)
    out = torch.stack([left_edges, top_edges, right_edges, bottom_edges], dim=-1)
    out = out * (~empty_filter).unsqueeze(-1)

    # Return to original shape
    if len(shape) > 2:
        out = out.reshape(*shape[:-2], 4)
    else:
        out = out[0]

    return out


def add_noise_to_boxes(boxes, max_noise=20):
    """
    Add noise to bounding box coordinates.

    Parameters:
    boxes (Tensor): a tensor of shape (B, N, 4) where B is the batch size, N is the number 
                    of boxes, and 4 represents the [x1, y1, x2, y2] coordinates of each box.
    max_noise (int): maximum noise to add.

    Returns:
    Tensor: a tensor with noise added to the boxes.
    """
    B, N, _ = boxes.size()

    # Calculate width, height, and 10% of side lengths
    width = boxes[:, :, 2] - boxes[:, :, 0]
    height = boxes[:, :, 3] - boxes[:, :, 1]
    noise_std_w = torch.clamp(width * 0.1, max=max_noise)
    noise_std_h = torch.clamp(height * 0.1, max=max_noise)

    # Generate random noise based on the standard deviation
    noise = torch.randn((B, N, 4), device=boxes.device) * torch.stack([noise_std_w, noise_std_h, noise_std_w, noise_std_h], dim=-1)
    
    boxes_noisy = boxes + noise
    return boxes_noisy


def calculate_miou(pred_masks, ground_truth):
    """
    Calculate mean Intersection over Union (mIoU) for predicted masks against ground truth.

    Parameters:
    pred_masks (Tensor): a tensor of shape (B, N, 3, H, W) representing predicted binary masks.
    ground_truth (Tensor): a tensor of shape (B, N, H, W) representing ground truth binary masks.

    Returns:
    Tensor: a tensor of mIoU values of shape (B, N, 3).
    """
    pred_masks = pred_masks.detach() > 0.0
    ground_truth = ground_truth.bool()

    # Ensure the ground truth is broadcastable to the shape of the pred_masks
    if len(pred_masks.shape) == 4:
        pred_masks = pred_masks.unsqueeze(2)
    ground_truth = ground_truth.unsqueeze(2)  # Shape becomes (B, N, 1, H, W)

    # Calculate intersection and union
    intersection = torch.logical_and(pred_masks, ground_truth).sum((-2, -1)).float()  # Shape (B, N, 3)
    union = torch.logical_or(pred_masks, ground_truth).sum((-2, -1)).float()  # Shape (B, N, 3)

    # Calculate IoU
    iou = (intersection + 1e-6) / (union + 1e-6)  # Prevent division by zero

    return iou  # This is mean IoU since it's averaged across all instances in N dimension


def show_result(
        point_coords, point_values, boxes, 
        mask_pred, mask_gt, image, original_size,
        title, output_dir,
        mean=[123.675, 116.28, 103.53], 
        std=[58.395, 57.12, 57.375]
    ):
    # visualize the result for one batch
    # point_coords: [b=1, n_masks, n_points, 2]; point_values: [b=1, n_masks, n_points]; boxes: [b=1, n_masks, 4]
    # mask_pred: [n_masks, h, w]; mask_gt: [n_masks, h, w]; image: [3, h, w]
    # original_size: [2, ]
        
    def show_points(coords, labels, ax, marker_size=375):
        pos_points = coords[labels==1]
        neg_points = coords[labels==0]
        ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
        ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25) 

    def show_box(box, ax):
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2)) 

    def show_mask(mask, ax, color='white'):
        if color == 'white':
            color = np.array([255/255, 255/255, 255/255, 0.4])
        elif color == 'blue':
            color = np.array([30/255, 144/255, 255/255, 0.4])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)


    if point_coords is not None and point_values is not None:
        point_coords = point_coords[0].cpu().numpy() # [N, m, 2]
        point_values = point_values[0].cpu().numpy() # [N, m]
    if boxes is not None:
        boxes = boxes[0].cpu().numpy() #[N, 4]

    mean = torch.tensor(mean, device=image.device)
    std = torch.tensor(std, device=image.device)
    
    image = image[:, :original_size[0].item(), :original_size[1].item()]
    mask_pred = mask_pred[:, :original_size[0].item(), :original_size[1].item()]
    mask_gt = mask_gt[:, :original_size[0].item(), :original_size[1].item()]

    image = image * std[..., None, None] + mean[..., None, None]
    image = image.permute(1, 2, 0).cpu().numpy().astype(np.uint8) # [1024, 1024, 3]
    mask_pred = mask_pred.cpu().numpy() # [N, 1024, 1024]
    mask_gt = mask_gt.cpu().numpy() # [N, 1024, 1024]
    
    plt.figure(figsize = (10, 10))
    plt.imshow(image)
    plt.title(f"{title}", fontsize=18)
    plt.axis('off')

    for n in range(mask_pred.shape[0]):
        if point_coords is not None and point_values is not None:
            show_points(point_coords[n], point_values[n], plt.gca())
        if boxes is not None:
            show_box(boxes[n], plt.gca())
        show_mask(mask_gt[n], plt.gca(), color='white')
        show_mask(mask_pred[n], plt.gca(), color='blue')

    output_dir = Path(output_dir)
    with open(output_dir / f'{title}.jpg', 'wb') as outfile:
            plt.savefig(outfile, format='jpg')

    plt.close('all')
    return


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return 0.0 if self.count==0 else d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return 0.0 if self.count==0 else d.mean().item()

    @property
    def global_avg(self):
        return 0.0 if self.count==0 else self.total / self.count

    @property
    def max(self):
        return 0.0 if self.count==0 else max(self.deque)

    @property
    def value(self):
        return 0.0 if self.count==0 else self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, n=1, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v, n)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    
#     if "SLURM_STEPS_GPUS" in os.environ:
#         gpu_ids = os.environ["SLURM_STEP_GPUS"].split(",")
#         os.environ["MASTER_PORT"] = str(12345 + int(min(gpu_ids)))
#     else:
#         os.environ["MASTER_PORT"] = str(12345)

#     if "SLURM_JOB_NODELIST" in os.environ:
#         hostnames = hostlist.expand_hostlist(os.environ["SLURM_JOB_NODELIST"])
#         os.environ["MASTER_ADDR"] = hostnames[0]
#     else:
#         os.environ["MASTER_ADDR"] = "127.0.0.1"

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)
    