import os
import ot
import cv2
import sys
sys.path.append('./')
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

import torch
from torch.nn import functional as F
from torchvision.ops import sigmoid_focal_loss as compute_focal_loss
from monai.losses import DiceLoss

from .custom_segment_anything import SamPredictor, SamPredictorTr


def point_selection(mask_sim, topk=1):
    # Top-1 point selection
    w, h = mask_sim.shape
    topk_xy = mask_sim.flatten(0).topk(topk)[1]
    topk_x = (topk_xy // h).unsqueeze(0)
    topk_y = (topk_xy - topk_x * h)
    topk_xy = torch.cat((topk_y, topk_x), dim=0).permute(1, 0)
    topk_label = np.array([1] * topk)
    topk_xy = topk_xy.cpu().numpy()
        
    # Top-last point selection
    last_xy = mask_sim.flatten(0).topk(topk, largest=False)[1]
    last_x = (last_xy // h).unsqueeze(0)
    last_y = (last_xy - last_x * h)
    last_xy = torch.cat((last_y, last_x), dim=0).permute(1, 0)
    last_label = np.array([0] * topk)
    last_xy = last_xy.cpu().numpy()
    
    return topk_xy, topk_label, last_xy, last_label


def compute_wasserstein_distance(A, B, B_weights=None):
    A_np = A.detach().cpu().numpy()
    B_np = B.detach().cpu().numpy()
    if B_weights is not None:
        B_weights_np = B_weights.detach().cpu().numpy()

    n_a = A_np.shape[0]
    a = ot.unif(n_a)
    if B_weights is not None:
        b = B_weights_np / np.sum(B_weights_np)
    else:
        n_b = B_np.shape[0]
        b = ot.unif(n_b)

    M = ot.dist(A_np, B_np)
    wasserstein_distance = ot.emd2(a, b, M, numItermax=10000000)
    return wasserstein_distance


def compute_dice(output, target):
    output = output.copy().astype(np.float)
    target = target.copy().astype(np.float)
    output = output > 0.0
    target = target > 0.0
    assert output.shape == target.shape

    true_pos = np.logical_and(output, target).sum()
    true_neg = np.logical_and(~output, ~target).sum()
    false_pos = np.logical_and(output, ~target).sum()
    false_neg = np.logical_and(~output, target).sum()
    dice = 2 * true_pos / (2 * true_pos + false_pos + false_neg + 1e-10)
    return dice


# path processing for lung data
def image2mask_path(image_path):
    temp_list = image_path.split('/')
    for i, elem in enumerate(temp_list):
        if elem == 'image':
            temp_list[i] = 'label'
    mask_path = '/'.join(temp_list)
    return mask_path


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


# direct transfer for medical image
def run_medical(args, sam, test_image_path, test_mask_path, output_path, slice_name):
    predictor = SamPredictor(sam)

    # Load test image and test mask
    test_image = cv2.imread(test_image_path)
    test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
    test_mask = cv2.imread(test_mask_path, cv2.IMREAD_GRAYSCALE)       
    test_mask = test_mask - (test_mask.max()+test_mask.min()) / 2.0
    test_mask = (test_mask > 0.0).astype(np.float)

    # get test point and box
    if args.box:
        test_mask_tensor = torch.Tensor(test_mask).unsqueeze(0).unsqueeze(0)
        test_box_tensor = batched_mask_to_box(test_mask_tensor)
        test_box_numpy = test_box_tensor.squeeze(0).cpu().numpy()
    else:
        test_box_numpy = None
    
    if args.point:
        test_mask_tensor = torch.Tensor(test_mask).unsqueeze(0).unsqueeze(0)
        mask_preds = torch.zeros_like(test_mask_tensor)
        point_coords, point_values = batched_mask_to_point(test_mask_tensor, mask_preds, num_points=1)
        point_coords_numpy = point_coords.squeeze(0).squeeze(0).cpu().numpy()
        point_values_numpy = point_values.squeeze(0).squeeze(0).cpu().numpy()
    else:
        point_coords_numpy = None
        point_values_numpy = None

    # Image feature encoding
    predictor.set_image(test_image, medsam=args.medsam)
    masks, _, _, _, _, _ = predictor.predict(
        point_coords=point_coords_numpy, 
        point_labels=point_values_numpy, 
        box = test_box_numpy,
        multimask_output=False,
        attn_sim=None,  # Target-guided Attention
        target_embedding=None  # Target-semantic Prompting
    )
    best_idx = 0

    # Save masks 
    final_mask = masks[best_idx]
    mask_colors = np.zeros((final_mask.shape[0], final_mask.shape[1], 3), dtype=np.uint8)
    mask_colors[final_mask, :] = np.array([[0, 0, 128]])
    mask_output_path = os.path.join(output_path, slice_name + '.png')
    cv2.imwrite(mask_output_path, mask_colors)
    return final_mask, compute_dice(final_mask, test_mask)


# p2sam for medical
def p2sam_medical(args, sam, ref_image_path, ref_mask_path, test_image_path, test_mask_path, output_path, slice_name):
    predictor = SamPredictor(sam)

    # Load images and masks
    ref_image = cv2.imread(ref_image_path)
    ref_image = cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB)
    ref_mask = cv2.imread(ref_mask_path)
    ref_mask = cv2.cvtColor(ref_mask, cv2.COLOR_BGR2RGB)

    # Image features encoding
    ref_mask = predictor.set_image(ref_image, ref_mask, medsam=args.medsam)
    ref_mask = ref_mask - (ref_mask.max()+ref_mask.min()) / 2.0
    ref_feat = predictor.features.squeeze().permute(1, 2, 0) # [h, w, c]

    ref_mask = F.interpolate(ref_mask, size=ref_feat.shape[0: 2], mode="bilinear")
    ref_mask = ref_mask.squeeze()[0]

    # Save the patch-level reference feature
    ref_feats_reg_patch = ref_feat[ref_mask > 0.0]
    if ref_feats_reg_patch.shape[0] == 0:
        return

    # Start testing
    final_mask = None
    final_point_coords = None
    final_point_labels = None
    best_reg_score = 1e+6

    # Load test image and test mask (only used for Dice score)
    test_image = cv2.imread(test_image_path)
    test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
    test_mask = cv2.imread(test_mask_path, cv2.IMREAD_GRAYSCALE)
    test_mask = test_mask - (test_mask.max()+test_mask.min()) / 2.0
    test_mask = (test_mask > 0.0).astype(np.float)

    # Image feature encoding
    predictor.set_image(test_image, medsam=args.medsam)
    test_feat = predictor.features.squeeze()
    C, h, w = test_feat.shape
    test_feat = test_feat / test_feat.norm(dim=0, keepdim=True)
    test_feat = test_feat.reshape(C, h * w)

    for n_fore_clusters in range(args.min_num_pos, args.max_num_pos + 1):
        # Reference feature extraction (foreground)
        target_fore_feat = ref_feat[ref_mask > 0.0]
        target_fore_embedding = target_fore_feat.mean(0).unsqueeze(0).unsqueeze(0)
        if n_fore_clusters > 1:
            if target_fore_feat.shape[0] < n_fore_clusters:
                continue
            kmeans = KMeans(n_clusters=n_fore_clusters, random_state=0)
            cluster = kmeans.fit_predict(target_fore_feat.cpu().numpy())
            cluster = torch.from_numpy(cluster)
            feats = []
            for c in range(n_fore_clusters):
                feat_c = target_fore_feat[cluster==c].mean(0)
                feat_c = feat_c / feat_c.norm(dim=-1, keepdim=True) # [c]
                feats.append(feat_c)
            target_fore_feat = torch.stack(feats, dim=0) # [n_clusters, c]
        else:
            target_fore_feat = target_fore_embedding / target_fore_embedding.norm(dim=-1, keepdim=True)
            target_fore_feat = target_fore_feat.squeeze(0) # [1, c]
        
        # Compute cosine similarity (foreground)
        sim = target_fore_feat @ test_feat # [n_clusters, h, w]
        sim = sim.reshape(1, n_fore_clusters, h, w)
        sim = F.interpolate(sim, scale_factor=4, mode="bilinear")
        sim = predictor.model.postprocess_masks(
                        sim,
                        input_size=predictor.input_size,
                        original_size=predictor.original_size)
        
        # Positive location prior
        pos_select_method = "mean" if n_fore_clusters == 1 else "max"
        if pos_select_method == "mean":
            sim = sim.mean(1).squeeze()
            topk_xy, topk_label, _, _ = point_selection(sim, topk=1)
            point_coords_1 = topk_xy
            point_values_1 = topk_label
        elif pos_select_method == "max":
            topk_xys = []
            topk_labels = []
            for c in range(n_fore_clusters):
                sim_c = sim[0, c]
                topk_xy_c, topk_label_c, _, _ = point_selection(sim_c, topk=1)
                topk_xys.append(topk_xy_c)
                topk_labels.append(topk_label_c)
            point_coords_1 = np.concatenate(topk_xys, axis=0)
            point_values_1 = np.concatenate(topk_labels, axis=0)
        
        for n_back_clusters in range(args.min_num_neg, args.max_num_neg + 1):
            if n_back_clusters > 0:
                # Reference feature extraction (background)
                target_back_feat = ref_feat[ref_mask <= 0.0]
                target_back_embedding = target_back_feat.mean(0).unsqueeze(0).unsqueeze(0)
                if n_back_clusters > 1:
                    if target_back_feat.shape[0] < n_back_clusters:
                        continue
                    kmeans = KMeans(n_clusters=n_back_clusters, random_state=0)
                    cluster = kmeans.fit_predict(target_back_feat.cpu().numpy())
                    cluster = torch.from_numpy(cluster)
                    feats = []
                    for c in range(n_back_clusters):
                        feat_c = target_back_feat[cluster==c].mean(0)
                        feat_c = feat_c / feat_c.norm(dim=-1, keepdim=True) # [c]
                        feats.append(feat_c)
                    target_back_feat = torch.stack(feats, dim=0) # [n_clusters, c]
                else:
                    target_back_feat = target_back_embedding / target_back_embedding.norm(dim=-1, keepdim=True)
                    target_back_feat = target_back_feat.squeeze(0) # [1, c]

                sim = target_back_feat @ test_feat # [n_clusters, h, w]
                sim = sim.reshape(1, n_back_clusters, h, w)
                sim = F.interpolate(sim, scale_factor=4, mode="bilinear")
                sim = predictor.model.postprocess_masks(
                                sim,
                                input_size=predictor.input_size,
                                original_size=predictor.original_size)
            
                # Negative location prior
                neg_select_method = "mean" if n_back_clusters == 1 else "max"
                if neg_select_method == "mean":
                    sim = sim.mean(1).squeeze()
                    topk_xy, _, _, topk_label = point_selection(sim, topk=1)
                    point_coords_2 = np.concatenate([point_coords_1, topk_xy], axis=0)
                    point_values_2 = np.concatenate([point_values_1, topk_label], axis=0)
                elif neg_select_method == "max":
                    topk_xys = []
                    topk_labels = []
                    for c in range(n_back_clusters):
                        sim_c = sim[0, c]
                        topk_xy_c, _, _, topk_label_c = point_selection(sim_c, topk=1)
                        topk_xys.append(topk_xy_c)
                        topk_labels.append(topk_label_c)
                    topk_xys = np.concatenate(topk_xys, axis=0)
                    topk_labels = np.concatenate(topk_labels, axis=0)
                    point_coords_2 = np.concatenate([point_coords_1, topk_xys], axis=0)
                    point_values_2 = np.concatenate([point_values_1, topk_labels], axis=0)
            else:
                point_coords_2 = point_coords_1
                point_values_2 = point_values_1
                
            # First-step prediction
            masks, scores, logits, _, _, _ = predictor.predict(
                point_coords=point_coords_2, 
                point_labels=point_values_2, 
                multimask_output=False,
                attn_sim=None,  # Target-guided Attention
                target_embedding=None  # Target-semantic Prompting
            )
            best_idx = 0

            # Cascaded Post-refinement-1
            multimask_output = False
            masks, scores, logits, _, _, _ = predictor.predict(
                        point_coords=point_coords_2,
                        point_labels=point_values_2,
                        mask_input=logits[best_idx: best_idx + 1, :, :], 
                        multimask_output=multimask_output)
            best_idx = 0

            # Cascaded Post-refinement-2
            y, x = np.nonzero(masks[best_idx])
            if x.shape[0] != 0 and y.shape[0] != 0:
                x_min = x.min()
                x_max = x.max()
                y_min = y.min()
                y_max = y.max()
                input_box = np.array([x_min, y_min, x_max, y_max])
            else:
                input_box = None
            multimask_output = False
            masks, scores, logits, _, _, _ = predictor.predict(
                point_coords=point_coords_2,
                point_labels=point_values_2,
                box=input_box[None, :] if input_box is not None else None,
                mask_input=logits[best_idx: best_idx + 1, :, :], 
                multimask_output=False)
            best_idx = 0

            # Regularization
            pred_mask = torch.Tensor(logits[best_idx][None, None, ...])
            pred_mask = F.interpolate(pred_mask, size=(64, 64), mode="bilinear").squeeze()
            test_feats_reg_patch = predictor.features.squeeze().permute(1, 2, 0)
            test_feats_reg_patch = test_feats_reg_patch[pred_mask > 0.0]
            weights = torch.sigmoid(pred_mask[pred_mask > 0.0]).flatten()
            
            # Patch-level regularization
            if args.reg_patch_weight:
                B_weights = weights
            else:
                B_weights = None

            if test_feats_reg_patch.shape[0] == 0:
                reg_score_patch = 1e+6 - 1
            else:
                reg_score_patch = compute_wasserstein_distance(ref_feats_reg_patch, test_feats_reg_patch, B_weights)

            curr_reg_score = reg_score_patch
            if curr_reg_score < best_reg_score:
                best_reg_score = curr_reg_score
                final_mask = masks[best_idx]
                final_point_coords = point_coords_2
                final_point_labels = point_values_2

    # Save masks 
    mask_colors = np.zeros((final_mask.shape[0], final_mask.shape[1], 3), dtype=np.uint8)
    mask_colors[final_mask, :] = np.array([[0, 0, 128]])
    mask_output_path = os.path.join(output_path, slice_name + '.png')
    cv2.imwrite(mask_output_path, mask_colors)

    return final_mask, final_point_coords, final_point_labels, compute_dice(final_mask, test_mask), best_reg_score


# p2sam for perseg
def p2sam_perseg(args, sam, ref_image_path, ref_mask_path, test_idx, test_image_path, output_path):
    predictor = SamPredictor(sam)

    # Load images and masks
    ref_image = cv2.imread(ref_image_path)
    ref_image = cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB)
    ref_mask = cv2.imread(ref_mask_path)
    ref_mask = cv2.cvtColor(ref_mask, cv2.COLOR_BGR2RGB)

    # Image features encoding
    ref_mask = predictor.set_image(ref_image, ref_mask) # resize and padding
    ref_mask = ref_mask - (ref_mask.max()+ref_mask.min()) / 2.0
    ref_feat = predictor.features.squeeze().permute(1, 2, 0) # [h, w, c]

    ref_mask = F.interpolate(ref_mask, size=ref_feat.shape[0: 2], mode="bilinear")
    ref_mask = ref_mask.squeeze()[0]

    # Save the patch-level reference feature
    ref_feats_reg_patch = ref_feat[ref_mask > 0.0]

    # Start testing
    final_mask = None
    final_point_coords = None
    final_point_labels = None
    best_reg_score = 1e+6

    # Load test image
    test_image = cv2.imread(test_image_path)
    test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)

    # Image feature encoding
    predictor.set_image(test_image)
    test_feat = predictor.features.squeeze()
    C, h, w = test_feat.shape
    test_feat = test_feat / test_feat.norm(dim=0, keepdim=True)
    test_feat = test_feat.reshape(C, h * w)
    for n_clusters in range(args.min_num_pos, args.max_num_pos + 1):
        # Reference feature extraction
        target_feat = ref_feat[ref_mask > 0.0]
        target_embedding = target_feat.mean(0).unsqueeze(0).unsqueeze(0)
        if n_clusters > 1:
            kmeans = KMeans(n_clusters=n_clusters, random_state=0)
            cluster = kmeans.fit_predict(target_feat.cpu().numpy())
            cluster = torch.from_numpy(cluster)
            target_feats = []
            for c in range(n_clusters):
                target_feat_c = target_feat[cluster==c].mean(0)
                target_feat_c = target_feat_c / target_feat_c.norm(dim=-1, keepdim=True) # [c]
                target_feats.append(target_feat_c)
            target_feat = torch.stack(target_feats, dim=0) # [n_clusters, c]
        elif n_clusters == 1:
            target_feat = target_embedding / target_embedding.norm(dim=-1, keepdim=True)
            target_feat = target_feat.squeeze(0) # [1, c]

        # Compute cosine similarity
        sim = target_feat @ test_feat # [n_clusters, h, w]
        sim = sim.reshape(1, n_clusters, h, w)
        sim = F.interpolate(sim, scale_factor=4, mode="bilinear")
        sim = predictor.model.postprocess_masks(
                        sim,
                        input_size=predictor.input_size,
                        original_size=predictor.original_size)
        
        # Positive location prior
        method = "mean" if n_clusters == 1 else "max"
        if method == "mean":
            sim = sim.mean(1).squeeze()
            topk_xy_i, topk_label_i, last_xy_i, last_label_i = point_selection(sim, topk=1)
            topk_xy = np.concatenate([topk_xy_i, last_xy_i], axis=0)
            topk_label = np.concatenate([topk_label_i, last_label_i], axis=0)
        elif method == "max":
            topk_xy = []
            topk_label = []
            for c in range(n_clusters):
                sim_c = sim[0, c]
                topk_xy_c, topk_label_c, _, _ = point_selection(sim_c, topk=1)
                topk_xy.append(topk_xy_c)
                topk_label.append(topk_label_c)
            sim = sim.mean(1).squeeze()
            _, _, last_xy_i, last_label_i = point_selection(sim, topk=1)
            topk_xy.append(last_xy_i)
            topk_label.append(last_label_i)
            topk_xy = np.concatenate(topk_xy, axis=0)
            topk_label = np.concatenate(topk_label, axis=0)

        # Obtain attention guidance (proposed by PerSAM)
        sim = (sim - sim.mean()) / torch.std(sim)
        sim = F.interpolate(sim.unsqueeze(0).unsqueeze(0), size=(64, 64), mode="bilinear")
        attn_sim = sim.sigmoid_().unsqueeze(0).flatten(3)

        # First-step prediction
        masks, scores, logits, _, _, _ = predictor.predict(
            point_coords=topk_xy, 
            point_labels=topk_label, 
            multimask_output=False,
            attn_sim=attn_sim,  # Target-guided Attention
            target_embedding=target_embedding  # Target-semantic Prompting
        )
        best_idx = 0

        # Cascaded Post-refinement-1
        masks, scores, logits, _, _, _ = predictor.predict(
                    point_coords=topk_xy,
                    point_labels=topk_label,
                    mask_input=logits[best_idx: best_idx + 1, :, :], 
                    multimask_output=True)
        best_idx = np.argmax(scores)

        # Cascaded Post-refinement-2
        y, x = np.nonzero(masks[best_idx])
        x_min = x.min()
        x_max = x.max()
        y_min = y.min()
        y_max = y.max()
        input_box = np.array([x_min, y_min, x_max, y_max])
        masks, scores, logits, _, _, _ = predictor.predict(
            point_coords=topk_xy,
            point_labels=topk_label,
            box=input_box[None, :],
            mask_input=logits[best_idx: best_idx + 1, :, :], 
            multimask_output=True)
        best_idx = np.argmax(scores)

        # Regularization
        pred_mask = torch.Tensor(logits[best_idx][None, None, ...])
        pred_mask = F.interpolate(pred_mask, size=(64, 64), mode="bilinear").squeeze()
        test_feats_reg_patch = predictor.features.squeeze().permute(1, 2, 0)
        test_feats_reg_patch = test_feats_reg_patch[pred_mask > 0.0]
        weights = torch.sigmoid(pred_mask[pred_mask > 0.0]).flatten()
        
        # Patch-level regularization
        if args.reg_patch_weight:
            B_weights = weights
        else:
            B_weights = None
        reg_score_patch = compute_wasserstein_distance(ref_feats_reg_patch, test_feats_reg_patch, B_weights)
        curr_reg_score = reg_score_patch

        if curr_reg_score < best_reg_score:
            best_reg_score = curr_reg_score
            final_mask = masks[best_idx]
            final_point_coords = topk_xy
            final_point_labels = topk_label

    # Save masks 
    mask_colors = np.zeros((final_mask.shape[0], final_mask.shape[1], 3), dtype=np.uint8)
    mask_colors[final_mask, :] = np.array([[0, 0, 128]])
    mask_output_path = os.path.join(output_path, test_idx + '.png')
    cv2.imwrite(mask_output_path, mask_colors)

    return final_mask, final_point_coords, final_point_labels, best_reg_score