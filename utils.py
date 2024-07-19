import os
import ot
import cv2
import numpy as np

from sklearn.cluster import KMeans

import torch
from torch.nn import functional as F

from per_segment_anything import SamPredictor


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


# p2sam for perseg
def p2sam_perseg(args, sam, ref_image_path, ref_mask_path, test_idx, test_image_path, output_path):
    predictor = SamPredictor(sam)

    # Load reference images and masks
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

    # 1. Save the patch-level reference feature
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

        # Cluster to get the local feature
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

        # Cosine Similarity
        sim = target_feat @ test_feat # [n_clusters, h, w]
        sim = sim.reshape(1, n_clusters, h, w)
        sim = F.interpolate(sim, scale_factor=4, mode="bilinear")
        sim = predictor.model.postprocess_masks(
                        sim,
                        input_size=predictor.input_size,
                        original_size=predictor.original_size)
        
        # Positive-negative location prior
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

        # Obtain the target guidance for cross-attention layers
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