from functools import partial

import torch
import torch.nn as nn

from segment_anything.build_sam import sam_model_registry
from timm.models import resolve_pretrained_cfg, load_pretrained

from .lora import Linear, MergedLinear
from .vision_transformer_sam import _create_vision_transformer, checkpoint_filter_fn


def make_timm_sam_encoder(pretrained=True, sam_type='vit_b', **kwargs):
    """ 
    ViT-B/16 for Segment-Anything
    """

    BASE_ARGS = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, global_attn_indexes=[2, 5, 8, 11],
        window_size=14, use_rel_pos=True, img_size=1024,
        drop_rate=0.0, pos_drop_rate=0.0, proj_drop_rate=0.0, attn_drop_rate=0.0, 
        drop_path_rate=0.1, norm_layer = partial(nn.LayerNorm, eps=1e-6),
    )

    LARGE_ARGS = dict(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, global_attn_indexes=[5, 11, 17, 23],
        window_size=14, use_rel_pos=True, img_size=1024,
        drop_rate=0.0, pos_drop_rate=0.0, proj_drop_rate=0.0, attn_drop_rate=0.0, 
        drop_path_rate=0.1, norm_layer = partial(nn.LayerNorm, eps=1e-6),
    )

    HUGE_ARGS = dict(
        patch_size=16, embed_dim=1280, depth=32, num_heads=16, global_attn_indexes=[7, 15, 23, 31],
        window_size=14, use_rel_pos=True, img_size=1024,
        drop_rate=0.0, pos_drop_rate=0.0, proj_drop_rate=0.0, attn_drop_rate=0.0, 
        drop_path_rate=0.1, norm_layer = partial(nn.LayerNorm, eps=1e-6),
    )

    if sam_type == 'vit_b':
        model_name = 'samvit_base_patch16'
        model_args = BASE_ARGS
    elif sam_type == 'vit_l':
        model_name = 'samvit_large_patch16'
        model_args = LARGE_ARGS
    elif sam_type == 'vit_h':
        model_name = 'samvit_huge_patch16'
        model_args = HUGE_ARGS

    model = _create_vision_transformer(
        model_name, pretrained=pretrained, **dict(model_args, **kwargs))
    
    class TimmSamImageEncoder(model.__class__):
        def forward(self, x):
            return self.forward_features(x)
        
    model.__class__ = TimmSamImageEncoder
    return model, model_name


def make_sam_training_class(sam_class):
    class SamTrain(sam_class):
        def forward(
            self,
            x,
            iteration,
            point_coords = None, # [B, N, m, 2]
            point_values = None, # [B, N, m]
            boxes = None, # [B, N, 4]
            masks = None, # [B, N, 1, 256, 256]
        ):
            if iteration == 0:
                image_embeddings = self.image_encoder(x)
            else:
                image_embeddings = x # [B, 256, H/4, W/4]

            batched_low_res_masks = []
            batched_iou_predictions = []
            for b in range(image_embeddings.shape[0]):
                curr_embedding = image_embeddings[b, ...] # [256, H/4, W/4]

                points = (point_coords[b, ...], point_values[b, ...]) if \
                    point_coords is not None and point_values is not None else None
                box = boxes[b, ...] if boxes is not None else None
                mask = masks[b, ...] if masks is not None else None

                sparse_embeddings, dense_embeddings = self.prompt_encoder(points=points, boxes=box, masks=mask)
                if points is None and box is None:
                    sparse_embeddings = sparse_embeddings.detach()
                if mask is None:
                    dense_embeddings = dense_embeddings.detach()
                
                low_res_masks, iou_predictions = self.mask_decoder(
                    image_embeddings = curr_embedding.unsqueeze(0),
                    image_pe = self.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings = sparse_embeddings,
                    dense_prompt_embeddings = dense_embeddings,
                    multimask_output = False
                ) # [N, 1, H/4, W/4], [N, 1]

                batched_low_res_masks.append(low_res_masks)
                batched_iou_predictions.append(iou_predictions)
            
            return image_embeddings, torch.stack(batched_low_res_masks, dim=0), torch.stack(batched_iou_predictions, dim=0)
    
    return SamTrain


def create_model(
    sam_type, checkpoint, encoder_type, 
    lora, r, enable_lora, 
    freeze_image_encoder=False,
    freeze_sparse_prompt=False,
    freeze_dense_prompt=False
):
    def freeze(model, not_freeze:list=None):
        for k, p in model.named_parameters():
            if not_freeze is not None:
                if all(name not in k for name in not_freeze):
                    p.requires_grad = False
            else:
                p.requires_grad = False

    model = sam_model_registry[sam_type](checkpoint=None)
    checkpoint = torch.load(checkpoint, map_location='cpu')

    if encoder_type == 'timm':
        model.load_state_dict(checkpoint, strict=False)
        
        timm_sam_encoder, name = make_timm_sam_encoder(sam_type=sam_type, pretrained=False)
        if lora:
            for block in timm_sam_encoder.blocks:
                in_features = block.attn.qkv.in_features
                out_features = block.attn.qkv.out_features
                block.attn.qkv = MergedLinear(in_features, out_features, r=r, enable_lora=enable_lora)
                block.attn.proj = Linear(in_features, in_features, r=r)

        pretrained_cfg = resolve_pretrained_cfg(name)
        pretrained_cfg = pretrained_cfg.to_dict()
        load_pretrained(model=timm_sam_encoder, pretrained_cfg=pretrained_cfg,filter_fn=checkpoint_filter_fn, strict=False)
        
        model.image_encoder = timm_sam_encoder

    elif encoder_type == 'meta':
        if lora:
            for block in model.image_encoder.blocks:
                in_features = block.attn.qkv.in_features
                out_features = block.attn.qkv.out_features
                block.attn.qkv = MergedLinear(in_features, out_features, r=r, enable_lora=enable_lora)
                block.attn.proj = Linear(in_features, in_features, r=r)
    
        model.load_state_dict(checkpoint, strict=False)

    SamTrain = make_sam_training_class(model.__class__)
    model.__class__ = SamTrain

    if lora:
        not_freeze = ['lora', 'patch_embed', 'neck']
        freeze(model.image_encoder, not_freeze)
    if freeze_image_encoder:
        freeze(model.image_encoder)
    if freeze_sparse_prompt:
        freeze(model.prompt_encoder, ['mask'])
    if freeze_dense_prompt:
        freeze(model.prompt_encoder, ['point'])
    return model
