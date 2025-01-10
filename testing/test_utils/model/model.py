from functools import partial
import sys
sys.path.append('./')
sys.path.append('../')

import torch
import torch.nn as nn

from ..custom_segment_anything.build_sam import sam_model_registry
from .lora import Linear, MergedLinear
from .vision_transformer_sam import _create_vision_transformer


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


def create_model(sam_type, checkpoint, encoder_type, lora, r, enable_lora):
    model = sam_model_registry[sam_type](checkpoint=None)
    checkpoint = torch.load(checkpoint, map_location='cpu')

    if encoder_type == 'timm':
        timm_sam_encoder, _ = make_timm_sam_encoder(sam_type=sam_type, pretrained=False)
        setattr(timm_sam_encoder, 'img_size', 1024)
        if lora:
            for block in timm_sam_encoder.blocks:
                in_features = block.attn.qkv.in_features
                out_features = block.attn.qkv.out_features
                block.attn.qkv = MergedLinear(in_features, out_features, r=r, enable_lora=enable_lora)
                block.attn.proj = Linear(in_features, in_features, r=r)
        model.image_encoder = timm_sam_encoder

    elif encoder_type == 'meta':
        if lora:
            for block in model.image_encoder.blocks:
                in_features = block.attn.qkv.in_features
                out_features = block.attn.qkv.out_features
                block.attn.qkv = MergedLinear(in_features, out_features, r=r, enable_lora=enable_lora)
                block.attn.proj = Linear(in_features, in_features, r=r)
    
    try:
        model.load_state_dict(checkpoint['model'], strict=True)
        print(f"validation: {checkpoint['result']} at {checkpoint['epoch']} epoch.")
    except KeyError:
        model.load_state_dict(checkpoint, strict=True)

    return model
