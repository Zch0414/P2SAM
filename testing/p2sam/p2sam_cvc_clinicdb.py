import os
import argparse
from pathlib import Path

import torch

import sys
sys.path.append('./')
sys.path.append('../')
import warnings
warnings.filterwarnings('ignore')

from test_utils.model import create_model
from test_utils.utils import *
from test_utils.utils_vis import show_mask, show_pos_points, show_neg_points
from test_utils.eval_dice_cvc_clinicdb import eval


def get_arguments():
    
    parser = argparse.ArgumentParser()

    # path
    parser.add_argument('--data', type=str, default='data/endoscopy_pro/cvc_clinicdb')
    parser.add_argument('--outdir', type=str, default='results/p2sam/cvc_clinicdb')
    parser.add_argument('--ref-frame', type=int, default=3)
    
    # model    
    parser.add_argument('--ckpt', type=str, default='pretrained_weights/kvasir_full_base/checkpoint.pth')
    parser.add_argument('--sam-type', type=str, default='vit_b')
    parser.add_argument('--encoder-type', type=str, default='timm')
    parser.add_argument('--medsam', action='store_true')
    parser.add_argument('--lora', action='store_true')
    parser.add_argument('--lora-rank', type=int, default=1)

    # p2sam
    parser.add_argument('--max-num-pos', type=int, default=1)
    parser.add_argument('--min-num-pos', type=int, default=1)
    parser.add_argument('--max-num-neg', type=int, default=1)
    parser.add_argument('--min-num-neg', type=int, default=1)
    parser.add_argument('--guided-attn', action='store_true')
    parser.add_argument('--reg-patch-weight', action='store_true')
    ## save point if needed
    parser.add_argument('--save-point', action='store_true')
    
    # vis for a specified case
    parser.add_argument('--vis', action='store_true')
    parser.add_argument('--vis-name', type=str, default='')

    args = parser.parse_args()
    return args


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # get arguments
    args = get_arguments()
    print("Args:", args)

    # data path
    images_path = os.path.join(args.data, 'image')
    masks_path = os.path.join(args.data, 'label')
    if args.outdir:
        Path(args.outdir).mkdir(parents=True, exist_ok=True)

    # model
    print("Create model...")
    sam_type, sam_ckpt = args.sam_type, args.ckpt
    sam = create_model(sam_type, sam_ckpt, encoder_type=args.encoder_type, lora=args.lora, r=args.lora_rank, enable_lora=[True, True, True])
    sam = sam.to(device)
    sam.eval()

    for obj_name in sorted(os.listdir(images_path)):
        if args.vis:
            if obj_name != args.vis_name:
                continue
        if ".DS" not in obj_name:
            print(f"\nSegment {obj_name}...")
            
            # video (images) path
            ref_image_path = sorted(os.listdir(os.path.join(images_path, obj_name)))[args.ref_frame-1]
            ref_image_path = os.path.join(images_path, obj_name, ref_image_path)
            ref_mask_path = sorted(os.listdir(os.path.join(masks_path, obj_name)))[args.ref_frame-1]
            ref_mask_path = os.path.join(masks_path, obj_name, ref_mask_path)
            test_images_path = os.path.join(images_path, obj_name)
            test_masks_path = os.path.join(masks_path, obj_name)
            output_path = os.path.join(args.outdir, obj_name)
            os.makedirs(output_path, exist_ok=True)
            for image_path, mask_path in zip(sorted(os.listdir(test_images_path)), sorted(os.listdir(test_masks_path))):
                
                # image path
                test_image_name = image_path.split('.')[0].split('_')[-1]
                test_mask_name = mask_path.split('.')[0].split('_')[-1]
                assert test_image_name == test_mask_name, f'{test_image_name} should be the same as {test_mask_name}'
                test_image_path = os.path.join(test_images_path, image_path)
                test_mask_path = os.path.join(test_masks_path, mask_path)
                
                # forward
                pred_mask, point_coords, point_labels, dice, dis = p2sam_medical(args, sam, ref_image_path, ref_mask_path, test_image_path, test_mask_path, output_path, test_image_name)
                
                # save point_coords and point_labels if needed
                if args.save_point:
                    point_coords_output_path = os.path.join(output_path, f'{test_image_name}_point_coords.npy')
                    point_labels_output_path = os.path.join(output_path, f'{test_image_name}_point_labels.npy')
                    np.save(point_coords_output_path, point_coords)
                    np.save(point_labels_output_path, point_labels)

                # vis
                if args.vis:
                    test_image = cv2.imread(test_image_path)
                    test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
                    
                    test_mask = cv2.imread(test_mask_path, cv2.IMREAD_GRAYSCALE)        
                    test_mask = test_mask - (test_mask.max()+test_mask.min()) / 2.0
                    gt_mask = test_mask > 0.0
                    
                    # plot groud truth
                    gt_output_path = os.path.join(output_path, f'{test_image_name}_gt.jpg')
                    plt.figure(figsize=(10, 10))
                    plt.imshow(test_image)
                    show_mask(gt_mask, plt.gca(), -1, linewidth=8) 
                    plt.axis('off')
                    plt.savefig(gt_output_path, bbox_inches='tight', pad_inches=0)

                    # plot prediction
                    pred_output_path = os.path.join(output_path, f'{test_image_name}_pred_{dice:.2f}_{dis:.2f}.jpg')
                    plt.figure(figsize=(10, 10))
                    plt.imshow(test_image)
                    show_mask(pred_mask, plt.gca(), -1, linewidth=8) 
                    show_pos_points(point_coords, point_labels, plt.gca(), -1, None, marker_size=2025, linewidth=3.75)
                    show_neg_points(point_coords, point_labels, plt.gca(), 'red', marker_size=2025, linewidth=3.75)
                    plt.axis('off')
                    plt.savefig(pred_output_path, bbox_inches='tight', pad_inches=0)                            

    # eval
    print("\nEvaluate...\n")
    eval(pred_path=args.outdir, gt_path=os.path.join(args.data, 'label'))

    
if __name__ == "__main__":
    main()
