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
from test_utils.eval_miou_perseg import eval

def get_arguments():
    
    parser = argparse.ArgumentParser()

    # path
    parser.add_argument('--data', type=str, default='data/perseg')
    parser.add_argument('--outdir', type=str, default='results/p2sam/perseg')
    parser.add_argument('--ref-idx', type=str, default='00')
    
    # model
    parser.add_argument('--ckpt', type=str, default='pretrained_weights/sam_vit_h.pth')
    parser.add_argument('--sam-type', type=str, default='vit_h')
    parser.add_argument('--guided-attn', action='store_true')

    # p2sam
    parser.add_argument('--max-num-pos', type=int, default=1)
    parser.add_argument('--min-num-pos', type=int, default=1)
    parser.add_argument('--reg-patch-weight', action='store_true')

    # vis for a specific case
    parser.add_argument('--vis', action='store_true')
    parser.add_argument('--vis-name', type=str, default='')

    args = parser.parse_args()
    return args


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # get
    args = get_arguments()
    print("Args:", args)

    # data path
    images_path = os.path.join(args.data, 'Images')
    masks_path = os.path.join(args.data, 'Annotations')
    if args.outdir:
        Path(args.outdir).mkdir(parents=True, exist_ok=True)

    print("Create model..." )
    sam_type, sam_ckpt = args.sam_type, args.ckpt
    sam = create_model(sam_type, sam_ckpt, encoder_type='meta', lora=False, r=None, enable_lora=None)
    sam = sam.to(device)
    sam.eval()
    
    for obj_name in os.listdir(images_path):
        if args.vis:
            if obj_name != args.vis_name:
                continue
        if ".DS" not in obj_name:
            print(f"\nSegment {obj_name}...")
            
            # video (images) path
            ref_image_path = os.path.join(images_path, obj_name, args.ref_idx + '.jpg')
            ref_mask_path = os.path.join(masks_path, obj_name, args.ref_idx + '.png')
            test_images_path = os.path.join(images_path, obj_name)
            output_path = os.path.join(args.outdir, obj_name)
            os.makedirs(output_path, exist_ok=True)
            for test_idx in range(len(os.listdir(test_images_path))):
                
                # image path
                test_idx = '%02d' % test_idx
                test_image_path = os.path.join(test_images_path, test_idx + '.jpg')

                # forward
                pred_mask, point_coords, point_labels, dis = p2sam_perseg(args, sam, ref_image_path, ref_mask_path, test_idx, test_image_path, output_path)

                # vis
                if args.vis:
                    if test_idx == args.ref_idx:
                        ref_image = cv2.imread(ref_image_path)
                        ref_image = cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB)
                        
                        ref_mask = cv2.imread(ref_mask_path)
                        ref_mask = cv2.cvtColor(ref_mask, cv2.COLOR_BGR2RGB)
                        ref_mask = ref_mask.transpose(2, 0, 1)[0]      
                        ref_mask = ref_mask - (ref_mask.max()+ref_mask.min()) / 2.0
                        ref_mask = ref_mask > 0.0

                        test_image = cv2.imread(test_image_path)
                        test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
                        
                        # plot groud truth
                        gt_output_path = os.path.join(output_path, f'{test_idx}_ref.jpg')
                        plt.figure(figsize=(10, 10))
                        plt.imshow(test_image)
                        show_mask(ref_mask, plt.gca(), -1, linewidth=8) 
                        plt.axis('off')
                        plt.savefig(gt_output_path, bbox_inches='tight', pad_inches=0)
                    
                    else:
                        test_image = cv2.imread(test_image_path)
                        test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
                        
                        # plot prediction
                        pred_output_path = os.path.join(output_path, f'{test_idx}_{dis:.2f}.jpg')
                        plt.figure(figsize=(10, 10))
                        plt.imshow(test_image)
                        show_mask(pred_mask, plt.gca(), -1, linewidth=8) 
                        show_pos_points(point_coords, point_labels, plt.gca(), -1, None, marker_size=2048, linewidth=3.75)
                        show_neg_points(point_coords, point_labels, plt.gca(), 'red', marker_size=2048, linewidth=3.75)
                        plt.axis('off')
                        plt.savefig(pred_output_path, bbox_inches='tight', pad_inches=0)                            

    # eval
    print("\nEvaluate...\n")
    eval(pred_path=args.outdir, gt_path=os.path.join(args.data, 'Annotations'))


if __name__ == "__main__":
    main()