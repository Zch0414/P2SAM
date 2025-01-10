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
from test_utils.utils_vis import show_mask
from test_utils.eval_dice_4d_lung import eval



def get_arguments():
    
    parser = argparse.ArgumentParser()

    # path
    parser.add_argument('--data', type=str, default='/data/lung_pro/4d_lung_multi_visits')
    parser.add_argument('--outdir', type=str, default='/results/direct_transfer/4d_lung')
    parser.add_argument('--gate', type=int, default=0)
    
    # model
    parser.add_argument('--ckpt', type=str, default='/pretrained_weights/nsclc_full_base/checkpoint.pth')
    parser.add_argument('--sam-type', type=str, default='vit_b')
    parser.add_argument('--encoder-type', type=str, default='timm')
    parser.add_argument('--medsam', action='store_true')
    parser.add_argument('--lora', action='store_true')
    parser.add_argument('--lora-rank', type=int, default=1)
    parser.add_argument('--point', action='store_true')
    parser.add_argument('--box', action='store_true')

    # evaluation
    parser.add_argument('--test-slice-num', type=int, default=10000)

    # vis for a specified case
    parser.add_argument('--vis', action='store_true')
    parser.add_argument('--vis-name', type=str, default='')

    args = parser.parse_args()
    return args


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # get arugments
    args = get_arguments()
    print("Args:", args)

    # data path
    images_path = os.path.join(args.data, 'image')
    if args.outdir:
        Path(args.outdir).mkdir(parents=True, exist_ok=True)

    # model
    print("Create model...")
    sam_type, sam_ckpt = args.sam_type, args.ckpt
    sam = create_model(sam_type, sam_ckpt, encoder_type=args.encoder_type, lora=args.lora, r=args.lora_rank, enable_lora=[True, True, True])
    sam = sam.to(device)
    sam.eval()
    
    for patient_name in sorted(os.listdir(images_path)):
        if args.vis:
            if patient_name != args.vis_name:
                continue

        if ".DS" not in patient_name:
            if len(os.listdir(os.path.join(images_path, patient_name))) <= 1:
                continue

            print(f"\nSegment {patient_name}...")
            # patient path
            patient_path = os.path.join(images_path, patient_name)
            ref_visit_date = sorted([i for i in os.listdir(patient_path) if ".DS" not in i])[0]
            ref_visit_path = os.path.join(patient_path, ref_visit_date)
            breath_gate = sorted([i for i in os.listdir(ref_visit_path) if ".DS" not in i])[args.gate]
            for visit_date in sorted(os.listdir(patient_path)):
                if ".DS" not in visit_date:
                    print(f"\nSegment {visit_date}...")
                    
                    # visit path
                    test_visit_path = os.path.join(patient_path, visit_date)
                    test_gate_path = os.path.join(test_visit_path, breath_gate)
                    output_path = os.path.join(args.outdir, patient_name, visit_date, breath_gate)
                    os.makedirs(output_path, exist_ok=True)

                    num_test_slices = len(os.listdir(test_gate_path))
                    begin = num_test_slices // 2 - args.test_slice_num // 2
                    end = num_test_slices // 2 + (args.test_slice_num + 1) // 2
                    if begin < 0:
                        begin = 0
                    if end > num_test_slices:
                        end = num_test_slices
                    for test_i in range(begin, end):
                        # slice path
                        test_slice = sorted(os.listdir(test_gate_path))[test_i]
                        test_slice_path = os.path.join(test_gate_path, test_slice)
                        test_mask_path = image2mask_path(test_slice_path)
                        slice_name = test_slice.split('.')[0]

                        # forward
                        pred_mask, dice_score = run_medical(args, sam, test_slice_path, test_mask_path, output_path, slice_name)
                        
                        # vis
                        if args.vis:
                            test_image = cv2.imread(test_slice_path)
                            test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
                            test_mask = cv2.imread(test_mask_path, cv2.IMREAD_GRAYSCALE)        
                            test_mask = test_mask - (test_mask.max()+test_mask.min()) / 2.0
                            gt_mask = test_mask > 0.0
                            
                            # plot groud truth
                            gt_output_path = os.path.join(output_path, f'{slice_name}_gt.jpg')
                            plt.figure(figsize=(10, 10))
                            plt.imshow(test_image)
                            show_mask(gt_mask, plt.gca(), -1, linewidth=4) 
                            plt.axis('off')
                            plt.savefig(gt_output_path, bbox_inches='tight', pad_inches=0)

                            # plot prediction
                            pred_output_path = os.path.join(output_path, f'{slice_name}_pred_{dice_score:.2f}.jpg')
                            plt.figure(figsize=(10, 10))
                            plt.imshow(test_image)
                            show_mask(pred_mask, plt.gca(), -1, linewidth=4) 
                            plt.axis('off')
                            plt.savefig(pred_output_path, bbox_inches='tight', pad_inches=0)                            

    # eval
    print("\nEvaluate...\n")
    eval(pred_path=args.outdir, gt_path=os.path.join(args.data, 'label'))

if __name__ == "__main__":
    main()
