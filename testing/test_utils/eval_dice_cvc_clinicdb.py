import os
import cv2
import numpy as np
from pathlib import Path
import sys
sys.path.append('./')
from .utils_eval import *


def eval(pred_path, gt_path='/data/endoscopy_pro/cvc_clinicdb/label/', ref_frame=3):
    global_video_dice_meter = AverageMeter()
    image_dice_meter = AverageMeter()

    # viedeo name
    video_names = sorted(os.listdir(gt_path))
    video_names = [video_name for video_name in video_names if ".DS" not in video_name]
    video_names.sort()
    for video_name in video_names:
        gt_path_video = os.path.join(gt_path, video_name)
        pred_path_video = os.path.join(pred_path, video_name)
        gt_images = [str(img_path) for img_path in sorted(Path(gt_path_video).rglob("*.png"))]
        pred_images = [str(img_path) for img_path in sorted(Path(pred_path_video).rglob("*.png"))]
        while len(gt_images) > len(pred_images):
            pred_images.insert(0, None)
        
        video_dice_meter = AverageMeter()
        for i, (gt_img, pred_img) in enumerate(zip(gt_images, pred_images)): 
            if i < ref_frame:
                continue
            
            # ground truth image
            gt_img = cv2.imread(gt_img)
            gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2GRAY) > 0
            gt_img = np.uint8(gt_img)

            # prediction
            pred_img = cv2.imread(pred_img)
            pred_img = cv2.cvtColor(pred_img, cv2.COLOR_BGR2GRAY) > 0
            pred_img = np.uint8(pred_img)

            true_pos, true_neg, false_pose, false_neg = compute_metrics(pred_img, gt_img)
            dice_image = 2 * true_pos / (2 * true_pos + false_pose + false_neg + 1e-10)

            # frame-wise dice
            image_dice_meter.update(dice_image)
            # video-wise dice
            video_dice_meter.update(dice_image)

        print(video_name + ',', "Dice: %.2f" %(100 * video_dice_meter.avg))
        global_video_dice_meter.update(video_dice_meter.avg)

    print("\nVideo Dice: %.2f" %(100 * global_video_dice_meter.avg))
    print("Image Dice: %.2f\n" %(100 * image_dice_meter.avg))


# for debug
# if __name__ == '__main__':
#     pred_path = '/test/transfer/cvc_clinicdb'
#     eval(pred_path=pred_path)
