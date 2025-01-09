import os
import cv2
import numpy as np
from pathlib import Path
import sys
sys.path.append('./')
from .utils_eval import *


def eval(pred_path, gt_path='/data/lung_pro/4d_lung_multi_visits/label/', ref_visit=1):
    global_patient_dice_meter = AverageMeter()
    global_visit_dice_meter = AverageMeter()
    image_dice_meter = AverageMeter()

    # patient name
    patient_names = sorted(os.listdir(gt_path))
    patient_names = [patient_name for patient_name in patient_names if ".DS" not in patient_name]
    patient_names.sort()
    for patient_name in patient_names:
        if len(os.listdir(os.path.join(gt_path, patient_name))) <= 1:
            continue
        patient_dice_meter = AverageMeter()

        # visit date
        patient_path = os.path.join(pred_path, patient_name)
        visit_dates = sorted(os.listdir(patient_path))
        visit_dates = [visit_date for visit_date in visit_dates if ".DS" not in visit_date]
        visit_dates.sort()
        for i, visit_date in enumerate(visit_dates):
            if i < ref_visit:
                continue
            visit_dice_meter = AverageMeter()
            visit_path = os.path.join(patient_path, visit_date)

            # slice
            pred_images = [str(img_path) for img_path in sorted(Path(visit_path).rglob("*.png"))]
            for i, pred_img in enumerate(pred_images): 
                # ground truth image
                gt_img = prepare_gt_path(pred_img, gt_path)
                gt_img = cv2.imread(gt_img)
                gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2GRAY) > 0
                gt_img = np.uint8(gt_img)
                
                # prediction
                pred_img = cv2.imread(pred_img)
                pred_img = cv2.cvtColor(pred_img, cv2.COLOR_BGR2GRAY) > 0
                pred_img = np.uint8(pred_img)

                true_pos, true_neg, false_pos, false_neg = compute_metrics(pred_img, gt_img)
                dice_image = 2 * true_pos / (2 * true_pos + false_pos + false_neg + 1e-10)

                # slice-wise dice
                image_dice_meter.update(dice_image)
                # volumn-wise (visit-wise) dice
                visit_dice_meter.update(dice_image)
            # patient_wise_dice
            patient_dice_meter.update(visit_dice_meter.avg)
            # dataset_wise_dice
            global_visit_dice_meter.update(visit_dice_meter.avg)

        print(patient_name + ',', "Dice: %.2f" %(100 * patient_dice_meter.avg))
        global_patient_dice_meter.update(patient_dice_meter.avg)

    print("\nPatient Dice: %.2f" %(100 * global_patient_dice_meter.avg))
    print("Visit Dice: %.2f" %(100 * global_visit_dice_meter.avg))
    print("Image Dice: %.2f\n" %(100 * image_dice_meter.avg))


# for debug
# if __name__ == '__main__':
#     pred_path = '/nfs/turbo/coe-liyues/chuizhao/experiments/test/transfer/4d_lung'
#     eval(pred_path=pred_path)
