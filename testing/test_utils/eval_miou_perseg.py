import os
import cv2
import numpy as np
from pathlib import Path
import sys
sys.path.append('./')
from .utils_eval import *


def eval(pred_path, gt_path='/data/perseg/Annotations', ref_idx='00'):


    class_names = sorted(os.listdir(gt_path))
    class_names = [class_name for class_name in class_names if ".DS" not in class_name]
    class_names.sort()

    mIoU, mAcc = 0, 0
    count = 0
    for class_name in class_names:
        count += 1
        gt_path_class = os.path.join(gt_path, class_name)
        pred_path_class = os.path.join(pred_path, class_name)

        gt_images = [str(img_path) for img_path in sorted(Path(gt_path_class).rglob("*.png"))]
        pred_images = [str(img_path) for img_path in sorted(Path(pred_path_class).rglob("*.png"))]

        intersection_meter = AverageMeter()
        union_meter = AverageMeter()
        target_meter = AverageMeter()

        for i, (gt_img, pred_img) in enumerate(zip(gt_images, pred_images)): 
            if ref_idx in gt_img:
                continue

            gt_img = cv2.imread(gt_img)
            gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2GRAY) > 0
            gt_img = np.uint8(gt_img)

            pred_img = cv2.imread(pred_img)
            pred_img = cv2.cvtColor(pred_img, cv2.COLOR_BGR2GRAY) > 0
            pred_img = np.uint8(pred_img)

            intersection, union, target = intersectionAndUnion(pred_img, gt_img)
            intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)

        iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
        accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)

        print(class_name + ',', "IoU: %.2f," %(100 * iou_class), "Acc: %.2f\n" %(100 * accuracy_class))

        mIoU += iou_class
        mAcc += accuracy_class

    print("\nmIoU: %.2f" %(100 * mIoU / count))
    print("mAcc: %.2f\n" %(100 * mAcc / count))


# for debug
# if __name__ == '__main__':
#     pred_path = '/nfs/turbo/coe-liyues/chuizhao/iclr_2025/test/transfer/cvc_clinicdb'
#     eval(pred_path=pred_path)