import os
import cv2
import numpy as np


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def intersectionAndUnion(output, target):
    assert (output.ndim in [1, 2, 3])
    assert output.shape == target.shape
    output = output.reshape(output.size).copy()
    target = target.reshape(target.size)
    
    area_intersection = np.logical_and(output, target).sum()
    area_union = np.logical_or(output, target).sum()
    area_target = target.sum()
    
    return area_intersection, area_union, area_target


def compute_metrics(output, target):
    assert (output.ndim in [1, 2, 3])
    assert output.shape == target.shape
    output = output.reshape(output.size).copy()
    target = target.reshape(target.size).copy()
    output = output.astype(bool)
    target = target.astype(bool)

    true_pos = np.logical_and(output, target).sum()
    true_neg = np.logical_and(~output, ~target).sum()
    false_pos = np.logical_and(output, ~target).sum()
    false_neg = np.logical_and(~output, target).sum()

    return true_pos, true_neg, false_pos, false_neg


def prepare_gt_path(ref_path, gt_path):
    ref_path = str(ref_path)
    image_name = ref_path.split('/')[-1]
    breath_gate = ref_path.split('/')[-2]
    visit_date = ref_path.split('/')[-3]
    patient_name = ref_path.split('/')[-4]

    return os.path.join(gt_path, patient_name, visit_date, breath_gate, image_name)


def prepare_pred_path(ref_path, gt_path):
    ref_path = str(ref_path)
    image_name = ref_path.split('/')[-1]
    breath_gate = ref_path.split('/')[-2]
    visit_date = ref_path.split('/')[-3]
    patient_name = ref_path.split('/')[-4]
    if image_name not in os.listdir(os.path.join(gt_path, patient_name, visit_date, 'Gated_00')):
        return None
    return os.path.join(gt_path, patient_name, visit_date, 'Gated_00', image_name)


def extract_contours(mask):
    mask = mask.copy().astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    points = [contour[:, 0, :] for contour in contours]
    return np.vstack(points)