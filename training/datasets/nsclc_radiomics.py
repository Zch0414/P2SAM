import albumentations as A

from .utils import *


PATH = '/data/lung_pro/nsclc_radiomics'
SPLIT = [331, 108, 312, 17, 262, 172, 221, 363, 393, 378, 149, 125, 126, 103, 93, 297, 36, 184, 163, 143, 96, 15, 323, 252, 207, 146, 137, 130,
         50, 360, 373, 121, 379, 314, 8, 349, 285, 381, 269, 279, 162, 383, 33, 244, 370, 168, 189, 0, 344, 294, 79, 70, 260, 320, 194, 53,
         141, 35, 414, 195, 72, 91, 420, 245, 265, 166, 84, 275, 4, 82, 59, 110, 3, 182, 259, 158, 74, 136, 23, 300, 217, 212, 351, 406,
         339, 87, 120, 135, 24, 154, 276, 198, 114, 234, 365, 2, 282, 81, 38, 63, 407, 374, 109, 144, 229, 71, 106, 25, 342, 47, 122, 51,
         340, 20, 77, 9, 216, 54, 43, 397, 11, 257, 315, 247, 263, 396, 12, 255, 111, 28, 41, 409, 230, 353, 196, 39, 1, 6, 105, 305,
         382, 26, 138, 206, 185, 375, 391, 88, 329, 332, 75, 273, 13, 167, 325, 19, 119, 73, 361, 286, 115, 281, 246, 118, 235, 197, 390, 309,
         218, 174, 306, 395, 208, 256, 225, 242, 45, 147, 328, 203, 347, 243, 301, 327, 335, 321, 161, 102, 233, 210, 415, 60, 188, 215, 107, 268,
         14, 224, 368, 34, 284, 52, 55, 85, 78, 64, 266, 302, 156, 5, 408, 186, 213, 22, 270, 385, 412, 56, 89, 399, 132, 219, 369, 376,
         181, 350, 171, 264, 377, 345, 318, 10, 150, 324, 148, 352, 405, 330, 112, 388, 272, 123, 326, 211, 227, 258, 254, 287, 7, 145, 170, 261,
         169, 417, 124, 31, 44, 298, 293, 69, 304, 354, 404, 299, 348, 101, 153, 392, 16, 271, 241, 201, 90, 364, 57, 366, 249, 173, 223, 278,
         290, 202, 367, 419, 322, 191, 46, 274, 237, 116, 236, 319, 190, 49, 100, 42, 222, 80, 411, 362, 316, 413, 277, 231, 251, 139, 346, 179,
         193, 288, 250, 209, 40, 113, 289, 334, 214, 30, 240, 204, 239, 129, 253, 192, 157, 160, 83, 317, 341, 267, 226, 232, 384, 117, 164, 134,
         356, 403, 95, 380, 99, 29, 104, 400, 200, 220, 92, 67, 187, 337, 389, 21, 18, 151, 238, 308, 418, 86, 338, 333, 133, 296, 313, 58,
         410, 152, 183, 307, 66, 98, 76, 343, 62, 68, 127, 165, 142, 401, 37, 355, 372, 402, 228, 32, 61, 295, 65, 97, 303, 311, 398, 140,
         177, 371, 180, 175, 27, 336, 359, 205, 386, 416, 248, 358, 159, 394, 291, 310, 131, 155, 176, 94, 292, 387, 199, 283, 48, 280, 178, 128, 357]
SAM_MEAN = [123.675, 116.28, 103.53]
SAM_STD = [58.395, 57.12, 57.375]


class NsclcRadiomicsDataset(BasicDataset):
    def __init__(self, data, transform, split='tr', pixel_mean=[123.675, 116.28, 103.53], pixel_std=[58.395, 57.12, 57.375]):
        super().__init__(data=data, transform=transform, split=split, pixel_mean=pixel_mean, pixel_std=pixel_std)
        self.resize = A.Compose([A.Resize(self.target_length, self.target_length)])
        
    def __getitem__(self, idx):
        batch = dict()

        # debug
        image_name = str(self.data[idx]['image']).split('/')[-1].split('.')[0]
        label_name = str(self.data[idx]['label']).split('/')[-1].split('.')[0]
        assert image_name == label_name
        batch['name'] = image_name

        # load
        image = Image.open(self.data[idx]['image'])
        label = Image.open(self.data[idx]['label'])
        image = np.array(image)[..., None] # [h, w, 1]
        label = np.array(label)[..., None] # [h, w, 1]

        # record
        if self.split == 'ts' or self.split == 'val':
            batch['original_size'] = torch.Tensor([image.shape[0], image.shape[1]]).int()
            batch['resize_size'] = torch.Tensor([self.target_length, self.target_length]).int()

        # resize
        resized = self.resize(image=image, mask=label)
        image = resized['image']
        label = resized['mask']

        # augmentation
        if self.transform is not None:
            transformed = self.transform(image=image, mask=label)
            image = transformed['image']
            label = transformed['mask']

        # totensor
        image = torch.from_numpy(image).permute(2, 0, 1).repeat(3, 1, 1)
        label = torch.from_numpy(label).permute(2, 0, 1)

        # normalization
        if self.pixel_mean is not None and self.pixel_std is not None:
            image = (image - self.pixel_mean) / self.pixel_std
        else:
            image = (image - image.min()) / torch.clamp(image.max() - image.min(), min=1e-8, max=None) 

        batch['image'] = image
        batch['label'] = label

        return batch


def get_file_list_nsclc_radiomics(patient_set, root=PATH):
    image_list = []
    label_list = []
    for patient in patient_set:
        image_sub_list = get_file_list(os.path.join(root, 'image', patient), suffix='*.png')
        image_list += image_sub_list
        label_sub_list = get_file_list(os.path.join(root, 'label', patient), suffix='*.png')
        label_list += label_sub_list
    data = [{'image': image, 'label': label} for (image, label) in zip(image_list, label_list)]
    return data


def build_dataset(pretrained_weight, root=PATH):
    patien_list = os.listdir(os.path.join(root, 'image'))
    patient_tr, patient_val, patient_ts = split_dataset(data=patien_list, sets=[337, 42, 42], split=SPLIT)
    tr_data = get_file_list_nsclc_radiomics(patient_set=patient_tr, root=root)
    val_data = get_file_list_nsclc_radiomics(patient_set=patient_val, root=root)
    ts_data = get_file_list_nsclc_radiomics(patient_set=patient_ts, root=root)
    
    train_transform = A.Compose(
        [   
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Affine(rotate=(-10, 10), scale=(0.8, 1.2), p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        ]
    )
    
    # pretrained_weight's format is crucial here
    pretrain_model_name = pretrained_weight.split('/')[-1].split('_')[0]
    pixel_mean = SAM_MEAN; pixel_std = SAM_STD
    if pretrain_model_name == 'medsam':
        pixel_mean = None; pixel_std = None

    tr_ds = NsclcRadiomicsDataset(data=tr_data, transform=train_transform, split='tr', pixel_mean=pixel_mean, pixel_std=pixel_std)
    vl_ds = NsclcRadiomicsDataset(data=val_data, transform=None, split='val', pixel_mean=pixel_mean, pixel_std=pixel_std)    
    ts_ds = NsclcRadiomicsDataset(data=ts_data, transform=None, split='ts', pixel_mean=pixel_mean, pixel_std=pixel_std)
    return tr_ds, vl_ds, ts_ds