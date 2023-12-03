import os
import numpy as np
import cv2
from typing import Any, Tuple
from torch.utils.data import Dataset


class GPRDataset(Dataset):
    def __init__(self, 
                 data_folder, 
                 transform = None):
        self.data_folder = data_folder
        self.image_folder = os.path.join(data_folder, "images")
        self.ann_folder = os.path.join(data_folder, "annotations")
        self.transform = transform

        self.data = list(sorted(os.listdir(os.path.join(data_folder, "images"))))
        self.targets = list(sorted(os.listdir(os.path.join(data_folder, "annotations"))))
        self.num_classes = 2

    def __getitem__(self, index: int) -> Tuple[Any, Any]:

        img_path = os.path.join(self.image_folder, self.data[index])
        annotation_path = os.path.join(self.ann_folder, self.targets[index])
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        norm_img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        annotation = np.loadtxt(annotation_path, ndmin=2)        

        data = {
            "img":norm_img, 
            "bboxes": annotation[:,1:5], # [x_center, y_center, w, h]
        }

        if self.transform is not None:
            data = self.transform(data)

        return data

    def __len__(self):
        return len(self.data)




