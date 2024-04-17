import torch
import numpy as np
import json
import h5py
import cv2
from PIL import Image
from torch.utils.data import Dataset

class CrowdCountingDataset(Dataset):
    def __init__(self, img_list_file, transform=None):
        with open(img_list_file, 'r') as f:
            img_list = json.load(f)
            
        self.img_list = img_list
        self.transform = transform

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        gt_path = img_path.replace('.jpg','.h5').replace('images','ground_truth')

        # 读取原图像
        img = Image.open(img_path).convert('RGB')

        # 读取密度图
        with h5py.File(gt_path, 'r') as hf:
            density = np.array(hf['density'])

        # 数据增强
        if self.transform:
            img = self.transform(img)
            
            original_size = density.shape[:2]  # 原始大小
            
            # 使用OpenCV缩放密度图
            density = cv2.resize(density, (224, 224), interpolation=cv2.INTER_LINEAR)

            # 调整密度图的值以保持总人数不变
            density *= (original_size[0] * original_size[1]) / (224*224)

            density_min = density.min()
            density_max = density.max()
            density_normalized = (density - density_min) / (density_max - density_min)
            density_normalized = density_normalized.round(4)

            # 将numpy数组转换为torch张量
            density = torch.from_numpy(density_normalized).float()

        return img, density
    
