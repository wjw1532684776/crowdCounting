import argparse
import os

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np

from dataset import CrowdCountingDataset
from model import CSRNet
from utils import evaluate_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-test', '--test_json', type=str, required=True, help='path to test JSON')
    parser.add_argument('-model', '--model_name', type=str, required=True, help='name of testing model')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载测试数据集
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(      
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )])

    test_data = CrowdCountingDataset('../../shanghaitech/'+args.test_json, transform=transform)
    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=8)

    # 加载模型
    model = CSRNet().to(device)
    checkpoint_path = f'checkpoints/{args.model_name}.pth'
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['state_dict'])
    else:
        print(f"No checkpoint found at '{checkpoint_path}'")
        exit()

    # 评估模型
    mae, mse = evaluate_model(model, test_dataloader, device)
    print(f'MAE: {mae:.3f}, MSE: {mse:.3f}')