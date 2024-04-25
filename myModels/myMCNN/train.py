import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from dataset import CrowdCountingDataset
from model import MCNN
from utils import evaluate_model, train_model

if __name__ == '__main__':
    # 读取命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument('-train', '--train_json', type=str, required=True, help='path to train JSON')
    parser.add_argument('-test', '--test_json', type=str, required=True, help='path ot test JSON')
    parser.add_argument('-data', '--dataset_name', type=str, required=True, help='name of the using dataset')
    args = parser.parse_args()

    # 训练参数设置
    learning_rate = 1e-5
    batch_size = 16
    epochs = 1000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    writer = SummaryWriter(f'runs/MCNN_{args.dataset_name}_{epochs}epoch_{learning_rate}lr_{batch_size}bs')
    
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(      
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )])
    
    # 加载数据集
    train_data = CrowdCountingDataset('../../shanghaitech/'+ args.train_json, transform=transform)
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=8)
    test_data = CrowdCountingDataset('../../shanghaitech/'+ args.test_json, transform=transform)
    test_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=8)
    
    model = MCNN().to(device)
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)
    
    # checkpoint
    checkpoint_path = 'checkpoints'
    
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    
    start_epoch = 0
    checkpoint_filename = f'MCNN_{args.dataset_name}_{epochs}epoch_{learning_rate}lr_{batch_size}bs.pth'    # model_name i.e. 'MCNN_ShanghaiTechA_1000epoch_1e-05lr_16bs.pth'
    checkpoint_filepath = os.path.join(checkpoint_path, checkpoint_filename)
    
    # 检查是否有checkpoint
    if os.path.exists(checkpoint_filepath):
        checkpoint = torch.load(checkpoint_filepath)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Checkpoint loaded from {checkpoint_filepath}, starting from epoch {start_epoch}")
    
    for epoch in range(start_epoch, epochs):

        epoch_loss = 0.0
        mae, mse = 0.0, 0.0

        # 训练
        model.train()
        epoch_loss = train_model(model, train_dataloader, criterion, optimizer, device)
        writer.add_scalar('Loss', epoch_loss, epoch)
        print(f"Epoch[{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}")

        # 每隔10个epoch在测试集上评估
        if (epoch + 1) % 10 == 0:
            mae, mse = evaluate_model(model, test_dataloader, device)
            writer.add_scalar('MAE', mae, epoch)
            writer.add_scalar('MSE', mse, epoch)
            print(f'MAE: {mae:.3f}, MSE: {mse:.3f}')
        
        # save checkpoint
        checkpoint = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        
        torch.save(checkpoint, checkpoint_filepath)
        print(f"Checkpoint saved to {checkpoint_filepath}")    

    writer.close()        
    print("finished training")