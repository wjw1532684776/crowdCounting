from model import CSRNet
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torchvision import transforms
from dataset import CrowdCountingDataset
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import os

def train_model(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for img, density in dataloader:
        img = img.to(device)
        density = density.to(device)
        
        optimizer.zero_grad()
        
        output = model(img)
        loss = criterion(output, density.unsqueeze(1))
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    epoch_loss = running_loss / len(dataloader)
    return epoch_loss

def validate_model(model, dataloader, device):
    model.eval()

    mae, mse = 0.0, 0.0
    with torch.no_grad(): 
        for img, density in dataloader:
            img = img.to(device)
            density = density.to(device)

            output = model(img)
            ground_truth_count = torch.sum(density).item()
            predicted_count = torch.sum(output).item()

            mae += abs(predicted_count - ground_truth_count)
            mse += ((predicted_count - ground_truth_count) ** 2)

    mae /= len(dataloader.dataset)
    mse /= len(dataloader.dataset)
    mse = np.sqrt(mse)
    return mae, mse

if __name__ == '__main__':
    
    # 读取命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--train_json', type=str, required=True, help='path to train JSON')
    parser.add_argument('-v', '--val_json', type=str, required=True, help='path to validation JSON')
    parser.add_argument('-p', '--model_prefix', type=str, required=True, help='prefix of model name')
    args = parser.parse_args()

    # 训练参数设置
    learning_rate = 1e-5
    batch_size = 1
    epochs = 1000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    writer = SummaryWriter(f'runs/{args.model_prefix}_{epochs}_{learning_rate}')

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(      
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )])
    
    # 加载数据集
    train_data = CrowdCountingDataset('../../shanghaitech/'+args.train_json, transform=transform)
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_data = CrowdCountingDataset('../../shanghaitech/'+args.val_json, transform=transform)
    val_dataloader = DataLoader(val_data, batch_size=1, shuffle=False)
    
    model = CSRNet().to(device)
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)
    
    # checkpoint
    checkpoint_path = 'checkpoints'
    
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    
    start_epoch = 0
    checkpoint_filename = f'{args.model_prefix}_{epochs}_{learning_rate}.pth'
    checkpoint_filepath = os.path.join(checkpoint_path, checkpoint_filename)
    
    # 检查是否有checkpoint
    if os.path.exists(checkpoint_filepath):
        checkpoint = torch.load(checkpoint_filepath, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Checkpoint loaded from {checkpoint_filepath}, starting from epoch {start_epoch}")
    
    for epoch in range(start_epoch, epochs):
        epoch_loss = train_model(model, train_dataloader, criterion, optimizer, device)
        mae, mse = validate_model(model, val_dataloader, device)

        writer.add_scalar('Loss', epoch_loss, epoch)
        writer.add_scalar('MAE', mae, epoch)
        writer.add_scalar('MSE', mse, epoch)

        print(f"Epoch[{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}")
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