from model import CSRNet
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torchvision import transforms
from dataset import CrowdCountingDataset
import os

def train_model(model, dataloader, criterion, optimizer, device, epoch, epochs):
    model.train()
    running_loss = 0.0
    for img, density in dataloader:
        img = img.to(device)
        density = density.to(device)
        
        optimizer.zero_grad()
        
        output = model(img)
        loss = criterion(output, density.unsqueeze(1))  # 添加一个通道维度
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    epoch_loss = running_loss / len(dataloader)
    print(f"Epoch[{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}")

def validate_model(model, dataloader, criterion, device, epoch, epochs):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for img, density in dataloader:
            img = img.to(device)
            density = density.to(device)
            
            output = model(img)
            loss = criterion(output, density.unsqueeze(1)).item()
            
            total_loss += loss
            
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch[{epoch+1}/{epochs}], Validation Loss:{avg_loss:.4f}")
    return avg_loss

if __name__ == '__main__':
    
    # 训练参数设置
    learning_rate = 1e-4
    batch_size = 16
    epochs = 30
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(      
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )])
    
    # 加载数据集
    train_data = CrowdCountingDataset('../../shanghaitech/part_A_train.json', transform=transform)
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_data = CrowdCountingDataset('../../shanghaitech/part_A_val.json', transform=transform)
    val_dataloader = DataLoader(val_data, batch_size=1, shuffle=False)
    
    model = CSRNet().to(device)
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)
    
    # checkpoint
    checkpoint_path = 'checkpoints'
    
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    
    start_epoch = 0
    checkpoint_filename = 'checkpointA.pth'
    checkpoint_filepath = os.path.join(checkpoint_path, checkpoint_filename)
    
    # 检查是否有checkpoint
    if os.path.exists(checkpoint_filepath):
        checkpoint = torch.load(checkpoint_filepath, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Checkpoint loaded from {checkpoint_filepath}, starting from epoch {start_epoch}")
    
    for epoch in range(start_epoch, epochs):
        train_model(model, train_dataloader, criterion, optimizer, device, epoch, epochs)
        validate_model(model, val_dataloader, criterion, device, epoch, epochs)

        # save checkpoint
        checkpoint = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        
        torch.save(checkpoint, checkpoint_filepath)
        print(f"Checkpoint saved to {checkpoint_filepath}")    

    print("finished training")