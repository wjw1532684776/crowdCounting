from model import MCNN
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torchvision import transforms
from dataset import CrowdCountingDataset
import os

if __name__ == '__main__':
    
    # 训练参数设置
    learning_rate = 1e-4
    batch_size = 16
    epochs = 30
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    
    # 加载数据集
    train_data = CrowdCountingDataset('../../shanghaitech/part_A_train.json', transform=transform)
    dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    
    model = MCNN().to(device)
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)
    
    # checkpoint
    checkpoint_path = 'checkpoints'
    
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    
    start_epoch = 0
    checkpoint_filename = 'checkpoint.pth'
    checkpoint_filepath = os.path.join(checkpoint_path, checkpoint_filename)
    
    # 检查是否有checkpoint
    if os.path.exists(checkpoint_filepath):
        checkpoint = torch.load(checkpoint_filepath, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Checkpoint loaded from {checkpoint_filepath}, starting from epoch {start_epoch}")
    
    for epoch in range(start_epoch, epochs):
        model.train()
        running_loss = 0.0
        for img, density in dataloader:
            img = img.to(device)
            density = density.to(device)
            
            # forward
            output = model(img)
            
            # loss
            loss = criterion(output, density.unsqueeze(1))  # 给密度图添加一个通道
            
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        epoch_loss = running_loss / len(dataloader)
        print(f"Epoch[{epoch+1}/{epochs}], Loss:{epoch_loss:.4f}")
        
        # save checkpoint
        checkpoint = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'loss': epoch_loss,
        }
        
        torch.save(checkpoint, checkpoint_filepath)
        print(f"Checkpoint saved to {checkpoint_filepath}")    

    print("finished training")