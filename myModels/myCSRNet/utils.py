import torch
import numpy as np

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

def evaluate_model(model, dataloader, device):
    model.eval()

    mae, mse = 0.0, 0.0
    with torch.no_grad():
        for img, density in dataloader:
            img = img.to(device)
            density = density.to(device)

            output = model(img)
            ground_truth_count = torch.sum(density).item()
            predicted_count = torch.sum(output).item()
            # print(f'真实人数:{ground_truth_count}, 预测人数:{predicted_count}')

            mae += abs(predicted_count - ground_truth_count)
            mse += ((predicted_count - ground_truth_count) ** 2)

    mae /= len(dataloader.dataset)
    mse /= len(dataloader.dataset)
    mse = np.sqrt(mse)  # 为了计算RMSE

    return mae, mse