import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import os
from dataset import CrowdCountingDataset
from model import CSRNet

def evaluate(model, dataloader, device):
    model.eval()  # 设置模型为评估模式

    mae, mse = 0.0, 0.0
    with torch.no_grad():  # 在评估过程中不计算梯度
        for img, density in dataloader:
            img = img.to(device)
            density = density.to(device)

            output = model(img)
            ground_truth_count = torch.sum(density).item()  # 真实人数
            predicted_count = torch.sum(output).item()  # 预测人数

            print(f'真实人数：{ground_truth_count}, 预测人数: {predicted_count}')

            mae += abs(predicted_count - ground_truth_count)
            mse += ((predicted_count - ground_truth_count) ** 2)

    mae /= len(dataloader.dataset)
    mse /= len(dataloader.dataset)
    mse = np.sqrt(mse)  # 为了计算RMSE

    return mae, mse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--test_json', type=str, required=True, help='path to test JSON')
    parser.add_argument('-n', '--model_name', type=str, required=True, help='name of testing model')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载测试数据集
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(      
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )])

    test_data = CrowdCountingDataset('../../shanghaitech/'+args.test_json, transform=transform)
    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False)

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
    mae, mse = evaluate(model, test_dataloader, device)
    print(f'MAE: {mae:.3f}, MSE: {mse:.3f}')