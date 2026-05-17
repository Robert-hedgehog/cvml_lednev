import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from unet_road import UNet, RoadsDataset

def main():
    path = Path("roads")
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Используется устройство: CUDA")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Используется устройство: MPS (Apple Silicon)")
    else:
        device = torch.device("cpu")
        print("Используется устройство: CPU")

    ds = RoadsDataset(path)

    indices = list(range(len(ds)))
    _, test_idx = train_test_split(indices, test_size=0.2)
    test_data = Subset(ds, test_idx)
    
    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False)

    model = UNet()
    model.to(device)
    model_path = Path("unet.pth")
    
    if model_path.exists():
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Модель успешно загружена!")
    else:
        print("Файл unet.pth не найден!")
        return

    model.eval()

    for images, masks in test_dataloader:
        images = images.to(device)
        
        with torch.no_grad():
            pred_probs = torch.sigmoid(model(images))
            pred_mask = (pred_probs > 0.5).float()

        img = images[0].cpu().permute(1, 2, 0).numpy()
        mask = masks[0, 0].cpu().numpy()
        pred = pred_mask[0, 0].cpu().numpy()
        difference = np.abs(mask - pred)

        fig, axes = plt.subplots(1, 4, figsize=(18, 5))
        
        axes[0].imshow(img)
        axes[0].set_title("Исходное изображение")
        
        axes[1].imshow(mask, cmap='gray')
        axes[1].set_title("Истинная маска")
        
        axes[2].imshow(pred, cmap='gray')
        axes[2].set_title("Предсказанная маска")
        
        axes[3].imshow(difference, cmap='hot')
        axes[3].set_title("Разница")
        
        for ax in axes: ax.axis('off')
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()