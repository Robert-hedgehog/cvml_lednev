import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from train_model import CyrrilicCNN, CustomImageDataset, test_transform

model_path = Path(__file__).parent / "CyrrilicCNN.pth"
model = CyrrilicCNN()
model.load_state_dict(torch.load(model_path, weights_only=True))
model.eval()

dataset_path = "./Cyrillic" 
full_dataset = CustomImageDataset(dataset_path)
indices = list(range(len(full_dataset)))

_, test_idx = train_test_split(
    indices, test_size=0.2, random_state=42, stratify=full_dataset.labels
)
test_data = Subset(CustomImageDataset(dataset_path, transform=test_transform), test_idx)
class_names = sorted(full_dataset.classes)

shuffled_test_indices = np.arange(len(test_data))
np.random.shuffle(shuffled_test_indices)

samples_to_show = {}
for i in shuffled_test_indices:
    _, label = test_data[i]
    if label not in samples_to_show:
        samples_to_show[label] = i
    if len(samples_to_show) == 34:
        break

rows, cols = 5, 7
fig, axes = plt.subplots(rows, cols, figsize=(22, 16)) 
axes = axes.flatten()

with torch.no_grad():
    for i in range(rows * cols):
        if i in samples_to_show:
            idx_in_test = samples_to_show[i]
            image, label = test_data[idx_in_test]
            
            output = model(image.unsqueeze(0))
            probabilities = torch.softmax(output, dim=1)
            prob_val, pred_idx = torch.max(probabilities, dim=1)
            
            prob_percent = prob_val.item() * 100
            prediction = pred_idx.item()

            img_np = image.squeeze().numpy() * 0.5 + 0.5
            
            axes[i].imshow(img_np, cmap='gray')
            is_correct = (prediction == label)
            color = 'green' if is_correct else 'red'

            title_text = (f"R: {class_names[label]}\n"
                          f"P: {class_names[prediction]}\n"
                          f"{prob_percent:.1f}%")
            
            axes[i].set_title(title_text, color=color, fontsize=11, fontweight='bold')
        
        axes[i].axis('off')

plt.tight_layout()

output_image = Path(__file__).parent / "pic.png"
plt.savefig(output_image, dpi=150, bbox_inches='tight')

plt.show()