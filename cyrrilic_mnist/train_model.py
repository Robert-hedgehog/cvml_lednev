import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from pathlib import Path
import torch.optim as optim
from torch.utils.data import Dataset
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torchvision.transforms import v2 as transforms
import time

if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Используется устройство: MPS (Apple Silicon)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Используется устройство: CUDA")
else:
    device = torch.device("cpu")
    print("Используется устройство: CPU")

print(f"{device=}")

save_path = Path(__file__).parent

torch.manual_seed(42)

class CustomImageDataset(Dataset):
    
    def __init__(self, img_dir, transform=None):

        self.path = Path(img_dir)
        self.transform = transform

        self.img_paths = []
        self.labels = []
        self.classes = []

        for folder in self.path.iterdir():
            if folder.is_dir():
                self.classes.append(folder.name)
        
        for idx, cls_name in enumerate(sorted(self.classes)):
            cls_dir = self.path / cls_name

            for img_paths in cls_dir.glob("*.png"):
                self.img_paths.append(img_paths)
                self.labels.append(idx)

    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, index):

        path = self.img_paths[index]
        label = self.labels[index]
        
        image = Image.open(path).convert('RGBA')
        background = Image.new("RGBA", image.size, (255, 255, 255))
        image = Image.alpha_composite(background, image).convert('L')

        if self.transform:
            image = self.transform(image)

        return image, label

class CyrrilicCNN(nn.Module):

    def __init__(self, num_classes=34):
        super(CyrrilicCNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2, 2) 

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.relu4 = nn.ReLU()
        self.dropout = nn.Dropout(0.4)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.pool3(x)
    
        x = self.flatten(x)

        x = self.fc1(x)
        x = self.relu4(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

train_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomRotation(degrees=15, fill=255),
    transforms.RandomAffine(degrees=5, translate=(0.1, 0.1), scale=(0.8, 1.0), shear=10, fill=255),
    transforms.ToImage(), 
    transforms.ToDtype(torch.float32, scale=True),
    transforms.Normalize((0.5, ), (0.5, ))
])

test_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToImage(), 
    transforms.ToDtype(torch.float32, scale=True),
    transforms.Normalize((0.5, ), (0.5, ))
])

if __name__ == "__main__":

    full_dataset = CustomImageDataset("./Cyrillic")
    indices = list(range(len(full_dataset)))
    train_idx, test_idx = train_test_split(
        indices, test_size=0.2, random_state=42, stratify=full_dataset.labels
    )

    train_data = Subset(CustomImageDataset("./Cyrillic", transform=train_transform), train_idx)
    test_data = Subset(CustomImageDataset("./Cyrillic", transform=test_transform), test_idx)

    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

    model = CyrrilicCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    num_epochs = 20
    train_loss = []
    train_acc = []

    model_path = save_path / "CyrrilicCNN.pth"

    if not model_path.exists():
        t = time.perf_counter()
        
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            
            for data, target in train_loader:
                data = data.to(device)
                target = target.to(device)

                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                
            scheduler.step()
            avg_train_loss = running_loss / len(train_loader)

            model.eval()
            correct = 0
            with torch.no_grad():
                for data, target in test_loader:
                    data = data.to(device)
                    target = target.to(device)
                    output = model(data)
                    pred = output.argmax(dim=1)
                    correct += (pred == target).sum().item()
            acc = 100.0 * correct / len(test_data)
            
            train_loss.append(avg_train_loss)
            train_acc.append(acc)
            
            print(f"Epoch {epoch+1}/{num_epochs} | Loss: {avg_train_loss:.4f} | Test Acc: {acc:.2f}%")
            
        torch.save(model.state_dict(), model_path)
        print(f"Elapsed time {time.perf_counter() - t}")
        
        plt.figure()
        plt.subplot(121)
        plt.title("Loss")
        plt.plot(train_loss)
        plt.subplot(122)
        plt.title("Acc")
        plt.plot(train_acc)
        plot_path = save_path / "training_results.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight') 
        plt.show()
        
    else:
        model.load_state_dict(torch.load(model_path))