import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torch import nn
import torch.optim as optim
import random
import string

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Используется устройство: CUDA")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Используется устройство: MPS (Apple Silicon)")
else:
    device = torch.device("cpu")
    print("Используется устройство: CPU")

class ImageDataset(Dataset):

    def __init__(self, n=200, size=128, option = 1):
        super().__init__()
        self.n = n
        self.size = size
        self.option = option
        self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return self.n
    
    def generate_random_text(self, length = 3):
        return ''.join(random.choice(string.ascii_uppercase) for _ in range(length))
    
    def __getitem__(self, idx):
        image = Image.new('L', (self.size, self.size), color = 255)
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()

        if self.option == 1:
            text = "ABC"
            x = np.random.randint(10, self.size - 30)
            y = np.random.randint(10, self.size - 20)
        
        elif self.option == 2:
            text = self.generate_random_text()
            x = 30
            y = 30

        elif self.option == 3:
            text = self.generate_random_text(np.random.randint(2, 12))
            x = 30
            y = 30

        elif self.option == 4:
            text = self.generate_random_text(np.random.randint(2, 12))
            x = np.random.randint(10, self.size - 30)
            y = np.random.randint(10, self.size - 20)

        draw.text((x, y), text, fill=0, font=font)
        tensor = self.transform(image)
        return tensor, tensor
    
ds = ImageDataset(2000, 256)

plt.imshow(ds[0][0][0])
plt.show()

class Encoder(nn.Module):
    def __init__(self, latent = 512):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, stride=2, kernel_size=4, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, stride=2, kernel_size=4, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, stride=2, kernel_size=4, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 256, stride=2, kernel_size=4, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        self.bottleneck = nn.Linear(256 * 16 * 16, latent)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.bottleneck(x)
        return x
    
class Decoder(nn.Module):

    def __init__(self, latent = 512):
        super().__init__()
        self.bottleneck = nn.Linear(latent, 256 * 16 * 16)
        self.features = nn.Sequential(
            nn.ConvTranspose2d(256, 128, stride = 2, kernel_size=4, padding = 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.ConvTranspose2d(128, 64, stride = 2, kernel_size=4, padding = 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 32, stride = 2, kernel_size=4, padding = 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.ConvTranspose2d(32, 1, stride = 2, kernel_size=4, padding = 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.bottleneck(x)
        x = x.view(x.size(0), 256, 16, 16)
        x = self.features(x)
        return x
    
if __name__ == "__main__":
    for option in range(1, 5):
        encoder = Encoder()
        decoder = Decoder()

        dataset = ImageDataset(2000, 256, option=option)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

        encoder.to(device)
        decoder.to(device)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()))

        encoder.train()
        decoder.train()

        epochs = 10
        for epoch in range(epochs):
            epoch_loss = 0.0
            for imgs, _ in dataloader:
                imgs = imgs.to(device)
                optimizer.zero_grad()
                latent = encoder(imgs)
                output = decoder(latent)
                loss = criterion(imgs, output)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            avg_loss = epoch_loss / len(dataloader)
            print(f"{epoch=}, {avg_loss=:.2f}")

        torch.save(encoder.state_dict(), f"encoder{option}.pth")
        torch.save(decoder.state_dict(), f"decoder{option}.pth")