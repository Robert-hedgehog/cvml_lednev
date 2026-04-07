import cv2
import numpy as np
import torch
import torchvision
from torchvision import transforms
from PIL import Image
import time
from collections import deque
import torch.nn as nn
from pathlib import Path
from torchvision.transforms import InterpolationMode

"""if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Используется устройство: MPS (Apple Silicon)")"""
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Используется устройство: CUDA")
else:
    device = torch.device("cpu")
    print("Используется устройство: CPU")

print(f"{device=}")

def build_model():
    weights = torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1
    model = torchvision.models.efficientnet_b0(weights)
    for param in model.features.parameters():
        param.requires_grad = False

    features = model.classifier[1].in_features 
    model.classifier[1] = nn.Linear(features, 1)

    model_path = Path(__file__).parent / "model_efficient.pth"

    if model_path.exists():
        print("Найдены сохраненные веса")
        model.load_state_dict(torch.load(model_path))

    return model.to(device)

def build_model_alex():
    weights = torchvision.models.AlexNet_Weights.IMAGENET1K_V1
    model = torchvision.models.alexnet(weights)
    for param in model.features.parameters():
        param.requires_grad = False

    features = model.classifier[6].in_features 
    model.classifier[6] = nn.Linear(features, 1)

    model_path = Path(__file__).parent / "model_alex.pth"

    if model_path.exists():
        print("Найдены сохраненные веса")
        model.load_state_dict(torch.load(model_path))

    return model.to(device)

criterion = nn.BCEWithLogitsLoss()

model = build_model()

optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr = 0.0001
)

model_alex = build_model_alex()

optimizer_alex = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model_alex.parameters()),
    lr = 0.0001
)

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
    # transforms.RandomCrop((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_alex = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def train(buffer):
    if len(buffer) < 10:
        return None
    
    model.train()
    images, labels = buffer.get_batch()
    optimizer.zero_grad()
    predictions = model(images).squeeze(1)
    loss = criterion(predictions, labels)
    loss.backward()
    optimizer.step()

    return loss.item()

def train(target_model, target_optimizer, target_buffer):
    if len(target_buffer) < 10:
        return None
    
    target_model.train()
    images, labels = target_buffer.get_batch()
    
    target_optimizer.zero_grad()
    predictions = target_model(images).squeeze(1)
    loss = criterion(predictions, labels)
    loss.backward()
    target_optimizer.step()

    return loss.item()

class Buffer():

    def __init__(self, maxsize=16):
        self.frames = deque(maxlen=maxsize)
        self.labels = deque(maxlen=maxsize)

    def append(self, tensor, label):
        self.frames.append(tensor)
        self.labels.append(label)
    
    def __len__(self):
        return len(self.frames)
    
    def get_batch(self):
        images = torch.stack(list(self.frames)).to(device)
        labels = torch.tensor(list(self.labels), dtype=torch.float32).to(device)
        return images, labels
    

cap = cv2.VideoCapture(0)
cv2.namedWindow("Camera", cv2.WINDOW_GUI_NORMAL)
buffer = Buffer()
buffer_alex = Buffer()

count_labeled = 0

while True:
    _, frame = cap.read()
    cv2.imshow("Camera", frame)
    key = cv2.waitKey(1) & 0xFF
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    if key == ord("q"):
        break

    elif key == ord("1"): # person
        tensor = transform(image)
        tensor_alex = transform_alex(image)
        buffer.append(tensor, 1.0)
        buffer_alex.append(tensor_alex, 1.0)
        count_labeled += 1
        print(count_labeled)

    elif key == ord("2"): # no person
        tensor = transform(image)
        tensor_alex = transform_alex(image)
        buffer.append(tensor, 0.0)
        buffer_alex.append(tensor_alex, 0.0)
        count_labeled += 1
        print(count_labeled)
        
    elif key == ord("s"): # save model
        save_path = Path(__file__).parent
        torch.save(model.state_dict(), save_path / "model_efficient.pth")
        torch.save(model_alex.state_dict(), save_path / "model_alex.pth")

    if count_labeled >= buffer.frames.maxlen:
        loss = train(model, optimizer, buffer)
        loss_alex = train(model_alex, optimizer_alex, buffer_alex)
        if loss:
            print(f"Loss = {loss}")
        if loss_alex:
            print(f"Loss = {loss_alex}")
        count_labeled = 0