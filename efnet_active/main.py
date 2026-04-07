import cv2
import numpy as np
import torch
import torchvision
from torchvision import transforms
from PIL import Image
import time
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
    model = torchvision.models.efficientnet_b0(weights=None)
    features = model.classifier[1].in_features 
    model.classifier[1] = nn.Linear(features, 1)

    model_path = Path(__file__).parent / "model_efficient.pth" # or "model_efficient_5.pth"

    if model_path.exists():
        print("Найдены сохраненные веса")
        model.load_state_dict(torch.load(model_path))
    else:
        print("Сохраненные веса не найдены")

    return model.to(device)

def build_model_alex():
    model = torchvision.models.alexnet(weights=None)
    features = model.classifier[6].in_features 
    model.classifier[6] = nn.Linear(features, 1)

    model_path = Path(__file__).parent / "model_alex.pth" # or "model_alex_5.pth"

    if model_path.exists():
        print("Найдены сохраненные веса")
        model.load_state_dict(torch.load(model_path))
    else:
        print("Сохраненные веса не найдены")

    return model.to(device)


model = build_model()
model.eval() 
model_alex = build_model_alex()
model_alex.eval() 

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256), interpolation=InterpolationMode.BICUBIC),
    transforms.CenterCrop((224, 224)), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_alex = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict(frame, target_transform, target_model):
    tensor = target_transform(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    tensor = tensor.unsqueeze(0).to(device)
    
    with torch.no_grad(): 
        predicted = target_model(tensor).squeeze()
        
        if predicted.dim() == 0:
            predicted = predicted.unsqueeze(0)
            
        prob = torch.sigmoid(predicted).item()
        
    label = "person" if prob > 0.5 else "no_person"
    return label, prob

cap = cv2.VideoCapture(0)
cv2.namedWindow("Camera", cv2.WINDOW_GUI_NORMAL)

while True:
    ret, frame = cap.read()
    if not ret: break

    label, confidence = predict(frame, transform, model)
    label_alex, confidence_alex = predict(frame, transform_alex, model_alex)

    color = (0, 255, 0) if label == "person" else (0, 0, 255)
    color_alex = (0, 255, 0) if label_alex == "person" else (0, 0, 255)
    cv2.putText(frame, f"efficient: {label}: {confidence:.2f}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.putText(frame, f"alex: {label_alex}: {confidence_alex:.2f}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, color_alex, 2)
    cv2.imshow("Camera", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()