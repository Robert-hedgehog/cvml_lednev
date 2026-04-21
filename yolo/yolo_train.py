from ultralytics import YOLO
from pathlib import Path
import yaml
import torch

classes = {0: "cube", 1: "neither", 2: "sphere"}

root = Path("spheres_and_cubes_new")

config = {
    "path": str(root.absolute()),
    "train": str((root / "images" / "train").absolute()),
    "val": str((root / "images" / "val").absolute()), 
    "nc": len(classes),
    "names": classes
}

with open(root / "dataset.yaml", "w") as f:
    yaml.dump(config, f, allow_unicode=True)

size = "n"
model = YOLO(f"yolo26{size}.pt")

result = model.train(
    data = str(root / "dataset.yaml"),
    imgsz = 640,
    batch = 16,
    workers = 4,

    epochs = 15,
    patience = 5,
    optimizer = "AdamW",
    lr0 = 0.01,
    warmup_epochs = 3,
    cos_lr = True,

    dropout = 0.2,

    hsv_h = 0.015,
    hsv_s = 0.7,
    hsv_v = 0.4,
    flipud = 0.0,
    fliplr = 0.5, 
    mosaic = 1.0,
    degrees = 5.0,
    scale = 0.5, 
    translate = 0.1,

    conf = 0.001, 
    iou = 0.7, 

    project = "figures",
    name = "yolo", 
    save = True, 
    save_period = 5,
    device = "mps", # 0 if torch.mps.is_available() else "cpu"

    verbose = True, 
    plots = True, 
    val = True, 
    close_mosaic = 8, 
    amp = True # FP16
)

print("Done")
print(result.save_dir)