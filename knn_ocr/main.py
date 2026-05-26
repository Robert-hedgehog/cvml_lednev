import cv2 
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from skimage.measure import regionprops, label
from skimage.io import imread

test_path = Path("./task")
train_path = test_path / "train"


def extractor(image):
    if image.ndim == 2:
        gray = image
        binary = gray > 0
    else:
        gray = np.mean(image, 2).astype("u1")
        binary = gray > 0
    lb = label(binary)
    props = regionprops(lb)
    features = None

    filtered_props = [p for p in props if p.extent <= 0.81]

    props1 = filtered_props if filtered_props else props
    
    features = [
        props1[0].eccentricity,
        props1[0].solidity,
        props1[0].extent,
        props1[0].euler_number,
        *props1[0].moments_hu
    ]                                                                                                                            
    return np.array(features,  dtype = "f4")

def make_train(path):
    train = []
    responses = []
    letters = []
    ncls = -1

    for cls in sorted(path.glob("*")):
        ncls += 1
        letters.append(str(cls)[-1])
        for p in sorted(cls.glob("*.png")):
            train.append(extractor(imread(p)))
            responses.append(ncls)
    train = np.array(train, dtype = "f4").reshape(-1, 11)
    responses = np.array(responses, dtype = 'f4').reshape(-1, 1)
    return train, responses, letters

train, responses, letters = make_train(train_path)
knn = cv2.ml.KNearest.create()
knn.train(train, cv2.ml.ROW_SAMPLE, responses)

for k in range(7):

    image = imread(test_path / f"{k}.png")

    gray = np.mean(image, 2).astype("u1")
    binary = gray > 0
    lb = label(binary.T)
    props = regionprops(lb)

    find = []

    for i, prop in enumerate(props):
        if props[i].extent < 0.7:
            find.append(extractor(props[i].image))
    find = np.array(find, dtype = "f4").reshape(-1,11)

    ret, result, neighbours, dist = knn.findNearest(find,  3)

    for res in result:
        print(letters[int(res.item())], end="")

    print("\n")