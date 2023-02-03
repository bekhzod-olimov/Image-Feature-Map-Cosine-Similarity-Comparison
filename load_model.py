import os, torch, cv2, time, timm
from glob import glob
from PIL import Image
import pandas as pd
import numpy as np
import torchvision.transforms as T
from imagenet_classes import get_classes

def load_model(model_name, num_classes):

    assert model_name in list(timm.list_models()), "Please choose the correct version of a model (refer to list(timm.list_models()))"
    model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
    print(f"{model_name} model is successfully loaded!")
    
    return model, model.pretrained_cfg["input_size"][1:] # tuple()

def apply_transformations(im_path, im_size):
    
    im = Image.open(im_path)
    tfs = T.Compose([T.ToTensor(), T.Resize(im_size), T.RandomHorizontalFlip(),
                     T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                     ])
    
    return tfs(im)

def predict(model, im, device):
    
    model.to(device)
    model.eval()
    pred = model(im.unsqueeze(0).to(device))
    
    values, indices = torch.topk(pred, k=15)

    return values.squeeze(), indices.squeeze()
    
classes = get_classes()
device = "cuda:3"
model, inp_size = load_model("rexnet_150", 1000)
im = apply_transformations("kuvasz.jpg", inp_size)
values, indices = predict(model, im, device)

for i, value in enumerate(values):
    # print(i)
    # print(value)
    # print(indices[i])
    print(f"{classes[indices[i].item()]} is predicted with {value.item()} probability!")

# print(model.pretrained_cfg["input_size"])
# model.eval()
# a = torch.rand(1,3,230,300)
# print(model(a).shape)

