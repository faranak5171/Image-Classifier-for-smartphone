import argparse
from torchvision import models
import torch
from torch import nn
import numpy as np
from PIL import Image
import json


def get_input_args():
    parse = argparse.ArgumentParser()
    parse.add_argument("path_to_image", type=str, default="flowers/test/100/image_07896.jpg", help="Path to image directory")
    parse.add_argument("checkpoint_directory", type=str, default="./SavingModel", help="Directory to load checkpoint")
    parse.add_argument("--top_k","--top_k", type=int, default=5 , help="Return top K most likely classes")
    parse.add_argument("--category_names","--category_names", type=str, default="cat_to_name.json", help="Mapping of categories to real names")
    parse.add_argument("--gpu","--gpu", type=str, default="gpu", help="Device mode")
    return parse.parse_args()


def LoadModel(chkpt_dir, device):
    model = models.vgg16(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    model.classifier = nn.Sequential(nn.Linear(25088,204),
                                nn.ReLU(),
                                nn.Dropout(p=0.2),
                                nn.Linear(204,102),
                                nn.LogSoftmax(dim=1))
    
    model.load_state_dict(torch.load(chkpt_dir + '/checkpoint.pth.tar'))
    model.to(device)
    model.eval()
    return model

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    mean_targets = [0.485, 0.456, 0.406]
    std_targets = [0.229, 0.224, 0.225]
    img = Image.open(image)
    img = img.resize((256,256))
    img = img.crop((16,16,240,240))
    img = np.array(img).astype(float)
    for i in range(3):
        img[...,i] = (img[...,i] - np.mean(img[...,i]))/ np.std(img[...,i])
        img[...,i] = (img[...,i] * std_targets[i]) + mean_targets[i]
    img = img.transpose((2,1,0))
    return torch.from_numpy(img)


def predict(image_path, model, device, topk):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    image = process_image(image_path)
    image = image.to(device, dtype=torch.float)
    with torch.no_grad():
        image = torch.unsqueeze(image,0)
        logps = model.forward(image)
        ps = torch.exp(logps)
    return ps.topk(topk, dim=1)





in_args = get_input_args()
if in_args.gpu == "gpu":
    device = "cuda" if torch.cuda.is_available() else "cpu"
else:
    device = "cpu"
model = LoadModel(in_args.checkpoint_directory, device)

probs , classes = predict(in_args.path_to_image ,model ,device, in_args.top_k)
print(f"\n\nPredicted probabilities: {probs.to('cpu').numpy().flatten()}")

with open(in_args.category_names, 'r') as f:
    cat_to_name = json.load(f)
    
class_names=[]
for item in classes.to('cpu').numpy().flatten():
    class_names.append(cat_to_name.get(str(item)))

print(f"Predicted classes: {class_names}\n\n")
