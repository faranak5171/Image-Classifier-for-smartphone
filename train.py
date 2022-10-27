import argparse
import torch
from torchvision import transforms, datasets, models
from torch import nn, optim
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as plimg
import seaborn as sns
import json

def get_input_args():
    parse = argparse.ArgumentParser()
    parse.add_argument("data_directory", type=str, default="flowers", help="Data Directory")
    parse.add_argument("--save_dir","--save_directory", type=str, default="./SavingModel", help="Directory to save checkpoint")
    parse.add_argument("--arch","--architecture", type=str, default="vgg16", help="architecture")
    parse.add_argument("--learning_rate","--learning_rate", type=float, default=0.01, help="Learning rate")
    parse.add_argument("--hidden_units","--hidden_units", type=int, default=204, help="hidden units")
    parse.add_argument("--epochs","--epochs", type=int, default=5, help="epochs")
    parse.add_argument("--gpu","--gpu", type=str, default="gpu", help="CUDA or CPU") 
    return parse.parse_args()


def TrainingModel():
    data_dir = in_args.data_directory
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    transform = transforms.Compose([transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    train_dataset = datasets.ImageFolder(train_dir, transform=transform)
    valid_dataset = datasets.ImageFolder(valid_dir, transform=transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=transform)

    trainloaders = torch.utils.data.DataLoader(train_dataset, batch_size=12, shuffle=True)
    validloaders = torch.utils.data.DataLoader(valid_dataset, batch_size=12, shuffle=True)
    testloaders = torch.utils.data.DataLoader(test_dataset, batch_size=12, shuffle=True)

    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)

    if in_args.arch == "vgg16":
        model = models.vgg16(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    
    
    hidden_units = in_args.hidden_units
    model.classifier = nn.Sequential(nn.Linear(25088,hidden_units),
                                nn.ReLU(),
                                nn.Dropout(p=0.2),
                                nn.Linear(hidden_units,102),
                                nn.LogSoftmax(dim=1))

    criterion = nn.NLLLoss()
    optimizer = optim.SGD(model.classifier.parameters(), lr=in_args.learning_rate)
    if in_args.gpu == "gpu":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = "cpu"
    model.to(device)
    
    epochs = in_args.epochs
    for epoch in range(epochs):
        model.train()
        running_loss = 0
        print("Start training the model")
        for data, labels in trainloaders:
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()
            logps = model.forward(data)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()        
            running_loss += loss.item()
    
        val_loss = 0
        accuracy = 0
        model.eval()
        print("Start Validation the model")
        for data, labels in validloaders:
            data, labels = data.to(device), labels.to(device)
            logps = model.forward(data)
            loss = criterion(logps, labels)
            val_loss += loss.item()
        
            ps = torch.exp(logps)
            top_k, top_class = logps.topk(1,dim=1)
            equal = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equal.type(torch.FloatTensor)).item()
    
        print(f"epochs: {epoch + 1}/{epochs}.. Average Train Loss:{running_loss/len(trainloaders):.3f}.."
            f"Average validation Loss:{val_loss/len(validloaders):.3f}.. Accuracy: {accuracy/len(validloaders):.3f}..")
        
        print("Start testing the model")
        accuracy = 0 
        for data, labels in testloaders:
            data, labels = data.to(device), labels.to(device)
            logps = model.forward(data)
            ps = torch.exp(logps)
            top_k, top_class = ps.topk(1, dim=1)
            equal = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equal.type(torch.FloatTensor)).item()

        print(f"Test Accuracy: {accuracy/len(testloaders):.3f}..")
        
        torch.save(model.state_dict(), in_args.save_dir + '/checkpoint.pth.tar')

in_args = get_input_args()
if len(os.listdir('./SavingModel')) >=1:
    os.remove('./SavingModel/checkpoint.pth.tar')
TrainingModel()