
import cv2
import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from google.colab.patches import cv2_imshow
import torch.nn as nn
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import torchvision
import pathlib

from google.colab import drive
drive.mount('/content/drive')

dataset_path_Test = '/content/drive/MyDrive/breast-cancer-dataset/Test'
dataset_path_Train='/content/drive/MyDrive/breast-cancer-dataset/Train'



images = []
for filename in os.listdir(dataset_path_Test):
    if filename.endswith('.jpg'):
        image_path = os.path.join(dataset_path_Test, filename)
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        images.append(img)

images = np.array(images)

imageTest = []
indexTest = []
for i, fo in enumerate((os.listdir(dataset_path_Test))):
    fp = os.path.join(dataset_path_Test, fo)
    for fname in os.listdir(fp):
       ip = os.path.join(fp, fname)
       image = cv2.imread(ip)
       image = cv2.resize(image, (244, 244))
       imageTest.append(image)
       indexTest.append(i)
imageTest=np.array(imageTest)
indexTest=np.array(indexTest)

#Trning images
imagest = []
for filename in os.listdir(dataset_path_Train):
    if filename.endswith('.jpg'):
        image_path = os.path.join(dataset_path_Train, filename)
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        imagest.append(img)

imagest = np.array(images)

imageTrain = []
indexTrain = []
for i, fo in enumerate((os.listdir(dataset_path_Train))):
    fp = os.path.join(dataset_path_Train, fo)
    for fname in os.listdir(fp):
       ip = os.path.join(fp, fname)
       image = cv2.imread(ip)
       image = cv2.resize(image, (244, 244))
       imageTrain.append(image)
       indexTrain.append(i)
imageTrain=np.array(imageTrain)
indexTrain=np.array(indexTrain)

cv2_imshow(im[223])
 print(index[223])

tensor_images_Test = torch.tensor(imageTest, dtype=torch.float32)
tensor_images_Test /= 255.0
tensor_images_Train = torch.tensor(imageTest, dtype=torch.float32)
tensor_images_Train /= 255.0

train_loader=DataLoader(
    torchvision.datasets.ImageFolder(dataset_path_Test,transform=tensor_images_Test),
    batch_size=64, shuffle=True
)
test_loader=DataLoader(
    torchvision.datasets.ImageFolder(dataset_path_Train,transform=tensor_images_Train),
    batch_size=32, shuffle=True
)

root=pathlib.Path(dataset_path_Train)
classes=sorted([j.name.split('/')[-1] for j in root.iterdir()])

print(classes)

class Network(nn.Module):
    def __init__(num_classes=2):

        c1=nn.Conv2d(in_channels=1,out_channels=16,kernel_size=3,stride=1,padding=1)
        bn1=nn.BatchNorm2d(num_features=16)
        relu1=nn.ReLU()
        pool1=nn.MaxPool2d(kernel_size=3)

        c2=nn.Conv2d(in_channels=16,out_channels=32,kernel_size=3,stride=1,padding=1)
        bn1=nn.BatchNorm2d(num_features=32)
        relu2=nn.ReLU()
        pool2=nn.MaxPool2d(kernel_size=3)

        c3=nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,stride=1,padding=1)
        bn1=nn.BatchNorm2d(num_features=64)
        relu2=nn.ReLU()
        pool3=nn.AvgPool2d(kernel_size=3)

        fc=nn.Linear(in_features=64*9*9,out_features=num_classes)





