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
import torch.optim as optim

from google.colab import drive
drive.mount('/content/drive')

dataset_path_Test = '/content/drive/MyDrive/breast-cancer-dataset/Test'
dataset_path_Train='/content/drive/MyDrive/breast-cancer-dataset/Train'

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

tensor_images_Test = torch.tensor(imageTest, dtype=torch.float32)
tensor_images_Train = torch.tensor(imageTrain, dtype=torch.float32)

batch_norm_test = nn.BatchNorm2d(num_features=244)
batch_norm_train = nn.BatchNorm2d(num_features=244)


tensor_images_Test_normalized = batch_norm_test(tensor_images_Test)

tensor_images_Train_normalized = batch_norm_train(tensor_images_Train)


transform = transforms.Compose([
    transforms.ToTensor(),
])

train_loader = DataLoader(
    dataset=datasets.ImageFolder(root=dataset_path_Train, transform=transform),
    batch_size=64,
    shuffle=True
)

test_loader = DataLoader(
    dataset=datasets.ImageFolder(root=dataset_path_Test, transform=transform),
    batch_size=32,
    shuffle=True
)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3,stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(2, 2),

            nn.Flatten(),
            nn.Linear(64*70, 256),
            nn.ReLU(),
            nn.Linear(256, 2),
             nn.ReLU(),

            )

    def forward(self, xb):
        return self.network(xb)

model = Model()
print(model)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

start.record()

for epoch in range(2):  

    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
       
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 2000 == 1999:    
            running_loss = 0.0


end.record()
torch.cuda.synchronize()

print(start.elapsed_time(end))


dataiter = iter(test_loader)
images, labels = next(dataiter)


print('GroundTruth: ', ['Benign' if label.item() == 0 else 'Malignant' for label in labels])

outputs = model(images)

_, predicted = torch.max(outputs, 1)

print('Predicted: ', ['Benign' if label.item() == 0 else 'Malignant' for label in predicted])


correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print('Accuracy is: %d %%' % (100 * correct / total))