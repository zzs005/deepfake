import os


import torch
import torch.nn as nn
from torchvision import transforms
from torchvision import datasets 
from torch.utils.data import DataLoader ,  Subset
import torch.optim as optim
from sklearn.model_selection import train_test_split


class deepfake_model (nn.Module):
    def __init__(self):
        super(deepfake_model,self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3,32, kernel_size= 3, stride= 1 , padding = 1 ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32,32, kernel_size = 3,stride = 1,padding =1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size = 2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(32,64,kernel_size = 3,stride= 1,padding = 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace = True),
            nn.Conv2d(64,64, kernel_size = 3,stride = 1,padding = 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size =2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(64,128, kernel_size= 3, stride= 1 , padding = 1 ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128,128, kernel_size = 3,stride = 1,padding =1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size = 2)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(128,256, kernel_size= 3, stride= 1 , padding = 1 ),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256,256, kernel_size = 3,stride = 1,padding =1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size = 2)
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(256,512, kernel_size= 3, stride= 1 , padding = 1 ),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size = 2)
        )
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.gap(x)
        x = self.classifier(x)
        return x



script_dir = os.path.dirname(os.path.abspath(__file__))
test_path = os.path.join(os.path.dirname(script_dir), "deep_face", "test")


test_transform = transforms.Compose(
    [
        transforms.Resize([256,256]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ]
)

test_data = datasets.ImageFolder(root = test_path,transform = test_transform)

test_loader = DataLoader(test_data, batch_size=32, shuffle=False)
