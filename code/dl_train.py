import os

import tqdm
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision import datasets 
from torch.utils.data import DataLoader ,  Subset
import torch.optim as optim
from sklearn.model_selection import train_test_split
import copy

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
train_path = os.path.join(os.path.dirname(script_dir), "deep_face", "train")
#数据增强
train_transform = transforms.Compose(
    [
        transforms.Resize([256,256]),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ]
)
#验证集
val_transform = transforms.Compose(
    [
        transforms.Resize([256,256]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ]
)


train_data = datasets.ImageFolder(root = train_path,transform = train_transform)
val_data = datasets.ImageFolder(root = train_path , transform = val_transform)

val_size = int(len(val_data)*0.15+1)
indices = list(range(len(val_data)))

labels = [label for _, label in val_data.samples]
train_idx, val_idx = train_test_split(
        indices, test_size=val_size, stratify=labels,
        random_state=42,
    )

train_dataset = Subset(train_data, train_idx)
val_dataset = Subset(val_data, val_idx)

train_loader = DataLoader(
    dataset = train_dataset,
    batch_size = 32,
    shuffle=True
)
val_loader = DataLoader(
    dataset = val_dataset,
    batch_size = 32
)
print(f"  训练集: {len(train_dataset)} 张")
print(f"  验证集: {len(val_dataset)} 张")



model = deepfake_model()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001,weight_decay=1e-4)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

EPOCHS = 30
best_loss = float('inf')
best_model_weights = None
for epoch in range(1,EPOCHS+1):

    model.train()  
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for images , labels in tqdm(train_loader, desc=f"Epoch {epoch:03d} Training", leave=False):
        images , labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()

    train_avg_loss = train_loss / train_total
    train_acc = 100.0 * train_correct / train_total

    model.eval()
    test_loss = 0.0
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for images,labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            test_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
    
    test_avg_loss = test_loss / test_total
    test_acc = 100.0 * test_correct / test_total

    print(f"\nEpoch {epoch:03d} ")
    print(f"  Train Loss: {train_avg_loss:.4f} | Train Acc: {train_acc:.2f}%")
    print(f"  Valid Loss: {test_avg_loss:.4f} | Valid Acc: {test_acc:.2f}%")

    if test_avg_loss < best_loss:
        best_loss = test_avg_loss
        best_model_weights = copy.deepcopy(model.state_dict())


model.load_state_dict(best_model_weights)
os.makedirs("models", exist_ok=True)
MODEL_PATH = os.path.join("models","dl_model.pth")
torch.save(
    model.state_dict(),
    MODEL_PATH
)




