import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import csv
from torch.utils.data import Subset

# Device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Transforms
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2470, 0.2435, 0.2616)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2470, 0.2435, 0.2616)),
])

# Data — train / val / test split 
full_trainset = torchvision.datasets.CIFAR10(root='./data', train=True,  download=True, transform=transform_train)
testset       = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

np.random.seed(42)
indices  = np.random.permutation(len(full_trainset))
trainset = Subset(full_trainset, indices[:45000])
valset   = Subset(full_trainset, indices[45000:])

trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True,  num_workers=0)
valloader   = torch.utils.data.DataLoader(valset,   batch_size=128, shuffle=False, num_workers=0)
testloader  = torch.utils.data.DataLoader(testset,  batch_size=128, shuffle=False, num_workers=0)

print(f"Train: {len(trainset)} | Val: {len(valset)} | Test: {len(testset)}")

classes = ('plane','car','bird','cat','deer','dog','frog','horse','ship','truck')

# Model 
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Block 1
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)
        # Block 2
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2   = nn.BatchNorm2d(64)
        # Block 3
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3   = nn.BatchNorm2d(128)

        self.pool    = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)

        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  
        x = torch.flatten(x, 1)                         
        x = self.dropout(F.relu(self.fc1(x)))           
        x = self.fc2(x)                                 
        return x

model = SimpleCNN().to(device)

# Dummy input test
dummy = torch.randn(1, 3, 32, 32).to(device)
out   = model(dummy)
print(f"Dummy output shape: {out.shape}")

# Loss & Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

# Training loop
def train(epoch):
    model.train()
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(trainloader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 100 == 99:
            print(f"Epoch {epoch+1}, step {i+1}: loss {running_loss/100:.3f}")
            running_loss = 0.0

# Validation loop
def evaluate(epoch):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for inputs, labels in valloader:   # val set, not test
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total   += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    acc = 100 * correct / total
    print(f"Epoch {epoch+1}: Val accuracy: {acc:.2f}%")
    return acc

# Run
if __name__ == "__main__":
    # Setup CSV log
    log_path = 'models/training_log.csv'
    with open(log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'val_acc', 'best_acc'])

    best_acc = 0
    for epoch in range(30):
        train(epoch)
        acc = evaluate(epoch)
        scheduler.step()

        # Log to CSV
        with open(log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch+1, round(acc, 2), round(best_acc, 2)])

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), 'models/best_model.pth')
            print(f"  ✓ Saved best model ({best_acc:.2f}%)")

    print(f"\nFinished. Best accuracy: {best_acc:.2f}%")