import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Model definition (must match train.py exactly) 
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2   = nn.BatchNorm2d(64)
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

# Load model 
model = SimpleCNN().to(device)
checkpoint = torch.load('models/best_checkpoint.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
print(f"Loaded checkpoint from epoch {checkpoint['epoch']+1} "
      f"with val acc {checkpoint['val_acc']:.2f}%")

# Test data
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2470, 0.2435, 0.2616)),
])
testset    = torchvision.datasets.CIFAR10(root='./data', train=False,
                                          download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=128,
                                          shuffle=False, num_workers=0)

classes = ('plane','car','bird','cat','deer','dog','frog','horse','ship','truck')

# Test set accuracy
model.eval()
all_preds  = []
all_labels = []
all_images = []

with torch.no_grad():
    for inputs, labels in testloader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = outputs.max(1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_images.extend(inputs.cpu().numpy())

all_preds  = np.array(all_preds)
all_labels = np.array(all_labels)
all_images = np.array(all_images)

test_acc = 100 * (all_preds == all_labels).sum() / len(all_labels)
print(f"\nTest set accuracy: {test_acc:.2f}%")

# Precision, Recall, F1
print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=classes))

# Confusion matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=classes, yticklabels=classes)
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('models/confusion_matrix.png', dpi=150)
plt.show()
print("Saved: models/confusion_matrix.png")

# Training curves
import csv
epochs, val_accs = [], []
with open('models/training_log.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        epochs.append(int(row['epoch']))
        val_accs.append(float(row['val_acc']))

plt.figure(figsize=(8, 5))
plt.plot(epochs, val_accs, 'b-o', markersize=4, label='Val Accuracy')
plt.axhline(y=75, color='r', linestyle='--', label='75% target')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Validation Accuracy over Training')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('models/training_curves.png', dpi=150)
plt.show()
print("Saved: models/training_curves.png")

# Misclassified examples
wrong_idx = np.where(all_preds != all_labels)[0][:16]

fig, axes = plt.subplots(4, 4, figsize=(10, 10))
mean = np.array([0.4914, 0.4822, 0.4465])
std  = np.array([0.2470, 0.2435, 0.2616])

for i, ax in enumerate(axes.flat):
    idx = wrong_idx[i]
    img = all_images[idx].transpose(1, 2, 0)  # CHW → HWC
    img = (img * std) + mean                   # unnormalize
    img = np.clip(img, 0, 1)
    ax.imshow(img)
    ax.set_title(f"True: {classes[all_labels[idx]]}\n"
                 f"Pred: {classes[all_preds[idx]]}", fontsize=8)
    ax.axis('off')

plt.suptitle('Misclassified Examples', fontsize=14)
plt.tight_layout()
plt.savefig('models/misclassified.png', dpi=150)
plt.show()
print("Saved: models/misclassified.png")