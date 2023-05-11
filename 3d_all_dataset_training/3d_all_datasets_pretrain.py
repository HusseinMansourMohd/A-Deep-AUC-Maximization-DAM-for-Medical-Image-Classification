import torch.optim as optim
from torch.utils import data
from torchvision.models import resnet18
from medmnist import INFO
import medmnist

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms



BATCH_SIZE = 64
NUM_EPOCHS = 100
lr=0.0001
n_classes = 2

def load_and_update_labels(data_flag, offset):
    DataClass = getattr(medmnist, INFO[data_flag]['python_class'])
    train_dataset = DataClass(split='train', download=True)
    test_dataset = DataClass(split='test', download=True)

    train_labels = np.array(train_dataset.labels)
    test_labels = np.array(test_dataset.labels)

    train_labels += offset
    test_labels += offset

    train_dataset.labels = list(train_labels)
    test_dataset.labels = list(test_labels)

    return train_dataset, test_dataset


data_flags = ['nodulemnist3d', 'adrenalmnist3d', 'vesselmnist3d', 'synapsemnist3d']

train_datasets = []
test_datasets = []

for i, data_flag in enumerate(data_flags):
    offset = i * n_classes
    train_dataset, test_dataset = load_and_update_labels(data_flag, offset)
    train_datasets.append(train_dataset)
    test_datasets.append(test_dataset)

combined_train_dataset = data.ConcatDataset(train_datasets)
combined_test_dataset = data.ConcatDataset(test_datasets)

train_loader = data.DataLoader(dataset=combined_train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = data.DataLoader(dataset=combined_test_dataset, batch_size=BATCH_SIZE, shuffle=False)

total_n_classes = n_classes * len(data_flags)

# BATCH_SIZE = 16
# NUM_EPOCHS = 10
# lr = 0.0001

from torchvision.models.video import r3d_18

class ResNet3D(nn.Module):
    def __init__(self, num_classes):
        super(ResNet3D, self).__init__()
        self.resnet3d = r3d_18(pretrained=False)
        self.resnet3d.stem[0] = nn.Conv3d(1, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False)
        self.resnet3d.fc = nn.Linear(self.resnet3d.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet3d(x)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ResNet3D(num_classes=total_n_classes)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
best_test_accuracy = 0.0
for epoch in range(NUM_EPOCHS):
    train_correct = 0
    train_total = 0
    test_correct = 0
    test_total = 0
    
    model.train()
    train_loss = 0
    for inputs, targets in train_loader:
        #inputs = inputs.float()
        inputs = inputs.float().to(device)
        targets = targets.squeeze().long().to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        targets = targets.squeeze().long()
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        train_total += targets.size(0)
        train_correct += predicted.eq(targets).sum().item()

    train_loss /= len(train_loader)
    train_accuracy = 100.0 * train_correct / train_total

    model.eval()
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.float()
            inputs = inputs.float().to(device)
            targets = targets.squeeze().long().to(device)
            outputs = model(inputs)
            targets = targets.squeeze().long()
            _, predicted = outputs.max(1)
            test_total += targets.size(0)
            test_correct += predicted.eq(targets).sum().item()

    test_accuracy = 100.0 * test_correct / test_total

    print(f"Epoch: {epoch + 1}/{NUM_EPOCHS}, Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Test Accuracy: {test_accuracy:.2f}%")
    
    if test_accuracy > best_test_accuracy:
        best_test_accuracy = test_accuracy
        torch.save(model.state_dict(), 'best_model.pth')
        print("Checkpoint saved.")