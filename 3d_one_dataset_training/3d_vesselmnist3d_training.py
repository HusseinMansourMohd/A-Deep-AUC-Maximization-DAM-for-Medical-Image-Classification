import torch
import torch.optim as optim
import torch.nn as nn
import torch.utils.data as data
from torchvision.models.video import r3d_18
from medmnist import INFO
import medmnist
from torchvision.transforms import Compose, RandomHorizontalFlip, RandomVerticalFlip, RandomRotation
from sklearn.metrics import roc_auc_score

# Set the best hyperparameters obtained from previous training
best_hyperparameters = {
    'optimizer': 'RMSprop',  # Replace with the best optimizer name
    'learning_rate': 0.001,  # Replace with the best learning rate
    'momentum': 0.9,  # Replace with the best momentum value
    'loss_function': 'CrossEntropyLoss'  # Replace with the best loss function name
}
#{'optimizer': 'RMSprop', 'learning_rate': 0.001, 'momentum': 0.9, 'loss_function': 'CrossEntropyLoss'}
num_epochs = 100

class ResNet3D(nn.Module):
    def __init__(self, num_classes):
        super(ResNet3D, self).__init__()
        self.resnet3d = r3d_18(pretrained=False)
        self.resnet3d.stem[0] = nn.Conv3d(1, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False)
        self.resnet3d.fc = nn.Linear(self.resnet3d.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet3d(x)

from torchvision.transforms import Lambda

def load_nodulemnist3d():
    DataClass = getattr(medmnist, INFO['vesselmnist3d']['python_class'])
    train_dataset = DataClass(split='train', download=True, transform=Lambda(lambda x: torch.tensor(x).unsqueeze(0).float().squeeze(1)))
    val_dataset = DataClass(split='val', download=True, transform=Lambda(lambda x: torch.tensor(x).unsqueeze(0).float().squeeze(1)))
    test_dataset = DataClass(split='test', download=True, transform=Lambda(lambda x: torch.tensor(x).unsqueeze(0).float().squeeze(1)))
    return train_dataset, val_dataset, test_dataset

train_dataset, val_dataset, test_dataset = load_nodulemnist3d()
train_loader = data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
val_loader = data.DataLoader(dataset=val_dataset, batch_size=64, shuffle=False)
test_loader = data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = ResNet3D(num_classes=8)
pretrained_model_path = '/local/musaeed/Assign/2DMultiTask/3d_data_augmentation_best_model_16_batch_size.pth'
model.load_state_dict(torch.load(pretrained_model_path))

model.resnet3d.fc = nn.Linear(model.resnet3d.fc.in_features, 2)
model = model.to(device)

for i, (name, param) in enumerate(model.named_parameters()):
    if i >= len(list(model.named_parameters())) - 10:
        param.requires_grad = True
    else:
        param.requires_grad = False

model.resnet3d.fc = nn.Linear(model.resnet3d.fc.in_features, 2)
model = model.to(device)

if best_hyperparameters['optimizer'] == 'Adam':
    optimizer = optim.Adam(model.parameters(), lr=best_hyperparameters['learning_rate'])
elif best_hyperparameters['optimizer'] == 'SGD' or best_hyperparameters['optimizer'] == 'RMSprop':
    optimizer = getattr(optim, best_hyperparameters['optimizer'])(model.parameters(), lr=best_hyperparameters['learning_rate'], momentum=best_hyperparameters['momentum'])
else:
    raise ValueError(f"Unsupported optimizer: {best_hyperparameters['optimizer']}")

criterion = nn.CrossEntropyLoss()

best_auc = 0.0 # To save the model with the best validation AUC
for epoch in range(num_epochs):
    train_correct = 0
    train_total = 0
    val_correct = 0
    val_total = 0
    test_correct = 0
    test_total = 0
    train_loss = 0

    model.train()
    for inputs, targets in train_loader:
        inputs = inputs.float().to(device)
        targets = targets.squeeze().long().to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        train_total += targets.size(0)
        train_correct += predicted.eq(targets).sum().item()

    train_loss /= len(train_loader)
    train_accuracy = 100.0 * train_correct / train_total
    print(f"Epoch: {epoch:.4f}, train_loss: {train_loss:.4f}, train_accuracy: {train_accuracy:.4f}")


    model.eval()
    with torch.no_grad():
        val_true = []
        val_scores = []
        for inputs, targets in val_loader:
            inputs = inputs.float().to(device)
            targets = targets.squeeze().long().to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            val_total += targets.size(0)
            val_correct += predicted.eq(targets).sum().item()
            val_true.extend(targets.cpu().numpy())
            val_scores.extend(outputs[:, 1].cpu().numpy())

        val_accuracy = 100.0 * val_correct / val_total
        val_auc = roc_auc_score(val_true, val_scores)
        print(f"Epoch: {epoch:.4f}, val accuracy: {val_accuracy:.4f}, validation auc: {val_auc:.4f}")

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), 'trained_vesselmnist3d_model_with_augmentation_best_auc.pth')
            print(f"Model saved with best validation AUC: {best_auc:.4f}")

    with torch.no_grad():
        test_true = []
        test_scores = []
        for inputs, targets in test_loader:
            inputs = inputs.float().to(device)
            targets = targets.squeeze().long().to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            test_total += targets.size(0)
            test_correct += predicted.eq(targets).sum().item()
            test_true.extend(targets.cpu().numpy())
            test_scores.extend(outputs[:, 1].cpu().numpy())

        test_accuracy = 100.0 * test_correct / test_total
        test_auc = roc_auc_score(test_true, test_scores)
        print(f"Epoch: {epoch:.4f}, test accuracy: {test_accuracy:.4f}, test auc: {test_auc:.4f}")

print(f"Epoch: {epoch + 1}/{num_epochs}, Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Validation Accuracy: {val_accuracy:.2f}%, Validation AUC: {val_auc:.4f}, Test Accuracy: {test_accuracy:.2f}%, Test AUC: {test_auc:.4f}")
print("Training finished.")

