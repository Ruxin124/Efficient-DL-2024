from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
import torch
from torch.utils.data.dataloader import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn as nn
from models_cifar100.resnet import ResNet18
from torchvision.models import densenet161
from torch.optim import Adam, lr_scheduler
from torch.nn import functional as F


# Load the CIFAR-10 dataset
## Normalization adapted for CIFAR10
normalize_scratch = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
# Transforms is a list of transformations applied on the 'raw' dataset before the data is fed to the network. 
# Here, Data augmentation (RandomCrop and Horizontal Flip) are applied to each batch, differently at each epoch, on the training set data only
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize_scratch,
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    normalize_scratch,
])
### The data from CIFAR10 will be downloaded in the following folder
rootdir = './data/cifar10'
# Load the CIFAR-10 dataset
c10train = CIFAR10(rootdir,train=True,download=True,transform=transform_train)
c10test = CIFAR10(rootdir,train=False,download=True,transform=transform_test)
# Create DataLoaders
trainloader = DataLoader(c10train,batch_size=32,shuffle=True)
testloader = DataLoader(c10test,batch_size=32)

# device check
if torch.cuda.is_available():
    print("CUDA is available. Using GPU.")
    device = torch.device("cuda")
else:
    print("CUDA is not available. Using CPU.")
    device = torch.device("cpu")

# Build model
model = densenet161().to(device)

# Adjust the final classifier layer for 10 classes instead of 100
model.classifier = nn.Linear(model.classifier.in_features, 10)
model.to(device)

# Training configuration
optimizer = Adam(model.parameters(), lr=0.001)
scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=0.001, last_epoch=-1)

num_epochs = 100
best_test_accuracy = 0.0
best_model_state = None

# Training and evaluation loop
for epoch in range(num_epochs):
    model.train()
    train_loss, correct, total = 0, 0, 0
    for data, targets in trainloader:
        data, targets = data.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(data)
        loss = F.cross_entropy(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()
    train_accuracy = 100 * correct / total
    print(f'Epoch {epoch}: Train Loss: {train_loss / len(trainloader)}, Accuracy: {train_accuracy:.2f}%')

    # Evaluate on test set
    model.eval()
    test_correct, test_total = 0, 0
    with torch.no_grad():
        for data, targets in testloader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            test_total += targets.size(0)
            test_correct += (predicted == targets).sum().item()
    test_accuracy = 100 * test_correct / test_total
    print(f'Test Accuracy: {test_accuracy:.2f}%')

    # Save the best model based on test accuracy
    if test_accuracy > best_test_accuracy:
        best_test_accuracy = test_accuracy
        best_model_state = model.state_dict()
        torch.save({
            'epoch': epoch,
            'model_state_dict': best_model_state,
            'optimizer_state_dict': optimizer.state_dict(),
            'test_accuracy': best_test_accuracy
        }, 'best_model_trained_densnet.pth')
    scheduler.step()