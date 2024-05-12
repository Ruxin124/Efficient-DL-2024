from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
import torch
from torch.utils.data.dataloader import DataLoader
import torch.optim as optim
import torch.nn as nn
from models_cifar100.resnet import ResNet18

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

# Hyperparameters
hparam_currentvalue = {
    'epochs': 3,
    'initial_lr': 0.01,
    'lr_decay_factor': 0.2,
    'lr_decay_epoch': 6,
}

# Early stopping parameters
early_stopping_patience = 20
early_stopping_delta = 0.001
best_test_loss = float('inf')
counter_early_stopping = 0

# Build model
model = ResNet18().to(device)

# Define optimizer and loss function
optimizer = optim.SGD(model.parameters(), lr=hparam_currentvalue['initial_lr'], momentum=0.9, weight_decay=5e-4)
criterion = nn.CrossEntropyLoss()

# Learning rate scheduler
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=hparam_currentvalue['lr_decay_epoch'], gamma=hparam_currentvalue['lr_decay_factor'])

# Training
model.train()
epoch_train_losses = []
epoch_test_losses = []
epoch_accuracies = []

# Training loop
for epoch in range(hparam_currentvalue['epochs']):
    training_loss = 0.0
    correct = 0
    total = 0
    
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        training_loss += loss.item()
        
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
     
    # Calculate and print statistics for the epoch
    epoch_train_loss = training_loss / len(trainloader)
    epoch_accuracy = 100 * correct / total
    print(f'Epoch: {epoch}, Loss: {epoch_train_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%')
    
    # Append loss and accuracy for visualization later
    epoch_train_losses.append(epoch_train_loss)
    epoch_accuracies.append(epoch_accuracy)

    # test loss Evaluate on test set 
    model.eval()  # Set model to evaluation mode
    test_loss = 0.0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

    # Early Stopping check
    if test_loss < best_test_loss - early_stopping_delta:
        best_test_loss = test_loss
        counter_early_stopping = 0
        best_model_state = model.state_dict()  # Save the best model state
    else:
        counter_early_stopping += 1
        if counter_early_stopping >= early_stopping_patience:
            print(f"Early stopping triggered after epoch {epoch} with test loss {test_loss:.4f}")
            model.load_state_dict(best_model_state)  # Load the best model state
            break  # Stop training
    
    epoch_test_loss = test_loss / len(testloader)
    epoch_test_losses.append(epoch_test_loss)
    print(f'Epoch {epoch}: Train Loss: {epoch_train_loss:.4f}, Test Loss: {epoch_test_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%')

    # Step the scheduler
    scheduler.step()

print('Finished Training')# build model

# After training, evaluate on test set to get final accuracy
model.eval()  # Set model to evaluation mode
test_correct = 0
test_total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()

test_accuracy = 100 * test_correct / test_total
print(f'Final Test Accuracy: {test_accuracy:.2f}%')

# Save model along with accuracy and loss history
state = {
    'net': model.state_dict(),
    'hyperparam': hparam_currentvalue,
    'epoch_train_losses': epoch_train_losses,
    'epoch_test_losses': epoch_test_losses,
    'epoch_accuracies': epoch_accuracies,
    'final_test_accuracy': test_accuracy  # include final test accuracy
}

torch.save(state, 'test_resnet18_cifar10_6.pth')