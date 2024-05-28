import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.optim import Adam, lr_scheduler
from torchvision.models import resnet101
from models_cifar100.resnet import ResNet18
from torch import nn

# Hyperparameters and configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_epochs = 20
learning_rate = 0.001
T = 10.0  # Temperature for distillation
alpha = 0.3  # Balance between CE and KL divergence


batch_size = 32

# Data loaders
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])
trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
testset = datasets.CIFAR10(root='./data', train=False, transform=transform)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

# Model setup
## load the teacher model
teacher_model = resnet101(pretrained=True)
teacher_model.to(device).train()  # Always in eval mode
# retrain the last layer for CIFAR-10
teacher_model.fc = nn.Linear(teacher_model.fc.in_features, 100)
teacher_model.to(device).eval()
print ('Teacher model loaded and adapted the Final Layer to CIFAR10.')

## load the sutdent model
## load our basic ResNet18 model already trained on CIFAR-10
model = ResNet18()  
model.to(device).eval()

# Optimizer and scheduler
optimizer = Adam(model.parameters(), lr=learning_rate)
scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# Loss function for distillation
def distillation_loss(y_student, y_teacher, y_true, T, alpha):
    loss_kl = F.kl_div(F.log_softmax(y_student/T, dim=1),
                       F.softmax(y_teacher/T, dim=1),
                       reduction='batchmean') * (T * T * alpha)
    loss_ce = F.cross_entropy(y_student, y_true) * (1 - alpha)
    return loss_kl + loss_ce

# Training and evaluation loop
best_test_accuracy = 0.0
best_model_state = None

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for data, target in trainloader:
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        
        # Teacher and student outputs
        with torch.no_grad():
            teacher_output = teacher_model(data)
        student_output = model(data)
    
        
        # Calculate loss
        loss = distillation_loss(student_output, teacher_output, target, T, alpha)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(student_output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
    
    epoch_accuracy = 100 * correct / total
    print(f'Epoch {epoch}: Train Loss: {running_loss / len(trainloader)}, Accuracy: {epoch_accuracy:.2f}%')
    
    # Evaluation on test set
    model.eval()
    test_accuracy = 0
    with torch.no_grad():
        for data, target in testloader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            test_accuracy += (predicted == target).sum().item()

    test_accuracy = 100 * test_accuracy / len(testset)
    print(f'Test Accuracy: {test_accuracy:.2f}%')
    
    # Update best model
    if test_accuracy > best_test_accuracy:
        best_test_accuracy = test_accuracy
        best_model_state = model.state_dict()
        print(f'Best model updated at epoch {epoch} with test accuracy {best_test_accuracy:.2f}%')

    scheduler.step()

# Save the best model
if best_model_state:
    torch.save({
        'model_state_dict': best_model_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'best_test_accuracy': best_test_accuracy
    }, 'best_distilled_model_4.pth')
    print("Best model saved.")

print('Finished Training')
