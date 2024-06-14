import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os

# Custom Dataset Class
class Comp472OriginalDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = [os.path.join(root_dir, fname) for fname in os.listdir(root_dir)]
        self.labels = [self.get_label(fname) for fname in os.listdir(root_dir)]

    def get_label(self, filename):
        # Implement logic to extract label from filename or other sources
        label = int(filename.split('_')[1])  # Example: extract label from filename
        return label

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path)
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

# Define data transformations
transform = transforms.Compose(
    [transforms.Resize((32, 32)),  # Resize to match the input size expected by the CNN
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Create datasets and dataloaders
trainset = Comp472OriginalDataset(root='./data/train', transform=transform)
trainloader = DataLoader(trainset, batch_size=100, shuffle=True, num_workers=2)

testset = Comp472OriginalDataset(root='./data/test', transform=transform)
testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

# Define class names (adjust according to your dataset)
classes = ('Angry', 'Happy', 'Engaged', 'Neutral')

# Base Model
class BaseCNN(nn.Module):
    def __init__(self):
        super(BaseCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# More Convolutional Layers Variant
class MoreConvLayersCNN(nn.Module):
    def __init__(self):
        super(MoreConvLayersCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Different Kernel Sizes Variant
class DifferentKernelSizesCNN(nn.Module):
    def __init__(self):
        super(DifferentKernelSizesCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding=2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Train the Model
def train_model(model, trainloader, criterion, optimizer, epochs=10, early_stopping_patience=3):
    model.train()
    patience_counter = 0
    best_loss = float('inf')

    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        epoch_loss = running_loss / len(trainloader)
        print(f'Epoch {epoch+1}, Loss: {epoch_loss:.4f}')
        
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print("Early stopping")
                break

    print('Finished Training')
    return model

# Evaluate the Model
def evaluate_model(model, testloader):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.numpy())
            all_labels.extend(labels.numpy())

    cm = confusion_matrix(all_labels, all_preds)
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average='macro')
    rec = recall_score(all_labels, all_preds, average='macro')
    f1 = f1_score(all_labels, all_preds, average='macro')

    print(f'Accuracy: {acc:.4f}')
    print(f'Precision: {prec:.4f}')
    print(f'Recall: {rec:.4f}')
    print(f'F1 Score: {f1:.4f}')
    
    return cm, acc, prec, rec, f1

# Main Function
if __name__ == "__main__":
    # Initialize model, loss function, and optimizer
    model = BaseCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Train the model
    trained_model = train_model(model, trainloader, criterion, optimizer, epochs=10)

    # Load the best model
    trained_model.load_state_dict(torch.load('best_model.pth'))

    # Evaluate the model
    cm, acc, prec, rec, f1 = evaluate_model(trained_model, testloader)

    # Visualize confusion matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

    # Train and evaluate more models
    for ModelClass in [MoreConvLayersCNN, DifferentKernelSizesCNN]:
        model = ModelClass()
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        trained_model = train_model(model, trainloader, criterion, optimizer, epochs=10)
        trained_model.load_state_dict(torch.load('best_model.pth'))
        cm, acc, prec, rec, f1 = evaluate_model(trained_model, testloader)
        print(f"Results for {ModelClass.__name__}")
        print(f'Accuracy: {acc:.4f}')
        print(f'Precision: {prec:.4f}')
        print(f'Recall: {rec:.4f}')
        print(f'F1 Score: {f1:.4f}')
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'Confusion Matrix for {ModelClass.__name__}')
        plt.show()
