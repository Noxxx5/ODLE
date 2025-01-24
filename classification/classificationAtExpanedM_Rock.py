import os
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from torch.cuda.amp import autocast, GradScaler

BATCH_SIZE = 256
LEARNING_RATE = 0.001
NUM_EPOCHS = 100
NUM_CLASSES = 15
DATASET_PATH = "data/m_rock_expansion/save/distdiff_batch_3x"


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

dataset = ImageFolder(DATASET_PATH, transform=transform)

def initialize_model(num_classes):
    model = models.resnet50(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
scaler = GradScaler()
def train_model(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()


        with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    
    return total_loss / len(dataloader), correct / total

def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return total_loss / len(dataloader), correct / total, all_labels, all_preds

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold = 0
class_accuracies = np.zeros((5, NUM_CLASSES))

for train_idx, val_idx in kf.split(dataset):
    fold += 1
    print(f"Training fold {fold}...")
    

    train_subset = torch.utils.data.Subset(dataset, train_idx)
    val_subset = torch.utils.data.Subset(dataset, val_idx)

    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)


    model = initialize_model(NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc = train_model(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_labels, val_preds = evaluate_model(model, val_loader, criterion, device)
        print(f"Fold {fold}, Epoch [{epoch+1}/{NUM_EPOCHS}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    for cls in range(NUM_CLASSES):
        cls_correct = np.sum((np.array(val_preds) == cls) & (np.array(val_labels) == cls))
        cls_total = np.sum(np.array(val_labels) == cls)
        class_accuracies[fold-1, cls] = cls_correct / cls_total if cls_total > 0 else 0.0

mean_accuracies = np.mean(class_accuracies, axis=0)
for cls in range(NUM_CLASSES):
    print(f"Class {cls} ({dataset.classes[cls]}): Accuracy: {mean_accuracies[cls]:.4f}")