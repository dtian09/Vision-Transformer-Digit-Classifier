import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import numpy as np
import wandb
from tqdm import tqdm
from vit import ViT

class Trainer:
    def __init__(self, model, train_loader, val_loader, test_loader, criterion, optimizer, device, num_epochs, patience):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.num_epochs = num_epochs
        self.patience = patience
        self.best_val_acc = 0.0
        self.patience_counter = 0

    def train_epoch(self):
        self.model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        for images, labels in tqdm(self.train_loader, desc="Training", leave=False):
            images, labels = images.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
        train_loss /= total
        train_acc = correct / total
        return train_loss, train_acc

    def train(self):
            for epoch in range(self.num_epochs):
                train_loss, train_acc = self.train_epoch()
                train_loss = round(train_loss, 2)
                train_acc = round(train_acc, 2)
                val_loss, val_acc = self.validate()
                val_loss = round(val_loss, 2)
                val_acc = round(val_acc, 2)
                print(f"Epoch {epoch+1}: Train Loss={train_loss}, Train Acc={train_acc}, Val Loss={val_loss}, Val Acc={val_acc}")
                wandb.log({
                    "epoch": epoch+1,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc
                })
                # Early stopping
                if val_acc > self.best_val_acc:
                    self.best_val_acc = val_acc
                    self.patience_counter = 0
                    torch.save(self.model.state_dict(), 'best_vit_mnist.pth')
                else:
                    self.patience_counter += 1
                    if self.patience_counter >= self.patience:
                        print(f"Early stopping at epoch {epoch+1}")
                        break
            # Load best model
            self.model.load_state_dict(torch.load('best_vit_mnist.pth'))

    def validate(self):
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in tqdm(self.val_loader, desc="Validation", leave=False):
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)
        val_loss /= total
        val_acc = correct / total
        return val_loss, val_acc

    
def test(model,test_loader,device):
        model.eval()
        correct = 0     
        total = 0
        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc="Testing", leave=False):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)
        test_acc = correct / total
        test_acc = round(test_acc, 2)
        print(f"Test Accuracy: {test_acc}")
        wandb.log({"test_acc": test_acc})
        return test_acc

if __name__ == "__main__":
    # Hyperparameters
    batch_size = 64 # number of images in a batch
    n_layers = 6  # Number of transformer encoder layers
    n_heads = 4 # number of attention heads
    num_epochs = 50  # Max epochs
    patience = 5  # Early stopping patience
    lr = 3e-4  # Learning rate

    # Initialize wandb
    wandb.init(project="vit-mnist", 
            entity="dtian",
            config={
                "batch_size": batch_size,
                "n_layers": n_layers,
                "n_heads": n_heads,
                "num_epochs": num_epochs,
                "patience": patience,
                "lr": lr
            })

    # Instantiate model and count parameters
    model = ViT(n_layers=n_layers, n_heads=n_heads)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {n_params}")

    # Data preparation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Download MNIST
    full_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_set = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # Split train into train/val (90%/10%)
    train_size = int(0.9 * len(full_train))
    val_size = len(full_train) - train_size
    train_set, val_set = random_split(full_train, [train_size, val_size], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Training and testing
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    trainer = Trainer(model, train_loader, val_loader, criterion, optimizer, device, num_epochs, patience)
    trainer.train()   
    test(model,test_loader,device) 
