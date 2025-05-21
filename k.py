import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.amp import GradScaler, autocast
import timm

# === Model Class ===
class BurnClassifier(nn.Module):
    def __init__(self, num_classes):
        super(BurnClassifier, self).__init__()
        self.backbone = timm.create_model('efficientnet_b2', pretrained=True, features_only=False)

        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.backbone.blocks[-4:].parameters():
            param.requires_grad = True

        num_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

# === MAIN ===
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Using device: {device}")
    if device.type == 'cuda':
        print(f"🚀 Training on: {torch.cuda.get_device_name(0)}")
    scaler = GradScaler(device.type)

    base_dir = r"C:\Users\rahma\OneDrive\Desktop\Work-based\burn skinnn\skin burn dataset"
    train_dir = os.path.join(base_dir, "train")
    valid_dir = os.path.join(base_dir, "valid")

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    valid_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    valid_dataset = datasets.ImageFolder(valid_dir, transform=valid_transform)

    print(f"📊 Train samples: {len(train_dataset)}")
    print(f"📊 Valid samples: {len(valid_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=False,
                              num_workers=4, pin_memory=True)

    model = BurnClassifier(num_classes=len(train_dataset.classes)).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.005,
                                              steps_per_epoch=len(train_loader),
                                              epochs=20, pct_start=0.3)

    best_val_acc = 0.0
    patience = 4
    no_improve = 0
    num_epochs = 20

    print("🔵 Starting training loop...\n")

    for epoch in range(num_epochs):
        print(f"🔁 Epoch {epoch + 1}/{num_epochs}")
        model.train()
        running_loss, correct = 0.0, 0
        train_bar = tqdm(train_loader, desc="Training")

        for images, labels in train_bar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            with autocast(device_type=device.type):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            train_bar.set_postfix(loss=loss.item())

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct / len(train_loader.dataset)
        print(f"✅ Train Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")

        model.eval()
        val_loss, val_correct = 0.0, 0
        val_bar = tqdm(valid_loader, desc="Validating")

        with torch.no_grad():
            for images, labels in val_bar:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_bar.set_postfix(loss=loss.item())

        val_loss /= len(valid_loader.dataset)
        val_acc = val_correct / len(valid_loader.dataset)
        print(f"🧪 Valid Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            no_improve = 0
            torch.save(model.state_dict(), "best_burn_model.pth")
            print("📀 Model improved. Saved new best model.")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"⛔ No improvement for {patience} epochs. Stopping early.")
                break

    torch.save(model.state_dict(), "final_burn_model.pth")
    print(f"\n🎉 Training complete. Best validation accuracy: {best_val_acc:.4f}")