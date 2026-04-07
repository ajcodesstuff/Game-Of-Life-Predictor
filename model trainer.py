import os
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import pickle

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

label_map = {
    "Stable": 0,
    "Oscillating": 1,
    "Dead": 2
}

class CGOLData(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        grid, label = self.data[idx]

        grid = torch.tensor(grid, dtype=torch.float32)
        grid = grid.unsqueeze(0)

        label = torch.tensor(label_map[label])

        return grid, label
    

train_dataset = []
with open("dataset.pkl", "rb") as f:
    while True:
        try:
            train_dataset.extend(pickle.load(f))
        except EOFError:
            break
    
train_dataset = CGOLData(train_dataset)
loader = DataLoader(train_dataset, batch_size=2048, shuffle=True)

test_dataset = []
with open("test_dataset.pkl", "rb") as f:
    while True:
        try:
            test_dataset.extend(pickle.load(f))
        except EOFError:
            break

test_dataset = CGOLData(test_dataset)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
print("Using device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

class CNN(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, 5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, 5, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.MaxPool2d(2),  # 20 → 10

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.MaxPool2d(2),  # 10 → 5

            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 5 * 5, 512),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.ReLU(),

            nn.Linear(256, 3)
        )
    
    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

model_path = "cgol_model.pt"

model = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
scaler = torch.amp.GradScaler()

start_epoch = 0
if os.path.exists(model_path):
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scaler.load_state_dict(checkpoint["scaler_state_dict"])
    start_epoch = checkpoint.get("epoch", 0)
    print(f"Loaded checkpoint from {model_path}, starting at epoch {start_epoch}")

EPOCHS = 50

for epoch in range(start_epoch, start_epoch+EPOCHS):
    for grids, labels in loader:
        grids = grids.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        with torch.amp.autocast('cuda'):
            outputs = model(grids)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
    if (epoch + 1) % 5 == 0:
        correct = 0
        total = 0
        with torch.no_grad():
            for grids, labels in test_loader:
                grids = grids.to(device)
                labels = labels.to(device)
                outputs = model(grids)
                _, predicted = torch.max(outputs, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            print("Accuracy:", (correct / total) * 100, "%")

torch.save({
    "epoch": start_epoch + EPOCHS,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "scaler_state_dict": scaler.state_dict()
}, model_path)

print(f"Saved checkpoint to {model_path}")