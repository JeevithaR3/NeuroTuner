# trainer.py
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

def get_data_loaders(batch_size):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train = datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)
    test  = datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader  = torch.utils.data.DataLoader(test,  batch_size=64, shuffle=False, num_workers=2)
    return train_loader, test_loader

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Flatten(),
            nn.Linear(32*7*7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
    def forward(self, x): return self.net(x)

def run_training(config):
    """
    config: dict { lr, batch_size, epochs, device (optional) }
    returns: dict { accuracy, latency_ms, loss }
    """
    device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = get_data_loaders(config["batch_size"])
    model = SimpleCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    criterion = nn.CrossEntropyLoss()

    # Quick training loop - keep epochs small for demo
    model.train()
    for epoch in range(config.get("epochs", 1)):
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

    # Evaluation + latency
    model.eval()
    correct = total = 0
    t0 = time.perf_counter()
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            preds = out.argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)
    t1 = time.perf_counter()
    latency_ms = (t1 - t0) * 1000 / len(test_loader)   # average batch latency (ms)
    acc = correct / total
    return {"accuracy": acc, "latency_ms": latency_ms, "loss": float(loss.item())}
