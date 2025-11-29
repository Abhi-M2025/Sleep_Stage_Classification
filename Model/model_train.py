import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from model_def import SleepStageCNN
import glob
import os

class SleepDataset(Dataset):
    def __init__(self, directory="./processed"):
        self.files = glob.glob(os.path.join(directory, "*.npz"))
        self.X = []
        self.Y = []

        for f in self.files:
            data = np.load(f)
            self.X.append(data["X"])
            self.Y.append(data["y"])

        # combine all subjects
        self.X = np.concatenate(self.X, axis=0)
        self.Y = np.concatenate(self.Y, axis=0)

        # convert to tensors
        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.Y = torch.tensor(self.Y, dtype=torch.long)

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

if __name__ == "__main__":
    # Create loader
    train_ds = SleepDataset("./processed")
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SleepStageCNN().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(10):
        model.train()
        total_loss = 0

        for X, y in train_loader:
            X, y = X.to(device), y.to(device)

            optimizer.zero_grad()
            logits = model(X)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}: Loss = {total_loss / len(train_loader):.4f}")

    torch.save(model.state_dict(), "sleep_stage_cnn.pth")
    print("Saved sleep_stage_cnn.pth")
