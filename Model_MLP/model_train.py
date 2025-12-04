import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from model_def import SleepStageMLP_ReLU, SleepStageMLP_LeakyReLU, SleepStageMLP_Tanh  # your MLP class
import glob
import os

class SleepDataset(Dataset):
    def __init__(self, directory="../processed"):
        self.files = glob.glob(os.path.join(directory, "*.npz"))
        self.X = []
        self.Y = []

        for f in self.files:
            data = np.load(f)
            self.X.append(data["X"])
            self.Y.append(data["y"])

        # combine all subjects
        print("start concatenating")
        self.X = np.concatenate(self.X, axis=0)
        self.Y = np.concatenate(self.Y, axis=0)
        print("finish concatenating")

        # flatten for MLP: (num_epochs, channels * samples)
        num_epochs, num_channels, num_samples = self.X.shape
        self.X = self.X.reshape(num_epochs, num_channels, num_samples//16, 16).mean(-1)  # (N, 4, 480)
        self.X = self.X.reshape(num_epochs, -1)
        # convert to tensors
        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.Y = torch.tensor(self.Y, dtype=torch.long)

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

if __name__ == "__main__":
    # Create dataset and loader
    train_ds = SleepDataset("../processed_train_data")
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize MLP
    input_size = train_ds.X.shape[1]  # 4*7680 = 30720
    print(f"input_size: {input_size}")
    num_classes = len(np.unique(train_ds.Y))  # should be 5
    model = SleepStageMLP_Tanh(input_dim=input_size, num_of_classes=num_classes, dropout=0.5).to(device)

    # Optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    # Training
    print("start training")
    for epoch in range(75):
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

    # Save model
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/sleep_stage_mlp.pth")
    print("Saved models/sleep_stage_mlp.pth")
