import numpy as np
import torch
from model_def import SleepStageMLP  # replace with your MLP class


data = np.load("test_data/SN005.npz")
X_np = data["X"]  # shape: (num_epochs, 4, 7680)
y = data["y"]

# Flatten for MLP
num_epochs, num_channels, num_samples = X_np.shape
X_np = X_np.reshape(num_epochs, num_channels * num_samples)

X = torch.tensor(X_np, dtype=torch.float32)

input_size = X.shape[1]
num_classes = len(np.unique(y))

model = SleepStageMLP(input_dim=input_size, num_of_classes=num_classes, dropout=0.0)
model.load_state_dict(torch.load("sleep_stage_mlp.pth", map_location="cpu"))
model.eval()  # disables dropout, no weight updates


with torch.no_grad():
    logits = model(X)
    preds = torch.argmax(logits, dim=1)

# -------------------------
# Accuracy
# -------------------------
accuracy = (preds.numpy() == y).mean()
print("PREDICTIONS PER EPOCH:")
print(preds.numpy())

print("\nGROUND TRUTH:")
print(y)

print("\nAccuracy:", accuracy)

# -------------------------
# Confusion matrix and classification report
# -------------------------
from sklearn.metrics import confusion_matrix, classification_report

cm = confusion_matrix(y, preds.numpy())
print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(y, preds.numpy(), target_names=["W","N1","N2","N3","R"]))