import numpy as np
import torch
from Other.model_def import SleepStageCNN   # same class used during training

# Load model
model = SleepStageCNN()
model.load_state_dict(torch.load("sleep_stage_cnn.pth", map_location="cpu"))
# Almost like activating "test" mode for the model - disables dropout and updating of weights during test phase
model.eval()

data = np.load("test_data/SN005.npz")

X = torch.tensor(data["X"], dtype=torch.float32)   # (num_epochs, 4, 7680)
y = data["y"]

with torch.no_grad():
    logits = model(X)
    preds = torch.argmax(logits, dim=1)

print("PREDICTIONS PER EPOCH:")
print(preds.numpy())

print("\nGROUND TRUTH:")
print(y)

print("\nAccuracy:", (preds.numpy() == y).mean())