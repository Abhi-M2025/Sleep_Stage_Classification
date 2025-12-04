import os
import glob
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

from Model_CNN.model_def import SleepStageCNN
from Model_MLP.model_def import SleepStageMLP_ReLU, SleepStageMLP_LeakyReLU, SleepStageMLP_Tanh

def load_testset(directory="./test_data"):
    files = glob.glob(os.path.join(directory, "*.npz"))
    X_all, Y_all = [], []

    for f in files:
        data = np.load(f)
        X_all.append(data["X"])   # (epochs, 4, 7680)
        Y_all.append(data["y"])   

    # Concatenate for consistent shapes
    X_raw = np.concatenate(X_all, axis=0)
    y = np.concatenate(Y_all, axis=0)

    # Normalize raw input for CNN
    mean = X_raw.mean(axis=-1, keepdims=True)
    std = X_raw.std(axis=-1, keepdims=True) + 1e-8
    X_raw = (X_raw - mean) / std

    # Downsample for MLP
    num_epochs, num_channels, num_samples = X_raw.shape
    X_down = X_raw.reshape(num_epochs, num_channels, num_samples//16, 16).mean(-1)  # (N, 4, 480)

    # Flatten for MLP
    X_mlp = X_down.reshape(num_epochs, -1)  # (N, 1920)

    # Convert to tensor
    X_cnn = torch.tensor(X_raw, dtype=torch.float32)
    X_mlp = torch.tensor(X_mlp, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)

    return X_cnn, X_mlp, y

def evaluate_model(model, X, y):
    criterion = nn.CrossEntropyLoss()
    
    model.eval()
    with torch.no_grad():
        logits = model(X)
        loss = criterion(logits, y).item()
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean().item()
    
    return loss, acc, preds.numpy()

if __name__ == "__main__":
    X_cnn, X_mlp, y = load_testset("./processed_test_data")

    input_dim_MLP = X_mlp.shape[1]
    num_classes = 5

    models_to_test = {
        "CNN": SleepStageCNN(),
        "MLP_ReLU": SleepStageMLP_ReLU(input_dim_MLP, num_classes, 0.5),
        "MLP_Tanh": SleepStageMLP_Tanh(input_dim_MLP, num_classes, 0.5),
        "MLP_LeakyReLU": SleepStageMLP_LeakyReLU(input_dim_MLP, num_classes, 0.5),
    }

    # Load saved weights
    models_to_test["CNN"].load_state_dict(torch.load("sleep_stage_cnn.pth"))
    models_to_test["MLP_ReLU"].load_state_dict(torch.load("sleep_stage_mlp_ReLU.pth"))
    models_to_test["MLP_Tanh"].load_state_dict(torch.load("sleep_stage_mlp_tanh.pth"))
    models_to_test["MLP_LeakyReLU"].load_state_dict(torch.load("sleep_stage_mlp_LeakyReLU.pth"))

    # Store results
    losses = {}
    accuracies = {}
    predictions = {}

    for name, model in models_to_test.items():
        print(f"Evaluating {name}...")

        X_input = X_cnn if name == "CNN" else X_mlp

        loss, acc, preds = evaluate_model(model, X_input, y)

        losses[name] = loss
        accuracies[name] = acc
        predictions[name] = preds

        print(f"  Loss = {loss:.4f}")
        print(f"  Acc  = {acc:.4f}")
        print("-" * 40)

    # Plot Accuracy line chart
    plt.figure(figsize=(8,5))
    plt.plot(list(accuracies.keys()), list(accuracies.values()), marker='o', linestyle='-', linewidth=2)
    plt.ylabel("Accuracy")
    plt.title("Accuracy Comparison")
    plt.ylim(0, 1.0)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.show()

    # Confusion matrices
    for name in models_to_test:
        cm = confusion_matrix(y, predictions[name])
        disp = ConfusionMatrixDisplay(cm)
        disp.plot(values_format="d", cmap="Blues")
        plt.title(f"Confusion Matrix: {name}")
        plt.show()

    # Classification report metrics for line charts
    all_metrics = {}
    for name in models_to_test:
        report = classification_report(y.numpy(), predictions[name], output_dict=True, zero_division=0)
        all_metrics[name] = report

    model_names = list(models_to_test.keys())
    metrics_to_plot = ["precision", "recall", "f1-score"]

    # Macro-average metrics
    for metric in metrics_to_plot:
        plt.figure(figsize=(8,5))
        values = [all_metrics[name]["macro avg"][metric] for name in model_names]
        plt.plot(model_names, values, marker='o', linestyle='-', linewidth=2)
        plt.ylim(0, 1.0)
        plt.ylabel(metric.capitalize())
        plt.title(f"Macro-average {metric.capitalize()} Comparison")
        plt.grid(True, linestyle="--", alpha=0.3)
        plt.show()

    # Per-class F1-score
    num_classes = len(all_metrics[model_names[0]]) - 3  # exclude 'accuracy', 'macro avg', 'weighted avg'
    classes = [str(i) for i in range(num_classes)]
    for cls in classes:
        plt.figure(figsize=(8,5))
        values = [all_metrics[name][cls]["f1-score"] for name in model_names]
        plt.plot(model_names, values, marker='o', linestyle='-', linewidth=2)
        plt.ylim(0, 1.0)
        plt.ylabel("F1-score")
        plt.title(f"Class {cls} F1-score Comparison")
        plt.grid(True, linestyle="--", alpha=0.3)
        plt.show()
