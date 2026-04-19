import argparse
import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import os
import random
import numpy as np

from preprocess import train_transform, CLASSES
from logger import get_logger

logger = get_logger("train")

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Model Definition
class MNISTClassifier(nn.Module):
    def __init__(self):
        super(MNISTClassifier, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = x.float()
        return self.fc(self.conv(x))


# Helpers 
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct = 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
    return total_loss / len(loader.dataset), correct / len(loader.dataset)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
    return total_loss / len(loader.dataset), correct / len(loader.dataset)


def save_prediction_plot(model, dataset, device, path="prediction_plot.png"):
    """Save a sample prediction plot as an MLflow artifact."""
    model.eval()
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    fig.suptitle("Sample Predictions", fontsize=14)
    with torch.no_grad():
        for i, ax in enumerate(axes.flat):
            img, label = dataset[i]
            output = model(img.unsqueeze(0).to(device))
            pred = output.argmax(1).item()
            ax.imshow(img.squeeze(), cmap="gray")
            color = "green" if pred == label else "red"
            ax.set_title(f"GT:{label} P:{pred}", color=color)
            ax.axis("off")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    logger.info(f"Prediction plot saved to {path}")
    return path


# Main Training Function 
def train(learning_rate, batch_size, epochs):
    device = torch.device("cpu")
    logger.info("Starting training...")
    logger.info(f"Config → lr={learning_rate}, batch_size={batch_size}, epochs={epochs}")

    # Data 
    logger.info("Loading MNIST dataset...")
    full_dataset = datasets.MNIST(
        root="data", train=True, download=True, transform=train_transform
    )
    test_dataset = datasets.MNIST(
        root="data", train=False, download=True, transform=train_transform
    )

    train_size = int(0.8 * len(full_dataset))
    val_size   = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    logger.info(f"Dataset split → Train: {train_size}, Val: {val_size}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)

    # Model, Loss, Optimizer 
    model     = MNISTClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    logger.info("Model, criterion and optimizer initialized.")

    # MLflow Run 
    mlflow.set_experiment("MNIST-Classifier")
    logger.info("MLflow experiment set: MNIST-Classifier")

    with mlflow.start_run(run_name=f"lr{learning_rate}_bs{batch_size}_ep{epochs}") as run:
        logger.info(f"MLflow Run ID: {run.info.run_id}")

        # Log Parameters
        mlflow.log_param("learning_rate",      learning_rate)
        mlflow.log_param("batch_size",         batch_size)
        mlflow.log_param("epochs",             epochs)
        mlflow.log_param("model_architecture", "CNN-2Conv-2FC")
        mlflow.log_param("optimizer",          "Adam")
        logger.info("Parameters logged to MLflow.")

        # Training Loop
        for epoch in range(1, epochs + 1):
            train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
            val_loss,   val_acc   = evaluate(model, val_loader, criterion, device)

            # Log Metrics
            mlflow.log_metric("train_loss",     train_loss, step=epoch)
            mlflow.log_metric("train_accuracy", train_acc,  step=epoch)
            mlflow.log_metric("val_loss",       val_loss,   step=epoch)
            mlflow.log_metric("val_accuracy",   val_acc,    step=epoch)

            logger.info(
                f"Epoch {epoch}/{epochs} | "
                f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}"
            )

        # Log Artifacts
        logger.info("Saving prediction plot artifact...")
        plot_path = save_prediction_plot(model, val_dataset, device)
        mlflow.log_artifact(plot_path)
        os.remove(plot_path)

        # Log Model
        logger.info("Logging model to MLflow...")
        mlflow.pytorch.log_model(model, artifact_path="model")

        logger.info("Training complete. Model and artifacts saved to MLflow.")
        logger.info(f"Run ID: {run.info.run_id}")


# Entry Point 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--batch_size",    type=int,   default=64)
    parser.add_argument("--epochs",        type=int,   default=5)
    args = parser.parse_args()

    train(args.learning_rate, args.batch_size, args.epochs)