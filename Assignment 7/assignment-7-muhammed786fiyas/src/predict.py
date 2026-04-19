import torch
import torch.nn as nn
import mlflow.pytorch
from logger import get_logger

logger = get_logger("predict")


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
        return self.fc(self.conv(x))


def load_model(run_id: str) -> torch.nn.Module:
    """
    Load a trained model from MLflow using its run ID.
    """
    model_uri = f"runs:/{run_id}/model"
    logger.info(f"Loading model from: {model_uri}")
    model = mlflow.pytorch.load_model(model_uri)
    model.eval()
    logger.info("Model loaded successfully.")
    return model


def predict(model: torch.nn.Module, tensor: torch.Tensor) -> dict:
    """
    Run inference on a preprocessed tensor.
    Returns predicted class and confidence scores.

    Args:
        model: loaded MLflow pytorch model
        tensor: preprocessed image tensor of shape (1, 1, 28, 28)

    Returns:
        dict with predicted digit and confidence scores
    """
    with torch.no_grad():
        outputs = model(tensor)                          # raw logits (1, 10)
        probabilities = torch.softmax(outputs, dim=1)    # convert to probabilities
        predicted_class = probabilities.argmax(dim=1).item()
        confidence = probabilities.max().item()

    result = {
        "predicted_digit": predicted_class,
        "confidence": round(confidence * 100, 2),
        "all_probabilities": {
            str(i): round(probabilities[0][i].item() * 100, 2)
            for i in range(10)
        }
    }

    logger.info(f"Prediction: {predicted_class} (confidence: {confidence*100:.2f}%)")
    return result


# Quick Test 
if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.dirname(__file__))
    from preprocess import preprocess_image

    if len(sys.argv) < 3:
        print("Usage: python predict.py <run_id> <image_path>")
        sys.exit(1)

    run_id     = sys.argv[1]
    image_path = sys.argv[2]

    model  = load_model(run_id)
    tensor = preprocess_image(image_path)
    result = predict(model, tensor)

    print(f"\nPredicted Digit : {result['predicted_digit']}")
    print(f"Confidence      : {result['confidence']}%")
    print(f"All Scores      : {result['all_probabilities']}")