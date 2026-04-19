import torch
from torchvision import transforms
from PIL import Image


# Constants 
IMG_SIZE = 28          # MNIST images are 28x28
MEAN     = (0.1307,)   # MNIST dataset mean
STD      = (0.3081,)   # MNIST dataset std

CLASSES  = [str(i) for i in range(10)]  # ['0','1',...,'9']


# Transform for Training 
train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # ensure single channel
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD)
])


# Transform for Inference (uploaded image) 
inference_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD)
])


def preprocess_image(image_path: str) -> torch.Tensor:
    """
    Load an image from disk and convert it to a
    model-ready tensor of shape (1, 1, 28, 28).
    The batch dimension (1) is added for inference.
    """
    img = Image.open(image_path).convert("L")   # convert to grayscale
    tensor = inference_transform(img)            # shape: (1, 28, 28)
    tensor = tensor.unsqueeze(0)                 # shape: (1, 1, 28, 28)
    return tensor


def tensor_to_list(tensor: torch.Tensor) -> list:
    """
    Convert a tensor to a nested Python list.
    Used when sending data to the MLflow REST API.
    """
    return tensor.float().numpy().tolist()  