import os
from typing import List, Tuple
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms

from models.face_parser.bisenet import BiSeNet

# eyes, eyebrows, nose, lips and teeth
FACE_COMPONENTS = (2, 3, 4, 5, 10, 11, 12, 13)
MODEL_NAME = "resnet34"
CHECKPOINT_PATH = "checkpoints/face_parser/%s.pt" % MODEL_NAME


def prepare_image(
    image: Image.Image, input_size: Tuple[int, int] = (512, 512)
) -> torch.Tensor:
    """
    Prepare an image for inference by resizing and normalizing it.

    Args:
        image: PIL Image to process
        input_size: Target size for resizing

    Returns:
        torch.Tensor: Preprocessed image tensor ready for model input
    """
    # Resize the image
    resized_image = image.resize(input_size, resample=Image.BILINEAR)

    # Define transformation pipeline
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    # Apply transformations
    image_tensor = transform(resized_image)
    image_batch = image_tensor.unsqueeze(0)

    return image_batch


def load_model(
    model_name: str, num_classes: int, weight_path: str, device: torch.device
) -> torch.nn.Module:
    """
    Load and initialize the BiSeNet model.

    Args:
        model_name: Name of the backbone model (e.g., "resnet18")
        num_classes: Number of segmentation classes
        weight_path: Path to the model weights file
        device: Device to load the model onto

    Returns:
        torch.nn.Module: Initialized and loaded model
    """
    model = BiSeNet(num_classes, backbone_name=model_name)
    model.to(device)

    if os.path.exists(weight_path):
        model.load_state_dict(torch.load(weight_path, map_location=device))
    else:
        raise ValueError(f"Weights not found from given path ({weight_path})")

    model.eval()
    return model


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model(MODEL_NAME, 19, CHECKPOINT_PATH, device)


def extract_masks(image: Image.Image):
    original_size = image.size  # (width, height)
    image_batch = prepare_image(image).to(device)

    output = model(image_batch)[0]  # use feat_out for inference only

    predicted_mask = output.squeeze(0).detach().cpu().numpy().argmax(0)
    masks = []

    for idx in FACE_COMPONENTS:
        mask = Image.fromarray((predicted_mask == idx).astype(np.uint8))
        mask = mask.resize(original_size, resample=Image.NEAREST)
        mask = np.array(mask)[..., np.newaxis]

        masks.append(mask)

    return masks
