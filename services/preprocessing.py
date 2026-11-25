"""
Image Preprocessing and Augmentation Service for Skin Lesion Analysis.
Supports:
- Image validation and decoding (file paths, byte streams, numpy arrays).
- RGB colour conversion.
- Hair artefact removal using DullRazor (Morphological BlackHat + inpainting).
- Pretrained Vision Transformer (ViT-B/16) input preprocessing (224x224, ImageNet normalization).
- U-Net segmentation preprocessing (256x256, nearest-neighbor mask interpolation, binary normalization).
- Coordinated training augmentations for images and masks (identical geometric transforms).
"""

import os
from pathlib import Path
from typing import Tuple, Optional, Union
import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF

# Standard ImageNet normalization parameters (required by pretrained ViT-B/16)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

CLASSIFICATION_IMAGE_SIZE = (224, 224)
SEGMENTATION_IMAGE_SIZE = (256, 256)


def validate_and_decode_image(
    image_input: Union[str, Path, bytes, np.ndarray]
) -> np.ndarray:
    """
    Validates and decodes input into a 3-channel RGB numpy uint8 image.
    Rejects corrupted images, empty files, or unsupported formats.
    """
    if isinstance(image_input, (str, Path)):
        path_str = str(image_input)
        if not os.path.exists(path_str):
            raise FileNotFoundError(f"Image file does not exist: {path_str}")
        bgr = cv2.imread(path_str, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"Failed to decode or corrupted image file: {path_str}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return rgb

    elif isinstance(image_input, (bytes, bytearray)):
        if len(image_input) == 0:
            raise ValueError("Empty image byte buffer provided.")
        np_arr = np.frombuffer(image_input, np.uint8)
        bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("Failed to decode image from byte buffer.")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return rgb

    elif isinstance(image_input, np.ndarray):
        if image_input.size == 0:
            raise ValueError("Empty numpy image array provided.")
        if image_input.ndim == 2:
            return cv2.cvtColor(image_input, cv2.COLOR_GRAY2RGB)
        elif image_input.ndim == 3:
            if image_input.shape[2] == 4:
                return cv2.cvtColor(image_input, cv2.COLOR_RGBA2RGB)
            elif image_input.shape[2] == 3:
                return image_input.copy()
        raise ValueError(f"Invalid image array shape: {image_input.shape}")

    raise TypeError(f"Unsupported image input type: {type(image_input)}")


def validate_and_decode_mask(
    mask_input: Union[str, Path, bytes, np.ndarray]
) -> np.ndarray:
    """
    Validates and decodes an expert segmentation mask into a single-channel uint8 binary mask (0 or 255).
    """
    if isinstance(mask_input, (str, Path)):
        path_str = str(mask_input)
        if not os.path.exists(path_str):
            raise FileNotFoundError(f"Mask file does not exist: {path_str}")
        mask = cv2.imread(path_str, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Failed to decode or corrupted mask file: {path_str}")
    elif isinstance(mask_input, (bytes, bytearray)):
        np_arr = np.frombuffer(mask_input, np.uint8)
        mask = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError("Failed to decode mask from byte buffer.")
    elif isinstance(mask_input, np.ndarray):
        if mask_input.ndim == 3:
            mask = cv2.cvtColor(mask_input, cv2.COLOR_RGB2GRAY)
        else:
            mask = mask_input.copy()
    else:
        raise TypeError(f"Unsupported mask input type: {type(mask_input)}")

    # Ensure binary values (0 and 255)
    binary_mask = np.where(mask > 127, 255, 0).astype(np.uint8)
    return binary_mask


def remove_hair_artefacts(
    rgb_image: np.ndarray,
    kernel_size: int = 17,
    threshold: int = 10
) -> np.ndarray:
    """
    DullRazor algorithm for dermatological hair artefact removal:
    1. Converts image to grayscale.
    2. Computes Morphological BlackHat transform to isolate dark linear hair strands.
    3. Thresholds the result to create a binary hair mask.
    4. Inpaints hair regions using fast marching method (Telea algorithm).
    """
    gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    # Structuring element with cross/ellipse kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    
    # Threshold to isolate hair strands
    _, hair_mask = cv2.threshold(blackhat, threshold, 255, cv2.THRESH_BINARY)
    
    # Convert RGB to BGR for OpenCV inpainting, then back to RGB
    bgr = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    inpainted_bgr = cv2.inpaint(bgr, hair_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    return cv2.cvtColor(inpainted_bgr, cv2.COLOR_BGR2RGB)


def apply_training_augmentation(
    rgb_image: np.ndarray,
    mask: Optional[np.ndarray] = None,
    p_flip_h: float = 0.5,
    p_flip_v: float = 0.5,
    max_rotation_deg: float = 15.0,
    brightness_range: Tuple[float, float] = (0.9, 1.1),
    contrast_range: Tuple[float, float] = (0.9, 1.1)
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Applies moderate training augmentations.
    When a segmentation mask is provided, identical geometric transformations are applied
    simultaneously to both the image and the mask.
    """
    augmented_img = rgb_image.copy()
    augmented_mask = mask.copy() if mask is not None else None

    # 1. Horizontal Flip
    if np.random.rand() < p_flip_h:
        augmented_img = cv2.flip(augmented_img, 1)
        if augmented_mask is not None:
            augmented_mask = cv2.flip(augmented_mask, 1)

    # 2. Vertical Flip
    if np.random.rand() < p_flip_v:
        augmented_img = cv2.flip(augmented_img, 0)
        if augmented_mask is not None:
            augmented_mask = cv2.flip(augmented_mask, 0)

    # 3. Small Rotation
    if max_rotation_deg > 0:
        angle = np.random.uniform(-max_rotation_deg, max_rotation_deg)
        h, w = augmented_img.shape[:2]
        center = (w // 2, h // 2)
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        augmented_img = cv2.warpAffine(
            augmented_img, rot_mat, (w, h),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101
        )
        if augmented_mask is not None:
            augmented_mask = cv2.warpAffine(
                augmented_mask, rot_mat, (w, h),
                flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0
            )

    # 4. Photometric adjustments (applied to image only, never to mask)
    b_factor = np.random.uniform(brightness_range[0], brightness_range[1])
    c_factor = np.random.uniform(contrast_range[0], contrast_range[1])
    img_float = augmented_img.astype(np.float32) * c_factor * b_factor
    augmented_img = np.clip(img_float, 0, 255).astype(np.uint8)

    return augmented_img, augmented_mask


def preprocess_for_classification(
    image_input: Union[str, Path, bytes, np.ndarray],
    apply_hair_removal: bool = False,
    is_training: bool = False
) -> torch.Tensor:
    """
    Full preprocessing pipeline for Vision Transformer (ViT-B/16) classification:
    1. Validation and decoding.
    2. Optional DullRazor hair removal.
    3. Optional training augmentation.
    4. Resize to 224x224.
    5. Normalization with ImageNet mean and standard deviation.
    6. Returns PyTorch float32 tensor of shape (3, 224, 224).
    """
    rgb = validate_and_decode_image(image_input)

    if apply_hair_removal:
        rgb = remove_hair_artefacts(rgb)

    if is_training:
        rgb, _ = apply_training_augmentation(rgb, mask=None)

    # Resize to 224x224
    resized = cv2.resize(rgb, CLASSIFICATION_IMAGE_SIZE, interpolation=cv2.INTER_AREA)

    # Convert to float tensor and normalize
    tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
    mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(3, 1, 1)
    normalized = (tensor - mean) / std

    return normalized


def preprocess_for_segmentation(
    image_input: Union[str, Path, bytes, np.ndarray],
    mask_input: Optional[Union[str, Path, bytes, np.ndarray]] = None,
    apply_hair_removal: bool = False,
    is_training: bool = False
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Full preprocessing pipeline for U-Net lesion segmentation:
    1. Validation and decoding of image (and optional mask).
    2. Optional DullRazor hair removal.
    3. Optional training augmentation (coordinated between image and mask).
    4. Image resized to 256x256 (bilinear/area); Mask resized to 256x256 using nearest-neighbor interpolation.
    5. Image normalized using ImageNet parameters (shape: 3, 256, 256).
    6. Mask normalized to binary [0.0, 1.0] (shape: 1, 256, 256).
    """
    rgb = validate_and_decode_image(image_input)
    mask = validate_and_decode_mask(mask_input) if mask_input is not None else None

    if apply_hair_removal:
        rgb = remove_hair_artefacts(rgb)

    if is_training and mask is not None:
        rgb, mask = apply_training_augmentation(rgb, mask=mask)
    elif is_training and mask is None:
        rgb, _ = apply_training_augmentation(rgb, mask=None)

    # Resize image to 256x256
    resized_img = cv2.resize(rgb, SEGMENTATION_IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    img_tensor = torch.from_numpy(resized_img).permute(2, 0, 1).float() / 255.0
    mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(3, 1, 1)
    normalized_img = (img_tensor - mean) / std

    if mask is not None:
        # Resize mask to 256x256 with nearest-neighbor interpolation
        resized_mask = cv2.resize(mask, SEGMENTATION_IMAGE_SIZE, interpolation=cv2.INTER_NEAREST)
        mask_binary = (resized_mask > 127).astype(np.float32)
        mask_tensor = torch.from_numpy(mask_binary).unsqueeze(0)  # Shape (1, 256, 256)
        return normalized_img, mask_tensor

    return normalized_img
