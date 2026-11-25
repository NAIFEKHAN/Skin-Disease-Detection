"""
Lightweight Image Quality Assessment Service.
Evaluates sharpness (Laplacian variance), illumination (darkness/glare),
and resolution using OpenCV heuristic checks.

NOTE: This is a technical heuristic checker for camera/image capture artifacts.
It does NOT validate clinical suitability or histological diagnostic quality.
"""

from typing import Dict, Any, Union
from pathlib import Path
import cv2
import numpy as np


class ImageQualityChecker:
    """
    Evaluates image quality metrics to provide prompt, constructive feedback
    to users uploading dermoscopic or smartphone skin lesion photos.
    """

    def __init__(
        self,
        blur_threshold: float = 60.0,
        dark_threshold: float = 38.0,
        bright_threshold: float = 222.0,
        min_dimension: int = 180,
    ):
        self.blur_threshold = blur_threshold
        self.dark_threshold = dark_threshold
        self.bright_threshold = bright_threshold
        self.min_dimension = min_dimension

    def assess(self, image_input: Union[str, Path, bytes, np.ndarray]) -> Dict[str, Any]:
        """
        Assesses an image across sharpness, brightness balance, and resolution.

        Args:
            image_input: File path, raw bytes, or decoded numpy BGR/RGB array.

        Returns:
            Structured dictionary with quality status, score metrics, and advice.
        """
        # 1. Decode image safely
        img_bgr = None
        if isinstance(image_input, (str, Path)):
            path_str = str(image_input)
            img_bgr = cv2.imread(path_str)
            if img_bgr is None:
                return {
                    "is_acceptable": False,
                    "has_warnings": True,
                    "status": "error",
                    "message": "Unable to read or decode image file. File may be corrupted or in an unsupported format.",
                    "details": {},
                    "warnings": ["Corrupted or invalid image file"],
                    "suggestions": ["Please select a genuine JPEG or PNG image."]
                }
        elif isinstance(image_input, bytes):
            nparr = np.frombuffer(image_input, np.uint8)
            img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img_bgr is None:
                return {
                    "is_acceptable": False,
                    "has_warnings": True,
                    "status": "error",
                    "message": "Failed to decode image from uploaded bytes. Please upload a valid JPG or PNG file.",
                    "details": {},
                    "warnings": ["Failed image decoding"],
                    "suggestions": ["Re-export or choose a standard image format."]
                }
        elif isinstance(image_input, np.ndarray):
            img_bgr = image_input
            if len(img_bgr.shape) == 2:
                img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
        else:
            return {
                "is_acceptable": False,
                "has_warnings": True,
                "status": "error",
                "message": f"Unsupported input type: {type(image_input)}",
                "details": {},
                "warnings": ["Unsupported input format"],
                "suggestions": []
            }

        height, width = img_bgr.shape[:2]
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # 2. Metric Calculations
        # A. Sharpness / Blur via Laplacian variance
        laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        
        # B. Illumination / Brightness
        mean_brightness = float(np.mean(gray))

        # C. Glare / Highlight saturation ratio
        bright_pixels = np.count_nonzero(gray >= 250)
        glare_ratio = float(bright_pixels / (height * width))

        warnings = []
        suggestions = []

        # 3. Heuristic Evaluations
        # Resolution
        if height < self.min_dimension or width < self.min_dimension:
            warnings.append(f"Low resolution: image dimensions ({width}×{height} px) are smaller than recommended ({self.min_dimension}×{self.min_dimension} px).")
            suggestions.append("Upload or capture a higher resolution image for clearer lesion inspection.")

        # Blur
        if laplacian_var < self.blur_threshold:
            warnings.append("Image appears blurry or out of focus.")
            suggestions.append("Hold the camera steady or retake the photo with autofocus centered on the skin lesion.")

        # Darkness
        if mean_brightness < self.dark_threshold:
            warnings.append("Image lighting is insufficient (too dark).")
            suggestions.append("Ensure the area is well lit with natural or soft direct lighting.")
        elif mean_brightness > self.bright_threshold or glare_ratio > 0.15:
            warnings.append("Image exhibits excessive brightness or glare reflection.")
            suggestions.append("Reduce camera flash reflection or avoid strong direct glare on the skin.")

        # 4. Status Determination
        has_warnings = len(warnings) > 0
        if not has_warnings:
            status = "acceptable"
            message = "Image quality is good. Sharpness and illumination are suitable for analysis."
        else:
            status = "warning"
            message = "Quality warning: " + "; ".join(warnings)

        return {
            "is_acceptable": True,  # Quality warnings allow continuing with user consent
            "has_warnings": has_warnings,
            "status": status,
            "message": message,
            "warnings": warnings,
            "suggestions": suggestions,
            "details": {
                "width": int(width),
                "height": int(height),
                "sharpness_score": round(laplacian_var, 2),
                "brightness_score": round(mean_brightness, 2),
                "glare_percentage": round(glare_ratio * 100, 2),
                "thresholds": {
                    "min_sharpness": self.blur_threshold,
                    "min_brightness": self.dark_threshold,
                    "max_brightness": self.bright_threshold,
                    "min_dimension": self.min_dimension
                }
            },
            "disclaimer": "Technical heuristic quality check only; does not validate dermatological diagnostic suitability."
        }


# Global singleton instance
_quality_checker = ImageQualityChecker()


def assess_image_quality(image_input: Union[str, Path, bytes, np.ndarray]) -> Dict[str, Any]:
    """Convenience helper for assessing image quality."""
    return _quality_checker.assess(image_input)
