"""
Converts a Grad-CAM++ heatmap into the structured spatial_features your
JSON contract expects: laterality, anatomical_zone, relative_area_percentage,
boundary_type.

HONEST CAVEATS — read these before trusting the output:

1. This describes WHERE THE HEATMAP ACTIVATED, not a verified tumor
   location. For pituitary specifically, your own Pointing Game evaluation
   found the heatmap rarely lands on the true tumor region (strict accuracy
   0.57%) — spatial_features for pituitary should be treated as unreliable,
   and the synthesizer prompt already says so explicitly.

2. "anatomical_zone" here is a coarse quadrant heuristic (anterior/posterior
   x superior/inferior), NOT a true anatomical-atlas lobe label. Going from
   a single 2D MRI slice's pixel coordinates to a real lobe name (frontal
   vs. temporal vs. parietal) properly requires registration against a
   brain atlas, which is out of scope here. Calling this "frontal lobe"
   would be overclaiming precision you don't have — the WHO paper excerpts
   this feeds into just need a rough zone, so a quadrant label is what this
   returns. If you want to upgrade this, say so and we can look at a
   lightweight atlas-registration approach, but don't present quadrant
   output as a verified anatomical lobe in the paper.

3. "laterality" (left/right) depends entirely on your preprocessing
   convention — whether images were loaded in radiological convention
   (patient's left appears on the image's right side, standard in clinical
   practice) or neurological convention (image left = patient left). This
   script assumes NEITHER and just reports which side of the IMAGE the
   activation is on ("image-left" / "image-right"), not a clinical
   left/right claim. If your dataset's convention is known, map it
   explicitly rather than assuming — get this wrong and you have a
   confidently incorrect clinical claim in the report, which is worse than
   no claim at all.

Usage: call extract_spatial_features(heatmap, image_shape) with the same
heatmap array your Grad-CAM++ code already produces (2D numpy array, values
roughly 0-1, same aspect ratio as the input image — resize first if needed).
"""

import cv2
import numpy as np


def extract_spatial_features(heatmap: np.ndarray, predicted_class: str) -> dict | None:
    """
    heatmap: 2D numpy array (H, W), float values (any range — will be normalized).
    predicted_class: used only to attach the pituitary unreliability flag.

    Returns None if no clear activation region is found (e.g. a near-uniform
    heatmap) rather than guessing — a null spatial_features is honest, a
    fabricated one is not.
    """
    h, w = heatmap.shape

    # normalize to 0-255 uint8 for Otsu
    norm = heatmap - heatmap.min()
    if norm.max() > 0:
        norm = (norm / norm.max() * 255).astype(np.uint8)
    else:
        return None  # flat heatmap, nothing to localize

    # Otsu adaptive threshold -> binary "high activation" mask
    _, binary = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    if area < 4:  # degenerate/noise-sized region, not worth reporting
        return None

    moments = cv2.moments(largest)
    if moments["m00"] == 0:
        return None
    cx = moments["m10"] / moments["m00"]
    cy = moments["m01"] / moments["m00"]

    # --- laterality: which side of the IMAGE, not a clinical claim (see caveat 3) ---
    laterality = "image-left" if cx < w / 2 else "image-right"
    if abs(cx - w / 2) < 0.08 * w:  # close enough to the midline to call it midline
        laterality = "midline"

    # --- anatomical_zone: coarse quadrant heuristic, NOT an atlas lobe label (see caveat 2) ---
    vertical = "superior" if cy < h / 2 else "inferior"
    horizontal = "anterior" if cx < w / 2 else "posterior"  # assumes a conventional axial/sagittal framing; verify against your actual scan orientation
    anatomical_zone = f"{vertical}-{horizontal} quadrant"

    # --- relative_area_percentage ---
    relative_area_percentage = round(100.0 * area / (h * w), 2)

    # --- boundary_type: solidity-based heuristic ---
    hull = cv2.convexHull(largest)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0
    boundary_type = "well-defined" if solidity >= 0.85 else "irregular"

    result = {
        "laterality": laterality,
        "anatomical_zone": anatomical_zone,
        "relative_area_percentage": relative_area_percentage,
        "boundary_type": boundary_type,
    }

    if predicted_class == "pituitary":
        result["reliability_warning"] = (
            "Grad-CAM++ localization is empirically unreliable for pituitary "
            "in this project's own evaluation (Pointing Game strict accuracy "
            "0.57%). Treat these spatial_features as low-confidence."
        )

    return result


if __name__ == "__main__":
    # smoke test with a synthetic heatmap — NOT a substitute for testing on a real one
    h, w = 300, 300
    fake_heatmap = np.zeros((h, w), dtype=np.float32)
    fake_heatmap[80:140, 180:260] = 1.0  # a synthetic "hot" blob, upper-right-ish
    import json

    features = extract_spatial_features(fake_heatmap, predicted_class="meningioma")
    print(json.dumps(features, indent=2))
