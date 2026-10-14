"""
Derives an empirical background-suppression threshold for the no_tumor class,
instead of picking an arbitrary cutoff (your own stated discipline: no
pre-picked number like "<=15%" — the threshold comes from YOUR validation
set's actual activation distribution).

WHAT THIS NEEDS THAT I DON'T HAVE HERE: your trained model and your BRISC
validation split, both of which live in your Kaggle environment. This script
is written to run there (or anywhere you have the checkpoint + val set
loaded) — I can't produce the actual threshold number myself, only the
method for deriving it honestly.

METHOD:
For every no_tumor validation image, run Grad-CAM++ and compute what
fraction of the heatmap is "active" (above a per-image Otsu threshold,
reusing spatial_features.py's logic) — this gives you a distribution of
activation-area percentages across true-negative scans. A well-behaved
no_tumor heatmap should have LOW activation area (little for Grad-CAM to
"explain" when there's nothing to classify). The threshold is then set at
a high percentile of that distribution (e.g. 95th), not an eyeballed number
— so "suppress/flag any no_tumor prediction whose own activation area
exceeds the 95th percentile of what normal no_tumor scans look like."

This also DOUBLES as a diagnostic: if the no_tumor activation-area
distribution turns out to be high/wide (meaning even correctly-classified
no_tumor scans show large "hot" regions), that itself is worth reporting as
a limitation — same spirit as your pituitary Pointing Game finding.
"""

import json
import numpy as np

from spatial_features import extract_spatial_features


def compute_activation_area_distribution(model, val_loader, gradcam_fn, class_names, device):
    """
    model: your trained model
    val_loader: your BRISC validation DataLoader (or any iterable of (image, label))
    gradcam_fn: your existing Grad-CAM++ function — must return a 2D heatmap
                array given (model, input_tensor, class_index)
    class_names: your CLASS_NAMES list, must include "no_tumor" / "notumor"
                 (check your actual label spelling — BRISC uses "notumor")

    Returns a list of activation-area percentages for correctly-classified
    no_tumor validation images.
    """
    no_tumor_label = None
    for i, name in enumerate(class_names):
        if "tumor" in name.lower() and "no" in name.lower():
            no_tumor_label = i
            break
    if no_tumor_label is None:
        raise ValueError(f"Couldn't find a no_tumor class in {class_names} — check spelling.")

    areas = []
    model.eval()
    for images, labels in val_loader:
        images = images.to(device)
        for i in range(images.shape[0]):
            if labels[i].item() != no_tumor_label:
                continue

            input_tensor = images[i:i + 1]
            heatmap, predicted_class_idx, _ = gradcam_fn(model, input_tensor, class_index=no_tumor_label)

            # only include scans the model actually got right — we're
            # characterizing "what does a correct no_tumor heatmap look
            # like," not polluting the distribution with misclassifications
            if predicted_class_idx != no_tumor_label:
                continue

            features = extract_spatial_features(heatmap, predicted_class="no_tumor")
            areas.append(features["relative_area_percentage"] if features else 0.0)

    return areas


def derive_threshold(areas: list, percentile: float = 95.0) -> dict:
    areas = np.array(areas)
    threshold = float(np.percentile(areas, percentile))

    return {
        "n_samples": len(areas),
        "percentile_used": percentile,
        "derived_threshold_pct": round(threshold, 2),
        "distribution_summary": {
            "mean": round(float(areas.mean()), 2),
            "median": round(float(np.median(areas)), 2),
            "std": round(float(areas.std()), 2),
            "min": round(float(areas.min()), 2),
            "max": round(float(areas.max()), 2),
        },
    }


if __name__ == "__main__":
    # Fill these in with your actual objects from the training notebook:
    #
    # from model_inference import load_model
    # model = load_model()
    # val_loader = ...               # your existing BRISC val DataLoader
    # from your_gradcam_module import make_gradcam_plus_plus as gradcam_fn
    # class_names = ["glioma", "meningioma", "notumor", "pituitary"]  # match your actual encoding
    #
    # areas = compute_activation_area_distribution(model, val_loader, gradcam_fn, class_names, device)
    # result = derive_threshold(areas)
    # print(json.dumps(result, indent=2))
    #
    # The printed "derived_threshold_pct" is the number to put in config.py /
    # your deferral logic as NO_TUMOR_ACTIVATION_THRESHOLD — not an arbitrary
    # pick.
    print(__doc__)
