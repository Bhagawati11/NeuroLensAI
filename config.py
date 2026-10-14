"""
Single place to put the numbers specific to YOUR trained run. Your team's
deferral threshold and temperature are measured per-training-run (you noted
training wasn't seeded until you added set_full_seed(), so these will drift
if you retrain) — update these three constants after your final run, don't
bury them inside model_inference.py.
"""

CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]  # must match your training label order exactly

CHECKPOINT_PATH = "best_efficientnet_b3.pth"

TEMPERATURE = 1.2402       # fitted Temperature Scaling value (BRISC val) — update if retrained
THETA_STAR = 0.0669        # frozen deferral threshold (hybrid risk/95th-pct rule) — update if retrained

MC_DROPOUT_PASSES = 20
IMAGE_SIZE = 300           # EfficientNet-B3 native input
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
