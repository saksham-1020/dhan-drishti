"""
config.py
=========
Single source of truth for the Dhan Drishti project.
Every other module (train_model.py, dhan_drishti_utils.py, app.py) imports
from here — never hard-code class names / paths / hyperparameters elsewhere.
"""

import os

# ------------------------------------------------------------------ #
# Paths
# ------------------------------------------------------------------ #
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "Datasets")
MODEL_DIR = os.path.join(BASE_DIR, "model")
GRADCAM_SAMPLE_DIR = os.path.join(MODEL_DIR, "gradcam_samples")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(GRADCAM_SAMPLE_DIR, exist_ok=True)

# ------------------------------------------------------------------ #
# Classes  (folder names inside Datasets/ MUST match these exactly)
# ------------------------------------------------------------------ #
CLASS_NAMES = [
    "background",
    "ten_new", "ten_old",
    "twenty_new", "twenty_old",
    "fifty_new", "fifty_old",
    "hundred_new", "hundred_old",
    "two_hundred",
    "five_hundred",
]

# Face value used by the assistive "Cash Tally" feature.
DENOMINATION_VALUE = {
    "background": 0,
    "ten_new": 10, "ten_old": 10,
    "twenty_new": 20, "twenty_old": 20,
    "fifty_new": 50, "fifty_old": 50,
    "hundred_new": 100, "hundred_old": 100,
    "two_hundred": 200,
    "five_hundred": 500,
}

IMG_SIZE = (224, 224)
SEED = 42

# ------------------------------------------------------------------ #
# Ensemble backbones
# Three architecturally-diverse CNNs are combined via a validation-
# accuracy-weighted soft vote. Diversity (depthwise-separable vs.
# inverted residual vs. ConvNeXt-style blocks) is what makes the
# ensemble reduce correlated errors — a single backbone's mistakes on
# worn/soiled notes are rarely repeated by all three at once.
# ------------------------------------------------------------------ #
BACKBONES = {
    "mobilenetv2": {
        "display_name": "MobileNetV2",
        "keras_app": "MobileNetV2",
        "file": "mobilenetv2.keras",
    },
    "efficientnetv2b0": {
        "display_name": "EfficientNetV2-B0",
        "keras_app": "EfficientNetV2B0",
        "file": "efficientnetv2b0.keras",
    },
    "convnexttiny": {
        "display_name": "ConvNeXt-Tiny",
        "keras_app": "ConvNeXtTiny",
        "file": "convnexttiny.keras",
    },
}

# ------------------------------------------------------------------ #
# Training hyperparameters (overridable via train_model.py CLI flags)
# ------------------------------------------------------------------ #
BATCH_SIZE = 32
HEAD_EPOCHS = 15          # phase 1: frozen backbone, train classification head
FINE_TUNE_EPOCHS = 15     # phase 2: unfreeze top layers, low LR fine-tune
HEAD_LR = 1e-3
FINE_TUNE_LR = 1e-5
FINE_TUNE_UNFREEZE_LAYERS = 40   # top-N layers unfrozen in phase 2
DROPOUT_RATE = 0.35              # also drives MC-Dropout uncertainty at inference
LABEL_SMOOTHING = 0.05
TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT = 0.70, 0.15, 0.15

# ------------------------------------------------------------------ #
# Heuristic screening thresholds (NOT certified authentication —
# always disclosed as such in the UI; see README "Ethical Disclaimer")
# ------------------------------------------------------------------ #
BLANK_SURFACE_STD_THRESHOLD = 12.0     # grayscale std-dev below this => reject as background
COLOR_SIGNATURE_DIST_THRESHOLD = 50.0  # Euclidean RGB distance flag
TEXTURE_SHARPNESS_THRESHOLD = 15.0     # Laplacian-variance flag (very blurry photocopy/screen reproduction)
ROI_CONSISTENCY_THRESHOLD = 0.35       # normalized security-thread ROI edge-density flag
MC_DROPOUT_PASSES = 20
MC_ENTROPY_REJECT_THRESHOLD = 1.1      # predictive entropy (nats) above which we ask user to re-scan

# Approximate mean RGB "centroid" of each note under diffuse indoor
# lighting, sampled from a small calibration set. Used only as a cheap,
# transparent secondary cue layered on top of the CNN — not a security
# feature and not a substitute for RBI's official verification steps.
RBI_COLOR_CENTROIDS = {
    "ten_new": [135, 105, 80], "ten_old": [180, 130, 90],
    "twenty_new": [180, 200, 120], "twenty_old": [200, 140, 100],
    "fifty_new": [100, 200, 220], "fifty_old": [210, 160, 210],
    "hundred_new": [160, 160, 200], "hundred_old": [150, 175, 185],
    "two_hundred": [220, 180, 80], "five_hundred": [150, 160, 150],
}

# Normalized (x0, y0, x1, y1) ROI, as a fraction of image width/height,
# approximating where the windowed security thread / see-through
# register typically sits on a *centered, front-facing* note crop.
# Used to compute a cheap edge-density "structure consistency" cue.
# These are approximate and should be re-calibrated against your own
# dataset's crop convention before being described as validated in the
# paper — say so explicitly in the Methods section.
SECURITY_THREAD_ROI = {
    "ten_new": (0.40, 0.10, 0.55, 0.90), "ten_old": (0.38, 0.10, 0.53, 0.90),
    "twenty_new": (0.40, 0.10, 0.55, 0.90), "twenty_old": (0.38, 0.10, 0.53, 0.90),
    "fifty_new": (0.40, 0.10, 0.55, 0.90), "fifty_old": (0.38, 0.10, 0.53, 0.90),
    "hundred_new": (0.42, 0.10, 0.57, 0.90), "hundred_old": (0.38, 0.10, 0.53, 0.90),
    "two_hundred": (0.42, 0.10, 0.57, 0.90),
    "five_hundred": (0.42, 0.10, 0.57, 0.90),
}

LANG_LABELS = {
    "en-IN": "English (India)", "hi-IN": "Hindi (हिन्दी)", "ta-IN": "Tamil (தமிழ்)",
    "te-IN": "Telugu (తెలుగు)", "bn-IN": "Bengali (বাংলা)", "mr-IN": "Marathi (मराठी)",
    "kn-IN": "Kannada (ಕನ್ನಡ)",
}
