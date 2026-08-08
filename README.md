# Dhan Drishti — Indian Banknote Recognition & Assistive Cash-Counting System

An ensemble deep-learning system for Indian banknote recognition, built around
five things that make it more than "yet another MobileNetV2 currency
classifier" — see **"Novelty for the paper"** below before you start writing
the manuscript.

---

## 1. Folder structure

```
dhan-drishti/
├── app.py                     # Streamlit UI (run this to demo)
├── train_model.py             # Trains the ensemble end-to-end
├── dhan_drishti_utils.py      # Shared model / Grad-CAM / TTA / MC-Dropout / heuristic code
├── config.py                  # ALL class names, paths, hyperparameters, thresholds
├── note_metadata.py           # Security-feature text + multilingual TTS phrase book
├── requirements.txt
├── README.md
├── .streamlit/
│   └── config.toml            # Dark theme
├── Datasets/                  # <-- put your images here (see §2)
│   ├── background/
│   ├── ten_new/            ├── ten_old/
│   ├── twenty_new/         ├── twenty_old/
│   ├── fifty_new/          ├── fifty_old/
│   ├── hundred_new/        ├── hundred_old/
│   ├── two_hundred/
│   └── five_hundred/
└── model/                     # auto-created by train_model.py
    ├── mobilenetv2.keras
    ├── efficientnetv2b0.keras
    ├── convnexttiny.keras
    ├── ensemble_config.json
    ├── metrics_summary.json
    ├── confusion_matrix_ensemble.png
    ├── classification_report_ensemble.txt
    ├── <backbone>_training_history.png
    ├── <backbone>_classification_report.txt
    └── gradcam_samples/*.png
```

## 2. Preparing the dataset

Put JPG/PNG images **directly** inside each `Datasets/<class_name>/` folder —
no further nesting. Class folder names must exactly match `config.CLASS_NAMES`:

```
background, ten_new, ten_old, twenty_new, twenty_old, fifty_new, fifty_old,
hundred_new, hundred_old, two_hundred, five_hundred
```

Recommendations for an IEEE Access-grade dataset section:
- **≥300–500 images per class** minimum; more for `background` (varied
  clutter: tables, hands, other objects, blurred/no-note frames).
- Vary lighting, distance, angle (small rotation only — see §4 on why we
  never horizontal-flip), background surface, and note condition
  (crisp/worn/creased) to avoid the classic "clean lab photos only"
  overfitting trap reviewers flag immediately.
- Keep an untouched **held-out test set** — `train_model.py` performs a
  stratified 70/15/15 split automatically, but if you already have a
  separate real-world capture set, evaluate on it separately too and report
  both numbers (in-distribution vs. field-condition accuracy is itself a
  publishable point).

## 3. Install & train

```bash
pip install -r requirements.txt

# Full ensemble (recommended for the paper's headline numbers)
python train_model.py --backbones mobilenetv2 efficientnetv2b0 convnexttiny \
                       --head-epochs 15 --fine-tune-epochs 15

# Faster ablation run with a single backbone (for the ablation table)
python train_model.py --backbones mobilenetv2 --head-epochs 10 --fine-tune-epochs 8
```

This writes every artifact listed under `model/` in §1, including per-backbone
and ensemble classification reports, a confusion matrix, and Grad-CAM sample
overlays — everything you need for the Results/Discussion figures.

## 4. Run the demo

```bash
streamlit run app.py
```

If `model/ensemble_config.json` doesn't exist yet, the app runs in a clearly
labelled **Demo Mode** (simulated predictions) so the UI is still explorable
before training finishes.

---

## Novelty for the paper (why this isn't "just another CNN classifier")

Plain single-CNN Indian-currency classification is indeed a saturated topic
for IEEE Access. Here's what to foreground as contributions:

1. **Validation-accuracy-weighted heterogeneous ensemble.** Three
   architecturally different backbones (depthwise-separable MobileNetV2,
   compound-scaled EfficientNetV2-B0, ConvNeXt-Tiny) are combined by a
   softmax-weighted soft vote over validation accuracy, rather than equal
   averaging. Report the ablation: single best backbone vs. equal-weight
   ensemble vs. accuracy-weighted ensemble — this table alone is a solid
   contribution section.

2. **Uncertainty-aware rejection via MC-Dropout**, rather than a bare
   softmax confidence threshold. Predictive entropy over stochastic forward
   passes is used to explicitly ask the user to re-scan instead of emitting
   a possibly-wrong top-1 label with false confidence — a meaningful
   reliability contribution for a system aimed at financial decisions.
   Report entropy calibration plots (reliability diagrams) in the paper.

3. **Currency-safe augmentation policy.** Unlike generic image classifiers,
   this pipeline deliberately excludes horizontal flips (which mirror
   printed numerals/portraits into physically impossible notes) and instead
   uses small rotation/zoom/brightness/contrast jitter calibrated to
   handheld-camera capture conditions. Document this design choice —
   reviewers who know currency data will notice if you *don't* explain it.

4. **Explainability via Grad-CAM tied to the highest-weighted ensemble
   member**, letting you show qualitatively that the model attends to the
   portrait/motif/security-thread region rather than background texture —
   directly supports a "trustworthy AI" framing.

5. **Assistive Cash-Tally accessibility mode.** Beyond single-note
   classification, the system supports a running session tally (count +
   total ₹ value) with multilingual (7 Indian languages) text-to-speech
   feedback — positioned for visually-impaired users counting cash, which is
   a genuinely under-served, real-world use case with its own literature
   (assistive fintech) to cite and differentiate against plain
   classification papers.

6. **Transparent, explicitly-non-forensic screening cues.** Be careful in
   the paper (and to reviewers) to never claim "counterfeit detection" from
   the color-signature / texture-sharpness / ROI-consistency heuristics —
   they are disclosed in the UI and here as lightweight secondary cues, not
   validated anti-counterfeiting technology. Overclaiming this is the #1 way
   this class of paper gets rejected or, worse, published and then
   criticized post-hoc.

### Suggested metrics table for the paper
Report, per backbone and for the ensemble: Accuracy, macro-F1,
macro-Precision/Recall, Cohen's Kappa, Matthews Correlation Coefficient, and
Top-3 Accuracy — all computed automatically by `train_model.py` into
`model/metrics_summary.json`.

---

## Ethical / accuracy disclaimer (keep this in the paper too)

This system performs **image-based denomination classification**, not
certified currency authentication. The heuristic screening cues
(color-signature distance, texture sharpness, security-thread ROI
consistency) are cheap, transparent sanity checks layered on top of the CNN
— they are **not** validated against RBI's official security-feature
specifications and must never be marketed or cited as counterfeit detection.
Always direct end users to verify high-value notes using the official RBI
security-feature checklist (surfaced in the "RBI Verification Guide" tab).
