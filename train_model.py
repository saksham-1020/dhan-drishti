# """
# train_model.py
# ================
# Trains a weighted, architecturally-diverse CNN ensemble for Indian banknote
# recognition, and exports every artifact app.py / the paper's Results section
# needs: per-backbone histories, an ensemble confusion matrix, classification
# reports, calibration-relevant metrics (Cohen's kappa, MCC, top-3 accuracy),
# and a handful of Grad-CAM sample overlays.

# Usage
# -----
#     python train_model.py --backbones mobilenetv2 efficientnetv2b0 convnexttiny \
#                            --head-epochs 15 --fine-tune-epochs 15

# Dataset layout expected (folder name == class name, see config.CLASS_NAMES):

#     Datasets/
#         background/*.jpg
#         ten_new/*.jpg
#         ten_old/*.jpg
#         ...
#         five_hundred/*.jpg
# """

# import argparse
# import glob
# import json
# import os

# import numpy as np
# from PIL import Image

# import config as cfg
# import dhan_drishti_utils as ddu


# def parse_args():
#     p = argparse.ArgumentParser(description="Train the Dhan Drishti ensemble.")
#     p.add_argument("--backbones", nargs="+", default=list(cfg.BACKBONES.keys()),
#                     choices=list(cfg.BACKBONES.keys()))
#     p.add_argument("--head-epochs", type=int, default=cfg.HEAD_EPOCHS)
#     p.add_argument("--fine-tune-epochs", type=int, default=cfg.FINE_TUNE_EPOCHS)
#     p.add_argument("--batch-size", type=int, default=cfg.BATCH_SIZE)
#     p.add_argument("--unfreeze-layers", type=int, default=cfg.FINE_TUNE_UNFREEZE_LAYERS)
#     return p.parse_args()


# def list_dataset_files():
#     """Returns (filepaths, labels) after validating the folder structure."""
#     filepaths, labels = [], []
#     missing = []
#     for cls in cfg.CLASS_NAMES:
#         cls_dir = os.path.join(cfg.DATASET_DIR, cls)
#         files = sorted(
#             glob.glob(os.path.join(cls_dir, "*.jpg"))
#             + glob.glob(os.path.join(cls_dir, "*.jpeg"))
#             + glob.glob(os.path.join(cls_dir, "*.png"))
#         )
#         if not files:
#             missing.append(cls)
#             continue
#         filepaths += files
#         labels += [cls] * len(files)

#     if missing:
#         raise FileNotFoundError(
#             "No images found for class(es): " + ", ".join(missing) +
#             f"\nExpected images directly inside Datasets/<class_name>/ for each of: {cfg.CLASS_NAMES}"
#         )
#     return filepaths, labels


# def stratified_split(filepaths, labels):
#     from sklearn.model_selection import train_test_split

#     fp_train, fp_temp, y_train, y_temp = train_test_split(
#         filepaths, labels, test_size=(cfg.VAL_SPLIT + cfg.TEST_SPLIT),
#         stratify=labels, random_state=cfg.SEED,
#     )
#     rel_test = cfg.TEST_SPLIT / (cfg.VAL_SPLIT + cfg.TEST_SPLIT)
#     fp_val, fp_test, y_val, y_test = train_test_split(
#         fp_temp, y_temp, test_size=rel_test, stratify=y_temp, random_state=cfg.SEED,
#     )
#     return (fp_train, y_train), (fp_val, y_val), (fp_test, y_test)


# def make_tf_dataset(filepaths, labels, backbone_name, batch_size, augment, class_to_idx):
#     import tensorflow as tf

#     preprocess_fn = ddu.BACKBONE_REGISTRY[backbone_name]["preprocess_input"]
#     label_idx = [class_to_idx[l] for l in labels]

#     def _load(path, label):
#         img = tf.io.read_file(path)
#         img = tf.image.decode_image(img, channels=3, expand_animations=False)
#         img.set_shape([None, None, 3])
#         img = tf.image.resize(img, cfg.IMG_SIZE)
#         return img, label

#     ds = tf.data.Dataset.from_tensor_slices((filepaths, label_idx))
#     ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)

#     if augment:
#         # Currency-safe augmentation only: NO horizontal flip (would mirror
#         # numerals/portrait), small rotation/zoom/brightness to emulate a
#         # handheld phone photo, plus mild JPEG-quality jitter to emulate
#         # compression noise from messaging-app-shared images.
#         aug_layer = tf.keras.Sequential([
#             tf.keras.layers.RandomRotation(0.03, fill_mode="constant", fill_value=255.0),
#             tf.keras.layers.RandomZoom(0.08, fill_mode="constant", fill_value=255.0),
#             tf.keras.layers.RandomBrightness(0.15, value_range=(0, 255)),
#             tf.keras.layers.RandomContrast(0.12),
#         ])
#         ds = ds.map(lambda x, y: (aug_layer(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)

#     def _preprocess(img, label):
#         img = preprocess_fn(img)
#         return img, tf.one_hot(label, depth=len(cfg.CLASS_NAMES))

#     ds = ds.map(_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
#     ds = ds.shuffle(1024, seed=cfg.SEED) if augment else ds
#     ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
#     return ds


# def compute_class_weights(labels, class_to_idx):
#     from sklearn.utils.class_weight import compute_class_weight
#     idx = np.array([class_to_idx[l] for l in labels])
#     classes = np.arange(len(cfg.CLASS_NAMES))
#     weights = compute_class_weight(class_weight="balanced", classes=classes, y=idx)
#     return {int(c): float(w) for c, w in zip(classes, weights)}


# def plot_history(history, out_path, title):
#     import matplotlib
#     matplotlib.use("Agg")
#     import matplotlib.pyplot as plt

#     fig, axes = plt.subplots(1, 2, figsize=(11, 4))
#     axes[0].plot(history["accuracy"], label="train")
#     axes[0].plot(history["val_accuracy"], label="val")
#     axes[0].set_title(f"{title} — Accuracy"); axes[0].legend(); axes[0].set_xlabel("epoch")
#     axes[1].plot(history["loss"], label="train")
#     axes[1].plot(history["val_loss"], label="val")
#     axes[1].set_title(f"{title} — Loss"); axes[1].legend(); axes[1].set_xlabel("epoch")
#     fig.tight_layout()
#     fig.savefig(out_path, dpi=150)
#     plt.close(fig)


# def train_one_backbone(backbone_name, splits, class_to_idx, args):
#     import tensorflow as tf

#     (fp_train, y_train), (fp_val, y_val), (fp_test, y_test) = splits
#     display_name = cfg.BACKBONES[backbone_name]["display_name"]
#     print(f"\n{'='*70}\nTraining backbone: {display_name}\n{'='*70}")

#     train_ds = make_tf_dataset(fp_train, y_train, backbone_name, args.batch_size, augment=True, class_to_idx=class_to_idx)
#     val_ds = make_tf_dataset(fp_val, y_val, backbone_name, args.batch_size, augment=False, class_to_idx=class_to_idx)
#     class_weights = compute_class_weights(y_train, class_to_idx)

#     model, base_model = ddu.build_model(backbone_name, num_classes=len(cfg.CLASS_NAMES))

#     loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=cfg.LABEL_SMOOTHING)
#     model.compile(optimizer=tf.keras.optimizers.Adam(cfg.HEAD_LR), loss=loss,
#                   metrics=["accuracy", tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top3_acc")])

#     ckpt_path = os.path.join(cfg.MODEL_DIR, cfg.BACKBONES[backbone_name]["file"])
#     callbacks = [
#         tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=6, restore_best_weights=True),
#         tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7),
#         tf.keras.callbacks.ModelCheckpoint(ckpt_path, monitor="val_accuracy", save_best_only=True),
#     ]

#     # ---- Phase 1: frozen backbone, train the head ----
#     h1 = model.fit(train_ds, validation_data=val_ds, epochs=args.head_epochs,
#                     class_weight=class_weights, callbacks=callbacks, verbose=2)

#     # ---- Phase 2: unfreeze top-N layers, fine-tune at low LR ----
#     base_model.trainable = True
#     for layer in base_model.layers[:-args.unfreeze_layers]:
#         layer.trainable = False
#     model.compile(optimizer=tf.keras.optimizers.Adam(cfg.FINE_TUNE_LR), loss=loss,
#                   metrics=["accuracy", tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top3_acc")])
#     h2 = model.fit(train_ds, validation_data=val_ds, epochs=args.fine_tune_epochs,
#                     class_weight=class_weights, callbacks=callbacks, verbose=2)

#     history = {k: h1.history.get(k, []) + h2.history.get(k, []) for k in h1.history}
#     plot_history(history, os.path.join(cfg.MODEL_DIR, f"{backbone_name}_training_history.png"), display_name)

#     model.save(ckpt_path)
#     val_accuracy = float(max(history["val_accuracy"]))

#     test_metrics = evaluate_model_on_test(model, fp_test, y_test, backbone_name, class_to_idx, display_name)
#     return model, val_accuracy, test_metrics, (fp_test, y_test)


# def evaluate_model_on_test(model, fp_test, y_test, backbone_name, class_to_idx, display_name):
#     from sklearn.metrics import (classification_report, accuracy_score, f1_score,
#                                   precision_score, recall_score, cohen_kappa_score,
#                                   matthews_corrcoef, top_k_accuracy_score)

#     y_true_idx, probs = [], []
#     for fp, label in zip(fp_test, y_test):
#         img = Image.open(fp)
#         p = ddu.single_predict(model, img, backbone_name)
#         probs.append(p)
#         y_true_idx.append(class_to_idx[label])
#     probs = np.array(probs)
#     y_true_idx = np.array(y_true_idx)
#     y_pred_idx = probs.argmax(axis=1)

#     report_txt = classification_report(y_true_idx, y_pred_idx, target_names=cfg.CLASS_NAMES, digits=4)
#     with open(os.path.join(cfg.MODEL_DIR, f"{backbone_name}_classification_report.txt"), "w", encoding="utf-8") as f:
#         f.write(report_txt)

#     metrics = {
#         "accuracy": float(accuracy_score(y_true_idx, y_pred_idx)),
#         "f1_macro": float(f1_score(y_true_idx, y_pred_idx, average="macro")),
#         "precision_macro": float(precision_score(y_true_idx, y_pred_idx, average="macro")),
#         "recall_macro": float(recall_score(y_true_idx, y_pred_idx, average="macro")),
#         "cohen_kappa": float(cohen_kappa_score(y_true_idx, y_pred_idx)),
#         "matthews_corrcoef": float(matthews_corrcoef(y_true_idx, y_pred_idx)),
#         "top3_accuracy": float(top_k_accuracy_score(y_true_idx, probs, k=3, labels=range(len(cfg.CLASS_NAMES)))),
#     }
#     print(f"[{display_name}] test accuracy = {metrics['accuracy']*100:.2f}%  f1_macro = {metrics['f1_macro']*100:.2f}%")
#     return metrics


# def evaluate_ensemble(models_dict, weights, fp_test, y_test, class_to_idx):
#     from sklearn.metrics import (classification_report, accuracy_score, f1_score,
#                                   precision_score, recall_score, cohen_kappa_score,
#                                   matthews_corrcoef, top_k_accuracy_score, confusion_matrix)

#     y_true_idx, ens_probs = [], []
#     for fp, label in zip(fp_test, y_test):
#         img = Image.open(fp)
#         p_sum = None
#         for name, model in models_dict.items():
#             p = ddu.single_predict(model, img, name)
#             p_sum = p * weights[name] if p_sum is None else p_sum + p * weights[name]
#         ens_probs.append(p_sum)
#         y_true_idx.append(class_to_idx[label])

#     ens_probs = np.array(ens_probs)
#     y_true_idx = np.array(y_true_idx)
#     y_pred_idx = ens_probs.argmax(axis=1)

#     report_txt = classification_report(y_true_idx, y_pred_idx, target_names=cfg.CLASS_NAMES, digits=4)
#     with open(os.path.join(cfg.MODEL_DIR, "classification_report_ensemble.txt"), "w", encoding="utf-8") as f:
#         f.write(report_txt)

#     cm = confusion_matrix(y_true_idx, y_pred_idx, normalize="true")
#     plot_confusion_matrix(cm, os.path.join(cfg.MODEL_DIR, "confusion_matrix_ensemble.png"))

#     metrics = {
#         "accuracy": float(accuracy_score(y_true_idx, y_pred_idx)),
#         "f1_macro": float(f1_score(y_true_idx, y_pred_idx, average="macro")),
#         "precision_macro": float(precision_score(y_true_idx, y_pred_idx, average="macro")),
#         "recall_macro": float(recall_score(y_true_idx, y_pred_idx, average="macro")),
#         "cohen_kappa": float(cohen_kappa_score(y_true_idx, y_pred_idx)),
#         "matthews_corrcoef": float(matthews_corrcoef(y_true_idx, y_pred_idx)),
#         "top3_accuracy": float(top_k_accuracy_score(y_true_idx, ens_probs, k=3, labels=range(len(cfg.CLASS_NAMES)))),
#     }
#     print(f"[ENSEMBLE] test accuracy = {metrics['accuracy']*100:.2f}%  f1_macro = {metrics['f1_macro']*100:.2f}%")
#     return metrics


# def plot_confusion_matrix(cm, out_path):
#     import matplotlib
#     matplotlib.use("Agg")
#     import matplotlib.pyplot as plt
#     import seaborn as sns

#     fig, ax = plt.subplots(figsize=(9, 8))
#     sns.heatmap(cm, annot=True, fmt=".2f", cmap="Purples",
#                 xticklabels=cfg.CLASS_NAMES, yticklabels=cfg.CLASS_NAMES, ax=ax)
#     ax.set_xlabel("Predicted"); ax.set_ylabel("True")
#     ax.set_title("Ensemble Confusion Matrix (row-normalized)")
#     plt.xticks(rotation=45, ha="right")
#     fig.tight_layout()
#     fig.savefig(out_path, dpi=150)
#     plt.close(fig)


# def export_gradcam_samples(models_dict, ensemble_config, fp_test, y_test, class_to_idx, n_samples=8):
#     """Exports a handful of Grad-CAM overlays from the highest-weighted
#     backbone for qualitative figures in the paper."""
#     best_name = max(ensemble_config["backbones"], key=lambda e: e["weight"])["name"]
#     best_model = models_dict[best_name]
#     head_model, base_model = ddu.build_head_submodel(best_model)
#     preprocess_fn = ddu.BACKBONE_REGISTRY[best_name]["preprocess_input"]

#     rng = np.random.default_rng(cfg.SEED)
#     idxs = rng.choice(len(fp_test), size=min(n_samples, len(fp_test)), replace=False)

#     for i in idxs:
#         fp, label = fp_test[i], y_test[i]
#         img = Image.open(fp).convert("RGB").resize(cfg.IMG_SIZE)
#         x = preprocess_fn(np.array(img).astype(np.float32).copy())
#         x = np.expand_dims(x, axis=0)
#         pred_idx = class_to_idx[label]
#         heatmap, _ = ddu.make_gradcam_heatmap(x, base_model, head_model, pred_index=pred_idx)
#         overlay = ddu.overlay_gradcam(img, heatmap)
#         out_name = f"{label}_{os.path.basename(fp).rsplit('.',1)[0]}_gradcam.png"
#         overlay.save(os.path.join(cfg.GRADCAM_SAMPLE_DIR, out_name))
#     print(f"Saved {len(idxs)} Grad-CAM sample overlays to {cfg.GRADCAM_SAMPLE_DIR}")


# def main():
#     args = parse_args()
#     class_to_idx = {c: i for i, c in enumerate(cfg.CLASS_NAMES)}

#     filepaths, labels = list_dataset_files()
#     print(f"Found {len(filepaths)} images across {len(cfg.CLASS_NAMES)} classes.")
#     splits = stratified_split(filepaths, labels)
#     (fp_train, y_train), (fp_val, y_val), (fp_test, y_test) = splits
#     print(f"Split sizes -> train: {len(fp_train)}  val: {len(fp_val)}  test: {len(fp_test)}")

#     models_dict, per_model_metrics, backbone_entries = {}, {}, []
#     for name in args.backbones:
#         model, val_acc, test_metrics, _ = train_one_backbone(name, splits, class_to_idx, args)
#         models_dict[name] = model
#         test_metrics["val_accuracy"] = val_acc
#         per_model_metrics[name] = test_metrics
#         backbone_entries.append({
#             "name": name,
#             "display_name": cfg.BACKBONES[name]["display_name"],
#             "file": cfg.BACKBONES[name]["file"],
#             "val_accuracy": val_acc,
#         })

#     # Softmax-normalize validation accuracies into ensemble weights so the
#     # stronger backbone(s) count for more in the soft vote.
#     val_accs = np.array([e["val_accuracy"] for e in backbone_entries])
#     exp_accs = np.exp(val_accs * 8.0)  # temperature=8 sharpens weighting toward the best model(s)
#     softmax_weights = exp_accs / exp_accs.sum()
#     for entry, w in zip(backbone_entries, softmax_weights):
#         entry["weight"] = float(w)
#     weights = {e["name"]: e["weight"] for e in backbone_entries}

#     ensemble_config = {
#         "class_names": cfg.CLASS_NAMES,
#         "img_size": list(cfg.IMG_SIZE),
#         "backbones": backbone_entries,
#     }
#     with open(os.path.join(cfg.MODEL_DIR, "ensemble_config.json"), "w", encoding="utf-8") as f:
#         json.dump(ensemble_config, f, indent=2)

#     ensemble_metrics = evaluate_ensemble(models_dict, weights, fp_test, y_test, class_to_idx)
#     export_gradcam_samples(models_dict, ensemble_config, fp_test, y_test, class_to_idx)

#     metrics_summary = {
#         "ensemble": ensemble_metrics,
#         "per_model": per_model_metrics,
#         "ensemble_weights": weights,
#     }
#     with open(os.path.join(cfg.MODEL_DIR, "metrics_summary.json"), "w", encoding="utf-8") as f:
#         json.dump(metrics_summary, f, indent=2)

#     print("\nAll done. Artifacts written to:", cfg.MODEL_DIR)
#     print("Run `streamlit run app.py` to launch the demo.")


# if __name__ == "__main__":
#     main()























"""
train_model.py
================
Trains a weighted, architecturally-diverse CNN ensemble for Indian banknote
recognition, and exports every artifact app.py / the paper's Results section
needs: per-backbone histories, an ensemble confusion matrix, classification
reports, calibration-relevant metrics (Cohen's kappa, MCC, top-3 accuracy),
and a handful of Grad-CAM sample overlays.

Usage
-----
    python train_model.py --backbones mobilenetv2 efficientnetv2b0 convnexttiny \
                           --head-epochs 15 --fine-tune-epochs 15

Dataset layout expected (folder name == class name, see config.CLASS_NAMES):

    Datasets/
        background/*.jpg
        ten_new/*.jpg
        ten_old/*.jpg
        ...
        five_hundred/*.jpg
"""

import argparse
import glob
import json
import os

import numpy as np
from PIL import Image

import config as cfg
import dhan_drishti_utils as ddu


def parse_args():
    p = argparse.ArgumentParser(description="Train the Dhan Drishti ensemble.")
    p.add_argument("--backbones", nargs="+", default=list(cfg.BACKBONES.keys()),
                    choices=list(cfg.BACKBONES.keys()))
    p.add_argument("--head-epochs", type=int, default=cfg.HEAD_EPOCHS)
    p.add_argument("--fine-tune-epochs", type=int, default=cfg.FINE_TUNE_EPOCHS)
    p.add_argument("--batch-size", type=int, default=cfg.BATCH_SIZE)
    p.add_argument("--unfreeze-layers", type=int, default=cfg.FINE_TUNE_UNFREEZE_LAYERS)
    return p.parse_args()


def list_dataset_files():
    """Returns (filepaths, labels) after validating the folder structure."""
    filepaths, labels = [], []
    missing = []
    for cls in cfg.CLASS_NAMES:
        cls_dir = os.path.join(cfg.DATASET_DIR, cls)
        files = sorted(
            glob.glob(os.path.join(cls_dir, "*.jpg"))
            + glob.glob(os.path.join(cls_dir, "*.jpeg"))
            + glob.glob(os.path.join(cls_dir, "*.png"))
        )
        if not files:
            missing.append(cls)
            continue
        filepaths += files
        labels += [cls] * len(files)

    if missing:
        raise FileNotFoundError(
            "No images found for class(es): " + ", ".join(missing) +
            f"\nExpected images directly inside Datasets/<class_name>/ for each of: {cfg.CLASS_NAMES}"
        )
    return filepaths, labels


def stratified_split(filepaths, labels):
    from sklearn.model_selection import train_test_split

    fp_train, fp_temp, y_train, y_temp = train_test_split(
        filepaths, labels, test_size=(cfg.VAL_SPLIT + cfg.TEST_SPLIT),
        stratify=labels, random_state=cfg.SEED,
    )
    rel_test = cfg.TEST_SPLIT / (cfg.VAL_SPLIT + cfg.TEST_SPLIT)
    fp_val, fp_test, y_val, y_test = train_test_split(
        fp_temp, y_temp, test_size=rel_test, stratify=y_temp, random_state=cfg.SEED,
    )
    return (fp_train, y_train), (fp_val, y_val), (fp_test, y_test)


def make_tf_dataset(filepaths, labels, backbone_name, batch_size, augment, class_to_idx):
    import tensorflow as tf

    preprocess_fn = ddu.BACKBONE_REGISTRY[backbone_name]["preprocess_input"]
    label_idx = [class_to_idx[l] for l in labels]

    def _load(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_image(img, channels=3, expand_animations=False)
        img.set_shape([None, None, 3])
        img = tf.image.resize(img, cfg.IMG_SIZE)
        return img, label

    ds = tf.data.Dataset.from_tensor_slices((filepaths, label_idx))
    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)

    if augment:
        # Currency-safe augmentation only: NO horizontal flip (would mirror
        # numerals/portrait), small rotation/zoom/brightness to emulate a
        # handheld phone photo, plus mild JPEG-quality jitter to emulate
        # compression noise from messaging-app-shared images.
        aug_layer = tf.keras.Sequential([
            tf.keras.layers.RandomRotation(0.03, fill_mode="constant", fill_value=255.0),
            tf.keras.layers.RandomZoom(0.08, fill_mode="constant", fill_value=255.0),
            tf.keras.layers.RandomBrightness(0.15, value_range=(0, 255)),
            tf.keras.layers.RandomContrast(0.12),
        ])
        ds = ds.map(lambda x, y: (aug_layer(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)

    def _preprocess(img, label):
        img = preprocess_fn(img)
        return img, tf.one_hot(label, depth=len(cfg.CLASS_NAMES))

    ds = ds.map(_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.shuffle(1024, seed=cfg.SEED) if augment else ds
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def compute_class_weights(labels, class_to_idx):
    from sklearn.utils.class_weight import compute_class_weight
    idx = np.array([class_to_idx[l] for l in labels])
    classes = np.arange(len(cfg.CLASS_NAMES))
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=idx)
    return {int(c): float(w) for c, w in zip(classes, weights)}


def plot_history(history, out_path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history["accuracy"], label="train")
    axes[0].plot(history["val_accuracy"], label="val")
    axes[0].set_title(f"{title} — Accuracy"); axes[0].legend(); axes[0].set_xlabel("epoch")
    axes[1].plot(history["loss"], label="train")
    axes[1].plot(history["val_loss"], label="val")
    axes[1].set_title(f"{title} — Loss"); axes[1].legend(); axes[1].set_xlabel("epoch")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def train_one_backbone(backbone_name, splits, class_to_idx, args):
    import tensorflow as tf

    (fp_train, y_train), (fp_val, y_val), (fp_test, y_test) = splits
    display_name = cfg.BACKBONES[backbone_name]["display_name"]
    print(f"\n{'='*70}\nTraining backbone: {display_name}\n{'='*70}")

    train_ds = make_tf_dataset(fp_train, y_train, backbone_name, args.batch_size, augment=True, class_to_idx=class_to_idx)
    val_ds = make_tf_dataset(fp_val, y_val, backbone_name, args.batch_size, augment=False, class_to_idx=class_to_idx)
    class_weights = compute_class_weights(y_train, class_to_idx)

    model, base_model = ddu.build_model(backbone_name, num_classes=len(cfg.CLASS_NAMES))

    loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=cfg.LABEL_SMOOTHING)
    model.compile(optimizer=tf.keras.optimizers.Adam(cfg.HEAD_LR), loss=loss,
                  metrics=["accuracy", tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top3_acc")])

    ckpt_path = os.path.join(cfg.MODEL_DIR, cfg.BACKBONES[backbone_name]["file"])
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=6, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7),
        tf.keras.callbacks.ModelCheckpoint(ckpt_path, monitor="val_accuracy", save_best_only=True),
    ]

    # ---- Phase 1: frozen backbone, train the head ----
    h1 = model.fit(train_ds, validation_data=val_ds, epochs=args.head_epochs,
                    class_weight=class_weights, callbacks=callbacks, verbose=2)

    # ---- Phase 2: unfreeze top-N layers, fine-tune at low LR ----
    base_model.trainable = True
    for layer in base_model.layers[:-args.unfreeze_layers]:
        layer.trainable = False
    model.compile(optimizer=tf.keras.optimizers.Adam(cfg.FINE_TUNE_LR), loss=loss,
                  metrics=["accuracy", tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top3_acc")])
    h2 = model.fit(train_ds, validation_data=val_ds, epochs=args.fine_tune_epochs,
                    class_weight=class_weights, callbacks=callbacks, verbose=2)

    history = {k: h1.history.get(k, []) + h2.history.get(k, []) for k in h1.history}
    plot_history(history, os.path.join(cfg.MODEL_DIR, f"{backbone_name}_training_history.png"), display_name)
    with open(os.path.join(cfg.MODEL_DIR, f"{backbone_name}_history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f)

    model.save(ckpt_path)
    val_accuracy = float(max(history["val_accuracy"]))

    test_metrics = evaluate_model_on_test(model, fp_test, y_test, backbone_name, class_to_idx, display_name)
    return model, val_accuracy, test_metrics, (fp_test, y_test)


def evaluate_model_on_test(model, fp_test, y_test, backbone_name, class_to_idx, display_name):
    from sklearn.metrics import (classification_report, accuracy_score, f1_score,
                                  precision_score, recall_score, cohen_kappa_score,
                                  matthews_corrcoef, top_k_accuracy_score)

    y_true_idx, probs = [], []
    for fp, label in zip(fp_test, y_test):
        img = Image.open(fp)
        p = ddu.single_predict(model, img, backbone_name)
        probs.append(p)
        y_true_idx.append(class_to_idx[label])
    probs = np.array(probs)
    y_true_idx = np.array(y_true_idx)
    y_pred_idx = probs.argmax(axis=1)

    report_txt = classification_report(y_true_idx, y_pred_idx, target_names=cfg.CLASS_NAMES, digits=4)
    with open(os.path.join(cfg.MODEL_DIR, f"{backbone_name}_classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(report_txt)

    metrics = {
        "accuracy": float(accuracy_score(y_true_idx, y_pred_idx)),
        "f1_macro": float(f1_score(y_true_idx, y_pred_idx, average="macro")),
        "precision_macro": float(precision_score(y_true_idx, y_pred_idx, average="macro")),
        "recall_macro": float(recall_score(y_true_idx, y_pred_idx, average="macro")),
        "cohen_kappa": float(cohen_kappa_score(y_true_idx, y_pred_idx)),
        "matthews_corrcoef": float(matthews_corrcoef(y_true_idx, y_pred_idx)),
        "top3_accuracy": float(top_k_accuracy_score(y_true_idx, probs, k=3, labels=range(len(cfg.CLASS_NAMES)))),
    }
    print(f"[{display_name}] test accuracy = {metrics['accuracy']*100:.2f}%  f1_macro = {metrics['f1_macro']*100:.2f}%")
    return metrics


def evaluate_ensemble(models_dict, weights, fp_test, y_test, class_to_idx):
    from sklearn.metrics import (classification_report, accuracy_score, f1_score,
                                  precision_score, recall_score, cohen_kappa_score,
                                  matthews_corrcoef, top_k_accuracy_score, confusion_matrix)

    y_true_idx, ens_probs = [], []
    for fp, label in zip(fp_test, y_test):
        img = Image.open(fp)
        p_sum = None
        for name, model in models_dict.items():
            p = ddu.single_predict(model, img, name)
            p_sum = p * weights[name] if p_sum is None else p_sum + p * weights[name]
        ens_probs.append(p_sum)
        y_true_idx.append(class_to_idx[label])

    ens_probs = np.array(ens_probs)
    y_true_idx = np.array(y_true_idx)
    y_pred_idx = ens_probs.argmax(axis=1)

    report_txt = classification_report(y_true_idx, y_pred_idx, target_names=cfg.CLASS_NAMES, digits=4)
    with open(os.path.join(cfg.MODEL_DIR, "classification_report_ensemble.txt"), "w", encoding="utf-8") as f:
        f.write(report_txt)

    cm = confusion_matrix(y_true_idx, y_pred_idx, normalize="true")
    plot_confusion_matrix(cm, os.path.join(cfg.MODEL_DIR, "confusion_matrix_ensemble.png"))

    metrics = {
        "accuracy": float(accuracy_score(y_true_idx, y_pred_idx)),
        "f1_macro": float(f1_score(y_true_idx, y_pred_idx, average="macro")),
        "precision_macro": float(precision_score(y_true_idx, y_pred_idx, average="macro")),
        "recall_macro": float(recall_score(y_true_idx, y_pred_idx, average="macro")),
        "cohen_kappa": float(cohen_kappa_score(y_true_idx, y_pred_idx)),
        "matthews_corrcoef": float(matthews_corrcoef(y_true_idx, y_pred_idx)),
        "top3_accuracy": float(top_k_accuracy_score(y_true_idx, ens_probs, k=3, labels=range(len(cfg.CLASS_NAMES)))),
    }
    print(f"[ENSEMBLE] test accuracy = {metrics['accuracy']*100:.2f}%  f1_macro = {metrics['f1_macro']*100:.2f}%")
    return metrics


def plot_confusion_matrix(cm, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(9, 8))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Purples",
                xticklabels=cfg.CLASS_NAMES, yticklabels=cfg.CLASS_NAMES, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Ensemble Confusion Matrix (row-normalized)")
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def export_gradcam_samples(models_dict, ensemble_config, fp_test, y_test, class_to_idx, n_samples=8):
    """Exports a handful of Grad-CAM overlays from the highest-weighted
    backbone for qualitative figures in the paper."""
    best_name = max(ensemble_config["backbones"], key=lambda e: e["weight"])["name"]
    best_model = models_dict[best_name]
    head_model, base_model = ddu.build_head_submodel(best_model)
    preprocess_fn = ddu.BACKBONE_REGISTRY[best_name]["preprocess_input"]

    rng = np.random.default_rng(cfg.SEED)
    idxs = rng.choice(len(fp_test), size=min(n_samples, len(fp_test)), replace=False)

    for i in idxs:
        fp, label = fp_test[i], y_test[i]
        img = Image.open(fp).convert("RGB").resize(cfg.IMG_SIZE)
        x = preprocess_fn(np.array(img).astype(np.float32).copy())
        x = np.expand_dims(x, axis=0)
        pred_idx = class_to_idx[label]
        heatmap, _ = ddu.make_gradcam_heatmap(x, base_model, head_model, pred_index=pred_idx)
        overlay = ddu.overlay_gradcam(img, heatmap)
        out_name = f"{label}_{os.path.basename(fp).rsplit('.',1)[0]}_gradcam.png"
        overlay.save(os.path.join(cfg.GRADCAM_SAMPLE_DIR, out_name))
    print(f"Saved {len(idxs)} Grad-CAM sample overlays to {cfg.GRADCAM_SAMPLE_DIR}")


def main():
    args = parse_args()
    class_to_idx = {c: i for i, c in enumerate(cfg.CLASS_NAMES)}

    filepaths, labels = list_dataset_files()
    print(f"Found {len(filepaths)} images across {len(cfg.CLASS_NAMES)} classes.")
    splits = stratified_split(filepaths, labels)
    (fp_train, y_train), (fp_val, y_val), (fp_test, y_test) = splits
    print(f"Split sizes -> train: {len(fp_train)}  val: {len(fp_val)}  test: {len(fp_test)}")

    models_dict, per_model_metrics, backbone_entries = {}, {}, []
    for name in args.backbones:
        model, val_acc, test_metrics, _ = train_one_backbone(name, splits, class_to_idx, args)
        models_dict[name] = model
        test_metrics["val_accuracy"] = val_acc
        per_model_metrics[name] = test_metrics
        backbone_entries.append({
            "name": name,
            "display_name": cfg.BACKBONES[name]["display_name"],
            "file": cfg.BACKBONES[name]["file"],
            "val_accuracy": val_acc,
        })

    # Softmax-normalize validation accuracies into ensemble weights so the
    # stronger backbone(s) count for more in the soft vote.
    val_accs = np.array([e["val_accuracy"] for e in backbone_entries])
    exp_accs = np.exp(val_accs * 8.0)  # temperature=8 sharpens weighting toward the best model(s)
    softmax_weights = exp_accs / exp_accs.sum()
    for entry, w in zip(backbone_entries, softmax_weights):
        entry["weight"] = float(w)
    weights = {e["name"]: e["weight"] for e in backbone_entries}

    ensemble_config = {
        "class_names": cfg.CLASS_NAMES,
        "img_size": list(cfg.IMG_SIZE),
        "backbones": backbone_entries,
    }
    with open(os.path.join(cfg.MODEL_DIR, "ensemble_config.json"), "w", encoding="utf-8") as f:
        json.dump(ensemble_config, f, indent=2)

    ensemble_metrics = evaluate_ensemble(models_dict, weights, fp_test, y_test, class_to_idx)
    export_gradcam_samples(models_dict, ensemble_config, fp_test, y_test, class_to_idx)

    metrics_summary = {
        "ensemble": ensemble_metrics,
        "per_model": per_model_metrics,
        "ensemble_weights": weights,
    }
    with open(os.path.join(cfg.MODEL_DIR, "metrics_summary.json"), "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)

    print("\nAll done. Artifacts written to:", cfg.MODEL_DIR)
    print("Run `streamlit run app.py` to launch the demo.")


if __name__ == "__main__":
    main()