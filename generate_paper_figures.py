"""
generate_paper_figures.py
==========================
Produces a rich set of publication-quality figures for the IEEE Access
Results/Discussion section, using the ALREADY-TRAINED models in model/ —
no retraining happens here, this is pure evaluation + plotting.

Figures written to model/paper_figures/:

  01_combined_training_curves.png   - accuracy/loss of all backbones overlaid
  02_model_comparison_bars.png      - Accuracy/F1/Precision/Recall/Kappa/MCC, per model vs ensemble
  03_ensemble_weights.png           - pie chart of the softmax ensemble weights
  04_roc_curves.png                 - one-vs-rest ROC curve per class + macro/micro AUC
  05_precision_recall_curves.png    - one-vs-rest PR curve per class
  06_per_class_f1.png               - sorted bar chart of per-class F1 (ensemble)
  07_confusion_matrix_counts.png    - raw-count confusion matrix (complements the
                                       normalized one train_model.py already writes)
  08_confidence_histogram.png       - top-1 confidence, correct vs incorrect predictions
  09_calibration_reliability.png    - reliability diagram + Expected Calibration Error
  10_mc_dropout_entropy.png         - predictive-entropy distribution, correct vs incorrect
  11_tsne_embeddings.png            - 2D t-SNE of penultimate-layer features, colored by class
  12_inference_time_comparison.png  - measured per-model / ensemble inference latency

Usage
-----
    python generate_paper_figures.py
    python generate_paper_figures.py --skip-tsne --skip-mc-dropout   # faster run
"""

import argparse
import json
import os
import time

import numpy as np
from PIL import Image

import config as cfg
import dhan_drishti_utils as ddu
import train_model as tm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG_DIR = os.path.join(cfg.MODEL_DIR, "paper_figures")
os.makedirs(FIG_DIR, exist_ok=True)

PALETTE = ["#8b5cf6", "#34d399", "#fbbf24", "#f87171", "#60a5fa", "#f472b6",
           "#a78bfa", "#4ade80", "#fb923c", "#38bdf8", "#e879f9"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--skip-tsne", action="store_true", help="Skip the t-SNE embedding plot (can be slow on CPU).")
    p.add_argument("--skip-mc-dropout", action="store_true", help="Skip the MC-Dropout entropy plot (extra forward passes).")
    p.add_argument("--tsne-max-samples", type=int, default=600)
    return p.parse_args()


def load_everything():
    config_path = os.path.join(cfg.MODEL_DIR, "ensemble_config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError("model/ensemble_config.json not found — train the ensemble first.")
    with open(config_path, "r", encoding="utf-8") as f:
        ensemble_config = json.load(f)

    metrics_path = os.path.join(cfg.MODEL_DIR, "metrics_summary.json")
    metrics_summary = None
    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics_summary = json.load(f)

    import tensorflow as tf
    models = {}
    for entry in ensemble_config["backbones"]:
        model_path = os.path.join(cfg.MODEL_DIR, entry["file"])
        if os.path.exists(model_path):
            models[entry["name"]] = tf.keras.models.load_model(model_path)

    return ensemble_config, metrics_summary, models


def get_test_split():
    filepaths, labels = tm.list_dataset_files()
    (fp_train, y_train), (fp_val, y_val), (fp_test, y_test) = tm.stratified_split(filepaths, labels)
    return fp_test, y_test


def collect_predictions(models, ensemble_config, fp_test, y_test, class_to_idx):
    """Runs every model + the ensemble once over the test set and returns
    everything downstream figures need, so we only pay the inference cost
    once instead of once per figure."""
    weights = {e["name"]: e["weight"] for e in ensemble_config["backbones"]}
    weights_sum = sum(weights.values())

    y_true = np.array([class_to_idx[l] for l in y_test])
    per_model_probs = {name: [] for name in models}
    ensemble_probs = []
    per_model_timings = {name: [] for name in models}

    for fp in fp_test:
        img = Image.open(fp)
        combined = None
        for entry in ensemble_config["backbones"]:
            name = entry["name"]
            if name not in models:
                continue
            t0 = time.perf_counter()
            p = ddu.single_predict(models[name], img, name, img_size=tuple(ensemble_config["img_size"]))
            per_model_timings[name].append((time.perf_counter() - t0) * 1000)
            per_model_probs[name].append(p)
            w = weights[name] / weights_sum
            combined = p * w if combined is None else combined + p * w
        ensemble_probs.append(combined)

    for name in per_model_probs:
        per_model_probs[name] = np.array(per_model_probs[name])
    ensemble_probs = np.array(ensemble_probs)

    return y_true, per_model_probs, ensemble_probs, per_model_timings


# ==================================================================== #
# 01. Combined training curves
# ==================================================================== #
def fig_combined_training_curves(ensemble_config):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for i, entry in enumerate(ensemble_config["backbones"]):
        # training_history.png was already saved per backbone; we don't have
        # the raw history arrays anymore post-hoc, so this figure instead
        # overlays validation accuracy/loss curves parsed back is not
        # possible without the raw numbers. We therefore re-plot from the
        # per-backbone PNG being unavailable -> skip gracefully if missing.
        pass
    plt.close(fig)


# The function above is intentionally replaced by a simpler, robust version
# that reads history from a companion JSON if present, else skips.
def fig_combined_training_curves_v2(ensemble_config):
    any_history = False
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for i, entry in enumerate(ensemble_config["backbones"]):
        hist_json = os.path.join(cfg.MODEL_DIR, f"{entry['name']}_history.json")
        if not os.path.exists(hist_json):
            continue
        with open(hist_json, "r", encoding="utf-8") as f:
            h = json.load(f)
        color = PALETTE[i % len(PALETTE)]
        axes[0].plot(h["val_accuracy"], label=entry["display_name"], color=color, linewidth=2)
        axes[1].plot(h["val_loss"], label=entry["display_name"], color=color, linewidth=2)
        any_history = True

    if not any_history:
        plt.close(fig)
        print("  (skip) 01_combined_training_curves.png — no *_history.json found "
              "(only available if you retrain with the updated train_model.py that saves history JSON).")
        return

    axes[0].set_title("Validation Accuracy — All Backbones"); axes[0].set_xlabel("epoch"); axes[0].legend(); axes[0].grid(alpha=0.2)
    axes[1].set_title("Validation Loss — All Backbones"); axes[1].set_xlabel("epoch"); axes[1].legend(); axes[1].grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "01_combined_training_curves.png"), dpi=160)
    plt.close(fig)
    print("  saved 01_combined_training_curves.png")


# ==================================================================== #
# 02. Per-model vs ensemble metric comparison
# ==================================================================== #
def fig_model_comparison_bars(metrics_summary):
    if not metrics_summary:
        print("  (skip) 02_model_comparison_bars.png — no metrics_summary.json")
        return
    per_model = metrics_summary["per_model"]
    ensemble = metrics_summary["ensemble"]
    names = list(per_model.keys()) + ["Ensemble"]
    display_names = [cfg.BACKBONES[n]["display_name"] for n in per_model.keys()] + ["Ensemble (ours)"]
    metric_keys = ["accuracy", "f1_macro", "precision_macro", "recall_macro"]
    metric_labels = ["Accuracy", "F1 (macro)", "Precision (macro)", "Recall (macro)"]

    data = np.array([[per_model[n][k] for k in metric_keys] for n in per_model.keys()] + [[ensemble[k] for k in metric_keys]])

    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(display_names))
    width = 0.2
    for j, mk in enumerate(metric_labels):
        ax.bar(x + (j - 1.5) * width, data[:, j] * 100, width, label=mk, color=PALETTE[j % len(PALETTE)])
    ax.set_xticks(x); ax.set_xticklabels(display_names, rotation=15, ha="right")
    ax.set_ylabel("Percent"); ax.set_ylim(0, 105)
    ax.set_title("Per-Model vs. Ensemble — Test-Set Metrics")
    ax.legend(loc="lower right"); ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "02_model_comparison_bars.png"), dpi=160)
    plt.close(fig)
    print("  saved 02_model_comparison_bars.png")


# ==================================================================== #
# 03. Ensemble weights pie
# ==================================================================== #
def fig_ensemble_weights(ensemble_config):
    labels = [e["display_name"] for e in ensemble_config["backbones"]]
    weights = [e["weight"] for e in ensemble_config["backbones"]]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(weights, labels=labels, autopct="%1.1f%%", colors=PALETTE[:len(labels)],
           wedgeprops={"edgecolor": "white", "linewidth": 1.5}, textprops={"fontsize": 11})
    ax.set_title("Validation-Accuracy-Weighted Ensemble Contribution")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "03_ensemble_weights.png"), dpi=160)
    plt.close(fig)
    print("  saved 03_ensemble_weights.png")


# ==================================================================== #
# 04 / 05. ROC + Precision-Recall curves (one-vs-rest)
# ==================================================================== #
def fig_roc_and_pr_curves(y_true, ensemble_probs, class_names):
    from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
    from sklearn.preprocessing import label_binarize

    y_bin = label_binarize(y_true, classes=list(range(len(class_names))))

    # ---- ROC ----
    fig, ax = plt.subplots(figsize=(8, 7))
    aucs = []
    for i, cls in enumerate(class_names):
        if y_bin[:, i].sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y_bin[:, i], ensemble_probs[:, i])
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        ax.plot(fpr, tpr, label=f"{cls} (AUC={roc_auc:.3f})", color=PALETTE[i % len(PALETTE)], linewidth=1.6)
    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title(f"One-vs-Rest ROC Curves (macro-avg AUC = {np.mean(aucs):.3f})")
    ax.legend(fontsize=7, loc="lower right"); ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "04_roc_curves.png"), dpi=160)
    plt.close(fig)
    print("  saved 04_roc_curves.png")

    # ---- Precision-Recall ----
    fig, ax = plt.subplots(figsize=(8, 7))
    aps = []
    for i, cls in enumerate(class_names):
        if y_bin[:, i].sum() == 0:
            continue
        precision, recall, _ = precision_recall_curve(y_bin[:, i], ensemble_probs[:, i])
        ap = average_precision_score(y_bin[:, i], ensemble_probs[:, i])
        aps.append(ap)
        ax.plot(recall, precision, label=f"{cls} (AP={ap:.3f})", color=PALETTE[i % len(PALETTE)], linewidth=1.6)
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title(f"One-vs-Rest Precision-Recall Curves (mean AP = {np.mean(aps):.3f})")
    ax.legend(fontsize=7, loc="lower left"); ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "05_precision_recall_curves.png"), dpi=160)
    plt.close(fig)
    print("  saved 05_precision_recall_curves.png")


# ==================================================================== #
# 06. Per-class F1 bar chart
# ==================================================================== #
def fig_per_class_f1(y_true, ensemble_probs, class_names):
    from sklearn.metrics import f1_score
    y_pred = ensemble_probs.argmax(axis=1)
    f1s = f1_score(y_true, y_pred, average=None, labels=range(len(class_names)))
    order = np.argsort(f1s)
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#f87171" if f1s[i] < 0.85 else "#34d399" for i in order]
    ax.barh(np.array(class_names)[order], f1s[order] * 100, color=colors)
    ax.set_xlabel("F1 Score (%)"); ax.set_xlim(0, 105)
    ax.set_title("Per-Class F1 — Ensemble")
    ax.axvline(85, color="gray", linestyle="--", linewidth=1, alpha=0.6)
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "06_per_class_f1.png"), dpi=160)
    plt.close(fig)
    print("  saved 06_per_class_f1.png")


# ==================================================================== #
# 07. Raw-count confusion matrix
# ==================================================================== #
def fig_confusion_matrix_counts(y_true, ensemble_probs, class_names):
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    y_pred = ensemble_probs.argmax(axis=1)
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(9, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Purples", xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title("Ensemble Confusion Matrix (raw counts)")
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "07_confusion_matrix_counts.png"), dpi=160)
    plt.close(fig)
    print("  saved 07_confusion_matrix_counts.png")


# ==================================================================== #
# 08. Confidence histogram (correct vs incorrect)
# ==================================================================== #
def fig_confidence_histogram(y_true, ensemble_probs):
    y_pred = ensemble_probs.argmax(axis=1)
    top1_conf = ensemble_probs.max(axis=1)
    correct_mask = y_pred == y_true

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0, 1, 25)
    ax.hist(top1_conf[correct_mask], bins=bins, alpha=0.75, label="Correct", color="#34d399")
    ax.hist(top1_conf[~correct_mask], bins=bins, alpha=0.75, label="Incorrect", color="#f87171")
    ax.set_xlabel("Top-1 Confidence"); ax.set_ylabel("Count")
    ax.set_title("Confidence Distribution — Correct vs. Incorrect Predictions")
    ax.legend(); ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "08_confidence_histogram.png"), dpi=160)
    plt.close(fig)
    print("  saved 08_confidence_histogram.png")


# ==================================================================== #
# 09. Calibration / reliability diagram + Expected Calibration Error
# ==================================================================== #
def fig_calibration_reliability(y_true, ensemble_probs, n_bins=10):
    y_pred = ensemble_probs.argmax(axis=1)
    confidences = ensemble_probs.max(axis=1)
    correct = (y_pred == y_true).astype(float)

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_acc, bin_conf, bin_count = [], [], []
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (confidences > lo) & (confidences <= hi) if i > 0 else (confidences >= lo) & (confidences <= hi)
        if mask.sum() == 0:
            bin_acc.append(np.nan); bin_conf.append((lo + hi) / 2); bin_count.append(0)
            continue
        acc = correct[mask].mean()
        conf = confidences[mask].mean()
        bin_acc.append(acc); bin_conf.append(conf); bin_count.append(mask.sum())
        ece += (mask.sum() / len(confidences)) * abs(acc - conf)

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    ax.bar(centers, np.nan_to_num(bin_acc), width=1.0 / n_bins * 0.9, color="#8b5cf6", alpha=0.85, label="Model accuracy")
    ax.set_xlabel("Confidence"); ax.set_ylabel("Accuracy")
    ax.set_title(f"Reliability Diagram (ECE = {ece:.4f})")
    ax.legend(); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "09_calibration_reliability.png"), dpi=160)
    plt.close(fig)
    print(f"  saved 09_calibration_reliability.png (ECE={ece:.4f})")
    return ece


# ==================================================================== #
# 10. MC-Dropout predictive-entropy distribution
# ==================================================================== #
def fig_mc_dropout_entropy(ensemble_config, models, fp_test, y_test, class_to_idx, max_samples=250):
    best_entry = max(ensemble_config["backbones"], key=lambda e: e["weight"])
    best_model = models.get(best_entry["name"])
    if best_model is None:
        print("  (skip) 10_mc_dropout_entropy.png — best-weighted backbone not loaded")
        return

    rng = np.random.default_rng(cfg.SEED)
    idxs = rng.choice(len(fp_test), size=min(max_samples, len(fp_test)), replace=False)

    entropies, correct_flags = [], []
    for i in idxs:
        img = Image.open(fp_test[i])
        mean_probs, entropy = ddu.mc_dropout_predict(best_model, img, best_entry["name"],
                                                       img_size=tuple(ensemble_config["img_size"]),
                                                       n_passes=15)
        pred = int(np.argmax(mean_probs))
        entropies.append(entropy)
        correct_flags.append(pred == class_to_idx[y_test[i]])

    entropies = np.array(entropies)
    correct_flags = np.array(correct_flags)

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0, max(entropies.max(), 0.1), 25)
    ax.hist(entropies[correct_flags], bins=bins, alpha=0.75, label="Correct", color="#34d399")
    ax.hist(entropies[~correct_flags], bins=bins, alpha=0.75, label="Incorrect", color="#f87171")
    ax.axvline(cfg.MC_ENTROPY_REJECT_THRESHOLD, color="#fbbf24", linestyle="--", label="Re-scan threshold")
    ax.set_xlabel("Predictive Entropy (nats)"); ax.set_ylabel("Count")
    ax.set_title(f"MC-Dropout Predictive Entropy — {best_entry['display_name']}")
    ax.legend(); ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "10_mc_dropout_entropy.png"), dpi=160)
    plt.close(fig)
    print("  saved 10_mc_dropout_entropy.png")


# ==================================================================== #
# 11. t-SNE of penultimate-layer embeddings
# ==================================================================== #
def fig_tsne_embeddings(ensemble_config, models, fp_test, y_test, class_to_idx, class_names, max_samples=600):
    from sklearn.manifold import TSNE
    import tensorflow as tf
    from tensorflow.keras import Model

    best_entry = max(ensemble_config["backbones"], key=lambda e: e["weight"])
    best_model = models.get(best_entry["name"])
    if best_model is None:
        print("  (skip) 11_tsne_embeddings.png — best-weighted backbone not loaded")
        return

    # Penultimate dense layer ("dense_head") activations, one forward pass per image.
    embed_model = Model(best_model.input, best_model.get_layer("dense_head").output)

    rng = np.random.default_rng(cfg.SEED)
    idxs = rng.choice(len(fp_test), size=min(max_samples, len(fp_test)), replace=False)

    preprocess_fn = ddu.BACKBONE_REGISTRY[best_entry["name"]]["preprocess_input"]
    img_size = tuple(ensemble_config["img_size"])
    batch, labels = [], []
    for i in idxs:
        img = Image.open(fp_test[i]).convert("RGB").resize(img_size)
        arr = preprocess_fn(np.array(img).astype(np.float32).copy())
        batch.append(arr)
        labels.append(class_to_idx[y_test[i]])
    batch = np.stack(batch, axis=0)
    labels = np.array(labels)

    embeddings = embed_model.predict(batch, verbose=0, batch_size=32)
    tsne = TSNE(n_components=2, random_state=cfg.SEED, perplexity=min(30, max(5, len(idxs) // 10)))
    proj = tsne.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(9, 8))
    for i, cls in enumerate(class_names):
        mask = labels == i
        if mask.sum() == 0:
            continue
        ax.scatter(proj[mask, 0], proj[mask, 1], s=18, alpha=0.8, color=PALETTE[i % len(PALETTE)], label=cls)
    ax.set_title(f"t-SNE of Penultimate-Layer Features — {best_entry['display_name']}")
    ax.legend(fontsize=7, loc="best", ncol=2); ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "11_tsne_embeddings.png"), dpi=160)
    plt.close(fig)
    print("  saved 11_tsne_embeddings.png")


# ==================================================================== #
# 12. Inference-time comparison
# ==================================================================== #
def fig_inference_time_comparison(per_model_timings, ensemble_config):
    names = [e["display_name"] for e in ensemble_config["backbones"] if e["name"] in per_model_timings]
    means = [np.mean(per_model_timings[e["name"]]) for e in ensemble_config["backbones"] if e["name"] in per_model_timings]
    stds = [np.std(per_model_timings[e["name"]]) for e in ensemble_config["backbones"] if e["name"] in per_model_timings]
    ensemble_mean = sum(means)  # sequential sum approximates the ensemble's per-image cost

    names.append("Ensemble (sum)")
    means.append(ensemble_mean)
    stds.append(0)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(names, means, yerr=stds, capsize=5, color=PALETTE[:len(names)])
    ax.set_ylabel("Per-Image Inference Time (ms, CPU)")
    ax.set_title("Inference Latency Comparison")
    plt.xticks(rotation=15, ha="right")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "12_inference_time_comparison.png"), dpi=160)
    plt.close(fig)
    print("  saved 12_inference_time_comparison.png")


def main():
    args = parse_args()
    class_to_idx = {c: i for i, c in enumerate(cfg.CLASS_NAMES)}

    print("Loading ensemble config, metrics summary, and trained models...")
    ensemble_config, metrics_summary, models = load_everything()
    class_names = ensemble_config["class_names"]

    print("Rebuilding the identical test split used during training...")
    fp_test, y_test = get_test_split()
    print(f"Test set size: {len(fp_test)}")

    print("\nRunning every model + the ensemble once over the test set (this is evaluation, not training)...")
    y_true, per_model_probs, ensemble_probs, per_model_timings = collect_predictions(
        models, ensemble_config, fp_test, y_test, class_to_idx
    )

    print("\nGenerating figures...")
    fig_combined_training_curves_v2(ensemble_config)
    fig_model_comparison_bars(metrics_summary)
    fig_ensemble_weights(ensemble_config)
    fig_roc_and_pr_curves(y_true, ensemble_probs, class_names)
    fig_per_class_f1(y_true, ensemble_probs, class_names)
    fig_confusion_matrix_counts(y_true, ensemble_probs, class_names)
    fig_confidence_histogram(y_true, ensemble_probs)
    fig_calibration_reliability(y_true, ensemble_probs)
    fig_inference_time_comparison(per_model_timings, ensemble_config)

    if not args.skip_mc_dropout:
        print("\nRunning MC-Dropout entropy analysis (extra stochastic forward passes, this takes a bit)...")
        fig_mc_dropout_entropy(ensemble_config, models, fp_test, y_test, class_to_idx)
    else:
        print("  (skipped MC-Dropout entropy figure per --skip-mc-dropout)")

    if not args.skip_tsne:
        print("\nRunning t-SNE embedding projection...")
        fig_tsne_embeddings(ensemble_config, models, fp_test, y_test, class_to_idx, class_names,
                             max_samples=args.tsne_max_samples)
    else:
        print("  (skipped t-SNE figure per --skip-tsne)")

    print(f"\nAll done. Figures saved to: {FIG_DIR}")
    print("Open the Streamlit app's Diagnostics tab to browse them, or use them directly in your paper.")


if __name__ == "__main__":
    main()