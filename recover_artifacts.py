# """
# recover_artifacts.py
# ======================
# Run this ONCE if train_model.py crashed on the very last step (Grad-CAM
# sample export) after training already finished successfully. It does
# NOT retrain anything — it loads your already-saved .keras models from
# model/ and regenerates only the missing artifacts:

#     model/metrics_summary.json
#     model/gradcam_samples/*.png

# It reproduces the exact same train/val/test split as the original run
# (same random seed + same dataset folder => identical split), so the
# reported metrics are the correct ones for your trained models.

# Usage
# -----
#     python recover_artifacts.py
# """

# import json
# import os

# import config as cfg
# import train_model as tm  # reuse list_dataset_files / stratified_split / evaluate_* / export_gradcam_samples


# def main():
#     class_to_idx = {c: i for i, c in enumerate(cfg.CLASS_NAMES)}

#     print("Rebuilding the identical train/val/test split used during training...")
#     filepaths, labels = tm.list_dataset_files()
#     splits = tm.stratified_split(filepaths, labels)
#     (fp_train, y_train), (fp_val, y_val), (fp_test, y_test) = splits
#     print(f"Split sizes -> train: {len(fp_train)}  val: {len(fp_val)}  test: {len(fp_test)}")

#     config_path = os.path.join(cfg.MODEL_DIR, "ensemble_config.json")
#     if not os.path.exists(config_path):
#         raise FileNotFoundError(
#             "model/ensemble_config.json not found — this recovery script expects "
#             "training to have already produced it. If it's missing, you do need to retrain."
#         )
#     with open(config_path, "r", encoding="utf-8") as f:
#         ensemble_config = json.load(f)

#     import tensorflow as tf
#     print("Loading already-trained models from disk (no training happens here)...")
#     models_dict, per_model_metrics = {}, {}
#     for entry in ensemble_config["backbones"]:
#         model_path = os.path.join(cfg.MODEL_DIR, entry["file"])
#         if not os.path.exists(model_path):
#             print(f"  ! Skipping {entry['name']}: {model_path} not found.")
#             continue
#         print(f"  Loading {entry['display_name']} from {model_path} ...")
#         model = tf.keras.models.load_model(model_path)
#         models_dict[entry["name"]] = model

#         print(f"  Evaluating {entry['display_name']} on the test split...")
#         test_metrics = tm.evaluate_model_on_test(model, fp_test, y_test, entry["name"], class_to_idx, entry["display_name"])
#         test_metrics["val_accuracy"] = entry["val_accuracy"]
#         per_model_metrics[entry["name"]] = test_metrics

#     weights = {e["name"]: e["weight"] for e in ensemble_config["backbones"]}

#     print("Evaluating the ensemble on the test split...")
#     ensemble_metrics = tm.evaluate_ensemble(models_dict, weights, fp_test, y_test, class_to_idx)

#     print("Exporting Grad-CAM sample overlays (this is what crashed before — now fixed)...")
#     try:
#         tm.export_gradcam_samples(models_dict, ensemble_config, fp_test, y_test, class_to_idx)
#     except Exception as e:
#         print(f"  ! Grad-CAM sample export still failed ({e}). Everything else below is saved regardless; "
#               f"the live in-app Grad-CAM tab uses the same fixed function, so try it in the Streamlit app too.")

#     metrics_summary = {
#         "ensemble": ensemble_metrics,
#         "per_model": per_model_metrics,
#         "ensemble_weights": weights,
#     }
#     with open(os.path.join(cfg.MODEL_DIR, "metrics_summary.json"), "w", encoding="utf-8") as f:
#         json.dump(metrics_summary, f, indent=2)

#     print("\nDone. model/metrics_summary.json and model/gradcam_samples/ are now populated.")
#     print("Run `streamlit run app.py` — the Model Diagnostics tab will be fully populated now.")


# if __name__ == "__main__":
#     main()











"""
recover_artifacts.py
======================
Run this ONCE if train_model.py crashed on the very last step (Grad-CAM
sample export) after training already finished successfully. It does
NOT retrain anything — it loads your already-saved .keras models from
model/ and regenerates only the missing artifacts:

    model/metrics_summary.json
    model/gradcam_samples/*.png

It reproduces the exact same train/val/test split as the original run
(same random seed + same dataset folder => identical split), so the
reported metrics are the correct ones for your trained models.

Usage
-----
    python recover_artifacts.py
"""

import json
import os

import config as cfg
import train_model as tm  # reuse list_dataset_files / stratified_split / evaluate_* / export_gradcam_samples


def main():
    class_to_idx = {c: i for i, c in enumerate(cfg.CLASS_NAMES)}

    print("Rebuilding the identical train/val/test split used during training...")
    filepaths, labels = tm.list_dataset_files()
    splits = tm.stratified_split(filepaths, labels)
    (fp_train, y_train), (fp_val, y_val), (fp_test, y_test) = splits
    print(f"Split sizes -> train: {len(fp_train)}  val: {len(fp_val)}  test: {len(fp_test)}")

    config_path = os.path.join(cfg.MODEL_DIR, "ensemble_config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            "model/ensemble_config.json not found — this recovery script expects "
            "training to have already produced it. If it's missing, you do need to retrain."
        )
    with open(config_path, "r", encoding="utf-8") as f:
        ensemble_config = json.load(f)

    import tensorflow as tf
    print("Loading already-trained models from disk (no training happens here)...")
    models_dict, per_model_metrics = {}, {}
    for entry in ensemble_config["backbones"]:
        model_path = os.path.join(cfg.MODEL_DIR, entry["file"])
        if not os.path.exists(model_path):
            print(f"  ! Skipping {entry['name']}: {model_path} not found.")
            continue
        print(f"  Loading {entry['display_name']} from {model_path} ...")
        model = tf.keras.models.load_model(model_path)
        models_dict[entry["name"]] = model

        print(f"  Evaluating {entry['display_name']} on the test split...")
        test_metrics = tm.evaluate_model_on_test(model, fp_test, y_test, entry["name"], class_to_idx, entry["display_name"])
        test_metrics["val_accuracy"] = entry["val_accuracy"]
        per_model_metrics[entry["name"]] = test_metrics

    weights = {e["name"]: e["weight"] for e in ensemble_config["backbones"]}

    print("Evaluating the ensemble on the test split...")
    ensemble_metrics = tm.evaluate_ensemble(models_dict, weights, fp_test, y_test, class_to_idx)

    print("Exporting Grad-CAM sample overlays (this is what crashed before — now fixed)...")
    try:
        tm.export_gradcam_samples(models_dict, ensemble_config, fp_test, y_test, class_to_idx)
    except Exception as e:
        print(f"  ! Grad-CAM sample export still failed ({e}). Everything else below is saved regardless; "
              f"the live in-app Grad-CAM tab uses the same fixed function, so try it in the Streamlit app too.")

    metrics_summary = {
        "ensemble": ensemble_metrics,
        "per_model": per_model_metrics,
        "ensemble_weights": weights,
    }
    with open(os.path.join(cfg.MODEL_DIR, "metrics_summary.json"), "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)

    print("\nDone. model/metrics_summary.json and model/gradcam_samples/ are now populated.")
    print("Run `streamlit run app.py` — the Model Diagnostics tab will be fully populated now.")


if __name__ == "__main__":
    main()