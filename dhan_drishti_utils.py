# # """
# # dhan_drishti_utils.py
# # ======================
# # Shared building blocks used by BOTH train_model.py and app.py, so the exact
# # same preprocessing / inference logic is guaranteed at train and serve time
# # (a common, reviewer-flagged bug in student CNN papers is train/serve skew).

# # Contents
# # --------
# # 1. BACKBONE_REGISTRY      -> per-architecture builder + preprocess_input
# # 2. build_model()           -> backbone + GAP + Dropout(MC) + Dense head
# # 3. build_head_submodel()   -> split model for Grad-CAM (conv-base vs head)
# # 4. make_gradcam_heatmap()  -> Grad-CAM heatmap
# # 5. overlay_gradcam()       -> heatmap -> RGB overlay image
# # 6. single_predict()        -> one deterministic forward pass
# # 7. tta_predict()           -> test-time augmentation averaged prediction
# # 8. mc_dropout_predict()    -> MC-Dropout: mean probs + predictive entropy
# # 9. color_signature_distance() / texture_sharpness_score() /
# #    security_roi_consistency_score()  -> cheap, explicitly-heuristic
# #    screening cues (never described as "fake note detection" — see README)
# # """

# # import io
# # import numpy as np
# # from PIL import Image, ImageOps

# # import config as cfg

# # # TensorFlow is imported lazily inside functions that need it so that pure
# # # heuristic/image-processing helpers (used by the Streamlit UI even before
# # # a model is loaded) stay fast and import-light.

# # CLASS_LABELS = cfg.CLASS_NAMES
# # RBI_COLOR_CENTROIDS = cfg.RBI_COLOR_CENTROIDS


# # # ============================================================== #
# # # 1. Backbone registry
# # # ============================================================== #
# # def _lazy_backbone_registry():
# #     import tensorflow as tf
# #     from tensorflow.keras.applications import (
# #         MobileNetV2, EfficientNetV2B0, ConvNeXtTiny,
# #     )
# #     from tensorflow.keras.applications import mobilenet_v2, efficientnet_v2, convnext

# #     return {
# #         "mobilenetv2": {
# #             "build": MobileNetV2,
# #             "preprocess_input": mobilenet_v2.preprocess_input,
# #         },
# #         "efficientnetv2b0": {
# #             "build": EfficientNetV2B0,
# #             "preprocess_input": efficientnet_v2.preprocess_input,
# #         },
# #         "convnexttiny": {
# #             "build": ConvNeXtTiny,
# #             "preprocess_input": convnext.preprocess_input,
# #         },
# #     }


# # class _LazyRegistry(dict):
# #     """Populates itself with real TF objects on first access, so importing
# #     this module doesn't force a slow TensorFlow import for callers that
# #     only need the pure-numpy heuristic functions."""

# #     def _ensure(self):
# #         if not self:
# #             self.update(_lazy_backbone_registry())

# #     def __getitem__(self, key):
# #         self._ensure()
# #         return super().__getitem__(key)

# #     def __contains__(self, key):
# #         self._ensure()
# #         return super().__contains__(key)


# # BACKBONE_REGISTRY = _LazyRegistry()


# # # ============================================================== #
# # # 2. Model builder
# # # ============================================================== #
# # def build_model(backbone_name: str, num_classes: int, dropout_rate: float = cfg.DROPOUT_RATE):
# #     """Backbone (ImageNet weights, initially frozen) + GAP + MC-Dropout +
# #     Dense softmax head. Returns (model, base_model) so callers can freeze /
# #     unfreeze `base_model` for the two-phase training schedule."""
# #     import tensorflow as tf
# #     from tensorflow.keras import layers, Model

# #     entry = BACKBONE_REGISTRY[backbone_name]
# #     base_model = entry["build"](
# #         include_top=False, weights="imagenet",
# #         input_shape=(*cfg.IMG_SIZE, 3), pooling=None,
# #     )
# #     base_model.trainable = False

# #     inputs = layers.Input(shape=(*cfg.IMG_SIZE, 3), name="input_image")
# #     x = base_model(inputs, training=False)
# #     x = layers.GlobalAveragePooling2D(name="gap")(x)
# #     x = layers.Dropout(dropout_rate, name="mc_dropout_1")(x)
# #     x = layers.Dense(256, activation="relu", name="dense_head")(x)
# #     x = layers.Dropout(dropout_rate, name="mc_dropout_2")(x)
# #     outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

# #     model = Model(inputs, outputs, name=f"dhan_drishti_{backbone_name}")
# #     return model, base_model


# # def build_head_submodel(model):
# #     """Splits a trained model into (base_model, head_model) for Grad-CAM:
# #     base_model: input -> last conv feature map of the backbone submodel
# #     head_model: that feature map -> final predictions
# #     Works generically by locating the nested backbone sub-model (the only
# #     layer of type Functional/Model inside our architecture)."""
# #     import tensorflow as tf
# #     from tensorflow.keras import Model

# #     backbone_layer = None
# #     for layer in model.layers:
# #         if isinstance(layer, tf.keras.Model):
# #             backbone_layer = layer
# #             break
# #     if backbone_layer is None:
# #         raise ValueError("Could not locate nested backbone sub-model for Grad-CAM.")

# #     last_conv_layer = None
# #     for layer in reversed(backbone_layer.layers):
# #         if len(layer.output_shape) == 4:
# #             last_conv_layer = layer
# #             break
# #     if last_conv_layer is None:
# #         raise ValueError("Could not locate a 4D conv feature map inside the backbone.")

# #     base_submodel = Model(backbone_layer.input, last_conv_layer.output, name="gradcam_base")

# #     # Head: feature map -> rest of the original model's layers (GAP onward)
# #     feat_input = tf.keras.Input(shape=last_conv_layer.output_shape[1:])
# #     x = feat_input
# #     started = False
# #     for layer in model.layers:
# #         if layer is backbone_layer:
# #             started = True
# #             continue
# #         if not started:
# #             continue
# #         x = layer(x)
# #     head_submodel = Model(feat_input, x, name="gradcam_head")
# #     return head_submodel, base_submodel


# # # ============================================================== #
# # # 3. Grad-CAM
# # # ============================================================== #
# # def make_gradcam_heatmap(preprocessed_batch, base_model, head_model, pred_index=None):
# #     import tensorflow as tf

# #     with tf.GradientTape() as tape:
# #         conv_output = base_model(preprocessed_batch)
# #         tape.watch(conv_output)
# #         preds = head_model(conv_output)
# #         if pred_index is None:
# #             pred_index = tf.argmax(preds[0])
# #         class_channel = preds[:, pred_index]

# #     grads = tape.gradient(class_channel, conv_output)
# #     pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
# #     conv_output = conv_output[0]
# #     heatmap = conv_output @ pooled_grads[..., tf.newaxis]
# #     heatmap = tf.squeeze(heatmap)
# #     heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
# #     return heatmap.numpy(), int(pred_index)


# # def overlay_gradcam(pil_img: Image.Image, heatmap: np.ndarray, alpha: float = 0.4) -> Image.Image:
# #     import matplotlib.cm as cm

# #     img = pil_img.convert("RGB")
# #     heatmap_img = Image.fromarray(np.uint8(255 * heatmap)).resize(img.size)
# #     heatmap_arr = np.array(heatmap_img)

# #     jet = cm.get_cmap("jet")
# #     jet_colors = jet(np.arange(256))[:, :3]
# #     jet_heatmap = jet_colors[heatmap_arr]
# #     jet_heatmap = np.uint8(jet_heatmap * 255)
# #     jet_heatmap_img = Image.fromarray(jet_heatmap).resize(img.size)

# #     blended = Image.blend(img, jet_heatmap_img, alpha=alpha)
# #     return blended


# # # ============================================================== #
# # # 4. Inference helpers (shared train/serve preprocessing)
# # # ============================================================== #
# # def _preprocess_pil(pil_img: Image.Image, backbone_name: str, img_size=None) -> np.ndarray:
# #     img_size = img_size or cfg.IMG_SIZE
# #     preprocess_fn = BACKBONE_REGISTRY[backbone_name]["preprocess_input"]
# #     resized = pil_img.convert("RGB").resize(img_size)
# #     arr = np.array(resized).astype(np.float32)
# #     arr = preprocess_fn(arr.copy())
# #     return np.expand_dims(arr, axis=0)


# # def single_predict(model, pil_img: Image.Image, backbone_name: str, img_size=None) -> np.ndarray:
# #     x = _preprocess_pil(pil_img, backbone_name, img_size)
# #     return model.predict(x, verbose=0)[0]


# # def tta_predict(model, pil_img: Image.Image, backbone_name: str, img_size=None, n_aug: int = 6) -> np.ndarray:
# #     """Test-time augmentation: small, currency-safe perturbations only.
# #     NOTE: we deliberately never horizontally flip a banknote — flipping
# #     mirrors the printed Devanagari/English numerals and portrait, which
# #     would corrupt rather than augment the signal a real user's phone
# #     camera would produce."""
# #     img_size = img_size or cfg.IMG_SIZE
# #     rng = np.random.default_rng(cfg.SEED)
# #     base = pil_img.convert("RGB")

# #     variants = [base]
# #     for _ in range(n_aug - 1):
# #         angle = float(rng.uniform(-8, 8))
# #         brightness = float(rng.uniform(0.85, 1.15))
# #         v = base.rotate(angle, resample=Image.BILINEAR, expand=False, fillcolor=(255, 255, 255))
# #         arr = np.array(v).astype(np.float32) * brightness
# #         arr = np.clip(arr, 0, 255).astype(np.uint8)
# #         variants.append(Image.fromarray(arr))

# #     preprocess_fn = BACKBONE_REGISTRY[backbone_name]["preprocess_input"]
# #     batch = np.stack([
# #         preprocess_fn(np.array(v.resize(img_size)).astype(np.float32).copy())
# #         for v in variants
# #     ], axis=0)
# #     preds = model.predict(batch, verbose=0)
# #     return preds.mean(axis=0)


# # def mc_dropout_predict(model, pil_img: Image.Image, backbone_name: str, img_size=None,
# #                         n_passes: int = cfg.MC_DROPOUT_PASSES):
# #     """Monte-Carlo Dropout: keeps Dropout layers ACTIVE at inference and
# #     runs n_passes stochastic forward passes. Returns:
# #       mean_probs      - averaged softmax vector (use for the prediction)
# #       predictive_entropy - epistemic-uncertainty proxy in nats; high value
# #                             means the model is unsure and the UI should
# #                             recommend a re-scan rather than trusting the
# #                             top-1 label."""
# #     import tensorflow as tf

# #     x = _preprocess_pil(pil_img, backbone_name, img_size)
# #     x_batch = np.repeat(x, n_passes, axis=0)
# #     probs = model(x_batch, training=True).numpy()  # training=True keeps Dropout active
# #     mean_probs = probs.mean(axis=0)
# #     eps = 1e-9
# #     predictive_entropy = float(-np.sum(mean_probs * np.log(mean_probs + eps)))
# #     return mean_probs, predictive_entropy


# # # ============================================================== #
# # # 5. Heuristic, explicitly-non-forensic screening cues
# # # ============================================================== #
# # def is_blank_surface(pil_img: Image.Image) -> tuple[bool, float]:
# #     gray = ImageOps.grayscale(pil_img).resize((64, 64))
# #     std_dev = float(np.std(np.array(gray)))
# #     return std_dev < cfg.BLANK_SURFACE_STD_THRESHOLD, std_dev


# # def color_signature_distance(pil_img: Image.Image, predicted_class: str) -> float:
# #     if predicted_class not in RBI_COLOR_CENTROIDS:
# #         return 0.0
# #     small = pil_img.convert("RGB").resize((100, 60))
# #     arr = np.array(small)
# #     center = arr[12:48, 20:80]
# #     mean_rgb = [float(np.mean(center[:, :, c])) for c in range(3)]
# #     target = RBI_COLOR_CENTROIDS[predicted_class]
# #     return float(np.sqrt(sum((mean_rgb[i] - target[i]) ** 2 for i in range(3))))


# # def texture_sharpness_score(pil_img: Image.Image) -> float:
# #     """Variance of Laplacian — a classic, cheap blur/quality proxy. Very
# #     low values often indicate a photo of a screen or a poor photocopy
# #     rather than a genuine note photographed directly."""
# #     gray = np.array(ImageOps.grayscale(pil_img).resize((300, 150)), dtype=np.float64)
# #     kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)
# #     from scipy.signal import convolve2d  # noqa: local import, optional dep
# #     lap = convolve2d(gray, kernel, mode="same", boundary="symm")
# #     return float(lap.var())


# # def security_roi_consistency_score(pil_img: Image.Image, predicted_class: str) -> float:
# #     """Cheap edge-density proxy inside the approximate security-thread /
# #     see-through-register ROI. Higher edge density in that narrow vertical
# #     band is *consistent with* the presence of print detail there; it is
# #     NOT a verified security-thread detector. Always disclosed as such."""
# #     if predicted_class not in cfg.SECURITY_THREAD_ROI:
# #         return 1.0
# #     w, h = pil_img.size
# #     x0f, y0f, x1f, y1f = cfg.SECURITY_THREAD_ROI[predicted_class]
# #     box = (int(x0f * w), int(y0f * h), int(x1f * w), int(y1f * h))
# #     roi = ImageOps.grayscale(pil_img.crop(box))
# #     if roi.size[0] < 2 or roi.size[1] < 2:
# #         return 1.0
# #     arr = np.array(roi, dtype=np.float64)
# #     gx = np.abs(np.diff(arr, axis=1)).mean()
# #     gy = np.abs(np.diff(arr, axis=0)).mean()
# #     edge_density = (gx + gy) / 2.0
# #     return float(edge_density / 255.0)





















# """
# dhan_drishti_utils.py
# ======================
# Shared building blocks used by BOTH train_model.py and app.py, so the exact
# same preprocessing / inference logic is guaranteed at train and serve time
# (a common, reviewer-flagged bug in student CNN papers is train/serve skew).

# Contents
# --------
# 1. BACKBONE_REGISTRY      -> per-architecture builder + preprocess_input
# 2. build_model()           -> backbone + GAP + Dropout(MC) + Dense head
# 3. build_head_submodel()   -> split model for Grad-CAM (conv-base vs head)
# 4. make_gradcam_heatmap()  -> Grad-CAM heatmap
# 5. overlay_gradcam()       -> heatmap -> RGB overlay image
# 6. single_predict()        -> one deterministic forward pass
# 7. tta_predict()           -> test-time augmentation averaged prediction
# 8. mc_dropout_predict()    -> MC-Dropout: mean probs + predictive entropy
# 9. color_signature_distance() / texture_sharpness_score() /
#    security_roi_consistency_score()  -> cheap, explicitly-heuristic
#    screening cues (never described as "fake note detection" — see README)
# """

# import io
# import numpy as np
# from PIL import Image, ImageOps

# import config as cfg

# # TensorFlow is imported lazily inside functions that need it so that pure
# # heuristic/image-processing helpers (used by the Streamlit UI even before
# # a model is loaded) stay fast and import-light.

# CLASS_LABELS = cfg.CLASS_NAMES
# RBI_COLOR_CENTROIDS = cfg.RBI_COLOR_CENTROIDS


# # ============================================================== #
# # 1. Backbone registry
# # ============================================================== #
# def _lazy_backbone_registry():
#     import tensorflow as tf
#     from tensorflow.keras.applications import (
#         MobileNetV2, EfficientNetV2B0, ConvNeXtTiny,
#     )
#     from tensorflow.keras.applications import mobilenet_v2, efficientnet_v2, convnext

#     return {
#         "mobilenetv2": {
#             "build": MobileNetV2,
#             "preprocess_input": mobilenet_v2.preprocess_input,
#         },
#         "efficientnetv2b0": {
#             "build": EfficientNetV2B0,
#             "preprocess_input": efficientnet_v2.preprocess_input,
#         },
#         "convnexttiny": {
#             "build": ConvNeXtTiny,
#             "preprocess_input": convnext.preprocess_input,
#         },
#     }


# class _LazyRegistry(dict):
#     """Populates itself with real TF objects on first access, so importing
#     this module doesn't force a slow TensorFlow import for callers that
#     only need the pure-numpy heuristic functions."""

#     def _ensure(self):
#         if not self:
#             self.update(_lazy_backbone_registry())

#     def __getitem__(self, key):
#         self._ensure()
#         return super().__getitem__(key)

#     def __contains__(self, key):
#         self._ensure()
#         return super().__contains__(key)


# BACKBONE_REGISTRY = _LazyRegistry()


# # ============================================================== #
# # 2. Model builder
# # ============================================================== #
# def build_model(backbone_name: str, num_classes: int, dropout_rate: float = cfg.DROPOUT_RATE):
#     """Backbone (ImageNet weights, initially frozen) + GAP + MC-Dropout +
#     Dense softmax head. Returns (model, base_model) so callers can freeze /
#     unfreeze `base_model` for the two-phase training schedule."""
#     import tensorflow as tf
#     from tensorflow.keras import layers, Model

#     entry = BACKBONE_REGISTRY[backbone_name]
#     base_model = entry["build"](
#         include_top=False, weights="imagenet",
#         input_shape=(*cfg.IMG_SIZE, 3), pooling=None,
#     )
#     base_model.trainable = False

#     inputs = layers.Input(shape=(*cfg.IMG_SIZE, 3), name="input_image")
#     x = base_model(inputs, training=False)
#     x = layers.GlobalAveragePooling2D(name="gap")(x)
#     x = layers.Dropout(dropout_rate, name="mc_dropout_1")(x)
#     x = layers.Dense(256, activation="relu", name="dense_head")(x)
#     x = layers.Dropout(dropout_rate, name="mc_dropout_2")(x)
#     outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

#     model = Model(inputs, outputs, name=f"dhan_drishti_{backbone_name}")
#     return model, base_model


# def build_head_submodel(model):
#     """Splits a trained model into (base_model, head_model) for Grad-CAM:
#     base_model: input -> last 4D feature map of the backbone submodel
#     head_model: that feature map -> final predictions
#     Works generically by locating the nested backbone sub-model (the only
#     layer of type Functional/Model inside our architecture).

#     NOTE: we intentionally read shapes via `layer.output` (a tensor) inside
#     a try/except rather than the older `layer.output_shape` property.
#     Some architectures (e.g. ConvNeXt, which uses LayerNormalization blocks
#     with multiple inbound call-sites) raise AttributeError on
#     `output_shape` under recent Keras/TF versions — wrapping the lookup
#     lets us just skip those layers and keep searching backwards for a
#     genuine 4D conv-style feature map."""
#     import tensorflow as tf
#     from tensorflow.keras import Model

#     backbone_layer = None
#     for layer in model.layers:
#         if isinstance(layer, tf.keras.Model):
#             backbone_layer = layer
#             break
#     if backbone_layer is None:
#         raise ValueError("Could not locate nested backbone sub-model for Grad-CAM.")

#     last_conv_layer = None
#     for layer in reversed(backbone_layer.layers):
#         try:
#             out_tensor = layer.output
#             shape = out_tensor.shape
#         except Exception:
#             continue
#         if shape is not None and len(shape) == 4:
#             last_conv_layer = layer
#             break
#     if last_conv_layer is None:
#         raise ValueError("Could not locate a 4D conv feature map inside the backbone.")

#     base_submodel = Model(backbone_layer.input, last_conv_layer.output, name="gradcam_base")

#     # Head: feature map -> rest of the original model's layers (GAP onward)
#     # Use the tensor's `.shape` (not the layer's `.output_shape` property,
#     # which is what raised AttributeError on ConvNeXt's LayerNormalization
#     # layers above) to build the matching Input spec.
#     feat_shape = tuple(last_conv_layer.output.shape[1:])
#     feat_input = tf.keras.Input(shape=feat_shape)
#     x = feat_input
#     started = False
#     for layer in model.layers:
#         if layer is backbone_layer:
#             started = True
#             continue
#         if not started:
#             continue
#         x = layer(x)
#     head_submodel = Model(feat_input, x, name="gradcam_head")
#     return head_submodel, base_submodel


# # ============================================================== #
# # 3. Grad-CAM
# # ============================================================== #
# def make_gradcam_heatmap(preprocessed_batch, base_model, head_model, pred_index=None):
#     import tensorflow as tf

#     with tf.GradientTape() as tape:
#         conv_output = base_model(preprocessed_batch)
#         tape.watch(conv_output)
#         preds = head_model(conv_output)
#         if pred_index is None:
#             pred_index = tf.argmax(preds[0])
#         class_channel = preds[:, pred_index]

#     grads = tape.gradient(class_channel, conv_output)
#     pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
#     conv_output = conv_output[0]
#     heatmap = conv_output @ pooled_grads[..., tf.newaxis]
#     heatmap = tf.squeeze(heatmap)
#     heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
#     return heatmap.numpy(), int(pred_index)


# def overlay_gradcam(pil_img: Image.Image, heatmap: np.ndarray, alpha: float = 0.4) -> Image.Image:
#     import matplotlib.cm as cm

#     img = pil_img.convert("RGB")
#     heatmap_img = Image.fromarray(np.uint8(255 * heatmap)).resize(img.size)
#     heatmap_arr = np.array(heatmap_img)

#     jet = cm.get_cmap("jet")
#     jet_colors = jet(np.arange(256))[:, :3]
#     jet_heatmap = jet_colors[heatmap_arr]
#     jet_heatmap = np.uint8(jet_heatmap * 255)
#     jet_heatmap_img = Image.fromarray(jet_heatmap).resize(img.size)

#     blended = Image.blend(img, jet_heatmap_img, alpha=alpha)
#     return blended


# # ============================================================== #
# # 4. Inference helpers (shared train/serve preprocessing)
# # ============================================================== #
# def _preprocess_pil(pil_img: Image.Image, backbone_name: str, img_size=None) -> np.ndarray:
#     img_size = img_size or cfg.IMG_SIZE
#     preprocess_fn = BACKBONE_REGISTRY[backbone_name]["preprocess_input"]
#     resized = pil_img.convert("RGB").resize(img_size)
#     arr = np.array(resized).astype(np.float32)
#     arr = preprocess_fn(arr.copy())
#     return np.expand_dims(arr, axis=0)


# def single_predict(model, pil_img: Image.Image, backbone_name: str, img_size=None) -> np.ndarray:
#     x = _preprocess_pil(pil_img, backbone_name, img_size)
#     return model.predict(x, verbose=0)[0]


# def tta_predict(model, pil_img: Image.Image, backbone_name: str, img_size=None, n_aug: int = 6) -> np.ndarray:
#     """Test-time augmentation: small, currency-safe perturbations only.
#     NOTE: we deliberately never horizontally flip a banknote — flipping
#     mirrors the printed Devanagari/English numerals and portrait, which
#     would corrupt rather than augment the signal a real user's phone
#     camera would produce."""
#     img_size = img_size or cfg.IMG_SIZE
#     rng = np.random.default_rng(cfg.SEED)
#     base = pil_img.convert("RGB")

#     variants = [base]
#     for _ in range(n_aug - 1):
#         angle = float(rng.uniform(-8, 8))
#         brightness = float(rng.uniform(0.85, 1.15))
#         v = base.rotate(angle, resample=Image.BILINEAR, expand=False, fillcolor=(255, 255, 255))
#         arr = np.array(v).astype(np.float32) * brightness
#         arr = np.clip(arr, 0, 255).astype(np.uint8)
#         variants.append(Image.fromarray(arr))

#     preprocess_fn = BACKBONE_REGISTRY[backbone_name]["preprocess_input"]
#     batch = np.stack([
#         preprocess_fn(np.array(v.resize(img_size)).astype(np.float32).copy())
#         for v in variants
#     ], axis=0)
#     preds = model.predict(batch, verbose=0)
#     return preds.mean(axis=0)


# def mc_dropout_predict(model, pil_img: Image.Image, backbone_name: str, img_size=None,
#                         n_passes: int = cfg.MC_DROPOUT_PASSES):
#     """Monte-Carlo Dropout: keeps Dropout layers ACTIVE at inference and
#     runs n_passes stochastic forward passes. Returns:
#       mean_probs      - averaged softmax vector (use for the prediction)
#       predictive_entropy - epistemic-uncertainty proxy in nats; high value
#                             means the model is unsure and the UI should
#                             recommend a re-scan rather than trusting the
#                             top-1 label."""
#     import tensorflow as tf

#     x = _preprocess_pil(pil_img, backbone_name, img_size)
#     x_batch = np.repeat(x, n_passes, axis=0)
#     probs = model(x_batch, training=True).numpy()  # training=True keeps Dropout active
#     mean_probs = probs.mean(axis=0)
#     eps = 1e-9
#     predictive_entropy = float(-np.sum(mean_probs * np.log(mean_probs + eps)))
#     return mean_probs, predictive_entropy


# # ============================================================== #
# # 5. Heuristic, explicitly-non-forensic screening cues
# # ============================================================== #
# def is_blank_surface(pil_img: Image.Image) -> tuple[bool, float]:
#     gray = ImageOps.grayscale(pil_img).resize((64, 64))
#     std_dev = float(np.std(np.array(gray)))
#     return std_dev < cfg.BLANK_SURFACE_STD_THRESHOLD, std_dev


# def color_signature_distance(pil_img: Image.Image, predicted_class: str) -> float:
#     if predicted_class not in RBI_COLOR_CENTROIDS:
#         return 0.0
#     small = pil_img.convert("RGB").resize((100, 60))
#     arr = np.array(small)
#     center = arr[12:48, 20:80]
#     mean_rgb = [float(np.mean(center[:, :, c])) for c in range(3)]
#     target = RBI_COLOR_CENTROIDS[predicted_class]
#     return float(np.sqrt(sum((mean_rgb[i] - target[i]) ** 2 for i in range(3))))


# def texture_sharpness_score(pil_img: Image.Image) -> float:
#     """Variance of Laplacian — a classic, cheap blur/quality proxy. Very
#     low values often indicate a photo of a screen or a poor photocopy
#     rather than a genuine note photographed directly."""
#     gray = np.array(ImageOps.grayscale(pil_img).resize((300, 150)), dtype=np.float64)
#     kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)
#     from scipy.signal import convolve2d  # noqa: local import, optional dep
#     lap = convolve2d(gray, kernel, mode="same", boundary="symm")
#     return float(lap.var())


# def security_roi_consistency_score(pil_img: Image.Image, predicted_class: str) -> float:
#     """Cheap edge-density proxy inside the approximate security-thread /
#     see-through-register ROI. Higher edge density in that narrow vertical
#     band is *consistent with* the presence of print detail there; it is
#     NOT a verified security-thread detector. Always disclosed as such."""
#     if predicted_class not in cfg.SECURITY_THREAD_ROI:
#         return 1.0
#     w, h = pil_img.size
#     x0f, y0f, x1f, y1f = cfg.SECURITY_THREAD_ROI[predicted_class]
#     box = (int(x0f * w), int(y0f * h), int(x1f * w), int(y1f * h))
#     roi = ImageOps.grayscale(pil_img.crop(box))
#     if roi.size[0] < 2 or roi.size[1] < 2:
#         return 1.0
#     arr = np.array(roi, dtype=np.float64)
#     gx = np.abs(np.diff(arr, axis=1)).mean()
#     gy = np.abs(np.diff(arr, axis=0)).mean()
#     edge_density = (gx + gy) / 2.0
#     return float(edge_density / 255.0)

























"""
dhan_drishti_utils.py
======================
Shared building blocks used by BOTH train_model.py and app.py, so the exact
same preprocessing / inference logic is guaranteed at train and serve time
(a common, reviewer-flagged bug in student CNN papers is train/serve skew).

Contents
--------
1. BACKBONE_REGISTRY      -> per-architecture builder + preprocess_input
2. build_model()           -> backbone + GAP + Dropout(MC) + Dense head
3. build_head_submodel()   -> split model for Grad-CAM (conv-base vs head)
4. make_gradcam_heatmap()  -> Grad-CAM heatmap
5. overlay_gradcam()       -> heatmap -> RGB overlay image
6. single_predict()        -> one deterministic forward pass
7. tta_predict()           -> test-time augmentation averaged prediction
8. mc_dropout_predict()    -> MC-Dropout: mean probs + predictive entropy
9. color_signature_distance() / texture_sharpness_score() /
   security_roi_consistency_score()  -> cheap, explicitly-heuristic
   screening cues (never described as "fake note detection" — see README)
"""

import io
import numpy as np
from PIL import Image, ImageOps

import config as cfg

# TensorFlow is imported lazily inside functions that need it so that pure
# heuristic/image-processing helpers (used by the Streamlit UI even before
# a model is loaded) stay fast and import-light.

CLASS_LABELS = cfg.CLASS_NAMES
RBI_COLOR_CENTROIDS = cfg.RBI_COLOR_CENTROIDS


# ============================================================== #
# 1. Backbone registry
# ============================================================== #
def _lazy_backbone_registry():
    import tensorflow as tf
    from tensorflow.keras.applications import (
        MobileNetV2, EfficientNetV2B0, ConvNeXtTiny,
    )
    from tensorflow.keras.applications import mobilenet_v2, efficientnet_v2, convnext

    return {
        "mobilenetv2": {
            "build": MobileNetV2,
            "preprocess_input": mobilenet_v2.preprocess_input,
        },
        "efficientnetv2b0": {
            "build": EfficientNetV2B0,
            "preprocess_input": efficientnet_v2.preprocess_input,
        },
        "convnexttiny": {
            "build": ConvNeXtTiny,
            "preprocess_input": convnext.preprocess_input,
        },
    }


class _LazyRegistry(dict):
    """Populates itself with real TF objects on first access, so importing
    this module doesn't force a slow TensorFlow import for callers that
    only need the pure-numpy heuristic functions."""

    def _ensure(self):
        if not self:
            self.update(_lazy_backbone_registry())

    def __getitem__(self, key):
        self._ensure()
        return super().__getitem__(key)

    def __contains__(self, key):
        self._ensure()
        return super().__contains__(key)


BACKBONE_REGISTRY = _LazyRegistry()


# ============================================================== #
# 2. Model builder
# ============================================================== #
def build_model(backbone_name: str, num_classes: int, dropout_rate: float = cfg.DROPOUT_RATE):
    """Backbone (ImageNet weights, initially frozen) + GAP + MC-Dropout +
    Dense softmax head. Returns (model, base_model) so callers can freeze /
    unfreeze `base_model` for the two-phase training schedule."""
    import tensorflow as tf
    from tensorflow.keras import layers, Model

    entry = BACKBONE_REGISTRY[backbone_name]
    base_model = entry["build"](
        include_top=False, weights="imagenet",
        input_shape=(*cfg.IMG_SIZE, 3), pooling=None,
    )
    base_model.trainable = False

    inputs = layers.Input(shape=(*cfg.IMG_SIZE, 3), name="input_image")
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(dropout_rate, name="mc_dropout_1")(x)
    x = layers.Dense(256, activation="relu", name="dense_head")(x)
    x = layers.Dropout(dropout_rate, name="mc_dropout_2")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = Model(inputs, outputs, name=f"dhan_drishti_{backbone_name}")
    return model, base_model


def build_head_submodel(model):
    """Splits a trained model into (base_model, head_model) for Grad-CAM:
    base_model: input -> last 4D feature map of the backbone submodel
    head_model: that feature map -> final predictions
    Works generically by locating the nested backbone sub-model (the only
    layer of type Functional/Model inside our architecture).

    NOTE: we intentionally read shapes via `layer.output` (a tensor) inside
    a try/except rather than the older `layer.output_shape` property.
    Some architectures (e.g. ConvNeXt, which uses LayerNormalization blocks
    with multiple inbound call-sites) raise AttributeError on
    `output_shape` under recent Keras/TF versions — wrapping the lookup
    lets us just skip those layers and keep searching backwards for a
    genuine 4D conv-style feature map."""
    import tensorflow as tf
    from tensorflow.keras import Model

    backbone_layer = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            backbone_layer = layer
            break
    if backbone_layer is None:
        raise ValueError("Could not locate nested backbone sub-model for Grad-CAM.")

    last_conv_layer = None
    for layer in reversed(backbone_layer.layers):
        try:
            out_tensor = layer.output
            shape = out_tensor.shape
        except Exception:
            continue
        if shape is not None and len(shape) == 4:
            last_conv_layer = layer
            break
    if last_conv_layer is None:
        raise ValueError("Could not locate a 4D conv feature map inside the backbone.")

    base_submodel = Model(backbone_layer.input, last_conv_layer.output, name="gradcam_base")

    # Head: feature map -> rest of the original model's layers (GAP onward)
    # Use the tensor's `.shape` (not the layer's `.output_shape` property,
    # which is what raised AttributeError on ConvNeXt's LayerNormalization
    # layers above) to build the matching Input spec.
    feat_shape = tuple(last_conv_layer.output.shape[1:])
    feat_input = tf.keras.Input(shape=feat_shape)
    x = feat_input
    started = False
    for layer in model.layers:
        if layer is backbone_layer:
            started = True
            continue
        if not started:
            continue
        x = layer(x)
    head_submodel = Model(feat_input, x, name="gradcam_head")
    return head_submodel, base_submodel


# ============================================================== #
# 3. Grad-CAM
# ============================================================== #
def make_gradcam_heatmap(preprocessed_batch, base_model, head_model, pred_index=None):
    import tensorflow as tf

    with tf.GradientTape() as tape:
        conv_output = base_model(preprocessed_batch)
        tape.watch(conv_output)
        preds = head_model(conv_output)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_output = conv_output[0]
    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy(), int(pred_index)


def overlay_gradcam(pil_img: Image.Image, heatmap: np.ndarray, alpha: float = 0.4) -> Image.Image:
    import matplotlib.cm as cm

    img = pil_img.convert("RGB")
    heatmap_img = Image.fromarray(np.uint8(255 * heatmap)).resize(img.size)
    heatmap_arr = np.array(heatmap_img)

    jet = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap_arr]
    jet_heatmap = np.uint8(jet_heatmap * 255)
    jet_heatmap_img = Image.fromarray(jet_heatmap).resize(img.size)

    blended = Image.blend(img, jet_heatmap_img, alpha=alpha)
    return blended


# ============================================================== #
# 4. Inference helpers (shared train/serve preprocessing)
# ============================================================== #
def _preprocess_pil(pil_img: Image.Image, backbone_name: str, img_size=None) -> np.ndarray:
    img_size = img_size or cfg.IMG_SIZE
    preprocess_fn = BACKBONE_REGISTRY[backbone_name]["preprocess_input"]
    resized = pil_img.convert("RGB").resize(img_size)
    arr = np.array(resized).astype(np.float32)
    arr = preprocess_fn(arr.copy())
    return np.expand_dims(arr, axis=0)


def single_predict(model, pil_img: Image.Image, backbone_name: str, img_size=None) -> np.ndarray:
    x = _preprocess_pil(pil_img, backbone_name, img_size)
    return model.predict(x, verbose=0)[0]


def tta_predict(model, pil_img: Image.Image, backbone_name: str, img_size=None, n_aug: int = 6) -> np.ndarray:
    """Test-time augmentation: small, currency-safe perturbations only.
    NOTE: we deliberately never horizontally flip a banknote — flipping
    mirrors the printed Devanagari/English numerals and portrait, which
    would corrupt rather than augment the signal a real user's phone
    camera would produce."""
    img_size = img_size or cfg.IMG_SIZE
    rng = np.random.default_rng(cfg.SEED)
    base = pil_img.convert("RGB")

    variants = [base]
    for _ in range(n_aug - 1):
        angle = float(rng.uniform(-8, 8))
        brightness = float(rng.uniform(0.85, 1.15))
        v = base.rotate(angle, resample=Image.BILINEAR, expand=False, fillcolor=(255, 255, 255))
        arr = np.array(v).astype(np.float32) * brightness
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        variants.append(Image.fromarray(arr))

    preprocess_fn = BACKBONE_REGISTRY[backbone_name]["preprocess_input"]
    batch = np.stack([
        preprocess_fn(np.array(v.resize(img_size)).astype(np.float32).copy())
        for v in variants
    ], axis=0)
    preds = model.predict(batch, verbose=0)
    return preds.mean(axis=0)


def mc_dropout_predict(model, pil_img: Image.Image, backbone_name: str, img_size=None,
                        n_passes: int = cfg.MC_DROPOUT_PASSES):
    """Monte-Carlo Dropout: keeps Dropout layers ACTIVE at inference and
    runs n_passes stochastic forward passes. Returns:
      mean_probs      - averaged softmax vector (use for the prediction)
      predictive_entropy - epistemic-uncertainty proxy in nats; high value
                            means the model is unsure and the UI should
                            recommend a re-scan rather than trusting the
                            top-1 label."""
    import tensorflow as tf

    x = _preprocess_pil(pil_img, backbone_name, img_size)
    x_batch = np.repeat(x, n_passes, axis=0)
    probs = model(x_batch, training=True).numpy()  # training=True keeps Dropout active
    mean_probs = probs.mean(axis=0)
    eps = 1e-9
    predictive_entropy = float(-np.sum(mean_probs * np.log(mean_probs + eps)))
    return mean_probs, predictive_entropy


# ============================================================== #
# 5. Heuristic, explicitly-non-forensic screening cues
# ============================================================== #
def is_blank_surface(pil_img: Image.Image) -> tuple[bool, float]:
    gray = ImageOps.grayscale(pil_img).resize((64, 64))
    std_dev = float(np.std(np.array(gray)))
    return std_dev < cfg.BLANK_SURFACE_STD_THRESHOLD, std_dev


def color_signature_distance(pil_img: Image.Image, predicted_class: str) -> float:
    if predicted_class not in RBI_COLOR_CENTROIDS:
        return 0.0
    small = pil_img.convert("RGB").resize((100, 60))
    arr = np.array(small)
    center = arr[12:48, 20:80]
    mean_rgb = [float(np.mean(center[:, :, c])) for c in range(3)]
    target = RBI_COLOR_CENTROIDS[predicted_class]
    return float(np.sqrt(sum((mean_rgb[i] - target[i]) ** 2 for i in range(3))))


def texture_sharpness_score(pil_img: Image.Image) -> float:
    """Variance of Laplacian — a classic, cheap blur/quality proxy. Very
    low values often indicate a photo of a screen or a poor photocopy
    rather than a genuine note photographed directly."""
    gray = np.array(ImageOps.grayscale(pil_img).resize((300, 150)), dtype=np.float64)
    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)
    from scipy.signal import convolve2d  # noqa: local import, optional dep
    lap = convolve2d(gray, kernel, mode="same", boundary="symm")
    return float(lap.var())


def security_roi_consistency_score(pil_img: Image.Image, predicted_class: str) -> float:
    """Cheap edge-density proxy inside the approximate security-thread /
    see-through-register ROI. Higher edge density in that narrow vertical
    band is *consistent with* the presence of print detail there; it is
    NOT a verified security-thread detector. Always disclosed as such."""
    if predicted_class not in cfg.SECURITY_THREAD_ROI:
        return 1.0
    w, h = pil_img.size
    x0f, y0f, x1f, y1f = cfg.SECURITY_THREAD_ROI[predicted_class]
    box = (int(x0f * w), int(y0f * h), int(x1f * w), int(y1f * h))
    roi = ImageOps.grayscale(pil_img.crop(box))
    if roi.size[0] < 2 or roi.size[1] < 2:
        return 1.0
    arr = np.array(roi, dtype=np.float64)
    gx = np.abs(np.diff(arr, axis=1)).mean()
    gy = np.abs(np.diff(arr, axis=0)).mean()
    edge_density = (gx + gy) / 2.0
    return float(edge_density / 255.0)