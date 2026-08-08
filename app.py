# # """
# # app.py
# # ======
# # Dhan Drishti — AI-Powered Indian Banknote Recognition & Assistive Cash-Tally
# # Portal. Streamlit front-end for the ensemble trained by train_model.py.

# # Unique contributions surfaced in this UI (see README for the full novelty
# # statement aimed at the IEEE Access submission):

# #   1. Weighted multi-backbone ensemble (diversity reduces correlated errors)
# #   2. MC-Dropout epistemic-uncertainty gating — low-confidence *and* high-
# #      entropy predictions are flagged for re-scan instead of silently shown
# #   3. Grad-CAM visual explanation of *why* the top backbone predicted a class
# #   4. Test-Time Augmentation (currency-safe: no horizontal flip)
# #   5. Transparent, clearly-labelled heuristic screening cues (color
# #      signature + texture sharpness + security-thread ROI consistency) —
# #      never oversold as "counterfeit detection"
# #   6. Assistive Cash-Tally mode: running count + total value of scanned
# #      notes, with multilingual voice feedback, for low-vision / accessibility
# #      use cases (a genuinely under-served angle vs. plain classification)
# # """

# # import os
# # import time

# # import numpy as np
# # import streamlit as st
# # from PIL import Image, ImageOps
# # import streamlit.components.v1 as components

# # import config as cfg
# # import dhan_drishti_utils as ddu
# # from note_metadata import NOTE_METADATA, AUDIO_DICTIONARY

# # st.set_page_config(
# #     page_title="Dhan Drishti — AI Currency Detector",
# #     page_icon="🪙",
# #     layout="wide",
# #     initial_sidebar_state="expanded",
# # )

# # # ------------------------------------------------------------------ #
# # # Styling
# # # ------------------------------------------------------------------ #
# # st.markdown("""
# # <style>
# #     .stApp { background: linear-gradient(135deg, #120e1e 0%, #1a152e 100%); color: #f0ecf9; }
# #     .banner {
# #         background: linear-gradient(135deg, rgba(138,87,230,0.15) 0%, rgba(74,20,140,0.25) 100%);
# #         border: 1px solid rgba(138,87,230,0.25); border-radius: 16px; padding: 28px;
# #         margin-bottom: 22px; text-align: center; box-shadow: 0 8px 32px 0 rgba(0,0,0,0.3);
# #     }
# #     .banner h1 {
# #         font-size: 3rem !important; font-weight: 800 !important;
# #         background: linear-gradient(90deg, #a78bfa 0%, #c084fc 100%);
# #         -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 4px;
# #     }
# #     .banner p { font-size: 1.05rem; color: #c4b5fd; margin: 0; }
# #     .result-display {
# #         text-align: center; padding: 18px; border-radius: 12px;
# #         background: linear-gradient(135deg, rgba(138,87,230,0.1) 0%, rgba(74,20,140,0.1) 100%);
# #         border: 1px solid rgba(138,87,230,0.2);
# #     }
# #     .result-val { font-size: 2.6rem; font-weight: 800; color: #ffb84d; margin: 8px 0; }
# #     .result-series { font-size: 1.1rem; color: #a78bfa; font-weight: 500; }
# #     .stat-row { display: flex; justify-content: space-between; margin-top: 12px; padding: 9px 14px;
# #                 background: rgba(10,5,20,0.4); border-radius: 8px; }
# #     .stat-label { color: #c4b5fd; font-weight: 500; }
# #     .stat-val { color: #34d399; font-weight: 700; }
# #     .stat-val.red { color: #f87171; }
# #     .model-row { display: flex; justify-content: space-between; padding: 7px 14px;
# #                  background: rgba(10,5,20,0.3); border-radius: 6px; margin-top: 5px; font-size: 0.88rem; }
# #     .alert-banner { padding: 14px; border-radius: 10px; margin-bottom: 14px; font-weight: 600;
# #                     display: flex; align-items: center; gap: 10px; }
# #     .alert-banner.warning { background-color: rgba(251,191,36,0.15); border: 1px solid rgba(251,191,36,0.3); color: #fbbf24; }
# #     .alert-banner.error { background-color: rgba(239,68,68,0.15); border: 1px solid rgba(239,68,68,0.3); color: #f87171; }
# #     .alert-banner.success { background-color: rgba(52,211,153,0.15); border: 1px solid rgba(52,211,153,0.3); color: #34d399; }
# #     .tally-item { display: flex; justify-content: space-between; padding: 6px 10px;
# #                   background: rgba(10,5,20,0.35); border-radius: 6px; margin-bottom: 4px; font-size: 0.9rem; }
# #     .tally-total { text-align: center; padding: 14px; border-radius: 10px; margin-top: 8px;
# #                    background: linear-gradient(135deg, rgba(52,211,153,0.15), rgba(52,211,153,0.05));
# #                    border: 1px solid rgba(52,211,153,0.3); font-size: 1.6rem; font-weight: 800; color: #34d399; }
# #     h3 { color: #e2dcf4 !important; font-weight: 600 !important; }
# # </style>
# # """, unsafe_allow_html=True)


# # # ------------------------------------------------------------------ #
# # # Model / ensemble loading
# # # ------------------------------------------------------------------ #
# # @st.cache_resource(show_spinner=False)
# # def load_ensemble():
# #     import json
# #     config_path = os.path.join(cfg.MODEL_DIR, "ensemble_config.json")
# #     if not os.path.exists(config_path):
# #         return None, None
# #     with open(config_path, "r", encoding="utf-8") as f:
# #         econfig = json.load(f)

# #     import tensorflow as tf
# #     models = {}
# #     for entry in econfig["backbones"]:
# #         model_path = os.path.join(cfg.MODEL_DIR, entry["file"])
# #         if os.path.exists(model_path):
# #             models[entry["name"]] = tf.keras.models.load_model(model_path)
# #     if not models:
# #         return None, None
# #     return econfig, models


# # @st.cache_resource(show_spinner=False)
# # def load_metrics_summary():
# #     import json
# #     path = os.path.join(cfg.MODEL_DIR, "metrics_summary.json")
# #     if os.path.exists(path):
# #         with open(path, "r", encoding="utf-8") as f:
# #             return json.load(f)
# #     return None


# # def simulate_prediction(image: Image.Image, class_names):
# #     """Demo-mode fallback (no trained ensemble present yet) so the UI stays
# #     explorable. Clearly labelled everywhere it is used — never silently
# #     substituted for a real prediction."""
# #     dist = ddu.color_signature_distance
# #     best_class, best_score = "background", float("inf")
# #     for cls in cfg.RBI_COLOR_CENTROIDS:
# #         score = dist(image, cls)
# #         if score < best_score:
# #             best_score, best_class = score, cls
# #     idx = class_names.index(best_class)
# #     rng = np.random.default_rng()
# #     probs = rng.dirichlet(np.ones(len(class_names)) * 0.4)
# #     probs[idx] += 1.2
# #     return probs / probs.sum()


# # def speak(text: str, lang: str = "en-IN"):
# #     safe_text = text.replace("\\", "\\\\").replace('"', '\\"')
# #     js_code = f"""
# #     <script>
# #     if ('speechSynthesis' in window) {{
# #         window.speechSynthesis.cancel();
# #         var utterance = new SpeechSynthesisUtterance("{safe_text}");
# #         utterance.lang = "{lang}";
# #         var voices = window.speechSynthesis.getVoices();
# #         var matchVoice = voices.find(v => v.lang.startsWith("{lang[:2]}"));
# #         if (matchVoice) utterance.voice = matchVoice;
# #         window.speechSynthesis.speak(utterance);
# #     }}
# #     </script>
# #     """
# #     components.html(js_code, height=0, width=0)


# # # ------------------------------------------------------------------ #
# # # Banner
# # # ------------------------------------------------------------------ #
# # st.markdown("""
# # <div class="banner">
# #     <h1>Dhan Drishti</h1>
# #     <p>AI-Powered Indian Banknote Recognition &amp; Assistive Cash-Counting Portal</p>
# # </div>
# # """, unsafe_allow_html=True)

# # with st.spinner("Initializing ensemble..."):
# #     ensemble_config, ensemble_models = load_ensemble()

# # DEMO_MODE = ensemble_models is None
# # CLASS_NAMES = ensemble_config["class_names"] if ensemble_config else cfg.CLASS_NAMES

# # if DEMO_MODE:
# #     st.warning(
# #         "🧪 **Demo Mode** — no trained ensemble found in `model/ensemble_config.json`. "
# #         "The scanner below shows a **simulated** prediction (color-heuristic only) so the UI "
# #         "stays fully explorable. Populate `Datasets/<class>/` with images and run "
# #         "`python train_model.py` to enable real ensemble predictions."
# #     )

# # metrics_summary = load_metrics_summary()

# # # ------------------------------------------------------------------ #
# # # Session state — assistive cash tally
# # # ------------------------------------------------------------------ #
# # if "wallet" not in st.session_state:
# #     st.session_state["wallet"] = []   # list of {"class": str, "value": int, "confidence": float}
# # if "target_note" not in st.session_state:
# #     st.session_state["target_note"] = "background"
# # if "last_uncertain" not in st.session_state:
# #     st.session_state["last_uncertain"] = False

# # # ------------------------------------------------------------------ #
# # # Sidebar
# # # ------------------------------------------------------------------ #
# # st.sidebar.markdown("### 🎛️ Portal Settings")

# # selected_lang = st.sidebar.selectbox(
# #     "Announcement Language:", options=list(cfg.LANG_LABELS.keys()),
# #     format_func=lambda x: cfg.LANG_LABELS[x],
# # )
# # confidence_thresh = st.sidebar.slider("Confidence Threshold:", 30, 95, 75, 5) / 100.0
# # auto_announce = st.sidebar.checkbox("Auto-Announce Results", value=True)
# # use_tta = st.sidebar.checkbox("Enable Test-Time Augmentation (slower, steadier)", value=False, disabled=DEMO_MODE)
# # use_mc_uncertainty = st.sidebar.checkbox("Enable MC-Dropout Uncertainty Check", value=True, disabled=DEMO_MODE)
# # show_gradcam = st.sidebar.checkbox("Show Grad-CAM Explainability", value=True, disabled=DEMO_MODE)

# # st.sidebar.markdown("---")
# # st.sidebar.markdown("### 💰 Assistive Cash Tally")
# # wallet = st.session_state["wallet"]
# # if wallet:
# #     for i, item in enumerate(wallet[-8:]):
# #         st.sidebar.markdown(
# #             f'<div class="tally-item"><span>{NOTE_METADATA[item["class"]]["title"]}</span>'
# #             f'<span>₹{item["value"]}</span></div>', unsafe_allow_html=True,
# #         )
# #     total = sum(item["value"] for item in wallet)
# #     st.sidebar.markdown(f'<div class="tally-total">Total: ₹{total}</div>', unsafe_allow_html=True)
# #     st.sidebar.caption(f"{len(wallet)} note(s) tallied this session.")
# #     if st.sidebar.button("🗑️ Reset Tally"):
# #         st.session_state["wallet"] = []
# #         st.rerun()
# # else:
# #     st.sidebar.info("Scan notes and tap **Add to Tally** to build a running count — handy for quickly totalling cash, e.g. for visually impaired users.")

# # st.sidebar.markdown("---")
# # st.sidebar.markdown("### 🔬 Ensemble Information")
# # if ensemble_config:
# #     lines = [f"**Backbones ({len(ensemble_config['backbones'])}):**"]
# #     for entry in sorted(ensemble_config["backbones"], key=lambda e: -e["weight"]):
# #         lines.append(f"- {entry['display_name']}: weight `{entry['weight']:.3f}`, val-acc `{entry['val_accuracy']*100:.1f}%`")
# #     lines.append(f"\n**Input**: {ensemble_config['img_size'][0]}x{ensemble_config['img_size'][1]}x3 RGB")
# #     lines.append(f"**Classes**: {len(CLASS_NAMES)}")
# #     st.sidebar.info("\n".join(lines))
# # else:
# #     st.sidebar.info(
# #         "**Planned architecture**: MobileNetV2 + EfficientNetV2-B0 + ConvNeXt-Tiny "
# #         "(validation-accuracy-weighted soft-vote ensemble)\n\n**Input**: 224x224x3 RGB"
# #     )

# # # ------------------------------------------------------------------ #
# # # Tabs
# # # ------------------------------------------------------------------ #
# # tab1, tab2, tab3 = st.tabs(["🔍 Currency Scanner", "🛡️ RBI Verification Guide", "📊 Model Diagnostics"])

# # # ============================= TAB 1 =============================== #
# # with tab1:
# #     col1, col2 = st.columns([1.2, 1])

# #     with col1:
# #         st.markdown("### 📸 Input Banknote Scan")
# #         input_mode = st.radio("Select Input Mode:", ["Live Camera Capture", "Upload Single Image"], horizontal=True)

# #         raw_image = None
# #         if input_mode == "Live Camera Capture":
# #             camera_image = st.camera_input("Scan your banknote")
# #             if camera_image:
# #                 raw_image = Image.open(camera_image)
# #         else:
# #             file_uploaded = st.file_uploader("Choose a note image...", type=["jpg", "jpeg", "png"])
# #             if file_uploaded:
# #                 raw_image = Image.open(file_uploaded)
# #                 st.image(raw_image, caption="Uploaded Image Preview", use_container_width=True)

# #     with col2:
# #         st.markdown("### 🏷️ Verification Analysis")

# #         if raw_image is not None:
# #             is_blank, std_dev = ddu.is_blank_surface(raw_image)
# #             inference_start = time.time()
# #             probs, predictive_entropy, per_model_probs = None, None, {}

# #             if is_blank:
# #                 predicted_class, confidence, inference_ms = "background", 1.0, 0
# #                 st.markdown('<div class="alert-banner warning">⚠️ Textureless plain surface detected. Please scan a real banknote.</div>', unsafe_allow_html=True)
# #                 st.markdown(f"""
# #                 <div class="result-display">
# #                     <span class="result-series">No Currency Profile Detected</span>
# #                     <div class="result-val">Background</div>
# #                 </div>""", unsafe_allow_html=True)
# #                 st.markdown(f"""
# #                 <div class="stat-row"><span class="stat-label">Grayscale Variance</span><span class="stat-val red">{std_dev:.2f} (low)</span></div>
# #                 """, unsafe_allow_html=True)
# #                 st.session_state["target_note"] = "background"

# #             else:
# #                 if DEMO_MODE:
# #                     probs = simulate_prediction(raw_image, CLASS_NAMES)
# #                     time.sleep(0.15)
# #                 else:
# #                     weights_sum = sum(e["weight"] for e in ensemble_config["backbones"])
# #                     ens_probs = None
# #                     for entry in ensemble_config["backbones"]:
# #                         model = ensemble_models.get(entry["name"])
# #                         if model is None:
# #                             continue
# #                         if use_tta:
# #                             p = ddu.tta_predict(model, raw_image, entry["name"], img_size=tuple(ensemble_config["img_size"]))
# #                         else:
# #                             p = ddu.single_predict(model, raw_image, entry["name"], img_size=tuple(ensemble_config["img_size"]))
# #                         per_model_probs[entry["name"]] = p
# #                         w = entry["weight"] / weights_sum
# #                         ens_probs = p * w if ens_probs is None else ens_probs + p * w
# #                     probs = ens_probs

# #                     if use_mc_uncertainty:
# #                         best_entry = max(ensemble_config["backbones"], key=lambda e: e["weight"])
# #                         best_model = ensemble_models[best_entry["name"]]
# #                         _, predictive_entropy = ddu.mc_dropout_predict(
# #                             best_model, raw_image, best_entry["name"], img_size=tuple(ensemble_config["img_size"])
# #                         )

# #                 max_idx = int(np.argmax(probs))
# #                 predicted_class = CLASS_NAMES[max_idx]
# #                 confidence = float(probs[max_idx])
# #                 inference_ms = int((time.time() - inference_start) * 1000)

# #                 if DEMO_MODE:
# #                     st.markdown('<div class="alert-banner warning">🧪 Demo Mode — simulated prediction, not a real ensemble output.</div>', unsafe_allow_html=True)

# #                 # ---- heuristic screening cues (explicitly non-forensic) ----
# #                 color_dist = ddu.color_signature_distance(raw_image, predicted_class)
# #                 sharpness = ddu.texture_sharpness_score(raw_image) if not DEMO_MODE else None
# #                 roi_score = ddu.security_roi_consistency_score(raw_image, predicted_class)
# #                 color_flag = predicted_class != "background" and color_dist > cfg.COLOR_SIGNATURE_DIST_THRESHOLD
# #                 texture_flag = (sharpness is not None) and sharpness < cfg.TEXTURE_SHARPNESS_THRESHOLD
# #                 roi_flag = predicted_class != "background" and roi_score < cfg.ROI_CONSISTENCY_THRESHOLD
# #                 is_uncertain = (predictive_entropy is not None) and (predictive_entropy > cfg.MC_ENTROPY_REJECT_THRESHOLD)
# #                 st.session_state["last_uncertain"] = is_uncertain

# #                 flags = [n for n, f in [("color", color_flag), ("texture", texture_flag), ("ROI structure", roi_flag)] if f]
# #                 validation_passed = len(flags) == 0
# #                 verification_msg = "No screening flags raised" if validation_passed else f"Screening flag(s): {', '.join(flags)}"

# #                 meta = NOTE_METADATA.get(predicted_class, NOTE_METADATA["background"])
# #                 st.session_state["target_note"] = predicted_class

# #                 if is_uncertain:
# #                     st.markdown(
# #                         '<div class="alert-banner warning">🤔 The model is uncertain about this prediction '
# #                         '(high MC-Dropout entropy). Please re-scan with better lighting/framing.</div>',
# #                         unsafe_allow_html=True,
# #                     )
# #                 elif confidence < confidence_thresh:
# #                     st.markdown('<div class="alert-banner warning">⚠️ Low confidence score. Improve lighting and center the note.</div>', unsafe_allow_html=True)
# #                 elif not validation_passed:
# #                     st.markdown(
# #                         '<div class="alert-banner error">🛑 Heuristic screening flag raised. This is a screening '
# #                         'cue, NOT certified counterfeit detection — verify manually against the RBI guide tab.</div>',
# #                         unsafe_allow_html=True,
# #                     )
# #                 else:
# #                     st.markdown('<div class="alert-banner success">✅ No screening flags raised. Still verify security features manually for high-value transactions.</div>', unsafe_allow_html=True)

# #                 st.markdown(f"""
# #                 <div class="result-display">
# #                     <span class="result-series">{meta['series']}</span>
# #                     <div class="result-val">{meta['title']}</div>
# #                     <span class="badge" style="background-color:{meta['accent']}; padding:5px 12px; border-radius:20px;">Ensemble Match: {confidence*100:.1f}%</span>
# #                 </div>""", unsafe_allow_html=True)

# #                 stat_rows = f"""
# #                 <div class="stat-row"><span class="stat-label">Inference Speed{' (TTA)' if use_tta else ''}</span><span class="stat-val">{inference_ms} ms</span></div>
# #                 <div class="stat-row"><span class="stat-label">Screening Engine</span><span class="stat-val {'red' if not validation_passed else ''}">{verification_msg}</span></div>
# #                 """
# #                 if predictive_entropy is not None:
# #                     stat_rows += f'<div class="stat-row"><span class="stat-label">Predictive Entropy (MC-Dropout)</span><span class="stat-val {"red" if is_uncertain else ""}">{predictive_entropy:.3f} nats</span></div>'
# #                 st.markdown(stat_rows, unsafe_allow_html=True)

# #                 if per_model_probs:
# #                     rows = ""
# #                     for entry in sorted(ensemble_config["backbones"], key=lambda e: -e["weight"]):
# #                         name = entry["name"]
# #                         if name not in per_model_probs:
# #                             continue
# #                         p = per_model_probs[name]
# #                         model_pred = CLASS_NAMES[int(np.argmax(p))]
# #                         model_conf = float(np.max(p))
# #                         agree = "✅" if model_pred == predicted_class else "⚠️"
# #                         rows += (f'<div class="model-row">{agree} <b>{entry["display_name"]}</b> (w={entry["weight"]:.2f}) '
# #                                  f'→ {NOTE_METADATA.get(model_pred, {}).get("title", model_pred)} @ {model_conf*100:.1f}%</div>')
# #                     st.markdown("##### 🧩 Per-Model Breakdown")
# #                     st.markdown(rows, unsafe_allow_html=True)

# #                 # ---- assistive cash tally ----
# #                 if predicted_class != "background" and not is_uncertain:
# #                     if st.button(f"➕ Add {meta['title']} to Tally"):
# #                         st.session_state["wallet"].append({
# #                             "class": predicted_class, "value": cfg.DENOMINATION_VALUE[predicted_class],
# #                             "confidence": confidence,
# #                         })
# #                         st.rerun()

# #                 # ---- voice announcement ----
# #                 phrase_book = AUDIO_DICTIONARY.get(selected_lang, AUDIO_DICTIONARY["en-IN"])
# #                 announce_phrase = phrase_book.get(predicted_class, phrase_book["background"])
# #                 if is_uncertain:
# #                     announce_phrase += ". " + phrase_book["warning_uncertain"]
# #                 if st.button("🔊 Play Voice Announcement"):
# #                     speak(announce_phrase, selected_lang)
# #                 if auto_announce:
# #                     speak(announce_phrase, selected_lang)

# #                 # ---- Grad-CAM ----
# #                 if show_gradcam and not DEMO_MODE and ensemble_models:
# #                     try:
# #                         best_entry = max(ensemble_config["backbones"], key=lambda e: e["weight"])
# #                         best_model = ensemble_models[best_entry["name"]]
# #                         head_model, base_model = ddu.build_head_submodel(best_model)
# #                         preprocess_fn = ddu.BACKBONE_REGISTRY[best_entry["name"]]["preprocess_input"]
# #                         img_resized = raw_image.convert("RGB").resize(tuple(ensemble_config["img_size"]))
# #                         x = preprocess_fn(np.array(img_resized).astype(np.float32).copy())
# #                         x = np.expand_dims(x, axis=0)
# #                         heatmap, _ = ddu.make_gradcam_heatmap(x, base_model, head_model, pred_index=max_idx)
# #                         overlay = ddu.overlay_gradcam(img_resized, heatmap)
# #                         st.markdown(f"##### 🔥 Grad-CAM — where **{best_entry['display_name']}** looked")
# #                         st.image(overlay, use_container_width=True, caption="Warmer colors = regions most influential to the prediction.")
# #                     except Exception as e:
# #                         st.caption(f"(Grad-CAM unavailable for this image: {e})")

# #             if probs is not None:
# #                 st.markdown("### 📊 Ensemble Probability Distribution")
# #                 chart_data = {NOTE_METADATA.get(CLASS_NAMES[i], {}).get("title", CLASS_NAMES[i]): float(probs[i]) for i in range(len(CLASS_NAMES))}
# #                 chart_data = dict(sorted(chart_data.items(), key=lambda item: item[1], reverse=True))
# #                 st.bar_chart(chart_data, horizontal=True)
# #         else:
# #             st.info("💡 Scan a note or upload an image to view verification and diagnostics.")
# #             st.session_state["target_note"] = "background"

# # # ============================= TAB 2 =============================== #
# # with tab2:
# #     st.markdown("### 🛡️ RBI Official Banknote Security Verification Guide")
# #     target_class = st.session_state.get("target_note", "background")
# #     meta = NOTE_METADATA.get(target_class, NOTE_METADATA["background"])

# #     if target_class == "background":
# #         st.info("💡 Details will auto-load here once a note is scanned in the Scanner tab.")

# #     col1, col2 = st.columns([1.5, 1])
# #     with col1:
# #         st.markdown(f"#### 🔎 Security Specifications — **{meta['title']}**")
# #         st.write(f"**Year of Release:** {meta['year']}")
# #         st.write(f"**Dimensions:** {meta['dimensions']}")
# #         st.write(f"**Dominant Color:** {meta['color']}")
# #         st.write(f"**Motif (Reverse):** {meta['motif']}")
# #         st.markdown("##### 📍 Mandatory Security Markings to Verify Manually:")
# #         for feature in meta["features"]:
# #             st.markdown(f"- 🔳 **{feature}**")
# #     with col2:
# #         st.markdown("#### 📜 Historical Context")
# #         st.markdown(f"""
# #         <div style="background-color: rgba(138,87,230,0.1); border-left: 4px solid {meta['accent']}; padding: 15px; border-radius: 4px;">
# #             <p style="font-style: italic; color: #e2dcf4;">{meta['funFact']}</p>
# #         </div>""", unsafe_allow_html=True)

# #     if target_class != "background":
# #         st.caption(
# #             "⚠️ The color-signature, texture-sharpness, and ROI-structure checks in the Scanner tab are "
# #             "lightweight screening heuristics — an educational aid, not a certified RBI authentication "
# #             "procedure. Always cross-check the security markings above against the physical note."
# #         )

# # # ============================= TAB 3 =============================== #
# # with tab3:
# #     st.markdown("### 📊 Ensemble Training & Evaluation Diagnostics")
# #     st.write("Accuracy/loss curves, confusion matrix, and classification reports from the last `train_model.py` run.")

# #     if metrics_summary:
# #         ens = metrics_summary["ensemble"]
# #         st.markdown("#### 🏆 Ensemble Test-Set Metrics")
# #         m1, m2, m3, m4 = st.columns(4)
# #         m1.metric("Accuracy", f"{ens['accuracy']*100:.2f}%")
# #         m2.metric("F1 (macro)", f"{ens['f1_macro']*100:.2f}%")
# #         m3.metric("Cohen's Kappa", f"{ens['cohen_kappa']:.3f}")
# #         m4.metric("MCC", f"{ens['matthews_corrcoef']:.3f}")
# #         m5, m6, m7 = st.columns(3)
# #         m5.metric("Precision (macro)", f"{ens['precision_macro']*100:.2f}%")
# #         m6.metric("Recall (macro)", f"{ens['recall_macro']*100:.2f}%")
# #         m7.metric("Top-3 Accuracy", f"{ens['top3_accuracy']*100:.2f}%")

# #         st.markdown("#### 🧩 Per-Model Test-Set Metrics")
# #         per_model = metrics_summary["per_model"]
# #         weights = metrics_summary.get("ensemble_weights", {})
# #         rows = [{
# #             "Model": name, "Ensemble Weight": round(weights.get(name, 0), 3),
# #             "Val Acc": f"{m.get('val_accuracy', 0)*100:.2f}%", "Test Acc": f"{m['accuracy']*100:.2f}%",
# #             "F1 (macro)": f"{m['f1_macro']*100:.2f}%", "Kappa": round(m["cohen_kappa"], 3), "MCC": round(m["matthews_corrcoef"], 3),
# #         } for name, m in per_model.items()]
# #         st.table(rows)
# #     else:
# #         st.info("No `metrics_summary.json` found yet. Run `train_model.py` to generate diagnostics.")

# #     col1, col2 = st.columns(2)
# #     with col1:
# #         st.markdown("#### 📈 Per-Backbone Training History")
# #         if ensemble_config:
# #             for entry in ensemble_config["backbones"]:
# #                 hist_path = os.path.join(cfg.MODEL_DIR, f"{entry['name']}_training_history.png")
# #                 if os.path.exists(hist_path):
# #                     st.image(hist_path, caption=f"{entry['display_name']} convergence", use_container_width=True)
# #         else:
# #             st.info("Training history plots not found. Run train_model.py to generate them.")
# #     with col2:
# #         st.markdown("#### 🎯 Ensemble Confusion Matrix")
# #         cm_path = os.path.join(cfg.MODEL_DIR, "confusion_matrix_ensemble.png")
# #         if os.path.exists(cm_path):
# #             st.image(cm_path, caption="Row-normalized confusion matrix — weighted ensemble", use_container_width=True)
# #         else:
# #             st.info("Confusion matrix not found. Run train_model.py to generate it.")

# #     st.markdown("#### 🔥 Grad-CAM Sample Explanations")
# #     if os.path.isdir(cfg.GRADCAM_SAMPLE_DIR):
# #         sample_files = sorted(f for f in os.listdir(cfg.GRADCAM_SAMPLE_DIR) if f.lower().endswith(".png"))
# #         if sample_files:
# #             cols = st.columns(4)
# #             for i, fname in enumerate(sample_files):
# #                 with cols[i % 4]:
# #                     st.image(os.path.join(cfg.GRADCAM_SAMPLE_DIR, fname), caption=fname.replace("_gradcam.png", ""), use_container_width=True)
# #         else:
# #             st.info("No Grad-CAM samples found yet.")
# #     else:
# #         st.info("Grad-CAM sample folder not found. Run train_model.py to generate it.")

# #     ens_report_path = os.path.join(cfg.MODEL_DIR, "classification_report_ensemble.txt")
# #     if os.path.exists(ens_report_path):
# #         st.markdown("#### 📄 Ensemble Classification Report")
# #         with open(ens_report_path, "r", encoding="utf-8") as f:
# #             st.text_area("Ensemble Report", f.read(), height=300)

# #     if ensemble_config:
# #         st.markdown("#### 📄 Per-Model Classification Reports")
# #         for entry in ensemble_config["backbones"]:
# #             report_path = os.path.join(cfg.MODEL_DIR, f"{entry['name']}_classification_report.txt")
# #             if os.path.exists(report_path):
# #                 with open(report_path, "r", encoding="utf-8") as f:
# #                     with st.expander(f"{entry['display_name']} report"):
# #                         st.text(f.read())























# """
# app.py
# ======
# Dhan Drishti — AI-Powered Indian Banknote Recognition & Assistive Cash-Tally
# Portal. Mobile-first, professional Streamlit front-end for the ensemble
# trained by train_model.py.
# """

# import json
# import os
# import time

# import numpy as np
# import streamlit as st
# from PIL import Image, ImageOps
# import streamlit.components.v1 as components

# import config as cfg
# import dhan_drishti_utils as ddu
# from note_metadata import NOTE_METADATA, AUDIO_DICTIONARY

# st.set_page_config(
#     page_title="Dhan Drishti — AI Currency Detector",
#     page_icon="🪙",
#     layout="wide",
#     initial_sidebar_state="auto",   # auto-collapses on narrow/mobile screens
# )

# # ==================================================================== #
# # Global styling — mobile-first, card-based, custom type scale
# # ==================================================================== #
# st.markdown("""
# <link rel="preconnect" href="https://fonts.googleapis.com">
# <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@500;700;800&family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
# <style>
#     html, body, [class*="css"] { font-family: 'Inter', -apple-system, sans-serif; }
#     .stApp {
#         background: radial-gradient(circle at 15% 0%, #1c1533 0%, #100c1c 45%, #0b0814 100%);
#         color: #eee7fb;
#     }
#     #MainMenu, footer, header { visibility: hidden; }
#     .block-container { padding-top: 1.2rem; padding-bottom: 4rem; max-width: 1180px; }

#     /* ---------- Hero banner ---------- */
#     .hero {
#         position: relative; overflow: hidden;
#         background: linear-gradient(135deg, rgba(147,97,255,0.22) 0%, rgba(88,28,155,0.32) 100%);
#         border: 1px solid rgba(167,139,250,0.28); border-radius: 22px;
#         padding: 30px 26px; margin-bottom: 18px; text-align: center;
#         box-shadow: 0 12px 40px rgba(76,29,149,0.35);
#     }
#     .hero::before {
#         content: ""; position: absolute; inset: 0;
#         background: radial-gradient(circle at 80% -20%, rgba(192,132,252,0.25), transparent 55%);
#     }
#     .hero-eyebrow {
#         display: inline-block; font-size: 0.72rem; font-weight: 700; letter-spacing: 0.14em;
#         text-transform: uppercase; color: #c4b5fd; background: rgba(167,139,250,0.14);
#         border: 1px solid rgba(167,139,250,0.3); padding: 5px 14px; border-radius: 999px; margin-bottom: 12px;
#     }
#     .hero h1 {
#         font-family: 'Outfit', sans-serif; font-size: clamp(2rem, 6vw, 3.1rem); font-weight: 800 !important;
#         background: linear-gradient(90deg, #c9b6ff 0%, #f0abfc 100%);
#         -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin: 0 0 6px 0; line-height: 1.1;
#     }
#     .hero p { font-size: clamp(0.9rem, 2.4vw, 1.08rem); color: #d8cef7; margin: 0; }
#     .badge-row { display: flex; flex-wrap: wrap; gap: 8px; justify-content: center; margin-top: 16px; }
#     .stat-badge {
#         background: rgba(10,6,20,0.5); border: 1px solid rgba(167,139,250,0.25); border-radius: 12px;
#         padding: 8px 16px; min-width: 92px;
#     }
#     .stat-badge b { display: block; font-family: 'Outfit', sans-serif; font-size: 1.25rem; color: #34d399; }
#     .stat-badge span { font-size: 0.68rem; color: #b9a9e8; text-transform: uppercase; letter-spacing: 0.05em; }

#     /* ---------- Cards ---------- */
#     .card {
#         background: rgba(28,22,48,0.55); border: 1px solid rgba(167,139,250,0.14); border-radius: 18px;
#         padding: 20px; margin-bottom: 16px; backdrop-filter: blur(10px);
#         box-shadow: 0 6px 24px rgba(0,0,0,0.25);
#     }
#     .card-title { font-family: 'Outfit', sans-serif; font-weight: 700; font-size: 1.05rem; color: #f1eaff; margin-bottom: 10px; display: flex; align-items: center; gap: 8px; }

#     .result-display {
#         text-align: center; padding: 22px 18px; border-radius: 16px;
#         background: linear-gradient(160deg, rgba(147,97,255,0.16) 0%, rgba(88,28,155,0.12) 100%);
#         border: 1px solid rgba(167,139,250,0.25);
#     }
#     .result-series { font-size: 0.98rem; color: #c4b5fd; font-weight: 600; letter-spacing: 0.02em; }
#     .result-val { font-family: 'Outfit', sans-serif; font-size: clamp(1.8rem, 6vw, 2.7rem); font-weight: 800; color: #ffc266; margin: 6px 0; }
#     .match-pill { display: inline-block; padding: 6px 16px; border-radius: 999px; font-weight: 700; font-size: 0.85rem; color: #120e1e; }

#     .stat-row {
#         display: flex; justify-content: space-between; align-items: center; margin-top: 10px;
#         padding: 10px 14px; background: rgba(8,5,16,0.45); border-radius: 10px; font-size: 0.88rem;
#     }
#     .stat-label { color: #b9a9e8; font-weight: 500; }
#     .stat-val { color: #34d399; font-weight: 700; }
#     .stat-val.red { color: #f87171; }
#     .stat-val.amber { color: #fbbf24; }

#     .model-row {
#         display: flex; justify-content: space-between; align-items: center; padding: 9px 14px;
#         background: rgba(8,5,16,0.35); border-radius: 10px; margin-top: 6px; font-size: 0.85rem;
#     }

#     .alert-banner {
#         padding: 13px 16px; border-radius: 12px; margin-bottom: 12px; font-weight: 600; font-size: 0.92rem;
#         display: flex; align-items: center; gap: 10px; border: 1px solid transparent;
#     }
#     .alert-banner.warning { background: rgba(251,191,36,0.13); border-color: rgba(251,191,36,0.3); color: #fbbf24; }
#     .alert-banner.error   { background: rgba(239,68,68,0.13); border-color: rgba(239,68,68,0.3); color: #f87171; }
#     .alert-banner.success { background: rgba(52,211,153,0.13); border-color: rgba(52,211,153,0.3); color: #34d399; }
#     .alert-banner.info    { background: rgba(96,165,250,0.13); border-color: rgba(96,165,250,0.3); color: #93c5fd; }

#     .tally-item { display: flex; justify-content: space-between; padding: 8px 12px; background: rgba(8,5,16,0.4);
#                   border-radius: 9px; margin-bottom: 5px; font-size: 0.87rem; }
#     .tally-total { text-align: center; padding: 16px; border-radius: 14px; margin-top: 10px;
#                    background: linear-gradient(135deg, rgba(52,211,153,0.16), rgba(52,211,153,0.05));
#                    border: 1px solid rgba(52,211,153,0.32); font-family:'Outfit',sans-serif; font-size: 1.7rem; font-weight: 800; color: #34d399; }

#     /* ---------- Tabs styling ---------- */
#     .stTabs [data-baseweb="tab-list"] { gap: 4px; background: rgba(20,15,35,0.5); padding: 5px; border-radius: 14px; }
#     .stTabs [data-baseweb="tab"] { border-radius: 10px; padding: 10px 16px; font-weight: 600; font-size: 0.92rem; }
#     .stTabs [aria-selected="true"] { background: linear-gradient(135deg, #8b5cf6, #a855f7) !important; color: white !important; }

#     /* ---------- Buttons: big + tappable on mobile ---------- */
#     .stButton>button {
#         border-radius: 12px !important; font-weight: 700 !important; padding: 0.6rem 1rem !important;
#         border: 1px solid rgba(167,139,250,0.35) !important; width: 100%;
#     }

#     h3, h4, h5 { font-family: 'Outfit', sans-serif !important; color: #f1eaff !important; }

#     /* ---------- Mobile tightening ---------- */
#     @media (max-width: 640px) {
#         .block-container { padding-left: 0.8rem; padding-right: 0.8rem; }
#         .hero { padding: 22px 16px; border-radius: 18px; }
#         .card { padding: 15px; border-radius: 14px; }
#         .stat-badge { min-width: 78px; padding: 6px 10px; }
#     }
# </style>
# """, unsafe_allow_html=True)


# # ==================================================================== #
# # Model / ensemble loading
# # ==================================================================== #
# @st.cache_resource(show_spinner=False)
# def load_ensemble():
#     config_path = os.path.join(cfg.MODEL_DIR, "ensemble_config.json")
#     if not os.path.exists(config_path):
#         return None, None
#     with open(config_path, "r", encoding="utf-8") as f:
#         econfig = json.load(f)
#     import tensorflow as tf
#     models = {}
#     for entry in econfig["backbones"]:
#         model_path = os.path.join(cfg.MODEL_DIR, entry["file"])
#         if os.path.exists(model_path):
#             models[entry["name"]] = tf.keras.models.load_model(model_path)
#     if not models:
#         return None, None
#     return econfig, models


# @st.cache_resource(show_spinner=False)
# def load_metrics_summary():
#     path = os.path.join(cfg.MODEL_DIR, "metrics_summary.json")
#     if os.path.exists(path):
#         with open(path, "r", encoding="utf-8") as f:
#             return json.load(f)
#     return None


# def simulate_prediction(image: Image.Image, class_names):
#     best_class, best_score = "background", float("inf")
#     for cls in cfg.RBI_COLOR_CENTROIDS:
#         score = ddu.color_signature_distance(image, cls)
#         if score < best_score:
#             best_score, best_class = score, cls
#     idx = class_names.index(best_class)
#     rng = np.random.default_rng()
#     probs = rng.dirichlet(np.ones(len(class_names)) * 0.4)
#     probs[idx] += 1.2
#     return probs / probs.sum()


# def speak(text: str, lang: str = "en-IN"):
#     safe_text = text.replace("\\", "\\\\").replace('"', '\\"')
#     components.html(f"""
#     <script>
#     if ('speechSynthesis' in window) {{
#         window.speechSynthesis.cancel();
#         var u = new SpeechSynthesisUtterance("{safe_text}");
#         u.lang = "{lang}";
#         var voices = window.speechSynthesis.getVoices();
#         var m = voices.find(v => v.lang.startsWith("{lang[:2]}"));
#         if (m) u.voice = m;
#         window.speechSynthesis.speak(u);
#     }}
#     </script>
#     """, height=0, width=0)


# def alert(kind: str, icon: str, text: str):
#     st.markdown(f'<div class="alert-banner {kind}">{icon} {text}</div>', unsafe_allow_html=True)


# # ==================================================================== #
# # Load ensemble + metrics
# # ==================================================================== #
# with st.spinner("Initializing ensemble..."):
#     ensemble_config, ensemble_models = load_ensemble()

# DEMO_MODE = ensemble_models is None
# CLASS_NAMES = ensemble_config["class_names"] if ensemble_config else cfg.CLASS_NAMES
# metrics_summary = load_metrics_summary()

# # ==================================================================== #
# # Hero banner
# # ==================================================================== #
# ens_acc_display = f"{metrics_summary['ensemble']['accuracy']*100:.1f}%" if metrics_summary else "—"
# ens_f1_display = f"{metrics_summary['ensemble']['f1_macro']*100:.1f}%" if metrics_summary else "—"
# n_backbones = len(ensemble_config["backbones"]) if ensemble_config else 3

# st.markdown(f"""
# <div class="hero">
#     <span class="hero-eyebrow">🇮🇳 Ensemble Deep Learning · Assistive Fintech</span>
#     <h1>Dhan Drishti</h1>
#     <p>AI-Powered Indian Banknote Recognition &amp; Assistive Cash-Counting Portal</p>
#     <div class="badge-row">
#         <div class="stat-badge"><b>{ens_acc_display}</b><span>Test Accuracy</span></div>
#         <div class="stat-badge"><b>{ens_f1_display}</b><span>Macro F1</span></div>
#         <div class="stat-badge"><b>{n_backbones}</b><span>Backbones</span></div>
#         <div class="stat-badge"><b>{len(CLASS_NAMES)}</b><span>Classes</span></div>
#         <div class="stat-badge"><b>7</b><span>Languages</span></div>
#     </div>
# </div>
# """, unsafe_allow_html=True)

# if DEMO_MODE:
#     alert("warning", "🧪", "<b>Demo Mode</b> — no trained ensemble found in <code>model/ensemble_config.json</code>. "
#                             "Showing a simulated, clearly-labelled prediction. Run <code>python train_model.py</code> to enable real inference.")

# # ==================================================================== #
# # Session state
# # ==================================================================== #
# if "wallet" not in st.session_state:
#     st.session_state["wallet"] = []
# if "target_note" not in st.session_state:
#     st.session_state["target_note"] = "background"

# # ==================================================================== #
# # Sidebar — settings + cash tally
# # ==================================================================== #
# with st.sidebar:
#     st.markdown("### 🎛️ Settings")
#     selected_lang = st.selectbox("Announcement language", options=list(cfg.LANG_LABELS.keys()),
#                                   format_func=lambda x: cfg.LANG_LABELS[x])
#     confidence_thresh = st.slider("Confidence threshold", 30, 95, 75, 5) / 100.0
#     auto_announce = st.checkbox("Auto-announce results", value=True)
#     use_tta = st.checkbox("Test-Time Augmentation", value=False, disabled=DEMO_MODE,
#                            help="Averages predictions over small rotation/brightness variants for a steadier result. Slower.")
#     use_mc_uncertainty = st.checkbox("MC-Dropout uncertainty check", value=True, disabled=DEMO_MODE,
#                                       help="Flags predictions the model is statistically unsure about, instead of trusting a single confident-looking guess.")
#     show_gradcam = st.checkbox("Grad-CAM explainability", value=True, disabled=DEMO_MODE)

#     st.markdown("---")
#     st.markdown("### 💰 Cash Tally")
#     wallet = st.session_state["wallet"]
#     if wallet:
#         for item in wallet[-8:]:
#             st.markdown(f'<div class="tally-item"><span>{NOTE_METADATA[item["class"]]["title"]}</span><span>₹{item["value"]}</span></div>', unsafe_allow_html=True)
#         total = sum(i["value"] for i in wallet)
#         st.markdown(f'<div class="tally-total">₹{total}</div>', unsafe_allow_html=True)
#         st.caption(f"{len(wallet)} note(s) tallied this session")
#         if st.button("🗑️ Reset tally"):
#             st.session_state["wallet"] = []
#             st.rerun()
#     else:
#         st.info("Scan a note and tap **Add to Tally** to start a running count — useful for quickly totalling cash.")

#     st.markdown("---")
#     st.markdown("### 🔬 Ensemble")
#     if ensemble_config:
#         for entry in sorted(ensemble_config["backbones"], key=lambda e: -e["weight"]):
#             st.caption(f"**{entry['display_name']}** — weight `{entry['weight']:.3f}` · val-acc `{entry['val_accuracy']*100:.1f}%`")
#     else:
#         st.caption("MobileNetV2 + EfficientNetV2-B0 + ConvNeXt-Tiny (planned)")

# # ==================================================================== #
# # Tabs
# # ==================================================================== #
# tab_scan, tab_guide, tab_diag = st.tabs(["🔍  Scanner", "🛡️  RBI Guide", "📊  Diagnostics"])

# # =============================== SCANNER ============================= #
# with tab_scan:
#     col1, col2 = st.columns([1.05, 1], gap="medium")

#     with col1:
#         st.markdown('<div class="card"><div class="card-title">📸 Scan a Banknote</div>', unsafe_allow_html=True)
#         input_mode = st.radio("Input mode", ["Live Camera", "Upload Image"], horizontal=True, label_visibility="collapsed")
#         raw_image = None
#         if input_mode == "Live Camera":
#             camera_image = st.camera_input("Scan your banknote", label_visibility="collapsed")
#             if camera_image:
#                 raw_image = Image.open(camera_image)
#         else:
#             file_uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
#             if file_uploaded:
#                 raw_image = Image.open(file_uploaded)
#                 st.image(raw_image, use_container_width=True)
#         st.markdown('</div>', unsafe_allow_html=True)

#     with col2:
#         st.markdown('<div class="card"><div class="card-title">🏷️ Verification Analysis</div>', unsafe_allow_html=True)

#         if raw_image is not None:
#             is_blank, std_dev = ddu.is_blank_surface(raw_image)
#             inference_start = time.time()
#             probs, predictive_entropy, per_model_probs = None, None, {}

#             if is_blank:
#                 predicted_class, confidence = "background", 1.0
#                 alert("warning", "⚠️", "Textureless plain surface detected. Please scan a real banknote.")
#                 st.markdown('<div class="result-display"><span class="result-series">No Currency Detected</span><div class="result-val">Background</div></div>', unsafe_allow_html=True)
#                 st.markdown(f'<div class="stat-row"><span class="stat-label">Grayscale Variance</span><span class="stat-val red">{std_dev:.2f} (low)</span></div>', unsafe_allow_html=True)
#                 st.session_state["target_note"] = "background"
#             else:
#                 if DEMO_MODE:
#                     probs = simulate_prediction(raw_image, CLASS_NAMES)
#                     time.sleep(0.15)
#                 else:
#                     weights_sum = sum(e["weight"] for e in ensemble_config["backbones"])
#                     ens_probs = None
#                     for entry in ensemble_config["backbones"]:
#                         model = ensemble_models.get(entry["name"])
#                         if model is None:
#                             continue
#                         p = (ddu.tta_predict(model, raw_image, entry["name"], img_size=tuple(ensemble_config["img_size"]))
#                              if use_tta else
#                              ddu.single_predict(model, raw_image, entry["name"], img_size=tuple(ensemble_config["img_size"])))
#                         per_model_probs[entry["name"]] = p
#                         ens_probs = p * (entry["weight"] / weights_sum) if ens_probs is None else ens_probs + p * (entry["weight"] / weights_sum)
#                     probs = ens_probs
#                     if use_mc_uncertainty:
#                         best_entry = max(ensemble_config["backbones"], key=lambda e: e["weight"])
#                         _, predictive_entropy = ddu.mc_dropout_predict(
#                             ensemble_models[best_entry["name"]], raw_image, best_entry["name"],
#                             img_size=tuple(ensemble_config["img_size"]))

#                 max_idx = int(np.argmax(probs))
#                 predicted_class = CLASS_NAMES[max_idx]
#                 confidence = float(probs[max_idx])
#                 inference_ms = int((time.time() - inference_start) * 1000)

#                 if DEMO_MODE:
#                     alert("warning", "🧪", "Demo Mode — simulated prediction, not a real ensemble output.")

#                 color_dist = ddu.color_signature_distance(raw_image, predicted_class)
#                 sharpness = ddu.texture_sharpness_score(raw_image) if not DEMO_MODE else None
#                 roi_score = ddu.security_roi_consistency_score(raw_image, predicted_class)
#                 color_flag = predicted_class != "background" and color_dist > cfg.COLOR_SIGNATURE_DIST_THRESHOLD
#                 texture_flag = (sharpness is not None) and sharpness < cfg.TEXTURE_SHARPNESS_THRESHOLD
#                 roi_flag = predicted_class != "background" and roi_score < cfg.ROI_CONSISTENCY_THRESHOLD
#                 is_uncertain = (predictive_entropy is not None) and (predictive_entropy > cfg.MC_ENTROPY_REJECT_THRESHOLD)

#                 flags = [n for n, f in [("color", color_flag), ("texture", texture_flag), ("ROI", roi_flag)] if f]
#                 validation_passed = len(flags) == 0
#                 verification_msg = "No screening flags" if validation_passed else f"Flag(s): {', '.join(flags)}"

#                 meta = NOTE_METADATA.get(predicted_class, NOTE_METADATA["background"])
#                 st.session_state["target_note"] = predicted_class

#                 if is_uncertain:
#                     alert("warning", "🤔", "Model is uncertain (high MC-Dropout entropy). Please re-scan with better lighting/framing.")
#                 elif confidence < confidence_thresh:
#                     alert("warning", "⚠️", "Low confidence. Improve lighting and center the note.")
#                 elif not validation_passed:
#                     alert("error", "🛑", "Heuristic screening flag raised — a screening cue, not certified counterfeit detection. Verify manually in the RBI Guide tab.")
#                 else:
#                     alert("success", "✅", "No screening flags raised. Still verify security features for high-value transactions.")

#                 st.markdown(f"""
#                 <div class="result-display">
#                     <span class="result-series">{meta['series']}</span>
#                     <div class="result-val">{meta['title']}</div>
#                     <span class="match-pill" style="background-color:{meta['accent']};">Match: {confidence*100:.1f}%</span>
#                 </div>""", unsafe_allow_html=True)

#                 stat_rows = f"""
#                 <div class="stat-row"><span class="stat-label">Inference{' (TTA)' if use_tta else ''}</span><span class="stat-val">{inference_ms} ms</span></div>
#                 <div class="stat-row"><span class="stat-label">Screening</span><span class="stat-val {'red' if not validation_passed else ''}">{verification_msg}</span></div>
#                 """
#                 if predictive_entropy is not None:
#                     stat_rows += f'<div class="stat-row"><span class="stat-label">MC-Dropout Entropy</span><span class="stat-val {"amber" if is_uncertain else ""}">{predictive_entropy:.3f} nats</span></div>'
#                 st.markdown(stat_rows, unsafe_allow_html=True)

#                 if per_model_probs:
#                     rows = ""
#                     for entry in sorted(ensemble_config["backbones"], key=lambda e: -e["weight"]):
#                         name = entry["name"]
#                         if name not in per_model_probs:
#                             continue
#                         p = per_model_probs[name]
#                         model_pred = CLASS_NAMES[int(np.argmax(p))]
#                         model_conf = float(np.max(p))
#                         agree = "✅" if model_pred == predicted_class else "⚠️"
#                         rows += (f'<div class="model-row">{agree} <b>{entry["display_name"]}</b> (w={entry["weight"]:.2f})'
#                                  f'<span>{NOTE_METADATA.get(model_pred, {}).get("title", model_pred)} · {model_conf*100:.1f}%</span></div>')
#                     st.markdown("**🧩 Per-model breakdown**", unsafe_allow_html=True)
#                     st.markdown(rows, unsafe_allow_html=True)

#                 bcol1, bcol2 = st.columns(2)
#                 with bcol1:
#                     if predicted_class != "background" and not is_uncertain:
#                         if st.button(f"➕ Add ₹{cfg.DENOMINATION_VALUE[predicted_class]} to Tally"):
#                             st.session_state["wallet"].append({"class": predicted_class, "value": cfg.DENOMINATION_VALUE[predicted_class], "confidence": confidence})
#                             st.rerun()
#                 with bcol2:
#                     if st.button("🔊 Announce"):
#                         phrase_book = AUDIO_DICTIONARY.get(selected_lang, AUDIO_DICTIONARY["en-IN"])
#                         phrase = phrase_book.get(predicted_class, phrase_book["background"])
#                         if is_uncertain:
#                             phrase += ". " + phrase_book["warning_uncertain"]
#                         speak(phrase, selected_lang)

#                 if auto_announce:
#                     phrase_book = AUDIO_DICTIONARY.get(selected_lang, AUDIO_DICTIONARY["en-IN"])
#                     phrase = phrase_book.get(predicted_class, phrase_book["background"])
#                     if is_uncertain:
#                         phrase += ". " + phrase_book["warning_uncertain"]
#                     speak(phrase, selected_lang)

#                 if show_gradcam and not DEMO_MODE and ensemble_models:
#                     try:
#                         best_entry = max(ensemble_config["backbones"], key=lambda e: e["weight"])
#                         best_model = ensemble_models[best_entry["name"]]
#                         head_model, base_model = ddu.build_head_submodel(best_model)
#                         preprocess_fn = ddu.BACKBONE_REGISTRY[best_entry["name"]]["preprocess_input"]
#                         img_resized = raw_image.convert("RGB").resize(tuple(ensemble_config["img_size"]))
#                         x = preprocess_fn(np.array(img_resized).astype(np.float32).copy())
#                         x = np.expand_dims(x, axis=0)
#                         heatmap, _ = ddu.make_gradcam_heatmap(x, base_model, head_model, pred_index=max_idx)
#                         overlay = ddu.overlay_gradcam(img_resized, heatmap)
#                         st.markdown(f"**🔥 Grad-CAM — {best_entry['display_name']}**")
#                         st.image(overlay, use_container_width=True, caption="Warmer = more influential region")
#                     except Exception as e:
#                         st.caption(f"(Grad-CAM unavailable: {e})")

#             if probs is not None:
#                 st.markdown("**📊 Probability Distribution**")
#                 chart_data = {NOTE_METADATA.get(CLASS_NAMES[i], {}).get("title", CLASS_NAMES[i]): float(probs[i]) for i in range(len(CLASS_NAMES))}
#                 chart_data = dict(sorted(chart_data.items(), key=lambda item: item[1], reverse=True))
#                 st.bar_chart(chart_data, horizontal=True)
#         else:
#             st.info("💡 Scan a note or upload an image to see results here.")
#             st.session_state["target_note"] = "background"
#         st.markdown('</div>', unsafe_allow_html=True)

# # =============================== RBI GUIDE ============================ #
# with tab_guide:
#     target_class = st.session_state.get("target_note", "background")
#     meta = NOTE_METADATA.get(target_class, NOTE_METADATA["background"])

#     if target_class == "background":
#         alert("info", "💡", "Details will auto-load here once a note is scanned in the Scanner tab.")

#     col1, col2 = st.columns([1.5, 1], gap="medium")
#     with col1:
#         st.markdown(f'<div class="card"><div class="card-title">🔎 {meta["title"]}</div>', unsafe_allow_html=True)
#         st.write(f"**Year of Release:** {meta['year']}")
#         st.write(f"**Dimensions:** {meta['dimensions']}")
#         st.write(f"**Dominant Color:** {meta['color']}")
#         st.write(f"**Motif (Reverse):** {meta['motif']}")
#         st.markdown("##### 📍 Security Markings to Verify Manually")
#         for feature in meta["features"]:
#             st.markdown(f"- 🔳 {feature}")
#         st.markdown('</div>', unsafe_allow_html=True)
#     with col2:
#         st.markdown(f"""
#         <div class="card">
#             <div class="card-title">📜 Historical Context</div>
#             <div style="border-left: 4px solid {meta['accent']}; padding-left: 14px; color:#e2dcf4; font-style: italic;">
#                 {meta['funFact']}
#             </div>
#         </div>""", unsafe_allow_html=True)

#     if target_class != "background":
#         st.caption("⚠️ Color/texture/ROI checks in the Scanner tab are lightweight screening heuristics — an educational aid, not certified RBI authentication. Always verify manually.")

# # =============================== DIAGNOSTICS =========================== #
# with tab_diag:
#     if metrics_summary:
#         ens = metrics_summary["ensemble"]
#         st.markdown('<div class="card"><div class="card-title">🏆 Ensemble Test Metrics</div>', unsafe_allow_html=True)
#         m1, m2, m3, m4 = st.columns(4)
#         m1.metric("Accuracy", f"{ens['accuracy']*100:.2f}%")
#         m2.metric("F1 (macro)", f"{ens['f1_macro']*100:.2f}%")
#         m3.metric("Cohen's Kappa", f"{ens['cohen_kappa']:.3f}")
#         m4.metric("MCC", f"{ens['matthews_corrcoef']:.3f}")
#         m5, m6, m7 = st.columns(3)
#         m5.metric("Precision (macro)", f"{ens['precision_macro']*100:.2f}%")
#         m6.metric("Recall (macro)", f"{ens['recall_macro']*100:.2f}%")
#         m7.metric("Top-3 Accuracy", f"{ens['top3_accuracy']*100:.2f}%")
#         st.markdown('</div>', unsafe_allow_html=True)

#         st.markdown('<div class="card"><div class="card-title">🧩 Per-Model Metrics</div>', unsafe_allow_html=True)
#         per_model = metrics_summary["per_model"]
#         weights = metrics_summary.get("ensemble_weights", {})
#         rows = [{"Model": name, "Weight": round(weights.get(name, 0), 3), "Val Acc": f"{m.get('val_accuracy',0)*100:.2f}%",
#                  "Test Acc": f"{m['accuracy']*100:.2f}%", "F1": f"{m['f1_macro']*100:.2f}%",
#                  "Kappa": round(m["cohen_kappa"], 3), "MCC": round(m["matthews_corrcoef"], 3)} for name, m in per_model.items()]
#         st.table(rows)
#         st.markdown('</div>', unsafe_allow_html=True)
#     else:
#         alert("info", "ℹ️", "No <code>metrics_summary.json</code> yet. Run <code>train_model.py</code> or <code>recover_artifacts.py</code>.")

#     col1, col2 = st.columns(2, gap="medium")
#     with col1:
#         st.markdown('<div class="card"><div class="card-title">📈 Training History</div>', unsafe_allow_html=True)
#         if ensemble_config:
#             for entry in ensemble_config["backbones"]:
#                 p = os.path.join(cfg.MODEL_DIR, f"{entry['name']}_training_history.png")
#                 if os.path.exists(p):
#                     st.image(p, caption=f"{entry['display_name']} convergence", use_container_width=True)
#         else:
#             st.caption("Not available yet.")
#         st.markdown('</div>', unsafe_allow_html=True)
#     with col2:
#         st.markdown('<div class="card"><div class="card-title">🎯 Confusion Matrix</div>', unsafe_allow_html=True)
#         cm_path = os.path.join(cfg.MODEL_DIR, "confusion_matrix_ensemble.png")
#         if os.path.exists(cm_path):
#             st.image(cm_path, use_container_width=True)
#         else:
#             st.caption("Not available yet.")
#         st.markdown('</div>', unsafe_allow_html=True)

#     st.markdown('<div class="card"><div class="card-title">📁 Extended Paper Figures</div>', unsafe_allow_html=True)
#     fig_dir = os.path.join(cfg.MODEL_DIR, "paper_figures")
#     if os.path.isdir(fig_dir):
#         fig_files = sorted(f for f in os.listdir(fig_dir) if f.lower().endswith(".png"))
#         if fig_files:
#             cols = st.columns(2)
#             for i, fname in enumerate(fig_files):
#                 with cols[i % 2]:
#                     st.image(os.path.join(fig_dir, fname), caption=fname.replace("_", " ").replace(".png", ""), use_container_width=True)
#         else:
#             st.caption("No figures found yet.")
#     else:
#         st.caption("Run `python generate_paper_figures.py` to produce ROC curves, PR curves, calibration diagrams, t-SNE embeddings, and more for the paper.")
#     st.markdown('</div>', unsafe_allow_html=True)

#     st.markdown('<div class="card"><div class="card-title">🔥 Grad-CAM Samples</div>', unsafe_allow_html=True)
#     if os.path.isdir(cfg.GRADCAM_SAMPLE_DIR):
#         sample_files = sorted(f for f in os.listdir(cfg.GRADCAM_SAMPLE_DIR) if f.lower().endswith(".png"))
#         if sample_files:
#             cols = st.columns(4)
#             for i, fname in enumerate(sample_files):
#                 with cols[i % 4]:
#                     st.image(os.path.join(cfg.GRADCAM_SAMPLE_DIR, fname), caption=fname.replace("_gradcam.png", ""), use_container_width=True)
#         else:
#             st.caption("No samples found yet.")
#     else:
#         st.caption("Not available yet.")
#     st.markdown('</div>', unsafe_allow_html=True)

#     ens_report_path = os.path.join(cfg.MODEL_DIR, "classification_report_ensemble.txt")
#     if os.path.exists(ens_report_path):
#         with open(ens_report_path, "r", encoding="utf-8") as f:
#             with st.expander("📄 Ensemble Classification Report"):
#                 st.text(f.read())

#     if ensemble_config:
#         for entry in ensemble_config["backbones"]:
#             report_path = os.path.join(cfg.MODEL_DIR, f"{entry['name']}_classification_report.txt")
#             if os.path.exists(report_path):
#                 with open(report_path, "r", encoding="utf-8") as f:
#                     with st.expander(f"📄 {entry['display_name']} Report"):
#                         st.text(f.read())










"""
app.py
======
Dhan Drishti — AI-Powered Indian Banknote Recognition & Assistive Cash-Tally
Portal. Mobile-first, professional Streamlit front-end for the ensemble
trained by train_model.py.
"""

import json
import os
import time

import numpy as np
import streamlit as st
from PIL import Image, ImageOps
import streamlit.components.v1 as components

import config as cfg
import dhan_drishti_utils as ddu
from note_metadata import NOTE_METADATA, AUDIO_DICTIONARY

st.set_page_config(
    page_title="Dhan Drishti — AI Currency Detector",
    page_icon="🪙",
    layout="wide",
    initial_sidebar_state="auto",   # auto-collapses on narrow/mobile screens
)

# ==================================================================== #
# Global styling — mobile-first, card-based, custom type scale
# ==================================================================== #
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Outfit:wght@500;700;800&family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
    html, body, [class*="css"] { font-family: 'Inter', -apple-system, sans-serif; }
    .stApp {
        background: radial-gradient(circle at 15% 0%, #1c1533 0%, #100c1c 45%, #0b0814 100%);
        color: #eee7fb;
    }
    #MainMenu, footer, header { visibility: hidden; }
    .block-container { padding-top: 1.2rem; padding-bottom: 4rem; max-width: 1180px; }

    /* ---------- Hero banner ---------- */
    .hero {
        position: relative; overflow: hidden;
        background: linear-gradient(135deg, rgba(147,97,255,0.22) 0%, rgba(88,28,155,0.32) 100%);
        border: 1px solid rgba(167,139,250,0.28); border-radius: 22px;
        padding: 30px 26px; margin-bottom: 18px; text-align: center;
        box-shadow: 0 12px 40px rgba(76,29,149,0.35);
    }
    .hero::before {
        content: ""; position: absolute; inset: 0;
        background: radial-gradient(circle at 80% -20%, rgba(192,132,252,0.25), transparent 55%);
    }
    .hero-eyebrow {
        display: inline-block; font-size: 0.72rem; font-weight: 700; letter-spacing: 0.14em;
        text-transform: uppercase; color: #c4b5fd; background: rgba(167,139,250,0.14);
        border: 1px solid rgba(167,139,250,0.3); padding: 5px 14px; border-radius: 999px; margin-bottom: 12px;
    }
    .hero h1 {
        font-family: 'Outfit', sans-serif; font-size: clamp(2rem, 6vw, 3.1rem); font-weight: 800 !important;
        background: linear-gradient(90deg, #c9b6ff 0%, #f0abfc 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin: 0 0 6px 0; line-height: 1.1;
    }
    .hero p { font-size: clamp(0.9rem, 2.4vw, 1.08rem); color: #d8cef7; margin: 0; }
    .badge-row { display: flex; flex-wrap: wrap; gap: 8px; justify-content: center; margin-top: 16px; }
    .stat-badge {
        background: rgba(10,6,20,0.5); border: 1px solid rgba(167,139,250,0.25); border-radius: 12px;
        padding: 8px 16px; min-width: 92px;
    }
    .stat-badge b { display: block; font-family: 'Outfit', sans-serif; font-size: 1.25rem; color: #34d399; }
    .stat-badge span { font-size: 0.68rem; color: #b9a9e8; text-transform: uppercase; letter-spacing: 0.05em; }

    /* ---------- Cards ---------- */
    .card {
        background: rgba(28,22,48,0.55); border: 1px solid rgba(167,139,250,0.14); border-radius: 18px;
        padding: 20px; margin-bottom: 16px; backdrop-filter: blur(10px);
        box-shadow: 0 6px 24px rgba(0,0,0,0.25);
    }
    .card-title { font-family: 'Outfit', sans-serif; font-weight: 700; font-size: 1.05rem; color: #f1eaff; margin-bottom: 10px; display: flex; align-items: center; gap: 8px; }

    .result-display {
        text-align: center; padding: 22px 18px; border-radius: 16px;
        background: linear-gradient(160deg, rgba(147,97,255,0.16) 0%, rgba(88,28,155,0.12) 100%);
        border: 1px solid rgba(167,139,250,0.25);
    }
    .result-series { font-size: 0.98rem; color: #c4b5fd; font-weight: 600; letter-spacing: 0.02em; }
    .result-val { font-family: 'Outfit', sans-serif; font-size: clamp(1.8rem, 6vw, 2.7rem); font-weight: 800; color: #ffc266; margin: 6px 0; }
    .match-pill { display: inline-block; padding: 6px 16px; border-radius: 999px; font-weight: 700; font-size: 0.85rem; color: #120e1e; }

    .stat-row {
        display: flex; justify-content: space-between; align-items: center; margin-top: 10px;
        padding: 10px 14px; background: rgba(8,5,16,0.45); border-radius: 10px; font-size: 0.88rem;
    }
    .stat-label { color: #b9a9e8; font-weight: 500; }
    .stat-val { color: #34d399; font-weight: 700; }
    .stat-val.red { color: #f87171; }
    .stat-val.amber { color: #fbbf24; }

    .model-row {
        display: flex; justify-content: space-between; align-items: center; padding: 9px 14px;
        background: rgba(8,5,16,0.35); border-radius: 10px; margin-top: 6px; font-size: 0.85rem;
    }

    .alert-banner {
        padding: 13px 16px; border-radius: 12px; margin-bottom: 12px; font-weight: 600; font-size: 0.92rem;
        display: flex; align-items: center; gap: 10px; border: 1px solid transparent;
    }
    .alert-banner.warning { background: rgba(251,191,36,0.13); border-color: rgba(251,191,36,0.3); color: #fbbf24; }
    .alert-banner.error   { background: rgba(239,68,68,0.13); border-color: rgba(239,68,68,0.3); color: #f87171; }
    .alert-banner.success { background: rgba(52,211,153,0.13); border-color: rgba(52,211,153,0.3); color: #34d399; }
    .alert-banner.info    { background: rgba(96,165,250,0.13); border-color: rgba(96,165,250,0.3); color: #93c5fd; }

    .tally-item { display: flex; justify-content: space-between; padding: 8px 12px; background: rgba(8,5,16,0.4);
                  border-radius: 9px; margin-bottom: 5px; font-size: 0.87rem; }
    .tally-total { text-align: center; padding: 16px; border-radius: 14px; margin-top: 10px;
                   background: linear-gradient(135deg, rgba(52,211,153,0.16), rgba(52,211,153,0.05));
                   border: 1px solid rgba(52,211,153,0.32); font-family:'Outfit',sans-serif; font-size: 1.7rem; font-weight: 800; color: #34d399; }

    /* ---------- Tabs styling ---------- */
    .stTabs [data-baseweb="tab-list"] { gap: 4px; background: rgba(20,15,35,0.5); padding: 5px; border-radius: 14px; }
    .stTabs [data-baseweb="tab"] { border-radius: 10px; padding: 10px 16px; font-weight: 600; font-size: 0.92rem; }
    .stTabs [aria-selected="true"] { background: linear-gradient(135deg, #8b5cf6, #a855f7) !important; color: white !important; }

    /* ---------- Buttons: big + tappable on mobile ---------- */
    .stButton>button {
        border-radius: 12px !important; font-weight: 700 !important; padding: 0.6rem 1rem !important;
        border: 1px solid rgba(167,139,250,0.35) !important; width: 100%;
    }

    h3, h4, h5 { font-family: 'Outfit', sans-serif !important; color: #f1eaff !important; }

    /* ---------- Mobile tightening ---------- */
    @media (max-width: 640px) {
        .block-container { padding-left: 0.8rem; padding-right: 0.8rem; }
        .hero { padding: 22px 16px; border-radius: 18px; }
        .card { padding: 15px; border-radius: 14px; }
        .stat-badge { min-width: 78px; padding: 6px 10px; }
    }
</style>
""", unsafe_allow_html=True)


# ==================================================================== #
# Model / ensemble loading
# ==================================================================== #
@st.cache_resource(show_spinner=False)
def load_ensemble():
    config_path = os.path.join(cfg.MODEL_DIR, "ensemble_config.json")
    if not os.path.exists(config_path):
        return None, None
    with open(config_path, "r", encoding="utf-8") as f:
        econfig = json.load(f)
    import tensorflow as tf
    models = {}
    for entry in econfig["backbones"]:
        model_path = os.path.join(cfg.MODEL_DIR, entry["file"])
        if os.path.exists(model_path):
            models[entry["name"]] = tf.keras.models.load_model(model_path)
    if not models:
        return None, None
    return econfig, models


@st.cache_resource(show_spinner=False)
def load_metrics_summary():
    path = os.path.join(cfg.MODEL_DIR, "metrics_summary.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def simulate_prediction(image: Image.Image, class_names):
    best_class, best_score = "background", float("inf")
    for cls in cfg.RBI_COLOR_CENTROIDS:
        score = ddu.color_signature_distance(image, cls)
        if score < best_score:
            best_score, best_class = score, cls
    idx = class_names.index(best_class)
    rng = np.random.default_rng()
    probs = rng.dirichlet(np.ones(len(class_names)) * 0.4)
    probs[idx] += 1.2
    return probs / probs.sum()


def speak(text: str, lang: str = "en-IN"):
    safe_text = text.replace("\\", "\\\\").replace('"', '\\"')
    components.html(f"""
    <script>
    if ('speechSynthesis' in window) {{
        window.speechSynthesis.cancel();
        var u = new SpeechSynthesisUtterance("{safe_text}");
        u.lang = "{lang}";
        var voices = window.speechSynthesis.getVoices();
        var m = voices.find(v => v.lang.startsWith("{lang[:2]}"));
        if (m) u.voice = m;
        window.speechSynthesis.speak(u);
    }}
    </script>
    """, height=0, width=0)


def alert(kind: str, icon: str, text: str):
    st.markdown(f'<div class="alert-banner {kind}">{icon} {text}</div>', unsafe_allow_html=True)


# ==================================================================== #
# Load ensemble + metrics
# ==================================================================== #
with st.spinner("Initializing ensemble..."):
    ensemble_config, ensemble_models = load_ensemble()

DEMO_MODE = ensemble_models is None
CLASS_NAMES = ensemble_config["class_names"] if ensemble_config else cfg.CLASS_NAMES
metrics_summary = load_metrics_summary()

# ==================================================================== #
# Hero banner
# ==================================================================== #
ens_acc_display = f"{metrics_summary['ensemble']['accuracy']*100:.1f}%" if metrics_summary else "—"
ens_f1_display = f"{metrics_summary['ensemble']['f1_macro']*100:.1f}%" if metrics_summary else "—"
n_backbones = len(ensemble_config["backbones"]) if ensemble_config else 3

st.markdown(f"""
<div class="hero">
    <span class="hero-eyebrow">🇮🇳 Ensemble Deep Learning · Assistive Fintech</span>
    <h1>Dhan Drishti</h1>
    <p>AI-Powered Indian Banknote Recognition &amp; Assistive Cash-Counting Portal</p>
    <div class="badge-row">
        <div class="stat-badge"><b>{ens_acc_display}</b><span>Test Accuracy</span></div>
        <div class="stat-badge"><b>{ens_f1_display}</b><span>Macro F1</span></div>
        <div class="stat-badge"><b>{n_backbones}</b><span>Backbones</span></div>
        <div class="stat-badge"><b>{len(CLASS_NAMES)}</b><span>Classes</span></div>
        <div class="stat-badge"><b>7</b><span>Languages</span></div>
    </div>
</div>
""", unsafe_allow_html=True)

if DEMO_MODE:
    alert("warning", "🧪", "<b>Demo Mode</b> — no trained ensemble found in <code>model/ensemble_config.json</code>. "
                            "Showing a simulated, clearly-labelled prediction. Run <code>python train_model.py</code> to enable real inference.")

# ==================================================================== #
# Session state
# ==================================================================== #
if "wallet" not in st.session_state:
    st.session_state["wallet"] = []
if "target_note" not in st.session_state:
    st.session_state["target_note"] = "background"

# ==================================================================== #
# Sidebar — settings + cash tally
# ==================================================================== #
with st.sidebar:
    st.markdown("### 🎛️ Settings")
    selected_lang = st.selectbox("Announcement language", options=list(cfg.LANG_LABELS.keys()),
                                  format_func=lambda x: cfg.LANG_LABELS[x])
    confidence_thresh = st.slider("Confidence threshold", 30, 95, 75, 5) / 100.0
    auto_announce = st.checkbox("Auto-announce results", value=True)
    use_tta = st.checkbox("Test-Time Augmentation", value=False, disabled=DEMO_MODE,
                           help="Averages predictions over small rotation/brightness variants for a steadier result. Slower.")
    # NOTE: default OFF — MC-Dropout runs extra stochastic forward passes on top of
    # normal inference. On resource-constrained hosts (e.g. Streamlit Community
    # Cloud free tier) this adds meaningful memory/CPU pressure on every scan.
    # Users can still turn it on manually from the sidebar.
    use_mc_uncertainty = st.checkbox("MC-Dropout uncertainty check", value=False, disabled=DEMO_MODE,
                                      help="Flags predictions the model is statistically unsure about, instead of trusting a single confident-looking guess.")
    # NOTE: default OFF — Grad-CAM rebuilds a head/base sub-model and runs an
    # additional forward+backward pass per scan. Same resource-pressure reasoning
    # as above; still available on demand via the checkbox.
    show_gradcam = st.checkbox("Grad-CAM explainability", value=False, disabled=DEMO_MODE)

    st.markdown("---")
    st.markdown("### 💰 Cash Tally")
    wallet = st.session_state["wallet"]
    if wallet:
        for item in wallet[-8:]:
            st.markdown(f'<div class="tally-item"><span>{NOTE_METADATA[item["class"]]["title"]}</span><span>₹{item["value"]}</span></div>', unsafe_allow_html=True)
        total = sum(i["value"] for i in wallet)
        st.markdown(f'<div class="tally-total">₹{total}</div>', unsafe_allow_html=True)
        st.caption(f"{len(wallet)} note(s) tallied this session")
        if st.button("🗑️ Reset tally"):
            st.session_state["wallet"] = []
            st.rerun()
    else:
        st.info("Scan a note and tap **Add to Tally** to start a running count — useful for quickly totalling cash.")

    st.markdown("---")
    st.markdown("### 🔬 Ensemble")
    if ensemble_config:
        for entry in sorted(ensemble_config["backbones"], key=lambda e: -e["weight"]):
            st.caption(f"**{entry['display_name']}** — weight `{entry['weight']:.3f}` · val-acc `{entry['val_accuracy']*100:.1f}%`")
    else:
        st.caption("MobileNetV2 + EfficientNetV2-B0 + ConvNeXt-Tiny (planned)")

# ==================================================================== #
# Tabs
# ==================================================================== #
tab_scan, tab_guide, tab_diag = st.tabs(["🔍  Scanner", "🛡️  RBI Guide", "📊  Diagnostics"])

# =============================== SCANNER ============================= #
with tab_scan:
    col1, col2 = st.columns([1.05, 1], gap="medium")

    with col1:
        st.markdown('<div class="card"><div class="card-title">📸 Scan a Banknote</div>', unsafe_allow_html=True)
        input_mode = st.radio("Input mode", ["Live Camera", "Upload Image"], horizontal=True, label_visibility="collapsed")
        raw_image = None
        if input_mode == "Live Camera":
            camera_image = st.camera_input("Scan your banknote", label_visibility="collapsed")
            if camera_image:
                raw_image = Image.open(camera_image)
        else:
            file_uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
            if file_uploaded:
                raw_image = Image.open(file_uploaded)
                st.image(raw_image, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card"><div class="card-title">🏷️ Verification Analysis</div>', unsafe_allow_html=True)

        if raw_image is not None:
            is_blank, std_dev = ddu.is_blank_surface(raw_image)
            inference_start = time.time()
            probs, predictive_entropy, per_model_probs = None, None, {}

            if is_blank:
                predicted_class, confidence = "background", 1.0
                alert("warning", "⚠️", "Textureless plain surface detected. Please scan a real banknote.")
                st.markdown('<div class="result-display"><span class="result-series">No Currency Detected</span><div class="result-val">Background</div></div>', unsafe_allow_html=True)
                st.markdown(f'<div class="stat-row"><span class="stat-label">Grayscale Variance</span><span class="stat-val red">{std_dev:.2f} (low)</span></div>', unsafe_allow_html=True)
                st.session_state["target_note"] = "background"
            else:
                if DEMO_MODE:
                    probs = simulate_prediction(raw_image, CLASS_NAMES)
                    time.sleep(0.15)
                else:
                    weights_sum = sum(e["weight"] for e in ensemble_config["backbones"])
                    ens_probs = None
                    for entry in ensemble_config["backbones"]:
                        model = ensemble_models.get(entry["name"])
                        if model is None:
                            continue
                        p = (ddu.tta_predict(model, raw_image, entry["name"], img_size=tuple(ensemble_config["img_size"]))
                             if use_tta else
                             ddu.single_predict(model, raw_image, entry["name"], img_size=tuple(ensemble_config["img_size"])))
                        per_model_probs[entry["name"]] = p
                        ens_probs = p * (entry["weight"] / weights_sum) if ens_probs is None else ens_probs + p * (entry["weight"] / weights_sum)
                    probs = ens_probs
                    if use_mc_uncertainty:
                        best_entry = max(ensemble_config["backbones"], key=lambda e: e["weight"])
                        _, predictive_entropy = ddu.mc_dropout_predict(
                            ensemble_models[best_entry["name"]], raw_image, best_entry["name"],
                            img_size=tuple(ensemble_config["img_size"]))

                max_idx = int(np.argmax(probs))
                predicted_class = CLASS_NAMES[max_idx]
                confidence = float(probs[max_idx])
                inference_ms = int((time.time() - inference_start) * 1000)

                if DEMO_MODE:
                    alert("warning", "🧪", "Demo Mode — simulated prediction, not a real ensemble output.")

                color_dist = ddu.color_signature_distance(raw_image, predicted_class)
                sharpness = ddu.texture_sharpness_score(raw_image) if not DEMO_MODE else None
                roi_score = ddu.security_roi_consistency_score(raw_image, predicted_class)
                color_flag = predicted_class != "background" and color_dist > cfg.COLOR_SIGNATURE_DIST_THRESHOLD
                texture_flag = (sharpness is not None) and sharpness < cfg.TEXTURE_SHARPNESS_THRESHOLD
                roi_flag = predicted_class != "background" and roi_score < cfg.ROI_CONSISTENCY_THRESHOLD
                is_uncertain = (predictive_entropy is not None) and (predictive_entropy > cfg.MC_ENTROPY_REJECT_THRESHOLD)

                flags = [n for n, f in [("color", color_flag), ("texture", texture_flag), ("ROI", roi_flag)] if f]
                validation_passed = len(flags) == 0
                verification_msg = "No screening flags" if validation_passed else f"Flag(s): {', '.join(flags)}"

                meta = NOTE_METADATA.get(predicted_class, NOTE_METADATA["background"])
                st.session_state["target_note"] = predicted_class

                if is_uncertain:
                    alert("warning", "🤔", "Model is uncertain (high MC-Dropout entropy). Please re-scan with better lighting/framing.")
                elif confidence < confidence_thresh:
                    alert("warning", "⚠️", "Low confidence. Improve lighting and center the note.")
                elif not validation_passed:
                    alert("error", "🛑", "Heuristic screening flag raised — a screening cue, not certified counterfeit detection. Verify manually in the RBI Guide tab.")
                else:
                    alert("success", "✅", "No screening flags raised. Still verify security features for high-value transactions.")

                st.markdown(f"""
                <div class="result-display">
                    <span class="result-series">{meta['series']}</span>
                    <div class="result-val">{meta['title']}</div>
                    <span class="match-pill" style="background-color:{meta['accent']};">Match: {confidence*100:.1f}%</span>
                </div>""", unsafe_allow_html=True)

                stat_rows = f"""
                <div class="stat-row"><span class="stat-label">Inference{' (TTA)' if use_tta else ''}</span><span class="stat-val">{inference_ms} ms</span></div>
                <div class="stat-row"><span class="stat-label">Screening</span><span class="stat-val {'red' if not validation_passed else ''}">{verification_msg}</span></div>
                """
                if predictive_entropy is not None:
                    stat_rows += f'<div class="stat-row"><span class="stat-label">MC-Dropout Entropy</span><span class="stat-val {"amber" if is_uncertain else ""}">{predictive_entropy:.3f} nats</span></div>'
                st.markdown(stat_rows, unsafe_allow_html=True)

                if per_model_probs:
                    rows = ""
                    for entry in sorted(ensemble_config["backbones"], key=lambda e: -e["weight"]):
                        name = entry["name"]
                        if name not in per_model_probs:
                            continue
                        p = per_model_probs[name]
                        model_pred = CLASS_NAMES[int(np.argmax(p))]
                        model_conf = float(np.max(p))
                        agree = "✅" if model_pred == predicted_class else "⚠️"
                        rows += (f'<div class="model-row">{agree} <b>{entry["display_name"]}</b> (w={entry["weight"]:.2f})'
                                 f'<span>{NOTE_METADATA.get(model_pred, {}).get("title", model_pred)} · {model_conf*100:.1f}%</span></div>')
                    st.markdown("**🧩 Per-model breakdown**", unsafe_allow_html=True)
                    st.markdown(rows, unsafe_allow_html=True)

                bcol1, bcol2 = st.columns(2)
                with bcol1:
                    if predicted_class != "background" and not is_uncertain:
                        if st.button(f"➕ Add ₹{cfg.DENOMINATION_VALUE[predicted_class]} to Tally"):
                            st.session_state["wallet"].append({"class": predicted_class, "value": cfg.DENOMINATION_VALUE[predicted_class], "confidence": confidence})
                            st.rerun()
                with bcol2:
                    if st.button("🔊 Announce"):
                        phrase_book = AUDIO_DICTIONARY.get(selected_lang, AUDIO_DICTIONARY["en-IN"])
                        phrase = phrase_book.get(predicted_class, phrase_book["background"])
                        if is_uncertain:
                            phrase += ". " + phrase_book["warning_uncertain"]
                        speak(phrase, selected_lang)

                if auto_announce:
                    phrase_book = AUDIO_DICTIONARY.get(selected_lang, AUDIO_DICTIONARY["en-IN"])
                    phrase = phrase_book.get(predicted_class, phrase_book["background"])
                    if is_uncertain:
                        phrase += ". " + phrase_book["warning_uncertain"]
                    speak(phrase, selected_lang)

                if show_gradcam and not DEMO_MODE and ensemble_models:
                    try:
                        best_entry = max(ensemble_config["backbones"], key=lambda e: e["weight"])
                        best_model = ensemble_models[best_entry["name"]]
                        head_model, base_model = ddu.build_head_submodel(best_model)
                        preprocess_fn = ddu.BACKBONE_REGISTRY[best_entry["name"]]["preprocess_input"]
                        img_resized = raw_image.convert("RGB").resize(tuple(ensemble_config["img_size"]))
                        x = preprocess_fn(np.array(img_resized).astype(np.float32).copy())
                        x = np.expand_dims(x, axis=0)
                        heatmap, _ = ddu.make_gradcam_heatmap(x, base_model, head_model, pred_index=max_idx)
                        overlay = ddu.overlay_gradcam(img_resized, heatmap)
                        st.markdown(f"**🔥 Grad-CAM — {best_entry['display_name']}**")
                        st.image(overlay, use_container_width=True, caption="Warmer = more influential region")
                    except Exception as e:
                        st.caption(f"(Grad-CAM unavailable: {e})")

            if probs is not None:
                st.markdown("**📊 Probability Distribution**")
                chart_data = {NOTE_METADATA.get(CLASS_NAMES[i], {}).get("title", CLASS_NAMES[i]): float(probs[i]) for i in range(len(CLASS_NAMES))}
                chart_data = dict(sorted(chart_data.items(), key=lambda item: item[1], reverse=True))
                st.bar_chart(chart_data, horizontal=True)
        else:
            st.info("💡 Scan a note or upload an image to see results here.")
            st.session_state["target_note"] = "background"
        st.markdown('</div>', unsafe_allow_html=True)

# =============================== RBI GUIDE ============================ #
with tab_guide:
    target_class = st.session_state.get("target_note", "background")
    meta = NOTE_METADATA.get(target_class, NOTE_METADATA["background"])

    if target_class == "background":
        alert("info", "💡", "Details will auto-load here once a note is scanned in the Scanner tab.")

    col1, col2 = st.columns([1.5, 1], gap="medium")
    with col1:
        st.markdown(f'<div class="card"><div class="card-title">🔎 {meta["title"]}</div>', unsafe_allow_html=True)
        st.write(f"**Year of Release:** {meta['year']}")
        st.write(f"**Dimensions:** {meta['dimensions']}")
        st.write(f"**Dominant Color:** {meta['color']}")
        st.write(f"**Motif (Reverse):** {meta['motif']}")
        st.markdown("##### 📍 Security Markings to Verify Manually")
        for feature in meta["features"]:
            st.markdown(f"- 🔳 {feature}")
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="card">
            <div class="card-title">📜 Historical Context</div>
            <div style="border-left: 4px solid {meta['accent']}; padding-left: 14px; color:#e2dcf4; font-style: italic;">
                {meta['funFact']}
            </div>
        </div>""", unsafe_allow_html=True)

    if target_class != "background":
        st.caption("⚠️ Color/texture/ROI checks in the Scanner tab are lightweight screening heuristics — an educational aid, not certified RBI authentication. Always verify manually.")

# =============================== DIAGNOSTICS =========================== #
with tab_diag:
    if metrics_summary:
        ens = metrics_summary["ensemble"]
        st.markdown('<div class="card"><div class="card-title">🏆 Ensemble Test Metrics</div>', unsafe_allow_html=True)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Accuracy", f"{ens['accuracy']*100:.2f}%")
        m2.metric("F1 (macro)", f"{ens['f1_macro']*100:.2f}%")
        m3.metric("Cohen's Kappa", f"{ens['cohen_kappa']:.3f}")
        m4.metric("MCC", f"{ens['matthews_corrcoef']:.3f}")
        m5, m6, m7 = st.columns(3)
        m5.metric("Precision (macro)", f"{ens['precision_macro']*100:.2f}%")
        m6.metric("Recall (macro)", f"{ens['recall_macro']*100:.2f}%")
        m7.metric("Top-3 Accuracy", f"{ens['top3_accuracy']*100:.2f}%")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card"><div class="card-title">🧩 Per-Model Metrics</div>', unsafe_allow_html=True)
        per_model = metrics_summary["per_model"]
        weights = metrics_summary.get("ensemble_weights", {})
        rows = [{"Model": name, "Weight": round(weights.get(name, 0), 3), "Val Acc": f"{m.get('val_accuracy',0)*100:.2f}%",
                 "Test Acc": f"{m['accuracy']*100:.2f}%", "F1": f"{m['f1_macro']*100:.2f}%",
                 "Kappa": round(m["cohen_kappa"], 3), "MCC": round(m["matthews_corrcoef"], 3)} for name, m in per_model.items()]
        st.table(rows)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        alert("info", "ℹ️", "No <code>metrics_summary.json</code> yet. Run <code>train_model.py</code> or <code>recover_artifacts.py</code>.")

    col1, col2 = st.columns(2, gap="medium")
    with col1:
        st.markdown('<div class="card"><div class="card-title">📈 Training History</div>', unsafe_allow_html=True)
        if ensemble_config:
            for entry in ensemble_config["backbones"]:
                p = os.path.join(cfg.MODEL_DIR, f"{entry['name']}_training_history.png")
                if os.path.exists(p):
                    st.image(p, caption=f"{entry['display_name']} convergence", use_container_width=True)
        else:
            st.caption("Not available yet.")
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="card"><div class="card-title">🎯 Confusion Matrix</div>', unsafe_allow_html=True)
        cm_path = os.path.join(cfg.MODEL_DIR, "confusion_matrix_ensemble.png")
        if os.path.exists(cm_path):
            st.image(cm_path, use_container_width=True)
        else:
            st.caption("Not available yet.")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card"><div class="card-title">📁 Extended Paper Figures</div>', unsafe_allow_html=True)
    fig_dir = os.path.join(cfg.MODEL_DIR, "paper_figures")
    if os.path.isdir(fig_dir):
        fig_files = sorted(f for f in os.listdir(fig_dir) if f.lower().endswith(".png"))
        if fig_files:
            cols = st.columns(2)
            for i, fname in enumerate(fig_files):
                with cols[i % 2]:
                    st.image(os.path.join(fig_dir, fname), caption=fname.replace("_", " ").replace(".png", ""), use_container_width=True)
        else:
            st.caption("No figures found yet.")
    else:
        st.caption("Run `python generate_paper_figures.py` to produce ROC curves, PR curves, calibration diagrams, t-SNE embeddings, and more for the paper.")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card"><div class="card-title">🔥 Grad-CAM Samples</div>', unsafe_allow_html=True)
    if os.path.isdir(cfg.GRADCAM_SAMPLE_DIR):
        sample_files = sorted(f for f in os.listdir(cfg.GRADCAM_SAMPLE_DIR) if f.lower().endswith(".png"))
        if sample_files:
            cols = st.columns(4)
            for i, fname in enumerate(sample_files):
                with cols[i % 4]:
                    st.image(os.path.join(cfg.GRADCAM_SAMPLE_DIR, fname), caption=fname.replace("_gradcam.png", ""), use_container_width=True)
        else:
            st.caption("No samples found yet.")
    else:
        st.caption("Not available yet.")
    st.markdown('</div>', unsafe_allow_html=True)

    ens_report_path = os.path.join(cfg.MODEL_DIR, "classification_report_ensemble.txt")
    if os.path.exists(ens_report_path):
        with open(ens_report_path, "r", encoding="utf-8") as f:
            with st.expander("📄 Ensemble Classification Report"):
                st.text(f.read())

    if ensemble_config:
        for entry in ensemble_config["backbones"]:
            report_path = os.path.join(cfg.MODEL_DIR, f"{entry['name']}_classification_report.txt")
            if os.path.exists(report_path):
                with open(report_path, "r", encoding="utf-8") as f:
                    with st.expander(f"📄 {entry['display_name']} Report"):
                        st.text(f.read())
