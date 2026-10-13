import csv
import io
import os
import tempfile
from pathlib import Path

import requests
import streamlit as st
from PIL import Image, ImageDraw, ImageFont


ROBOFLOW_ENDPOINT = "https://detect.roboflow.com"
MAX_VIDEO_FRAMES = 80

st.set_page_config(page_title="Form Check | Roboflow", page_icon="FC", layout="wide")
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@500;600;700&family=DM+Sans:wght@400;500;600;700&display=swap');
    :root { --ink: #17211f; --muted: #64716d; --paper: #f4f5ef; --green: #bddb5a; --line: #d9ded6; }
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; color: var(--ink); }
    .stApp { background: var(--paper); }
    h1, h2, h3 { font-family: 'Barlow Condensed', sans-serif !important; letter-spacing: 0 !important; }
    h1 { font-size: 3.4rem !important; line-height: .98 !important; margin-bottom: .2rem !important; }
    .kicker { color: #587029; text-transform: uppercase; font-size: .75rem; font-weight: 700; letter-spacing: .08em; }
    .lede { color: var(--muted); max-width: 620px; }
    [data-testid="stSidebar"] { background: #e8ece3; border-right: 1px solid var(--line); }
    [data-testid="stMetric"] { background: #fff; border: 1px solid var(--line); padding: 14px 16px; border-radius: 6px; }
    [data-testid="stFileUploader"] section { background: #fff; border: 1px dashed #9da99f; border-radius: 6px; }
    div.stButton > button, div.stDownloadButton > button { border-radius: 4px; font-weight: 700; }
    .verdict { font-size: 1.2rem; font-weight: 700; padding: 12px 14px; border-left: 4px solid var(--green); background: #fff; }
    hr { border-color: var(--line); }
    </style>
    """,
    unsafe_allow_html=True,
)


def normalized_labels(value):
    return {part.strip().lower().replace(" ", "_") for part in value.split(",") if part.strip()}


def classify(predictions, confidence_threshold, correct_labels, incorrect_labels):
    matches = [
        prediction
        for prediction in predictions
        if float(prediction.get("confidence", 0)) >= confidence_threshold
    ]
    classes = {str(item.get("class", "")).strip().lower().replace(" ", "_") for item in matches}
    incorrect = classes & incorrect_labels
    correct = classes & correct_labels
    if incorrect:
        verdict = "Needs attention"
    elif correct:
        verdict = "Looks correct"
    else:
        verdict = "Unclear"
    relevant_classes = incorrect if incorrect else correct
    relevant = [
        item
        for item in matches
        if str(item.get("class", "")).strip().lower().replace(" ", "_") in relevant_classes
    ]
    strongest = max(relevant or matches, key=lambda item: float(item.get("confidence", 0)), default=None)
    signal = str(strongest.get("class", "No matching detection")) if strongest else "No matching detection"
    score = float(strongest.get("confidence", 0)) if strongest else 0
    return verdict, signal, score, matches


def infer_frame(image, api_key, model_id, confidence_threshold):
    payload = io.BytesIO()
    image.convert("RGB").save(payload, format="JPEG", quality=90)
    response = requests.post(
        f"{ROBOFLOW_ENDPOINT}/{model_id.strip().strip('/')}",
        params={"api_key": api_key, "confidence": round(confidence_threshold * 100)},
        data=payload.getvalue(),
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=45,
    )
    if not response.ok:
        raise RuntimeError(f"Roboflow returned HTTP {response.status_code}. Check the model ID, API key, and model deployment.")
    result = response.json()
    predictions = result.get("predictions", [])
    if not isinstance(predictions, list):
        raise RuntimeError("The model response did not contain an object-detection predictions list.")
    return predictions


def render_predictions(image, predictions, incorrect_labels):
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 16)
    except OSError:
        font = ImageFont.load_default()
    for prediction in predictions:
        try:
            x, y = float(prediction["x"]), float(prediction["y"])
            width, height = float(prediction["width"]), float(prediction["height"])
        except (KeyError, TypeError, ValueError):
            continue
        left, top = x - width / 2, y - height / 2
        right, bottom = x + width / 2, y + height / 2
        label = f"{prediction.get('class', 'detection')} {float(prediction.get('confidence', 0)):.0%}"
        label_class = str(prediction.get("class", "")).strip().lower().replace(" ", "_")
        color = "#d44b35" if label_class in incorrect_labels else "#50751c"
        draw.rectangle((left, top, right, bottom), outline=color, width=3)
        label_box = draw.textbbox((left, top), label, font=font)
        draw.rectangle(label_box, fill=color)
        draw.text((left, top), label, fill="white", font=font)
    return canvas


def analyze_video(video_bytes, suffix, api_key, model_id, confidence_threshold, correct_labels, incorrect_labels, sample_seconds, frame_limit, progress):
    temporary_path = None
    rows = []
    samples = []
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as video_file:
            video_file.write(video_bytes)
            temporary_path = video_file.name
        import cv2

        capture = cv2.VideoCapture(temporary_path)
        fps = capture.get(cv2.CAP_PROP_FPS) or 0
        if not capture.isOpened() or fps <= 0:
            raise RuntimeError("Could not read this video. Try an MP4, MOV, or AVI file.")
        frame_interval = max(1, round(fps * sample_seconds))
        frame_index = 0
        checked = 0
        while checked < frame_limit:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % frame_interval == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(rgb)
                predictions = infer_frame(image, api_key, model_id, confidence_threshold)
                verdict, signal, score, matches = classify(predictions, confidence_threshold, correct_labels, incorrect_labels)
                timestamp = frame_index / fps
                rows.append({"time_seconds": round(timestamp, 2), "verdict": verdict, "top_detection": signal, "confidence": round(score, 4)})
                if len(samples) < 8:
                        samples.append((timestamp, render_predictions(image, matches, incorrect_labels)))
                checked += 1
                progress.progress(min(checked / frame_limit, 1.0), text=f"Checking frame {checked} of up to {frame_limit}")
            frame_index += 1
        capture.release()
        if not rows:
            raise RuntimeError("No readable frames were found in this video.")
        return rows, samples
    finally:
        if temporary_path:
            Path(temporary_path).unlink(missing_ok=True)


def csv_bytes(rows):
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=["time_seconds", "verdict", "top_detection", "confidence"])
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("utf-8")


with st.sidebar:
    st.markdown("### Model connection")
    api_key = st.text_input("Roboflow API key", value=os.getenv("ROBOFLOW_API_KEY", ""), type="password", help="Your key is sent only to Roboflow's inference endpoint.")
    model_id = st.text_input("Model ID", value=os.getenv("ROBOFLOW_MODEL_ID", ""), placeholder="workspace/project/version")
    confidence_threshold = st.slider("Minimum confidence", min_value=0.05, max_value=0.95, value=0.4, step=0.05)
    st.markdown("### Class mapping")
    correct_text = st.text_input("Correct-form classes", value="correct, correct_form, good_form, proper_form")
    incorrect_text = st.text_input("Needs-attention classes", value="incorrect, incorrect_form, bad_form, poor_form")
    st.caption("Comma-separated labels must match your trained model's class names.")

st.markdown('<p class="kicker">Roboflow vision prototype / 01</p>', unsafe_allow_html=True)
st.title("Form Check")
st.markdown('<p class="lede">Review an exercise image or video against your trained form-detection model. Results are model signals, not medical or coaching advice.</p>', unsafe_allow_html=True)

media_type = st.radio("Media", ["Image", "Video"], horizontal=True, label_visibility="collapsed")
correct_labels = normalized_labels(correct_text)
incorrect_labels = normalized_labels(incorrect_text)

if media_type == "Image":
    uploaded = st.file_uploader("Choose an exercise image", type=["jpg", "jpeg", "png", "webp"])
    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        left, right = st.columns([1.1, 0.9], gap="large")
        with left:
            st.image(image, use_container_width=True)
        with right:
            st.markdown("#### Image assessment")
            if st.button("Analyze image", type="primary", disabled=not api_key or not model_id, use_container_width=True):
                try:
                    predictions = infer_frame(image, api_key, model_id, confidence_threshold)
                    verdict, signal, score, matches = classify(predictions, confidence_threshold, correct_labels, incorrect_labels)
                    st.markdown(f'<div class="verdict">{verdict}</div>', unsafe_allow_html=True)
                    st.metric("Strongest signal", signal, f"{score:.0%} confidence")
                    st.image(render_predictions(image, matches), caption="Detected regions")
                    if verdict == "Unclear":
                        st.info("No configured correct- or incorrect-form class cleared the confidence threshold.")
                except (requests.RequestException, ValueError, RuntimeError) as error:
                    st.error(str(error))
            elif not api_key or not model_id:
                st.info("Add your Roboflow API key and deployed model ID in the sidebar to run inference.")
else:
    uploaded = st.file_uploader("Choose a workout video", type=["mp4", "mov", "avi", "m4v"])
    sample_seconds = st.slider("Seconds between sampled frames", min_value=0.5, max_value=5.0, value=1.0, step=0.5)
    frame_limit = st.slider("Maximum frames to analyze", min_value=5, max_value=MAX_VIDEO_FRAMES, value=30, step=5)
    if uploaded:
        st.video(uploaded)
        if st.button("Analyze sampled frames", type="primary", disabled=not api_key or not model_id, use_container_width=True):
            progress = st.progress(0, text="Preparing video")
            try:
                rows, samples = analyze_video(uploaded.getvalue(), Path(uploaded.name).suffix, api_key, model_id, confidence_threshold, correct_labels, incorrect_labels, sample_seconds, frame_limit, progress)
                counts = {verdict: sum(row["verdict"] == verdict for row in rows) for verdict in ("Looks correct", "Needs attention", "Unclear")}
                st.markdown("#### Sampled-frame assessment")
                metrics = st.columns(3)
                for column, (verdict, count) in zip(metrics, counts.items()):
                    column.metric(verdict, count, f"{count / len(rows):.0%} of sampled frames")
                st.bar_chart(counts, y_label="Frames", x_label="Assessment")
                st.dataframe(rows, use_container_width=True, hide_index=True)
                st.download_button("Download frame results as CSV", csv_bytes(rows), file_name="form-check-results.csv", mime="text/csv")
                if samples:
                    st.markdown("#### Example frames")
                    for start in range(0, len(samples), 4):
                        columns = st.columns(min(4, len(samples) - start))
                        for column, (timestamp, sample) in zip(columns, samples[start:start + 4]):
                            column.image(sample, caption=f"{timestamp:.1f}s", use_container_width=True)
                st.caption("Video status is counted per sampled frame; it is not a full-motion or rep-counting assessment.")
            except (requests.RequestException, ValueError, RuntimeError, OSError) as error:
                st.error(str(error))
    elif not api_key or not model_id:
        st.info("Add your Roboflow API key and deployed model ID in the sidebar to run inference.")
