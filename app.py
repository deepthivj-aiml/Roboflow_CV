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


def landmark_angle(first, vertex, last):
    first_vector = first - vertex
    last_vector = last - vertex
    denominator = float((first_vector**2).sum() ** 0.5 * (last_vector**2).sum() ** 0.5)
    if denominator < 1e-6:
        return None
    cosine = max(-1.0, min(1.0, float((first_vector * last_vector).sum()) / denominator))
    return float(__import__("math").degrees(__import__("math").acos(cosine)))


def inspect_pose(points, confidence):
    required = [5, 6, 11, 12, 13, 14, 15, 16]
    if confidence is None or any(float(confidence[index]) < 0.3 for index in required):
        return None, {}, "Move back so your shoulders, hips, knees, and ankles are visible."

    left_knee = landmark_angle(points[11], points[13], points[15])
    right_knee = landmark_angle(points[12], points[14], points[16])
    hip_width = float(((points[11] - points[12]) ** 2).sum() ** 0.5)
    ankle_width = float(((points[15] - points[16]) ** 2).sum() ** 0.5)
    knee_width = float(((points[13] - points[14]) ** 2).sum() ** 0.5)
    expected_knee_width = (hip_width + ankle_width) / 2
    shoulder_center = (points[5] + points[6]) / 2
    hip_center = (points[11] + points[12]) / 2
    torso_lean = float(__import__("math").degrees(__import__("math").atan2(
        abs(float(shoulder_center[0] - hip_center[0])),
        max(abs(float(shoulder_center[1] - hip_center[1])), 1e-6),
    )))

    issues = []
    knee_ratio = knee_width / expected_knee_width if expected_knee_width > 1 else 1.0
    if knee_ratio < 0.62:
        issues.append("Knees may be moving inward")
    if left_knee is not None and right_knee is not None and abs(left_knee - right_knee) > 30:
        issues.append("Knee bend differs side to side")
    if torso_lean > 50:
        issues.append("Substantial torso lean")

    metrics = {
        "knee_width_ratio": round(knee_ratio, 2),
        "left_knee_angle": round(left_knee, 1) if left_knee is not None else None,
        "right_knee_angle": round(right_knee, 1) if right_knee is not None else None,
        "torso_lean_degrees": round(torso_lean, 1),
    }
    return issues, metrics, None


@st.cache_resource(show_spinner="Loading the pose model...")
def load_pose_detector():
    import threading
    from ultralytics import YOLO

    return YOLO("yolo11n-pose.pt"), threading.Lock()


def render_webcam():
    import time
    import cv2
    from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer

    st.markdown("#### Live pose check")
    st.caption("Possible issue cues use simple 2D landmark thresholds. Use a front-facing view for knee alignment; results are not coaching or medical advice.")
    try:
        pose_model, model_lock = load_pose_detector()
    except Exception as error:
        st.error(f"Could not load the pose model: {error}")
        return

    class PoseProcessor(VideoProcessorBase):
        def __init__(self):
            self.latest = None
            self.latest_lock = __import__("threading").Lock()

        def recv(self, frame):
            image = frame.to_ndarray(format="bgr24")
            with model_lock:
                result = pose_model.predict(image, verbose=False, imgsz=480, conf=0.35)[0]
            annotated = result.plot()
            issues = None
            metrics = {}
            message = "No person pose detected. Keep your full body in view."

            if result.keypoints is not None and len(result.keypoints.xy):
                person_index = int(result.boxes.conf.argmax().item()) if result.boxes is not None and len(result.boxes) else 0
                points = result.keypoints.xy[person_index].cpu().numpy()
                confidence = result.keypoints.conf[person_index].cpu().numpy() if result.keypoints.conf is not None else None
                issues, metrics, message = inspect_pose(points, confidence)
                if issues is not None:
                    message = "Possible issues: " + ", ".join(issues) if issues else "No configured issue cue detected"

            if issues is None:
                banner = message
                color = (30, 145, 205)
            elif issues:
                banner = "Possible: " + ", ".join(issues)
                color = (45, 65, 210)
            else:
                banner = "No configured issue cue detected"
                color = (60, 145, 40)

            cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 42), (24, 33, 31), -1)
            cv2.putText(annotated, banner[:90], (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)
            with self.latest_lock:
                self.latest = {"issues": issues, "metrics": metrics, "message": message}
            return av.VideoFrame.from_ndarray(annotated, format="bgr24")

        def snapshot(self):
            with self.latest_lock:
                return dict(self.latest) if self.latest else None

    import av

    context = webrtc_streamer(
        key="form-check-webcam",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=PoseProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
    status = st.empty()
    while context.state.playing:
        processor = context.video_processor
        snapshot = processor.snapshot() if processor else None
        if snapshot:
            if snapshot["issues"] is None:
                status.info(snapshot["message"])
            elif snapshot["issues"]:
                status.error(snapshot["message"])
            else:
                status.success(snapshot["message"])
            if snapshot["metrics"]:
                columns = status.container().columns(4)
                for column, (label, value) in zip(columns, snapshot["metrics"].items()):
                    column.metric(label.replace("_", " ").title(), value)
        time.sleep(0.2)
    if not context.state.playing:
        status.info("Start the camera above to see live pose landmarks and issue cues.")


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

media_type = st.radio("Input", ["Image", "Video", "Webcam"], horizontal=True, label_visibility="collapsed")
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
                    st.image(render_predictions(image, matches, incorrect_labels), caption="Detected regions")
                    if verdict == "Unclear":
                        st.info("No configured correct- or incorrect-form class cleared the confidence threshold.")
                except (requests.RequestException, ValueError, RuntimeError) as error:
                    st.error(str(error))
            elif not api_key or not model_id:
                st.info("Add your Roboflow API key and deployed model ID in the sidebar to run inference.")
elif media_type == "Video":
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
else:
    render_webcam()
