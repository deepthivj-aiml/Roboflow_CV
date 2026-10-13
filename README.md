# Form Check

Form Check is a Streamlit prototype for reviewing exercise form from images, uploaded clips, or a live webcam. It has two inference paths: uploaded images and videos use your deployed Roboflow object-detection model; webcam mode estimates body pose locally with Ultralytics YOLO pose and flags a few configurable geometric cues.

This repository does not include training photographs or a pretrained form-quality model. Collect and label your own exercise images in Roboflow before using the hosted image/video assessment. The webcam cues are heuristics, not a trained form classifier or a guarantee of correct technique.

## App modes

| Input | What it does | Processing |
| --- | --- | --- |
| Image | Sends one image to your Roboflow model; displays class, confidence, and predicted boxes. | Image is sent to Roboflow's hosted inference endpoint. |
| Video | Samples frames at the selected interval, summarizes frame verdicts, and offers a CSV download. | Each sampled frame is sent to Roboflow. The app caps an upload at 80 inference requests. |
| Webcam | Streams annotated pose landmarks and possible issue cues: knees moving inward, uneven knee bend, or substantial torso lean. | YOLO pose runs on the Streamlit server using the browser's WebRTC camera stream; it does not call Roboflow. |

Webcam checks need a mostly full-body view. Knee alignment is best viewed from the front; torso lean is estimated in 2D and is sensitive to camera angle. The app shows possible issue cues, not medical or coaching advice.

## Run locally

Use Python 3.10–3.14. The first webcam run downloads the YOLO pose weights and may take longer while the model initializes.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

For **Image** and **Video**, enter a Roboflow API key and deployed model ID in `workspace/project/version` format in the sidebar. Alternatively, set `ROBOFLOW_API_KEY` and `ROBOFLOW_MODEL_ID` before starting the app. Keep API keys private and never commit them.

For **Webcam**, allow camera access in the browser. WebRTC requires a secure context (HTTPS or localhost) and working network access between the browser and Streamlit server. Remote containers may need port forwarding and an HTTPS tunnel; the camera is the browser device, while pose inference runs on the app server.

## Training images for Roboflow

The hosted image/video path requires an object-detection model trained on images labeled for the exercise and form cues. No training images are stored in this repository; upload your licensed, consented examples to a Roboflow object-detection project. A Roboflow YOLO export will typically have this structure:

```text
exercise-form/
├── data.yaml
├── train/
│   ├── images/    # labeled training photos or extracted video frames
│   └── labels/    # one YOLO box-label file per image
├── valid/
│   ├── images/
│   └── labels/
└── test/
	├── images/
	└── labels/
```

Use real examples from the camera views your users will use. Include full-body and relevant close views, different participants, clothing, lighting, distances, and exercise phases. Include both accepted form and the specific issues you want to flag. Choose a consistent annotation policy: for whole-person classes, box the person; for issue-specific classes, box the relevant body region. Example labels are `correct_form`, `knees_caving`, and `back_rounding`.

Avoid near-duplicate frames across training, validation, and test sets. Split by participant or source video/session so a model is evaluated on people and recordings it did not train on. In Roboflow, create the dataset version, train an object-detection model, and deploy it. Enter that deployed model ID in the app. In **Class mapping**, list accepted-form labels under **Correct-form classes** and fault labels under **Needs-attention classes**. An issue-class detection takes priority if both kinds appear in the same frame.

## Notebook training data

[workout_form_live.ipynb](workout_form_live.ipynb) is a separate local pose-estimation experiment. It reads UI-PRMD 3D skeleton coordinate sequences, not training photographs, and requires the dataset to be downloaded separately. If its archive does not provide explicit correct/incorrect labels, add `data/UI-PRMD/labels.csv` with `file,label,subject_id`; the notebook refuses to infer correctness from exercise names. Follow the dataset's current access and use terms.

The notebook's live and YouTube checks use a locally trained frame-level classifier and a correct-pose reference. That experiment is separate from the Roboflow-hosted detector used by the app's image/video modes.

## Limitations

Roboflow verdicts depend on your dataset, annotations, model version, and confidence threshold. Webcam cues use simple 2D geometry and can be wrong when landmarks are occluded or the view is angled. Video mode counts sampled frame predictions; it does not understand full movement timing or count repetitions. Validate with held-out people and footage before use. Do not use this prototype for injury diagnosis or as a replacement for qualified coaching.