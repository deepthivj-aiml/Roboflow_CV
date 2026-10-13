# Form Check

A small Streamlit computer-vision app that sends an exercise image or sampled video frames to a deployed Roboflow object-detection model. It marks configured detections as **Looks correct**, **Needs attention**, or **Unclear** and shows the detected regions and confidence scores.

This is an inference app, not a pre-trained fitness coach. You need to train and deploy a model on examples of the exercise and form labels you care about. A single-frame detector cannot assess movement quality across a full repetition.

## Run locally

Use Python 3.10–3.14.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

In the app sidebar, enter your Roboflow API key and deployed model ID in `workspace/project/version` format. The key can instead be supplied through `ROBOFLOW_API_KEY`; `ROBOFLOW_MODEL_ID` can supply the model ID. The sidebar fields take precedence when changed. Keep keys private and do not commit them.

## Prepare a Roboflow model

1. Create a Roboflow object-detection project and upload representative images of the exercise, including different body types, camera angles, lighting, and both correct and incorrect examples.
2. Annotate the relevant form cues consistently. You can use class names such as `correct_form` and `incorrect_form`, or label specific issues such as `knees_caving` and `back_rounding`.
3. Split training and validation examples by person or recording session where possible, so near-identical frames do not leak across splits. Train a model and deploy a version.
4. Copy the deployed model ID into the sidebar. In **Class mapping**, set comma-separated class names exactly to your model's labels. Classes that indicate an issue should go in **Needs-attention classes**; accepted-form classes go in **Correct-form classes**. Issue classes take precedence if both types are detected in a frame.

The app calls Roboflow's hosted object-detection endpoint. Uploaded images and sampled video frames are sent to Roboflow for inference. Video analysis samples at a configurable interval and caps requests at 80 frames per upload; each sampled frame is a separate inference request.

## Interpret results carefully

`Looks correct` means at least one configured correct-form class cleared the confidence threshold and no configured issue class did. `Needs attention` means a configured issue class cleared it. `Unclear` means neither did. These are model outputs, not guarantees of correct technique. Validate the model with held-out footage before relying on it, and do not use it for injury diagnosis or as a substitute for qualified coaching.