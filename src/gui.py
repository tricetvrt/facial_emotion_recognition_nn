import gradio as gr
import torch
from PIL import Image
import mediapipe as mp
print(mp.__file__)
print(mp.__version__)
import numpy as np

import config
from model import get_resnet50
from transforms import val_transform
from dataset import get_class_names

model = get_resnet50(num_classes=config.NUM_CLASSES, pretrained=False, freeze_backbone=False)
model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location="cpu"))
model.eval()

class_names = get_class_names()

mp_face_detection = mp.solutions.face_detection
face_detector = mp_face_detection.FaceDetection(
    model_selection=1,
    min_detection_confidence=0.5
)


def crop_face(image: Image.Image):
    # PIL u numpy array
    img = np.array(image.convert("RGB"))

    results = face_detector.process(img)

    if not results.detections:
        return None

    detection = results.detections[0]
    bbox = detection.location_data.relative_bounding_box

    h, w, _ = img.shape

    x = int(bbox.xmin * w)
    y = int(bbox.ymin * h)
    bw = int(bbox.width * w)
    bh = int(bbox.height * h)

    # prosirenje malciceee oko lica
    margin = 20

    x1 = max(0, x - margin)
    y1 = max(0, y - margin)
    x2 = min(w, x + bw + margin)
    y2 = min(h, y + bh + margin)

    # numpy u PIL
    face = Image.fromarray(img[y1:y2, x1:x2])
    face.save("cropped_face.jpg")

    return face


def predict(image):
    face = crop_face(image)
    if face is None:
        return {"No face detected": 1.0}
    tensor = val_transform(face).unsqueeze(0)
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)[0]
    return {class_names[i]: float(probs[i]) for i in range(len(class_names))}

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=6),
    title="FER+ Emotion Classifier"
)

demo.launch()