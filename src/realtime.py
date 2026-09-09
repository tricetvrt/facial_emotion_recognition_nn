"""
realtime_emotion.py

Realtime prepoznavanje emocija preko web kamere:
  Kamera frejm -> MediaPipe detekcija lica -> crop lica -> MobileNetV4
  -> prikaz predikcije uz bounding box na frejmu, uzivo.

Instalacija (jednom):
    pip install mediapipe opencv-python

Pokretanje:
    python realtime_emotion.py

Pritisnite 'q' da zatvorite prozor.
"""

import time
import cv2
import torch
import numpy as np
import mediapipe as mp
from PIL import Image

import config
from model import get_mobilenetv4
from transforms import val_transform  # isti preprocessing kao pri evaluaciji - BEZ augmentacije
from dataset import get_class_names

DISPLAY_NAME = {
    "neutral": "neutral",
    "happiness": "happy",
    "surprise": "surprise",
    "sadness": "sad",
    "anger": "angry",
    "fear": "fear",
}

# boje po emociji (BGR format za OpenCV), radi lakseg vizuelnog razlikovanja
COLORS = {
    "neutral": (200, 200, 200),
    "happiness": (0, 255, 0),
    "surprise": (0, 200, 255),
    "sadness": (255, 0, 0),
    "anger": (0, 0, 255),
    "fear": (180, 0, 180),
}

CONFIDENCE_THRESHOLD = 0.30  # ispod ovoga prikazujemo "Nesigurno" umesto klase
FACE_DETECTION_CONFIDENCE = 0.6
FACE_MARGIN = 0.1  # dodatni prostor oko detektovanog lica (25% sirine/visine), da model vidi ceo kontekst lica


def load_model(path, device):
    model = get_mobilenetv4(
        num_classes=config.NUM_CLASSES,
        pretrained=False,
        freeze_backbone=False,  
    )
    model.load_state_dict(torch.load(path, map_location=device))
    model = model.to(device)
    model.eval()
    return model


def expand_and_clip_bbox(x, y, w, h, frame_w, frame_h, margin=FACE_MARGIN):
    # oznacava granice frejma i dodaje marginu oko boxa
    mx = int(w * margin)
    my = int(h * margin)
    x1 = max(0, x - mx)
    y1 = max(0, y - my)
    x2 = min(frame_w, x + w + mx)
    y2 = min(frame_h, y + h + my)
    return x1, y1, x2, y2


@torch.no_grad()
def predict_emotion(model, face_crop_bgr, device, class_names):
    
    #face_crop_bgr: numpy array (H, W, 3) u BGR formatu (OpenCV default)
    #vraca (predvidejna klasa, confidence, sve verovatnoce)

    # BGR u rgb
    face_rgb = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(face_rgb)

    tensor = val_transform(pil_image).unsqueeze(0).to(device)  # dodaje batch dimenziju [1, C, H, W]

    logits = model(tensor)
    probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

    pred_idx = int(np.argmax(probs))
    pred_class = class_names[pred_idx]
    confidence = float(probs[pred_idx])

    return pred_class, confidence, probs


def main():
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"koristimo: {device}")

    class_names = get_class_names()

    print(f"ucitavamo model: {config.MODEL_SAVE_PATH}")
    model = load_model(config.MODEL_SAVE_PATH, device)

    # MediaPipe Face Detection - lagan, brz, dobar za realtime
    mp_face_detection = mp.solutions.face_detection
    face_detector = mp_face_detection.FaceDetection(
        model_selection=0,  # 0 = optimizovano za lica blizu kamere (do ~2m), brze
        min_detection_confidence=FACE_DETECTION_CONFIDENCE,
    )

    camera = cv2.VideoCapture(0)  #pali kameruuuuuuuuuuuu
    if not camera.isOpened():
        print("GRESKA: Ne mogu da otvorim kameru.")
        return

    print("Kamera pokrenuta. Pritisnite 'q' da izadjete.")

    prev_time = time.time()
    fps = 0.0

    while True:
        ret, frame = camera.read()
        if not ret:
            print("GRESKA: nee mogu da procitam frejm sa kamereeeeeee")
            break

        frame_h, frame_w = frame.shape[:2]
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = face_detector.process(frame_rgb) # sa rgb verzije frejma pomocu medaia pipea detektujemo lice

        if results.detections: #ako ima detekcija
            for detection in results.detections:# za svaku analiziraj lice
                bbox = detection.location_data.relative_bounding_box # bounding mox za lice pamtimo i cropujemo sliku tu
                x = int(bbox.xmin * frame_w)
                y = int(bbox.ymin * frame_h)
                w = int(bbox.width * frame_w)
                h = int(bbox.height * frame_h)

                x1, y1, x2, y2 = expand_and_clip_bbox(x, y, w, h, frame_w, frame_h)
                #dodajemo malo marginu da ne bude skroz isecena faca

                if x2 <= x1 or y2 <= y1:
                    continue  # degenerisan bbox, preskoci

                face_crop = frame[y1:y2, x1:x2] # secemo frejm tu gde je andjeno lcie
                if face_crop.size == 0:
                    continue

                pred_class, confidence, _ = predict_emotion(model, face_crop, device, class_names)

                if confidence < CONFIDENCE_THRESHOLD: #ako nema pojma bolje da pise
                    label = f"unsure ({confidence*100:.0f}%)"
                    color = (128, 128, 128)
                else:
                    label = f"{DISPLAY_NAME[pred_class]} ({confidence*100:.0f}%)"
                    color = COLORS[pred_class]

                cv2.rectangle(frame, (x1, y1), (x2, y2), color,2)
                cv2.putText(
                    frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 1 ,
                )

        # FPS racunanje i prikaz
        curr_time = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(curr_time - prev_time, 1e-6))  # eksponencijalno glacanje
        prev_time = curr_time
        cv2.putText(
            frame, f"fps: {fps:.1f}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
        )

        cv2.imshow("Realtime Emotion Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()
    face_detector.close()


if __name__ == "__main__":
    main()