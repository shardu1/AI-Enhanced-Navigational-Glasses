import os
import time
import cv2
import numpy as np
import torch
import pyttsx3
from ultralytics import YOLO
from collections import defaultdict
import threading
import queue
from PIL import Image
from transformers import pipeline

# ------------------ CONFIG ------------------

CAM_INDEX = 0
CAP_WIDTH, CAP_HEIGHT = 640, 480
YOLO_MODEL = "yolov8n.pt"
VLM_MODEL = "Salesforce/blip-image-captioning-large"

CONF_THRES = 0.35
SHOW_WINDOWS = True

SPEAK_INTERVAL = 5.0
URGENT_COOLDOWN = 2.0

FOCAL_LENGTH_PIXELS = 800
AVERAGE_STEP_LENGTH_M = 0.6

KNOWN_WIDTHS = {
    "person": 0.5, "car": 1.8, "bus": 2.5, "truck": 2.5,
    "bicycle": 0.7, "motorcycle": 0.8,
    "chair": 0.5, "dog": 0.4, "cat": 0.3,
}

ALERT_PRIORITY = ["person", "car", "truck", "bus", "motorcycle", "bicycle"]

object_cooldown = {}

# ----------------------------------------------------

# =============== AUDIO WORKER =======================

def audio_worker(q, tts_engine):
    while True:
        text = q.get()
        if text is None:
            break
        try:
            tts_engine.stop()
            tts_engine.say(text)
            tts_engine.runAndWait()
        except Exception as e:
            print(f"[TTS ERROR] {e}")
        q.task_done()

def clear_audio_queue(q):
    while not q.empty():
        try:
            q.get_nowait()
        except queue.Empty:
            break

# =============== DISTANCE + LOGIC ===================

def estimate_metric_distance(box, label, frame_width):
    if label not in KNOWN_WIDTHS:
        return None
    pixel_width = box[2] - box[0]
    if pixel_width < 5:
        return None
    reference_width = KNOWN_WIDTHS[label]
    apparent_width = pixel_width / frame_width
    if apparent_width <= 0:
        return None
    distance = (reference_width * FOCAL_LENGTH_PIXELS) / (apparent_width * frame_width)
    return max(0.3, min(distance, 25.0))

def meters_to_steps(distance):
    if distance is None:
        return "unknown distance"
    steps = round(distance / AVERAGE_STEP_LENGTH_M)
    if steps <= 0:
        return "right in front of you"
    elif steps == 1:
        return "one step away"
    elif steps <= 5:
        return f"{steps} steps away"
    else:
        return f"about {steps} steps ahead"

def lane_of_box(box, frame_w):
    cx = 0.5 * (box[0] + box[2])
    if cx < frame_w * 0.35:
        return "on your right"
    elif cx < frame_w * 0.65:
        return "ahead"
    else:
        return "on your left"

def get_danger_level(label, distance):
    if distance is None:
        return "low"
    if label in ALERT_PRIORITY and distance < 3:
        return "high"
    elif distance < 4:
        return "medium"
    return "low"

def generate_alert(priority_objs):
    if not priority_objs:
        return "All clear."

    closest = priority_objs[0]
    label = closest["label"]
    lane = closest["lane"]
    distance = closest["dist_m"]

    danger = get_danger_level(label, distance)
    steps = meters_to_steps(distance)

    if danger == "high":
        return f"Warning! {label} {lane}, {steps}"
    else:
        return f"There is a {label} {lane}, {steps}"

# =============== VLM ===============================

def ask_vlm(frame, prompt, vlm_pipeline, audio_queue):
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    try:
        if "Read" in prompt:
            answer = vlm_pipeline(pil_image, text=prompt, max_new_tokens=75)[0]['generated_text']
        else:
            answer = vlm_pipeline(pil_image, max_new_tokens=75)[0]['generated_text']

        answer = answer.replace(prompt, "").strip()
        clear_audio_queue(audio_queue)
        audio_queue.put(answer)
    except:
        audio_queue.put("Sorry, I could not analyze the scene.")




class SpeechController:
    def __init__(self, rate=180):
        self.engine = pyttsx3.init()
        self.engine.setProperty("rate", rate)
        self.engine.startLoop(False)  # Non-blocking loop
        self.last_spoken = ""
        self.last_time = 0

    def update(self):
        self.engine.iterate()  # Must be called continuously

    def speak(self, text, interrupt=False):
        if interrupt:
            self.engine.stop()
        self.engine.say(text)
        self.last_spoken = text
        self.last_time = time.time()


# ===================== MAIN =========================

def main():
    speech = SpeechController()
    print("[INIT] Loading YOLO...")
    yolo = YOLO(YOLO_MODEL)

    print("[INIT] Loading VLM...")
    device = 0 if torch.cuda.is_available() else -1
    vlm_pipeline = pipeline("image-to-text", model=VLM_MODEL, device=device)

    print("[INIT] Initializing TTS...")
    audio_queue = queue.Queue()
    tts_engine = pyttsx3.init()
    tts_engine.setProperty("rate", 180)

    audio_thread = threading.Thread(
        target=audio_worker,
        args=(audio_queue, tts_engine),
        daemon=True
    )
    audio_thread.start()

    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_HEIGHT)

    last_speak_time = 0
    last_spoken = ""
    vlm_processing = False

    while True:
        speech.update()
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        current_time = time.time()

        results = yolo.predict(frame, conf=CONF_THRES, verbose=False)[0]

        boxes = results.boxes.xyxy.cpu().numpy() if results.boxes else []
        clses = results.boxes.cls.cpu().numpy().astype(int) if results.boxes else []

        priority_objs = []

        for (x1, y1, x2, y2), k in zip(boxes, clses):
            label = yolo.names.get(k, str(k))
            dist = estimate_metric_distance((x1, y1, x2, y2), label, w)
            lane = lane_of_box((x1, y1, x2, y2), w)

            if dist and dist < 8:
                priority_objs.append({
                    "label": label,
                    "dist_m": dist,
                    "lane": lane,
                    "box": (x1, y1, x2, y2)
                })

        priority_objs.sort(key=lambda x: x["dist_m"])

        instruction = generate_alert(priority_objs)

        current_time = time.time()
        is_urgent = instruction.startswith("Warning")

        # 🚨 Immediate urgent interrupt (always speak if urgent and changed)
        if is_urgent and instruction != speech.last_spoken:
            speech.speak(instruction, interrupt=True)

        # 🔁 Normal 5-second check
        elif current_time - speech.last_time >= 5.0:
            
            # Only speak if message changed
            if instruction != speech.last_spoken:
                speech.speak(instruction, interrupt=False)


        # -------- VLM KEY INPUT --------
        key = cv2.waitKey(1) & 0xFF
        if not vlm_processing:
            if key == ord('d'):
                vlm_processing = True
                threading.Thread(
                    target=lambda: [ask_vlm(frame.copy(), "A photo of", vlm_pipeline, audio_queue),
                                    setattr(globals(), 'vlm_processing', False)]
                ).start()

            elif key == ord('r'):
                vlm_processing = True
                threading.Thread(
                    target=lambda: [ask_vlm(frame.copy(),
                                            "Read any text or signs in this image:",
                                            vlm_pipeline, audio_queue),
                                    setattr(globals(), 'vlm_processing', False)]
                ).start()

        # -------- VISUALIZATION --------
        if SHOW_WINDOWS:
            for obj in priority_objs:
                x1, y1, x2, y2 = obj["box"]
                dist = obj["dist_m"]
                steps = round(dist / AVERAGE_STEP_LENGTH_M)
                color = (0, 0, 255) if dist < 3 else (0, 165, 255)

                text = f"{obj['label']} {dist:.1f}m ({steps} steps)"
                cv2.rectangle(frame, (int(x1), int(y1)),
                              (int(x2), int(y2)), color, 2)
                cv2.putText(frame, text,
                            (int(x1), int(y1) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, color, 2)

            cv2.imshow("Navigation Assistant", frame)

        if key == ord('q'):
            break

    audio_queue.put(None)
    audio_thread.join()
    cap.release()
    cv2.destroyAllWindows()

# ====================================================

if __name__ == "__main__":
    main()
