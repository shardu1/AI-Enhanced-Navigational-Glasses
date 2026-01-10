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

# ------------------ Config ------------------
CAM_INDEX = 0
CAP_WIDTH, CAP_HEIGHT, CAP_FPS = 640, 480, 30 
YOLO_MODEL = "yolov8m.pt"
VLM_MODEL = "Salesforce/blip-image-captioning-large"

CONF_THRES = 0.35
SHOW_WINDOWS = True
REMINDER_INTERVAL_SEC = 7.0 

# IMPROVED: Better distance estimation parameters
FOCAL_LENGTH_PIXELS = 800  # Adjusted for better accuracy
KNOWN_WIDTHS = {
    "person": 0.5, "car": 1.8, "bus": 2.5, "truck": 2.5, "bicycle": 0.7,
    "motorcycle": 0.8, "traffic light": 0.3, "stop sign": 0.3,
    "chair": 0.5, "bench": 1.0, "backpack": 0.4, "handbag": 0.4,
    "umbrella": 0.8, "dog": 0.4, "cat": 0.3, "bird": 0.2,
    "bottle": 0.1, "cup": 0.1, "book": 0.2, "cell phone": 0.15,
}

PRIORITY_DISTANCE_BUFFER_M = 1.0  # Increased buffer for better grouping
ALERT_PRIORITY = ["person", "car", "truck", "bus", "motorcycle", "bicycle"]

# NEW: Speech cooldown and tracking
MIN_SPEECH_COOLDOWN = 2.0
object_cooldown = {}  # Track when we last spoke about each object
# --------------------------------------------

# --- Audio Worker Thread ---
def audio_worker(q, tts_engine):
    """Handles all audio playback in a separate, non-blocking thread."""
    while True:
        try:
            text_to_speak = q.get()
            if text_to_speak is None: break
            tts_engine.say(text_to_speak)
            tts_engine.runAndWait()
            q.task_done()
        except Exception as e:
            print(f"[Audio Worker ERROR] {e}")

# --- VLM Interaction Function ---
def ask_vlm(frame, prompt, vlm_pipeline, audio_queue):
    """Gets a description or reads text from the VLM."""
    print(f"Asking VLM: '{prompt}'...")
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR_RGB))
    
    try:
        if "Read" in prompt:
            answer = vlm_pipeline(pil_image, text=prompt, max_new_tokens=75)[0]['generated_text']
        else: # For descriptions
            answer = vlm_pipeline(pil_image, max_new_tokens=75)[0]['generated_text']

        answer = answer.replace(prompt, "").strip()
        print(f"VLM Response: {answer}")
        
        clear_audio_queue(audio_queue)
        audio_queue.put(answer)
    except Exception as e:
        print(f"[VLM ERROR] {e}")
        audio_queue.put("Sorry, I could not analyze the scene.")

def clear_audio_queue(q):
    """Clears all pending messages from the audio queue."""
    while not q.empty():
        try: q.get_nowait()
        except queue.Empty: continue

# --- IMPROVED Distance Estimation ---
def estimate_metric_distance(box, label, frame_width):
    """Improved distance estimation using apparent size method"""
    if label not in KNOWN_WIDTHS:
        return None
    
    pixel_width = box[2] - box[0]
    if pixel_width < 5:  # Minimum pixel width threshold
        return None
    
    reference_width = KNOWN_WIDTHS[label]
    apparent_width = pixel_width / frame_width
    
    if apparent_width > 0:
        distance = (reference_width * FOCAL_LENGTH_PIXELS) / (apparent_width * frame_width)
        # Clamp distance to reasonable values
        return max(0.3, min(distance, 25.0))
    return None

def lane_of_box(box, frame_w):
    """Determine object position with more precision"""
    cx = 0.5 * (box[0] + box[2])
    if cx < frame_w * 0.35: 
        return "on your right"
    elif cx < frame_w * 0.65: 
        return "ahead"
    else: 
        return "on your left"

def get_danger_level(label, distance):
    """Determine danger level based on object type and distance"""
    dangerous_objects = ["person", "car", "truck", "bus", "motorcycle", "bicycle"]
    
    if label in dangerous_objects and distance < 3.0:
        return "high"
    elif distance < 1.5:
        return "high"
    elif distance < 4.0:
        return "medium"
    else:
        return "low"

def should_speak_about_object(label, position, distance, current_time):
    """Check if we should speak about this object (cooldown management)"""
    object_key = f"{label}_{position}"
    
    # Check cooldown
    last_spoken = object_cooldown.get(object_key, 0)
    if current_time - last_spoken < MIN_SPEECH_COOLDOWN:
        return False
    
    # Only speak about high/medium priority objects
    danger_level = get_danger_level(label, distance)
    if danger_level in ["high", "medium"]:
        object_cooldown[object_key] = current_time
        return True
    
    return False

# --- IMPROVED Alert Generation ---
def generate_yolo_alert(priority_objs, current_time):
    """Generate intelligent alerts with better object grouping"""
    if not priority_objs: 
        return "All clear."
    
    # Group objects by type and danger level
    object_groups = defaultdict(list)
    urgent_warnings = []
    
    for obj in priority_objs:
        label = obj["label"]
        distance = obj["dist_m"]
        position = obj["lane"]
        
        # Check for urgent warnings
        danger_level = get_danger_level(label, distance)
        if danger_level == "high" and should_speak_about_object(label, position, distance, current_time):
            if distance < 1.5:
                urgent_warnings.append(f"Warning! {label} very close {position}")
            else:
                urgent_warnings.append(f"Warning! {label} {position} at {distance:.1f} meters")
        
        object_groups[label].append(obj)
    
    # Return urgent warnings immediately
    if urgent_warnings:
        return urgent_warnings[0]  # Return the most urgent warning
    
    # Generate normal alerts for medium priority objects
    alert_parts = []
    for label in ALERT_PRIORITY:
        if label in object_groups:
            objects_in_group = object_groups[label]
            count = len(objects_in_group)
            
            # Get the closest object in this group
            closest_obj = min(objects_in_group, key=lambda x: x['dist_m'] if x['dist_m'] is not None else float('inf'))
            
            if closest_obj['dist_m'] is not None and should_speak_about_object(
                label, closest_obj['lane'], closest_obj['dist_m'], current_time
            ):
                obj_name = label if count == 1 else f"{label}s"
                distance_word = "close" if closest_obj['dist_m'] < 3.0 else "ahead"
                alert_parts.append(f"{obj_name} {distance_word} {closest_obj['lane']}")
    
    if alert_parts:
        return "There is " + ", ".join(alert_parts[:2])  # Limit to 2 alerts
    
    return "All clear."

def main():
    # --- Model and Audio Initialization ---
    print("[init] Loading YOLO model...")
    yolo = YOLO(YOLO_MODEL)

    print("[init] Loading Vision-Language Model (VLM)...")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[init] Using device: {device}")

    try:
        vlm_pipeline = pipeline("image-to-text", model=VLM_MODEL, device=device, torch_dtype=torch.float16 if device != "cpu" else torch.float32)
    except Exception as e:
        print(f"[FATAL] Could not load VLM model. Error: {e}")
        return
    
    audio_queue = queue.Queue()
    try:
        tts_engine = pyttsx3.init()
        tts_engine.setProperty("rate", 180)
        audio_thread = threading.Thread(target=audio_worker, args=(audio_queue, tts_engine), daemon=True)
        audio_thread.start()
    except Exception as e:
        print(f"[FATAL] Could not initialize TTS engine: {e}")
        return

    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_HEIGHT)
    
    last_spoken, last_speak_time = "", 0
    vlm_is_processing = False

    while True:
        ok, frame = cap.read()
        if not ok: break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        current_time = time.time()

        # --- Continuous YOLO Detections ---
        yolo_results = yolo.predict(source=frame, conf=CONF_THRES, verbose=False)[0]
        boxes = yolo_results.boxes.xyxy.cpu().numpy() if yolo_results.boxes is not None else []
        clses = yolo_results.boxes.cls.cpu().numpy().astype(int) if yolo_results.boxes is not None else []
        
        priority_objs = []
        all_objs = []
        
        for (x1, y1, x2, y2), k in zip(boxes, clses):
            label = yolo.names.get(k, str(k))
            
            # IMPROVED: Use frame width for better distance estimation
            dist_m = estimate_metric_distance((x1, y1, x2, y2), label, w)
            lane = lane_of_box((x1, y1, x2, y2), w)
            
            obj_data = {
                "box": (x1, y1, x2, y2), 
                "label": label, 
                "dist_m": dist_m, 
                "lane": lane,
                "danger_level": get_danger_level(label, dist_m) if dist_m else "low"
            }
            
            all_objs.append(obj_data)
            
            # Priority objects are those that are close enough to matter
            if dist_m is not None and dist_m < 8.0:  # Only consider objects within 8 meters
                priority_objs.append(obj_data)
        
        # IMPROVED: Sort by distance (closest first)
        priority_objs.sort(key=lambda x: x["dist_m"] if x["dist_m"] is not None else float('inf'))
        
        # IMPROVED: Use current time for better cooldown management
        yolo_instruction = generate_yolo_alert(priority_objs, current_time)
        
        # IMPROVED: Speech logic with better cooldown
        if yolo_instruction != last_spoken and yolo_instruction != "All clear.":
            if audio_queue.empty():  # Only speak if queue is empty
                clear_audio_queue(audio_queue)
                audio_queue.put(yolo_instruction)
                last_spoken = yolo_instruction
                last_speak_time = current_time
                print(f"ALERT: {yolo_instruction}")
        
        # Reminder system
        elif (yolo_instruction != "All clear." and 
              current_time - last_speak_time > REMINDER_INTERVAL_SEC and 
              audio_queue.empty()):
            audio_queue.put(yolo_instruction)
            last_speak_time = current_time
            print(f"REMINDER: {yolo_instruction}")

        # --- Handle User Input for VLM ---
        key = cv2.waitKey(1) & 0xFF
        if not vlm_is_processing:
            vlm_prompt = None
            if key == ord('d'): # Describe scene
                vlm_prompt = "A photo of"
            elif key == ord('r'): # Read text
                vlm_prompt = "Read any text or signs in this image out loud:"

            if vlm_prompt:
                clear_audio_queue(audio_queue)
                audio_queue.put("Analyzing, please wait.")
                
                def vlm_task_wrapper(frame, prompt, vlm, q):
                    global vlm_is_processing
                    ask_vlm(frame, prompt, vlm, q)
                    vlm_is_processing = False

                vlm_is_processing = True
                threading.Thread(target=vlm_task_wrapper, args=(frame.copy(), vlm_prompt, vlm_pipeline, audio_queue)).start()
        
        if SHOW_WINDOWS:
            # IMPROVED: Better visualization with color coding
            for obj in priority_objs:
                x1, y1, x2, y2 = obj["box"]
                label = obj["label"]
                distance = obj["dist_m"]
                danger_level = obj["danger_level"]
                
                # Color code based on danger level
                if danger_level == "high":
                    color = (0, 0, 255)  # Red
                elif danger_level == "medium":
                    color = (0, 165, 255)  # Orange
                else:
                    color = (0, 255, 0)  # Green
                
                display_text = f"{label}: {distance:.1f}m" if distance else f"{label}"
                
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(frame, display_text, (int(x1), max(15, int(y1)-5)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # Add position indicator
                cv2.putText(frame, obj["lane"], (int(x1), int(y2) + 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Display current alert
            on_screen_text = yolo_instruction.replace("Watch out. ", "").split(".")[0]
            cv2.putText(frame, on_screen_text, (10, h-20), cv2.FONT_HERSHEY_TRIPLEX, 0.8, (0,0,255), 2)
            cv2.putText(frame, "Press 'd' to Describe, 'r' to Read Text", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            if vlm_is_processing:
                 cv2.putText(frame, "VLM PROCESSING...", (w - 250, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            cv2.imshow("Enhanced Navigation Assistant", frame)
            
        if key == ord('q'):
            break

    # Cleanup
    audio_queue.put(None)
    audio_thread.join()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
