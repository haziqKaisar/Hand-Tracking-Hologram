import cv2
import numpy as np
import mediapipe as mp
import math
import os
import urllib.request

from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

# ==============================================================================
# Konfigurasi Model
# ==============================================================================
MODEL_PATH = "hand_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)

def ensure_model_downloaded():
    if not os.path.exists(MODEL_PATH):
        print("Model belum ada, mendownload hand_landmarker.task (sekali saja)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Download selesai.")

def dist(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

# ==============================================================================
# Kumpulan Fungsi Efek Visual
# ==============================================================================
def effect_edges(frame_bgr):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 150)
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

def effect_thermal(frame_bgr):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.applyColorMap(gray, cv2.COLORMAP_JET)

def effect_neon(frame_bgr):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 150)
    neon = np.zeros_like(frame_bgr)
    neon[edges > 0] = (255, 60, 220)
    glow = cv2.GaussianBlur(neon, (7, 7), 3)
    return cv2.addWeighted(neon, 1.0, glow, 0.6, 0)

def effect_matrix(frame_bgr):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 40, 120)
    matrix = np.zeros_like(frame_bgr)
    matrix[edges > 0] = (0, 255, 0)
    return matrix

def effect_blur(frame_bgr):
    # Menggunakan Gaussian Blur agar efek blur halus dan kentara
    return cv2.GaussianBlur(frame_bgr, (61, 61), 0)

def effect_invert(frame_bgr):
    return cv2.bitwise_not(frame_bgr)

def effect_ocean(frame_bgr):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.applyColorMap(gray, cv2.COLORMAP_OCEAN)

def effect_all_mixed(frame_bgr):
    edges = effect_edges(frame_bgr)
    thermal = effect_thermal(frame_bgr)
    return cv2.addWeighted(thermal, 0.6, edges, 0.4, 0)

EFFECTS = [
    ("edges", effect_edges),
    ("thermal", effect_thermal),
    ("neon", effect_neon),
    ("matrix", effect_matrix),
    ("blur", effect_blur), # Pixelated diganti jadi Blur
    ("negative", effect_invert),
    ("ocean", effect_ocean),
    ("all-mixed", effect_all_mixed)
]

# ==============================================================================
# Program Utama
# ==============================================================================
def main():
    ensure_model_downloaded()

    base_options = BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=2,
        running_mode=vision.RunningMode.VIDEO,
        min_hand_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )
    detector = vision.HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Webcam tidak dapat diakses.")
        return

    current_effect_idx = 0
    cooldown = 0
    frame_timestamp_ms = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        frame_timestamp_ms += 33
        result = detector.detect_for_video(mp_image, frame_timestamp_ms)

        output_frame = frame.copy()
        active_effect_name, active_effect_func = EFFECTS[current_effect_idx]

        if result.hand_landmarks:
            hand_data = []

            for hand_landmarks in result.hand_landmarks:
                px_landmarks = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
                
                wrist = px_landmarks[0]
                thumb_tip = px_landmarks[4]   # Ujung Jempol
                index_tip = px_landmarks[8]   # Ujung Telunjuk
                middle_mcp = px_landmarks[9]

                for pt in px_landmarks:
                    cv2.circle(output_frame, pt, 2, (200, 200, 200), -1)

                palm_size = dist(wrist, middle_mcp)
                pinch_dist = dist(thumb_tip, index_tip)
                is_pinching = palm_size > 0 and (pinch_dist < palm_size * 0.35)

                hand_data.append({
                    "cx": middle_mcp[0],
                    "thumb": thumb_tip,
                    "index": index_tip,
                    "is_pinching": is_pinching
                })

            if len(hand_data) == 2:
                hand_data.sort(key=lambda x: x["cx"])
                l = hand_data[0] 
                r = hand_data[1] 

                # Urutan paten: Jempol-Jempol-Telunjuk-Telunjuk
                clean_box_pts = np.array([
                    l["thumb"],
                    r["thumb"],
                    r["index"],
                    l["index"]
                ], dtype=np.int32)

                mask = np.zeros((h, w), dtype=np.uint8)
                effect_frame = active_effect_func(frame)

                # Gunakan fillPoly agar tidak glitch saat menyilang (hourglass shape)
                cv2.fillPoly(mask, [clean_box_pts], 255)
                mask_3ch = cv2.merge([mask, mask, mask])

                output_frame = np.where(mask_3ch == 255, effect_frame, output_frame)
                
                # Garis pinggiran hologram
                cv2.polylines(output_frame, [clean_box_pts], isClosed=True, color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)

                # Logika Ganti Efek
                if l["is_pinching"] and r["is_pinching"]:
                    if cooldown == 0:
                        current_effect_idx = (current_effect_idx + 1) % len(EFFECTS)
                        cooldown = 35 
                    cv2.putText(output_frame, "PINCH! SWITCHING...", (20, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

        if cooldown > 0:
            cooldown -= 1

        cv2.putText(output_frame, f"Effect: {active_effect_name.upper()} (Pinch 2 Tangan untuk ganti)", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(output_frame, "Tekan 'q' di keyboard untuk keluar", (15, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

        cv2.imshow("Finger Grip Hologram", output_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    detector.close()

if __name__ == "__main__":
    main()