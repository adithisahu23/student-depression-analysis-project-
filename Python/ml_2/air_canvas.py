import cv2
import time
import numpy as np
import mediapipe as mp
from collections import deque
from datetime import datetime

CAMERA_ID = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
MAX_BUFFER = 1024
DRAW_COLOR = (0, 0, 255)
BRUSH_THICKNESS = 8
ERASER_THICKNESS = 50
BACKGROUND_COLOR = (255, 255, 255)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.5,
)

def fingers_up(landmarks):
    tips_ids = [4, 8, 12, 16, 20]
    fingers = []
    if landmarks[tips_ids[0]].x < landmarks[2].x:
        fingers.append(True)
    else:
        fingers.append(False)
    for id in range(1, 5):
        tip_y = landmarks[tips_ids[id]].y
        pip_y = landmarks[tips_ids[id] - 2].y
        fingers.append(tip_y < pip_y)
    return fingers

def normalized_to_pixel(norm_x, norm_y, frame_w, frame_h):
    x = int(norm_x * frame_w)
    y = int(norm_y * frame_h)
    return x, y

def main():
    cap = cv2.VideoCapture(CAMERA_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        print("ERROR: Could not open camera.")
        return

    canvas = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
    last_x, last_y = None, None
    eraser_mode = False

    print("Press 'c' to clear, 's' to save, 'e' to toggle eraser, 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame.")
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        h, w, _ = frame.shape

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            lm = hand_landmarks.landmark

            fingers = fingers_up(lm)
            index_up = fingers[1]
            middle_up = fingers[2]

            ix, iy = normalized_to_pixel(lm[8].x, lm[8].y, w, h)

            if index_up and not middle_up:
                if last_x is None:
                    last_x, last_y = ix, iy

                if eraser_mode:
                    cv2.line(canvas, (last_x, last_y), (ix, iy), BACKGROUND_COLOR, ERASER_THICKNESS)
                else:
                    cv2.line(canvas, (last_x, last_y), (ix, iy), DRAW_COLOR, BRUSH_THICKNESS)

                last_x, last_y = ix, iy
            else:
                last_x, last_y = None, None

            cv2.circle(frame, (ix, iy), 8, (0, 255, 0), cv2.FILLED)

        overlay = cv2.addWeighted(frame, 0.5, canvas, 0.5, 0)
        cv2.imshow("Air Canvas", overlay)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            canvas[:] = 0
        elif key == ord('s'):
            save_img = np.full_like(canvas, BACKGROUND_COLOR)
            mask = np.any(canvas != 0, axis=2)
            save_img[mask] = canvas[mask]
            filename = f"air_canvas_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            cv2.imwrite(filename, save_img)
            print(f"Saved as {filename}")
        elif key == ord('e'):
            eraser_mode = not eraser_mode

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
