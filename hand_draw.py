"""
╔══════════════════════════════════════════════════════╗
║       HAND GESTURE DRAWING APP - by Claude           ║
║  Apni ungli se draw karo - Camera ke saamne!         ║
╚══════════════════════════════════════════════════════╝

CONTROLS:
  ☝️  Sirf Index Finger UP  → Drawing mode (draw karo)
  ✌️  Index + Middle UP     → Move mode (draw nahi hoga)
  ✊  Mutthi band karo       → Sab kuch erase (clear)
  [S] key                   → Drawing save karo
  [C] key                   → Canvas clear karo
  [Q] / [ESC]               → App band karo

COLOR ZONES (screen ke upar):
  LEFT  → Laal (Red)
  MID-L → Neela (Blue)
  MID-R → Hara (Green)
  RIGHT → Peela (Yellow)
  FAR-R → Eraser
"""

import cv2
import mediapipe as mp
import numpy as np
import os
from datetime import datetime

# ─── MediaPipe Setup ───────────────────────────────────────────────────────────
mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils

# ─── Constants ─────────────────────────────────────────────────────────────────
WINDOW_NAME   = "✏️  Hand Drawing App"
SAVE_FOLDER   = os.path.expanduser("~/Desktop/HandDrawings")
BRUSH_SIZE    = 8
ERASER_SIZE   = 40
SMOOTHING     = 5          # kitne points ka average lena hai

COLORS = {
    "Laal"   : (0,   0,   220),
    "Neela"  : (220, 60,  20 ),
    "Hara"   : (20,  180, 20 ),
    "Peela"  : (20,  220, 220),
    "Eraser" : None,           # None = eraser
}

COLOR_LIST = list(COLORS.items())   # ordered list

os.makedirs(SAVE_FOLDER, exist_ok=True)

# ─── Helper: finger tips ───────────────────────────────────────────────────────
TIPS = [4, 8, 12, 16, 20]   # thumb, index, middle, ring, pinky

def fingers_up(lm, hand_label):
    """Return list of 5 booleans: which fingers are up."""
    up = []
    # Thumb (direction depends on hand)
    if hand_label == "Right":
        up.append(lm[4].x < lm[3].x)
    else:
        up.append(lm[4].x > lm[3].x)
    # Other four fingers
    for tip, pip in zip(TIPS[1:], [6, 10, 14, 18]):
        up.append(lm[tip].y < lm[pip].y)
    return up

# ─── Helper: UI overlay ────────────────────────────────────────────────────────
def draw_ui(frame, canvas, sel_color_name, brush, mode_text, msg=""):
    h, w = frame.shape[:2]
    col_w = w // len(COLOR_LIST)

    # Color palette bar at top
    for i, (name, color) in enumerate(COLOR_LIST):
        x1, x2 = i * col_w, (i + 1) * col_w
        bar_color = color if color else (255, 255, 255)
        cv2.rectangle(frame, (x1, 0), (x2, 60), bar_color, -1)

        # Hatching for eraser
        if color is None:
            for k in range(0, 60, 8):
                cv2.line(frame, (x1, k), (x2, k + 8), (180, 180, 180), 1)
            cv2.putText(frame, "Eraser", (x1 + 5, 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 80, 80), 2)
        else:
            text_color = (255, 255, 255) if sum(color) < 400 else (30, 30, 30)
            cv2.putText(frame, name, (x1 + 8, 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, text_color, 2)

        # Highlight selected
        if name == sel_color_name:
            cv2.rectangle(frame, (x1 + 2, 2), (x2 - 2, 58), (255, 255, 255), 3)

    # Divider
    cv2.line(frame, (0, 60), (w, 60), (80, 80, 80), 2)

    # Status bar at bottom
    cv2.rectangle(frame, (0, h - 40), (w, h), (30, 30, 30), -1)
    status = f"Mode: {mode_text}  |  Color: {sel_color_name}  |  Brush: {brush}px  |  [S]=Save  [C]=Clear  [Q]=Quit"
    cv2.putText(frame, status, (10, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # Floating message
    if msg:
        (tw, th), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        cx = (w - tw) // 2
        cv2.rectangle(frame, (cx - 10, h // 2 - 40), (cx + tw + 10, h // 2 + 10),
                      (0, 0, 0), -1)
        cv2.putText(frame, msg, (cx, h // 2 - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 120), 2)

# ─── Main App ──────────────────────────────────────────────────────────────────
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Camera nahi mili! Webcam connect karo aur dobara chalao.")
        return

    ret, frame = cap.read()
    if not ret:
        print("❌ Camera se frame nahi aaya.")
        cap.release()
        return

    h, w = frame.shape[:2]
    canvas = np.zeros((h, w, 3), dtype=np.uint8)   # drawing surface

    sel_color_name = "Laal"
    brush          = BRUSH_SIZE
    prev_pts       = []       # smoothing buffer
    flash_msg      = ""
    flash_timer    = 0
    prev_x = prev_y = None

    with mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.6
    ) as hands:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)   # mirror
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            mode_text = "🕊️ Waiting..."

            if result.multi_hand_landmarks:
                for hlm, hinfo in zip(result.multi_hand_landmarks,
                                      result.multi_handedness):
                    lm         = hlm.landmark
                    hand_label = hinfo.classification[0].label
                    up         = fingers_up(lm, hand_label)

                    # Index finger tip pixel coords
                    ix = int(lm[8].x * w)
                    iy = int(lm[8].y * h)

                    # Draw hand skeleton
                    mp_draw.draw_landmarks(frame, hlm, mp_hands.HAND_CONNECTIONS,
                        mp_draw.DrawingSpec(color=(100, 255, 100), thickness=2, circle_radius=4),
                        mp_draw.DrawingSpec(color=(200, 200, 200), thickness=2))

                    # ── Color select zone (top 60 px) ──
                    if iy < 60:
                        zone = min(ix // (w // len(COLOR_LIST)), len(COLOR_LIST) - 1)
                        sel_color_name = COLOR_LIST[zone][0]
                        prev_x = prev_y = None
                        mode_text = f"🎨 Color: {sel_color_name}"

                    # ── Fist = Clear ──
                    elif all(not u for u in up[1:]):
                        canvas[:] = 0
                        prev_x = prev_y = None
                        flash_msg   = "🗑️  Canvas Clear!"
                        flash_timer = 40
                        mode_text   = "✊ Clear"

                    # ── Two fingers up = Move (no draw) ──
                    elif up[1] and up[2]:
                        prev_x = prev_y = None
                        mode_text = "✌️  Move Mode"
                        # show cursor dot
                        cv2.circle(frame, (ix, iy), 12, (255, 255, 0), 2)

                    # ── Only index finger = Draw ──
                    elif up[1] and not up[2]:
                        mode_text = "✏️  Drawing"
                        cur_color = COLORS[sel_color_name]

                        # Smoothing
                        prev_pts.append((ix, iy))
                        if len(prev_pts) > SMOOTHING:
                            prev_pts.pop(0)
                        sx = int(np.mean([p[0] for p in prev_pts]))
                        sy = int(np.mean([p[1] for p in prev_pts]))

                        if prev_x is not None:
                            if cur_color is None:
                                # Eraser
                                cv2.line(canvas, (prev_x, prev_y), (sx, sy),
                                         (0, 0, 0), ERASER_SIZE)
                                cv2.circle(frame, (sx, sy), ERASER_SIZE // 2,
                                           (255, 255, 255), 2)
                            else:
                                cv2.line(canvas, (prev_x, prev_y), (sx, sy),
                                         cur_color, brush)
                                cv2.circle(frame, (sx, sy), brush // 2 + 2,
                                           cur_color, -1)
                        prev_x, prev_y = sx, sy

                    else:
                        prev_x = prev_y = None
            else:
                prev_x = prev_y = None
                prev_pts.clear()

          

if __name__ == "__main__":
    print(__doc__)
    main()