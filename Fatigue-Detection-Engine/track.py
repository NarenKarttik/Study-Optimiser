"""
STEP 3 — Live Driver Risk Monitoring (webcam).

Uses the trained drowsy_classifier.pkl for real-time drowsiness detection.
Also runs YOLOv8 Nano for phone detection.
No GPU required — runs on CPU.

Usage:
    pip install opencv-python mediapipe ultralytics scikit-learn joblib pygame
    python step3_live_tracking.py

Controls: Q = quit  |  R = recalibrate  |  S = screenshot
"""
import os
import ssl

# This bypasses the certificate check for the model download
if (not os.environ.get('PYTHONHTTPSVERIFY', '') and 
    getattr(ssl, '_create_unverified_context', None)):
    ssl._create_default_https_context = ssl._create_unverified_context
import cv2
import mediapipe as mp
import numpy as np
import joblib
import time
import threading
import sys
from collections import deque
from pathlib import Path

# ── Load drowsiness classifier ────────────────────────────────────────────────
MODEL_PATH = "drowsy_classifier.pkl"
if not Path(MODEL_PATH).exists():
    print(f"ERROR: {MODEL_PATH} not found. Run step2_train_and_evaluate.py first.")
    sys.exit(1)

bundle    = joblib.load(MODEL_PATH)
clf       = bundle["clf"]
scaler    = bundle["scaler"]
FEATURES  = bundle["features"]
MODEL_NAME = bundle["model_name"]
print(f"Loaded: {MODEL_NAME}")

# ── YOLOv8 ───────────────────────────────────────────────────────────────────
try:
    from ultralytics import YOLO
    yolo = YOLO("yolov8n.pt")
    USE_YOLO = True
    print("YOLOv8 loaded.")
except Exception as e:
    USE_YOLO = False
    print(f"YOLOv8 not available ({e}) — phone detection disabled.")

PHONE_CLASS = 67   # COCO: cell phone

# ── MediaPipe ─────────────────────────────────────────────────────────────────
mp_fm   = mp.solutions.face_mesh
mp_pose = mp.solutions.pose

face_mesh = mp_fm.FaceMesh(max_num_faces=1, refine_landmarks=False,
                            min_detection_confidence=0.5, min_tracking_confidence=0.4)
pose_est  = mp_pose.Pose(model_complexity=0,
                          min_detection_confidence=0.5, min_tracking_confidence=0.4)

# ── Landmark indices ──────────────────────────────────────────────────────────
L_EYE  = [362, 385, 387, 263, 373, 380]
R_EYE  = [33,  160, 158, 133, 153, 144]
MOUTH  = [61, 291, 17, 0]
LM_IDX = [1, 152, 263, 33, 287, 57]
MODEL_PTS = np.array([
    [0.0,    0.0,    0.0],
    [0.0,   -330.0, -65.0],
    [-225.0,  170.0,-135.0],
    [225.0,   170.0,-135.0],
    [-150.0, -150.0,-125.0],
    [150.0,  -150.0,-125.0],
], dtype=np.float64)


def _ear(lm, idx, W, H):
    p = [(lm[i].x*W, lm[i].y*H) for i in idx]
    A = np.linalg.norm(np.array(p[1]) - np.array(p[5]))
    B = np.linalg.norm(np.array(p[2]) - np.array(p[4]))
    C = np.linalg.norm(np.array(p[0]) - np.array(p[3]))
    return (A+B)/(2*C+1e-7)


def _mar(lm, W, H):
    pts = [(lm[i].x*W, lm[i].y*H) for i in MOUTH]
    v = np.linalg.norm(np.array(pts[3]) - np.array(pts[2]))
    h = np.linalg.norm(np.array(pts[0]) - np.array(pts[1]))
    return v/(h+1e-7)


def _head_pose(lm, W, H):
    img_pts = np.array([(lm[i].x*W, lm[i].y*H) for i in LM_IDX], dtype=np.float64)
    cam  = np.array([[W, 0, W/2],[0, W, H/2],[0, 0, 1]], dtype=np.float64)
    dist = np.zeros((4,1))
    ok, rvec, _ = cv2.solvePnP(MODEL_PTS, img_pts, cam, dist,
                                flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return 0.0, 0.0, 0.0
    rmat, _ = cv2.Rodrigues(rvec)
    sy = np.sqrt(rmat[0,0]**2 + rmat[1,0]**2)
    return (np.degrees(np.arctan2(rmat[1,0], rmat[0,0])),   # yaw
            np.degrees(np.arctan2(-rmat[2,0], sy)),           # pitch
            np.degrees(np.arctan2(rmat[2,1], rmat[2,2])))    # roll


# ── State ─────────────────────────────────────────────────────────────────────
class S:
    ear_buf       = deque(maxlen=30)
    perclos       = 0.0
    ear_avg       = 0.30
    mar           = 0.0
    yaw = pitch = roll = 0.0
    drowsy_prob   = 0.0        # classifier output
    phone         = False
    phone_score   = 0.0
    phone_start   = None
    phone_dur     = 0.0
    posture_score = 0.0
    visual_risk   = 0.0
    total_risk    = 0.0
    level         = "SAFE"
    calibrating   = True
    calib_n       = 0
    CALIB         = 30
    calib_ears    = []
    thr           = 0.22       # dynamic EAR threshold, updated after calib
    last_alert    = 0.0
    fps           = 30.0


def compute_risk():
    drowsy_sub = min(100, S.drowsy_prob * 100 * 0.6 + S.perclos * 100 * 0.4)
    S.visual_risk = min(100, drowsy_sub * 0.55 + S.phone_score * 0.35
                        + S.posture_score * 0.10)
    S.total_risk  = S.visual_risk
    if S.total_risk >= 75:
        S.level = "CRITICAL"
    elif S.total_risk >= 50:
        S.level = "MODERATE"
    else:
        S.level = "SAFE"


# ── Audio ─────────────────────────────────────────────────────────────────────
try:
    import pygame
    pygame.mixer.init(frequency=22050, size=-16, channels=1, buffer=512)
    HAS_AUDIO = True
except Exception:
    HAS_AUDIO = False


def beep(critical=False):
    if not HAS_AUDIO:
        return
    def _b():
        freq = 1400 if critical else 900
        reps = 3 if critical else 1
        for _ in range(reps):
            t   = np.linspace(0, 0.15, int(22050*0.15), False)
            buf = (np.sin(2*np.pi*freq*t) * 32767).astype(np.int16)
            pygame.sndarray.make_sound(buf).play()
            time.sleep(0.2)
    threading.Thread(target=_b, daemon=True).start()


# ── HUD ───────────────────────────────────────────────────────────────────────
FONT   = cv2.FONT_HERSHEY_SIMPLEX
COLORS = {"SAFE": (50,205,50), "MODERATE": (0,165,255), "CRITICAL": (0,0,220)}


def draw_hud(frame):
    h, w = frame.shape[:2]
    ov = frame.copy()
    cv2.rectangle(ov, (0,0), (268, h), (15,15,15), -1)
    cv2.addWeighted(ov, 0.62, frame, 0.38, 0, frame)

    rc = COLORS[S.level]

    cv2.putText(frame, "DRIVER RISK MONITOR", (6, 22), FONT, 0.48, (180,220,255), 1, cv2.LINE_AA)

    # Risk bar
    by = 32; bh = 26
    cv2.rectangle(frame, (6, by), (262, by+bh), (55,55,55), -1)
    bw = int(256 * S.total_risk / 100)
    cv2.rectangle(frame, (6, by), (6+bw, by+bh), rc, -1)
    cv2.rectangle(frame, (6, by), (262, by+bh), (180,180,180), 1)
    cv2.putText(frame, f"RISK {S.total_risk:.0f}%  [{S.level}]",
                (10, by+18), FONT, 0.48, (255,255,255), 1, cv2.LINE_AA)

    if S.calibrating:
        prog = S.calib_n / S.CALIB
        cv2.rectangle(frame, (0, by+bh+2), (int(268*prog), by+bh+12), (0,255,200), -1)
        cv2.putText(frame, f"Calibrating {S.calib_n}/{S.CALIB}",
                    (6, by+bh+28), FONT, 0.42, (0,255,200), 1)
        return

    y = 80
    def row(lbl, val, col=(200,220,255)):
        nonlocal y
        cv2.putText(frame, lbl, (8,y),   FONT, 0.40, (150,150,150), 1, cv2.LINE_AA)
        cv2.putText(frame, val, (140,y), FONT, 0.40, col, 1, cv2.LINE_AA)
        y += 19

    warn = (0,80,220)
    row("EAR avg",  f"{S.ear_avg:.3f}",
        warn if S.ear_avg < S.thr else (200,220,255))
    row("PERCLOS",  f"{S.perclos*100:.1f}%",
        warn if S.perclos > 0.15 else (200,220,255))
    row("MAR",      f"{S.mar:.3f}",
        warn if S.mar > 0.55 else (200,220,255))
    row("Yaw",      f"{S.yaw:.1f}°",
        warn if abs(S.yaw) > 20 else (200,220,255))
    row("Pitch",    f"{S.pitch:.1f}°",
        warn if S.pitch < -10 else (200,220,255))
    row("Drowsy P", f"{S.drowsy_prob*100:.0f}%",
        warn if S.drowsy_prob > 0.5 else (80,200,80))
    y += 4
    cv2.line(frame, (6,y), (260,y), (70,70,70), 1); y += 14
    row("Phone",    "DETECTED" if S.phone else "clear",
        warn if S.phone else (80,200,80))
    if S.phone:
        row("  duration", f"{S.phone_dur:.1f}s", warn)
    y += 4
    cv2.line(frame, (6,y), (260,y), (70,70,70), 1); y += 14
    row("Visual Risk", f"{S.visual_risk:.0f}/100", rc)
    row("TOTAL RISK",  f"{S.total_risk:.0f}/100",  rc)
    cv2.putText(frame, f"FPS {S.fps:.0f}", (w-75, 18), FONT, 0.42, (120,120,120), 1)

    # Alert banners
    alerts = []
    if S.level == "CRITICAL":
        alerts.append(("! CRITICAL RISK — PULL OVER !", (0,0,200)))
    elif S.level == "MODERATE":
        alerts.append(("MODERATE RISK — STAY ALERT", (0,120,255)))
    if S.phone and S.phone_dur > 2:
        alerts.append((f"PHONE {S.phone_dur:.0f}s — DANGEROUS", (0,0,180)))
    if S.drowsy_prob > 0.7:
        alerts.append(("DROWSINESS DETECTED", (0,50,200)))

    bx = 276; ay = h - len(alerts)*34 - 8
    for txt, col in alerts:
        bg = frame.copy()
        cv2.rectangle(bg, (bx, ay), (w-6, ay+28), col, -1)
        cv2.addWeighted(bg, 0.72, frame, 0.28, 0, frame)
        cv2.putText(frame, txt, (bx+6, ay+19), FONT, 0.50, (255,255,255), 1, cv2.LINE_AA)
        ay += 34


# ── Main loop ─────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
if not cap.isOpened():
    print("Cannot open camera."); sys.exit(1)

print("\nLive tracking started. Q=quit  R=recalibrate  S=screenshot")
fc = 0; t0 = time.time()

while True:
    ret, frame = cap.read()
    if not ret: break
    fc += 1
    if fc % 15 == 0:
        S.fps = 15 / (time.time() - t0); t0 = time.time()

    H, W = frame.shape[:2]
    rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # ── Face mesh ─────────────────────────────────────────────────────────────
    res = face_mesh.process(rgb)
    if res.multi_face_landmarks:
        lm = res.multi_face_landmarks[0].landmark
        el = _ear(lm, L_EYE, W, H)
        er = _ear(lm, R_EYE, W, H)
        ea = (el+er)/2
        S.ear_avg = ea
        S.mar = _mar(lm, W, H)
        S.yaw, S.pitch, S.roll = _head_pose(lm, W, H)

        # Calibration
        if S.calibrating:
            S.calib_ears.append(ea)
            S.calib_n += 1
            if S.calib_n >= S.CALIB:
                S.thr = np.percentile(S.calib_ears, 80) * 0.75
                S.calibrating = False
                print(f"[Calibrated] EAR threshold = {S.thr:.3f}")

        S.ear_buf.append(ea)
        if len(S.ear_buf) >= 5:
            S.perclos = sum(1 for e in S.ear_buf if e < S.thr) / len(S.ear_buf)

        # Classifier inference
        if not S.calibrating:
            feat = np.array([[el, er, ea, S.mar, S.yaw, S.pitch, S.roll, S.perclos]])
            feat_s = scaler.transform(feat)
            try:
                S.drowsy_prob = clf.predict_proba(feat_s)[0][1]
            except Exception:
                S.drowsy_prob = float(clf.predict(feat_s)[0])

        # Draw eye landmarks
        for idx in L_EYE + R_EYE:
            cx, cy = int(lm[idx].x*W), int(lm[idx].y*H)
            cv2.circle(frame, (cx, cy), 2, (0,255,100), -1)
    else:
        S.ear_buf.append(0.30)

    # ── YOLOv8 phone ──────────────────────────────────────────────────────────
    if USE_YOLO and fc % 2 == 0:
        yres = yolo(frame, verbose=False, classes=[PHONE_CLASS], conf=0.45)
        S.phone = False
        for r in yres:
            if r.boxes:
                for box in r.boxes:
                    if int(box.cls[0]) == PHONE_CLASS:
                        S.phone = True
                        x1,y1,x2,y2 = map(int, box.xyxy[0])
                        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,220),2)
                        cv2.putText(frame,f"PHONE {float(box.conf[0])*100:.0f}%",
                                    (x1,y1-8),FONT,0.52,(0,0,220),2)
        if S.phone:
            if S.phone_start is None: S.phone_start = time.time()
            S.phone_dur   = time.time() - S.phone_start
            S.phone_score = min(100, 60 + S.phone_dur*5)
        else:
            S.phone_start = None; S.phone_dur = 0
            S.phone_score = max(0, S.phone_score-3)

    # ── Posture (every 3rd frame) ─────────────────────────────────────────────
    if fc % 3 == 0:
        pres = pose_est.process(rgb)
        if pres.pose_landmarks:
            lp = pres.pose_landmarks.landmark
            lean   = abs(lp[11].y - lp[12].y)
            slouch = max(0, (lp[11].y + lp[12].y)/2 - lp[0].y - 0.25)
            S.posture_score = min(100, slouch*300 + lean*180)
        else:
            S.posture_score = max(0, S.posture_score-2)

    compute_risk()

    # ── Alert ─────────────────────────────────────────────────────────────────
    now = time.time()
    if S.level == "CRITICAL" and now - S.last_alert > 5:
        beep(critical=True); S.last_alert = now
    elif S.level == "MODERATE" and now - S.last_alert > 15:
        beep(critical=False); S.last_alert = now

    draw_hud(frame)
    cv2.imshow("Driver Risk Monitor", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): break
    elif key == ord('r'):
        S.calibrating = True; S.calib_n = 0; S.calib_ears = []
        print("[Recalibrating...]")
    elif key == ord('s'):
        fn = f"screenshot_{int(time.time())}.png"
        cv2.imwrite(fn, frame); print(f"Saved {fn}")

cap.release()
cv2.destroyAllWindows()
face_mesh.close(); pose_est.close()
print("Done.")