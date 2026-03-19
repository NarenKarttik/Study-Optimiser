"""
STEP 1 — Feature Extraction from NTHU-DDD videos.

Reads every video + its .txt label file, extracts per-frame:
  EAR_left, EAR_right, EAR_avg, MAR, yaw, pitch, roll, perclos_30

Saves: features_train.csv  features_test.csv

Usage:
    python step1_extract_features.py --root /path/to/NTHU-DDD

NTHU-DDD expected layout:
    NTHU-DDD/
    ├── Training_Evaluation_Dataset/
    │   ├── BareFace/
    │   │   ├── 001.avi  (or .mp4)
    │   │   ├── 001.txt  ← frame labels: one integer per line (0=alert,1=drowsy)
    │   │   └── ...
    │   ├── Glasses/ NightBareFace/ NightGlasses/ Sunglasses/
    └── Testing_Dataset/
        └── (same)
"""

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
from collections import deque

# ── MediaPipe setup ────────────────────────────────────────────────────────
mp_fm = mp.solutions.face_mesh

# EAR landmark indices (MediaPipe 478-pt)
L_EYE = [362, 385, 387, 263, 373, 380]
R_EYE = [33,  160, 158, 133, 153, 144]
MOUTH = [61, 291, 17, 0]           # left, right, bottom, top

# 3-D model points for PnP head pose
MODEL_PTS = np.array([
    [0.0,    0.0,    0.0],    # Nose tip  (1)
    [0.0,   -330.0, -65.0],   # Chin       (152)
    [-225.0,  170.0,-135.0],  # L eye corner (263)
    [225.0,   170.0,-135.0],  # R eye corner (33)
    [-150.0, -150.0,-125.0],  # L mouth (287)
    [150.0,  -150.0,-125.0],  # R mouth (57)
], dtype=np.float64)

LM_IDX = [1, 152, 263, 33, 287, 57]   # face mesh indices for MODEL_PTS

SCENARIOS = ["BareFace", "Glasses", "Sunglasses",
             "NightBareFace", "NightGlasses",
             "Night_BareFace", "Night_Glasses"]   # handle both naming styles

SAMPLE = 3   # keep every Nth frame (reduces file size while preserving temporal info)


def ear(lm, idx, W, H):
    p = [(lm[i].x * W, lm[i].y * H) for i in idx]
    A = np.linalg.norm(np.array(p[1]) - np.array(p[5]))
    B = np.linalg.norm(np.array(p[2]) - np.array(p[4]))
    C = np.linalg.norm(np.array(p[0]) - np.array(p[3]))
    return (A + B) / (2.0 * C + 1e-7)


def mar(lm, W, H):
    pts = [(lm[i].x * W, lm[i].y * H) for i in MOUTH]
    vert  = np.linalg.norm(np.array(pts[3]) - np.array(pts[2]))
    horiz = np.linalg.norm(np.array(pts[0]) - np.array(pts[1]))
    return vert / (horiz + 1e-7)


def head_pose(lm, W, H):
    img_pts = np.array(
        [(lm[i].x * W, lm[i].y * H) for i in LM_IDX], dtype=np.float64)
    focal = W
    cam = np.array([[focal, 0, W/2],
                    [0, focal, H/2],
                    [0,     0,   1]], dtype=np.float64)
    dist = np.zeros((4, 1))
    ok, rvec, _ = cv2.solvePnP(MODEL_PTS, img_pts, cam, dist,
                                flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return 0.0, 0.0, 0.0
    rmat, _ = cv2.Rodrigues(rvec)
    sy = np.sqrt(rmat[0,0]**2 + rmat[1,0]**2)
    pitch = np.degrees(np.arctan2(-rmat[2,0], sy))
    yaw   = np.degrees(np.arctan2(rmat[1,0], rmat[0,0]))
    roll  = np.degrees(np.arctan2(rmat[2,1], rmat[2,2]))
    return yaw, pitch, roll


def read_labels(txt_path):
    labels = []
    try:
        with open(txt_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    labels.append(int(line.split(',')[0].split()[0]))
                except Exception:
                    labels.append(0)
    except FileNotFoundError:
        pass
    return labels

def process_video(video_path, face_mesh):
    """
    Modified to handle both images (.jpg) and videos.
    """
    # 1. Determine if it's an image or video
    is_image = video_path.suffix.lower() in ['.jpg', '.jpeg', '.png']
    
    # 2. Smart Labeling: If no .txt file exists, use the filename
    label_path = video_path.with_suffix(".txt")
    labels = read_labels(label_path)
    if not labels:
        # If filename contains 'notdrowsy', label is 0 (Alert). Otherwise 1 (Drowsy).
        label = 0 if "notdrowsy" in video_path.name.lower() else 1
    else:
        label = labels[0]

    rows = []
    
    # 3. Reading Logic
    if is_image:
        frame = cv2.imread(str(video_path))
        if frame is None:
            return []
        # We wrap it in a list so the loop below still works
        frames_to_process = [(frame, 0)]
    else:
        # Standard video logic (kept for compatibility)
        cap = cv2.VideoCapture(str(video_path))
        frames_to_process = []
        while True:
            ret, f = cap.read()
            if not ret: break
            frames_to_process.append((f, len(frames_to_process)))
        cap.release()

    # 4. Feature Extraction (The EAR/MAR Math)
    for frame, frame_idx in frames_to_process:
        H, W = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = face_mesh.process(rgb)

        if not res.multi_face_landmarks:
            # If no face is found, we can't extract features
            continue

        lm = res.multi_face_landmarks[0].landmark

        # Calculate the 3 Pillars of Drowsiness Detection
        el = ear(lm, L_EYE, W, H)
        er = ear(lm, R_EYE, W, H)
        ea = (el + er) / 2.0
        ma = mar(lm, W, H)
        yaw, pitch, roll = head_pose(lm, W, H)

        # Note: perclos is 0 for static images because it needs a time-window
        rows.append({
            "ear_l": el, "ear_r": er, "ear_avg": ea,
            "mar": ma, "yaw": yaw, "pitch": pitch, "roll": roll,
            "perclos": 0.0, 
            "label": label,
            "video": video_path.stem, 
            "frame": frame_idx
        })

    return rows

def process_split(root, split_folder, out_csv):
    split_path = root / split_folder
    if not split_path.exists():
        print(f"  [SKIP] {split_folder} not found at {split_path}")
        return

    face_mesh = mp_fm.FaceMesh(
        max_num_faces=1, refine_landmarks=False,
        min_detection_confidence=0.5, min_tracking_confidence=0.4)

    # Use rglob to find all images inside '001', '002', etc.
    images = list(split_path.rglob("*.jpg")) + list(split_path.rglob("*.png"))
    print(f"  Found {len(images)} images in {split_folder}")

    all_rows = []
    for ip in images:
        rows = process_video(ip, face_mesh) # Our modified function handles images
        for r in rows:
            r["scenario"] = ip.parent.name # Records '001', '002', etc.
        all_rows.extend(rows)
        if len(all_rows) % 100 == 0:
            print(f"    Processed {len(all_rows)} frames...", end='\r')

    face_mesh.close()

    if not all_rows:
        print("  No data extracted! Check if images are in the right path.")
        return

    df = pd.DataFrame(all_rows)
    df.to_csv(out_csv, index=False)
    print(f"\n  Saved {len(df):,} frames → {out_csv}")
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True,
                        help="Path to NTHU-DDD root folder")
    parser.add_argument("--out", default=".",
                        help="Output directory for CSV files")
    args = parser.parse_args()

    root = Path(args.root)
    out  = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print("=== Extracting TRAINING features ===")
    process_split(root, "Training_Evaluation_Dataset",
                  out / "features_train.csv")

    print("\n=== Extracting TEST features ===")
    process_split(root, "Testing_Dataset",
                  out / "features_test.csv")

    print("\nDone. Run step2_train_and_evaluate.py next.")


if __name__ == "__main__":
    main()