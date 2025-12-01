import cv2
import torch
import numpy as np
import json
import time
from collections import deque

from ultralytics import YOLO

from model.stgcn import STGCN  # your existing model
# if you have NUM_CLASSES in config.py, you can import it:
# from config import NUM_CLASSES


# -----------------------------
# CONFIG (ADAPT AS NEEDED)
# -----------------------------

# Number of classes your STGCN was trained on
NUM_CLASSES = 8  # change if your model uses a different number

# Temporal window (number of frames in a sequence)
T_WINDOW = 30  # e.g. 30 frames ≈ 1 second at 30 FPS

# Number of joints used by YOLO-Pose (COCO has 17 keypoints)
V_JOINTS = 17

# ST-GCN standard input shape: (N, C, T, V, M)
# N = batch, C = channels, T = frames, V = joints, M = persons (we use 1)
M_PERSONS = 1

# Path to model weights
MODEL_PATH = "stgcn.pth"

# Optional: path to label mapping (id -> name)
LABELS_PATH = "dataset/processed/labels.json"

# Number of joints used in the *training dataset* (from STGCN -> 256 * 24)
DATASET_JOINTS = 24



def load_label_map(path):
    """
    Try to load label mapping: index -> class name.
    Supports either:
      - list: ["person1", "person2", ...]
      - dict: {"0": "person1", "1": "person2", ...}
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return {i: name for i, name in enumerate(data)}
        elif isinstance(data, dict):
            # keys may be strings "0", "1", ...
            return {int(k): v for k, v in data.items()}
    except Exception as e:
        print(f"Could not load labels from {path}: {e}")
    # fallback: numeric labels
    return {i: f"Class {i}" for i in range(NUM_CLASSES)}


def prepare_sequence_buffer():
    """
    Create a deque buffer to hold last T_WINDOW pose frames.
    Each element will be shape (V_JOINTS, 2) for (x, y).
    """
    return deque(maxlen=T_WINDOW)


def pose_to_array(result, frame_width, frame_height):
    """
    Extract skeleton keypoints from YOLO-Pose result for the first person.
    Returns normalized coordinates in [0,1] as (V_JOINTS, 2)
    or None if no person detected.
    """
    if len(result.keypoints) == 0:
        return None

    # Take first detected person
    kpts = result.keypoints[0].xy  # shape: (V, 2)
    kpts = kpts.cpu().numpy()

    # Optionally, limit to first V_JOINTS (in case model uses more)
    kpts = kpts[:V_JOINTS, :]

    # Normalize by image size to keep scale consistent
    kpts[:, 0] /= frame_width   # x / W
    kpts[:, 1] /= frame_height  # y / H

    return kpts  # shape: (V_JOINTS, 2)




def build_stgcn_input(sequence):
    """
    Convert YOLO-Pose sequence → ST-GCN input (1, T, V, C).
    If the sequence does not match the dataset spec (e.g., wrong number of joints),
    return None so the caller can treat it as "Not match".
    """

    # Filter None frames
    clean_seq = [f for f in sequence if f is not None]

    if len(clean_seq) < T_WINDOW:
        print(f"[WARN] Not enough valid frames: {len(clean_seq)}/{T_WINDOW}")
        return None

    try:
        # Usually (T, 1, V, 2) or (T, V, 2)
        seq = np.stack(clean_seq, axis=0)
    except Exception as e:
        print("[ERROR] Could not stack frames:", e)
        return None

    print("DEBUG seq raw shape:", seq.shape)

    # If there is a person dimension (1), squeeze it: (T, 1, V, 2) → (T, V, 2)
    if seq.ndim == 4 and seq.shape[1] == 1:
        seq = seq[:, 0, :, :]  # (T, V, 2)

    if seq.ndim != 3:
        print("[ERROR] Unexpected seq ndim after squeeze:", seq.shape)
        return None

    T, V, C = seq.shape  # e.g. (30, 17, 2)

    # ❗ Check joint count against dataset
    if V != DATASET_JOINTS:
        print(f"[WARN] Joint count mismatch: got V={V}, expected {DATASET_JOINTS}")
        return None  # caller will treat this as "Not match"

    # Ensure C = 3 (x, y, confidence) as STGCN uses in_channels = 3
    if C == 2:
        # Add dummy confidence = 1
        conf = np.ones((T, V, 1), dtype=seq.dtype)
        seq = np.concatenate([seq, conf], axis=-1)   # (T, V, 3)
        C = 3
    elif C != 3:
        print("[ERROR] Unexpected channel count C:", C)
        return None

    print("DEBUG seq final (T, V, C):", seq.shape)  # (T, 24, 3) if compatible

    # Add batch dimension → (1, T, V, C)
    seq = np.expand_dims(seq, axis=0)

    return torch.from_numpy(seq).float()







def main():
    # -----------------------------
    # Device & Model Setup
    # -----------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Load label map
    label_map = load_label_map(LABELS_PATH)
    print("Label map:", label_map)

    # Load STGCN model
    model = STGCN(NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print(f"Loaded STGCN weights from {MODEL_PATH}")

    # Load YOLO Pose model (use small model for speed)
    pose_model = YOLO("yolov8n-pose.pt")
    print("Loaded YOLO-Pose model (yolov8n-pose.pt)")

    # -----------------------------
    # Webcam Setup
    # -----------------------------
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open camera")
        return

    seq_buffer = prepare_sequence_buffer()
    last_pred_name = "N/A"
    last_pred_time = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            h, w = frame.shape[:2]

            # Run YOLO-Pose (single frame, no verbose)
            results = pose_model(frame, verbose=False)

            if len(results) > 0:
                # Draw YOLO-Pose skeletons for visualization
                drawn = results[0].plot()  # numpy image with drawings
                frame = drawn

                pose_arr = pose_to_array(results[0], w, h)
                if pose_arr is not None:
                    seq_buffer.append(pose_arr)

            # If we have enough frames, run STGCN prediction
            if len(seq_buffer) == T_WINDOW:
                with torch.no_grad():
                    x = build_stgcn_input(list(seq_buffer))

                    # If skeleton does not match dataset → treat as "Not match"
                    if x is None:
                        last_pred_name = "Not match"
                        last_pred_time = time.time()
                        print("[INFO] Sequence incompatible with dataset. Marked as Not match.")
                        # Optionally clear buffer so it tries again
                        seq_buffer.clear()
                        continue

                    x = x.to(device)

                    try:
                        logits = model(x)  # shape: (1, NUM_CLASSES)
                        pred_id = logits.argmax(dim=1).item()
                        last_pred_name = label_map.get(pred_id, f"Class {pred_id}")
                    except RuntimeError as e:
                        # Catch any unexpected shape or FC errors
                        print("[ERROR] Model forward failed:", e)
                        last_pred_name = "Not match"

                    last_pred_time = time.time()
                    # You may also clear the buffer here to start fresh
                    seq_buffer.clear()


            # -----------------------------
            # Overlay prediction on frame
            # -----------------------------
            now = time.time()
            if now - last_pred_time < 2.0:  # show for 2 seconds after last prediction
                text = f"Prediction: {last_pred_name}"
            else:
                text = "Collecting gait sequence..."

            cv2.putText(
                frame, text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA
            )

            cv2.imshow("Real-time Gait Recognition (YOLO-Pose + STGCN)", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
