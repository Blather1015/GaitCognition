#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate one image per JSON file using OUMVLP skeleton (24 joints).
"""

import json, os, numpy as np, matplotlib.pyplot as plt

# ---------- Config ----------
ROOT_FOLDER = r"C:\Users\Ment\Downloads\10307"
OUTPUT_FOLDER = r"C:\Users\Ment\Downloads\output"

# --------- OUMVLP Skeleton (24 joints) ----------
# From OUMVLP-Mesh paper, Figure 5 (24-joint model)
OUMVLP_EDGES = [
    (0,1), (0,2),           
    (1,4), (2,5),           
    (4,7), (5,8),           
    (7,10), (8,11),         
    (0,3), (3,6), (6,9),    
    (9,12), (12,15),        
    (12,13), (12,14),       
    (13,16), (14,17),       
    (16,18), (17,19),       
    (18,20), (19,21),       
    (20,22), (21,23)        
]
#yes mae

def read_frame_from_file(path):
    try:
        data = open(path, "r", encoding="utf-8").read()
        obj = json.loads(data)
        if "hc3d" in obj:
            arr = np.array(obj["hc3d"], dtype=float)
            return arr[:, :3]   # X,Y,Z
    except:
        return None


def draw_single_frame(frame, out_path):
    # ใช้แกนที่ถูกต้องของ SMPL/OUMVLP = X,Y
    X = frame[:, 0]
    Y = frame[:, 1]

    fig, ax = plt.subplots(figsize=(5, 7))

    # joints
    ax.scatter(X, Y, s=25)

    # bones
    for a, b in OUMVLP_EDGES:
        if a < len(frame) and b < len(frame):
            ax.plot([X[a], X[b]], [Y[a], Y[b]], linewidth=2)

    ax.set_title("OUMVLP Skeleton (X–Y view)")
    ax.axis("equal")

    # กลับด้าน
    # ax.invert_yaxis()

    ax.axis("off")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"📸 Saved: {out_path}")


# ---------- Main ----------
if __name__ == "__main__":

    for subfolder in sorted(os.listdir(ROOT_FOLDER)):
        sub_path = os.path.join(ROOT_FOLDER, subfolder)
        if not os.path.isdir(sub_path):
            continue

        print(f"\n📂 Folder: {subfolder}")

        out_dir = os.path.join(OUTPUT_FOLDER, subfolder)
        os.makedirs(out_dir, exist_ok=True)

        for filename in sorted(os.listdir(sub_path)):
            if filename.endswith(".json"):
                full_json = os.path.join(sub_path, filename)
                frame = read_frame_from_file(full_json)

                if frame is not None:
                    out_file = os.path.join(out_dir, filename.replace(".json", ".png"))
                    draw_single_frame(frame, out_file)
    
