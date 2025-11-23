#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot gait storyboard directly with default parameters.
Run simply:  python plot_gait_storyboard.py
"""

import zipfile, json, os, numpy as np, matplotlib.pyplot as plt

# ---------- Default Configuration ----------
DEFAULT_ZIP = r"C:\Work\GaitCognition\000_00-20251026T164710Z-1-001.zip"
DEFAULT_OUT = r"C:\Work\GaitCognition\storyboard_cob.png"
DEFAULT_SKEL = "0-1,0-2,1-3,5-7,7-9,6-8,8-10,5-6,11-12,5-11,6-12,11-13,12-14,13-15,14-16"
DEFAULT_COLS = 20
HIP_CENTER, SPINE, HIP_RIGHT, HIP_LEFT = 0, 1, 2, 3
# -------------------------------------------------------

def parse_skel_edges(s):
    edges = []
    for part in s.split(","):
        a, b = part.split("-")
        edges.append((int(a), int(b)))
    return edges

def safe_norm(v):
    n = np.linalg.norm(v)
    return v / (n + 1e-12)

def gram_schmidt_basis(hip_center, spine, hip_right, hip_left):
    vs = spine - hip_center
    vh = hip_left - hip_right
    vh_orth = vh - (np.dot(vs, vh) / (np.dot(vs, vs) + 1e-12)) * vs
    vd = np.cross(vh_orth, vs)
    uh = safe_norm(vh_orth)
    us = safe_norm(vs)
    ud = safe_norm(vd)
    return np.vstack([uh, us, ud])

def to_cob_coords(joints_xyz):
    hc, sp, hr, hl = joints_xyz[HIP_CENTER], joints_xyz[SPINE], joints_xyz[HIP_RIGHT], joints_xyz[HIP_LEFT]
    B = gram_schmidt_basis(hc, sp, hr, hl)
    P = joints_xyz - hc
    return (B @ P.T).T

def read_frame_from_text(txt):
    txt = txt.strip()
    try:
        obj = json.loads(txt)
        if "hc3d" in obj:
            arr = np.array(obj["hc3d"], dtype=float)
            return arr[:, :3]
    except Exception:
        pass
    return None

def load_frames_from_zip(zip_path):
    frames = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        for name in sorted(zf.namelist()):
            if not name.lower().endswith(".json"):
                continue
            data = zf.read(name).decode("utf-8", errors="ignore")
            arr = read_frame_from_text(data)
            if arr is not None:
                frames.append(arr)
    return frames

def draw_skeleton(ax, pts, edges, title):
    ax.scatter(pts[:, 0], pts[:, 1], s=20)
    for a, b in edges:
        if a < len(pts) and b < len(pts):
            ax.plot([pts[a, 0], pts[b, 0]], [pts[a, 1], pts[b, 1]], linewidth=2)
    ax.set_title(title, fontsize=10)
    ax.axis("off")
    ax.set_aspect("equal")

def plot_storyboard(frames, edges, out_path, cols=4):
    n = min(cols, len(frames))
    idxs = np.linspace(0, len(frames) - 1, n, dtype=int)
    sel = [frames[i] for i in idxs]
    sel_cob = [to_cob_coords(f) for f in sel]

    fig, axes = plt.subplots(2, n, figsize=(3.5 * n, 6))
    if n == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    for i, f in enumerate(sel):
        draw_skeleton(axes[0, i], f[:, :2], edges, f"Frame {i}")
    for i, f in enumerate(sel_cob):
        draw_skeleton(axes[1, i], f[:, :2], edges, f"CoB {i}")

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Saved: {out_path}")

# ---------- Main ----------
if __name__ == "__main__":
    zip_path = DEFAULT_ZIP
    out_path = DEFAULT_OUT
    edges = parse_skel_edges(DEFAULT_SKEL)
    frames = load_frames_from_zip(zip_path)

    if not frames:
        print(f"❌ No frames found in ZIP: {zip_path}")
    else:
        print(f"📦 Loaded {len(frames)} frames from {os.path.basename(zip_path)}")
        plot_storyboard(frames, edges, out_path, cols=DEFAULT_COLS)
