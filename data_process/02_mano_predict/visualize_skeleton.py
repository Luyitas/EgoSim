"""
3D skeleton visualization using PyRender (sphere + cylinder rasterization).

Reads the same HaMeR JSON annotation format as visualize_skeleton.py but renders
real 3D geometry (icosphere joints + cylinder bones) via offscreen PyRender.

Outputs per clip:
  {output_dir}/{uuid}/{clip}_overlay.mp4   skeleton composited on video
  {output_dir}/{uuid}/{clip}_black.mp4     skeleton on black background
"""

import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'
os.environ['PYOPENGL_ERROR_ON_COPY'] = '0'

import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm
import trimesh
import pyrender


# ─── Skeleton topology (21 MANO joints) ─────────────────────────────────────
FINGER_CONNECTIONS = {
    "thumb":  [0,  1,  2,  3,  4],
    "index":  [0,  5,  6,  7,  8],
    "middle": [0,  9, 10, 11, 12],
    "ring":   [0, 13, 14, 15, 16],
    "little": [0, 17, 18, 19, 20],
}

FINGER_COLORS = {
    "thumb":  np.array([238, 130, 238]) / 255.0,
    "index":  np.array([255,  99,  71]) / 255.0,
    "middle": np.array([230, 245, 250]) / 255.0,
    "ring":   np.array([173, 255,  47]) / 255.0,
    "little": np.array([  0, 152, 191]) / 255.0,
}

# HaMeR default camera parameters
HAMER_FOCAL = 5000.0
HAMER_IMG_SIZE = 256.0

# Rendering geometry sizes (meters in camera space)
JOINT_RADIUS = 0.005
BONE_RADIUS = 0.003


# ─── Mesh creation helpers ───────────────────────────────────────────────────

def create_sphere(position: np.ndarray, radius: float, color: np.ndarray) -> trimesh.Trimesh:
    sphere = trimesh.creation.icosphere(subdivisions=2, radius=radius)
    sphere.apply_translation(position)
    sphere.visual.vertex_colors = np.tile(
        (color * 255).astype(np.uint8), (len(sphere.vertices), 1)
    )
    return sphere


def create_cylinder(p1: np.ndarray, p2: np.ndarray, radius: float, color: np.ndarray) -> Optional[trimesh.Trimesh]:
    direction = p2 - p1
    height = np.linalg.norm(direction)
    if height < 1e-6:
        return None

    cylinder = trimesh.creation.cylinder(radius=radius, height=height, sections=16)

    direction_normalized = direction / height
    z_axis = np.array([0, 0, 1])
    v = np.cross(z_axis, direction_normalized)
    c = np.dot(z_axis, direction_normalized)

    if np.linalg.norm(v) < 1e-6:
        rotation_matrix = np.eye(3) if c > 0 else np.diag([1, -1, -1])
    else:
        s = np.linalg.norm(v)
        kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        rotation_matrix = np.eye(3) + kmat + kmat @ kmat * ((1 - c) / (s ** 2))

    transform = np.eye(4)
    transform[:3, :3] = rotation_matrix
    cylinder.apply_transform(transform)
    cylinder.apply_translation((p1 + p2) / 2)
    cylinder.visual.vertex_colors = np.tile(
        (color * 255).astype(np.uint8), (len(cylinder.vertices), 1)
    )
    return cylinder


# ─── Skeleton mesh from MANO 21 keypoints ───────────────────────────────────

def build_hand_mesh(kp3d_cam: np.ndarray) -> Optional[trimesh.Trimesh]:
    """Build combined sphere+cylinder mesh for one hand (21 joints in camera space)."""
    meshes: List[trimesh.Trimesh] = []

    # Wrist (joint 0) rendered once with middle-finger color, matching batch_render_skeleton_3d.py
    wrist_color = FINGER_COLORS["middle"]
    meshes.append(create_sphere(kp3d_cam[0], JOINT_RADIUS, wrist_color))

    for finger, indices in FINGER_CONNECTIONS.items():
        color = FINGER_COLORS[finger]
        # indices[0] is always wrist (0); start cylinders from wrist but skip its sphere
        for i in range(len(indices) - 1):
            idx_a, idx_b = indices[i], indices[i + 1]
            pa, pb = kp3d_cam[idx_a], kp3d_cam[idx_b]
            cyl = create_cylinder(pa, pb, BONE_RADIUS, color)
            if cyl is not None:
                meshes.append(cyl)
            # add sphere for the distal joint only (not the wrist end)
            meshes.append(create_sphere(pb, JOINT_RADIUS, color))

    if not meshes:
        return None
    return trimesh.util.concatenate(meshes)


def kp3d_to_camera_space(kp3d: np.ndarray, cam_t: np.ndarray, is_right: int) -> np.ndarray:
    """Convert HaMeR keypoints_3d + cam_t to camera-space coordinates."""
    kp = kp3d.copy()
    multiplier = 2 * is_right - 1
    kp[:, 0] *= multiplier
    return kp + cam_t.reshape(1, 3)


# ─── Per-frame rendering ─────────────────────────────────────────────────────

def render_skeleton_frame(
    renderer: pyrender.OffscreenRenderer,
    hand_meshes: List[trimesh.Trimesh],
    camera: pyrender.IntrinsicsCamera,
    width: int,
    height: int,
) -> np.ndarray:
    """Render hand meshes and return RGB frame (black background)."""
    scene = pyrender.Scene(ambient_light=[1.0, 1.0, 1.0], bg_color=[0, 0, 0, 0])

    for mesh in hand_meshes:
        pyrender_mesh = pyrender.Mesh.from_trimesh(mesh)
        scene.add(pyrender_mesh)

    camera_pose = np.eye(4)
    camera_pose[1, 1] = -1
    camera_pose[2, 2] = -1
    scene.add(camera, pose=camera_pose)

    color, _ = renderer.render(scene)
    return color


# ─── Process one clip ────────────────────────────────────────────────────────

def process_clip(video_path: Path, annot_path: Path, output_dir: Path, no_resume: bool = False):
    with open(annot_path, "r") as f:
        data = json.load(f)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[WARN] Cannot open video: {video_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 16.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = video_path.stem
    overlay_path = output_dir / f"{stem}_overlay.mp4"
    black_path = output_dir / f"{stem}_black.mp4"

    if not no_resume and overlay_path.exists() and black_path.exists():
        return

    scaled_fl = HAMER_FOCAL / HAMER_IMG_SIZE * max(width, height)
    cx, cy = width / 2.0, height / 2.0

    camera = pyrender.IntrinsicsCamera(
        fx=scaled_fl, fy=scaled_fl, cx=cx, cy=cy,
        znear=0.01, zfar=100.0
    )
    renderer = pyrender.OffscreenRenderer(width, height)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    w_overlay = cv2.VideoWriter(str(overlay_path), fourcc, fps, (width, height))
    w_black = cv2.VideoWriter(str(black_path), fourcc, fps, (width, height))

    frame_map = {f["frame_idx"]: f for f in data.get("frames", [])}

    for fidx in range(total):
        ret, frame = cap.read()
        if not ret:
            break

        frame_annot = frame_map.get(fidx, {})
        hand_meshes: List[trimesh.Trimesh] = []

        for hand in frame_annot.get("hands", []):
            kp3d = hand.get("keypoints_3d")
            if kp3d is None:
                continue
            kp3d = np.array(kp3d, dtype=np.float64)
            if kp3d.shape[0] < 21:
                continue

            cam_t_key = "cam_t_full" if "cam_t_full" in hand else "cam_t"
            cam_t = np.array(hand[cam_t_key], dtype=np.float64)
            is_right = int(hand["is_right"])

            kp3d_cam = kp3d_to_camera_space(kp3d, cam_t, is_right)
            mesh = build_hand_mesh(kp3d_cam)
            if mesh is not None:
                hand_meshes.append(mesh)

        if hand_meshes:
            rendered = render_skeleton_frame(renderer, hand_meshes, camera, width, height)
            black_bgr = cv2.cvtColor(rendered, cv2.COLOR_RGB2BGR)

            mask = (rendered.sum(axis=2) > 0).astype(np.uint8)[:, :, np.newaxis]
            overlay = frame.copy()
            rendered_bgr = cv2.cvtColor(rendered, cv2.COLOR_RGB2BGR)
            overlay = overlay * (1 - mask) + rendered_bgr * mask
            overlay = overlay.astype(np.uint8)
        else:
            black_bgr = np.zeros((height, width, 3), dtype=np.uint8)
            overlay = frame.copy()

        w_overlay.write(overlay)
        w_black.write(black_bgr)

    cap.release()
    w_overlay.release()
    w_black.release()
    renderer.delete()


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="3D skeleton visualization (PyRender) for HaMeR MANO annotations"
    )
    parser.add_argument("--video_dir", type=str, required=True,
                        help="Directory tree with per-clip mp4 files")
    parser.add_argument("--annot_dir", type=str, required=True,
                        help="Directory tree with per-clip full MANO json")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for overlay / black-bg videos")
    parser.add_argument("--no_resume", action="store_true",
                        help="Re-process all clips even if output already exists")
    args = parser.parse_args()

    video_dir = Path(args.video_dir)
    annot_dir = Path(args.annot_dir)
    output_dir = Path(args.output_dir)

    annot_files = sorted(annot_dir.rglob("*.json"))
    annot_files = [a for a in annot_files if a.name != "source.json"]

    pairs = []
    for ap in annot_files:
        rel = ap.relative_to(annot_dir)
        vp = video_dir / rel.with_suffix(".mp4")
        if not vp.exists():
            print(f"[SKIP] Video not found: {vp}")
            continue
        out_sub = output_dir / rel.parent
        pairs.append((vp, ap, out_sub))

    print(f"Total annotation files: {len(annot_files)}")
    print(f"Clips to process: {len(pairs)}")

    for vp, ap, out_sub in tqdm(pairs, desc="3D Rendering"):
        try:
            process_clip(vp, ap, out_sub, no_resume=args.no_resume)
        except Exception as e:
            print(f"[ERROR] {vp.name}: {e}")


if __name__ == "__main__":
    main()
