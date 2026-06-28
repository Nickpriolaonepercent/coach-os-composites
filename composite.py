"""
Coach OS — before/after composite builder (Replicate-deployable).

Single entry point:
    build_composite(before_path, after_path, pose, out_path) -> str

Implements Nick Priola's permanent v4 ruleset:

OUTPUT
- One combined image per pose, two panels (BEFORE | AFTER), exactly equal W x H.
- No padding, no letterboxing. Crop out extra background, never pad.

SUBJECT SCALE
- Same body size in both panels.
- Match by fixed landmark: head_top near the top of the frame,
  bottom crop at the same body landmark (mid-thigh or hip) in both panels.

ZOOM & CROP
- Same zoom level on both sides (matched body_height_in_panel).
- Same crop type: head-to-mid-thigh both sides (falls back to head-to-hip
  if either photo doesn't show the legs).

FRAMING
- Body centered horizontally in each panel.
- ~4:5 portrait per panel (480 x 600 px).
- Thin seam (2 px) between panels.

MIRRORING
- BEFORE is horizontally mirrored if its facing direction differs from AFTER.

ERROR HANDLING
- Raises CompositeError if MediaPipe can't detect usable landmarks on either
  input photo. Caller decides what to show the user.
"""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("GLOG_minloglevel", "3")
import warnings
warnings.filterwarnings("ignore")

from io import BytesIO
from pathlib import Path
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
import mediapipe as mp

mp_pose = mp.solutions.pose

# Panel geometry (4:5 portrait per panel)
PANEL_W = 480
PANEL_H = 600
HEAD_OFFSET = 20   # head_top sits at y=20 in panel (near top, not touching edge)
SEAM_PX = 2        # thin divider between panels
MIN_PHOTO_W = 400  # reject thumbnails


class CompositeError(Exception):
    """Raised when a composite can't be built (typically pose detection failure)."""


def get_landmarks(pil_img):
    """Detect: head_top, mid_thigh_y, body_center_x (for horizontal centering),
    and the facing direction. Returns None if pose can't be read."""
    arr = np.array(pil_img.convert("RGB"))
    h, w = arr.shape[:2]
    with mp_pose.Pose(static_image_mode=True, model_complexity=1) as pose:
        results = pose.process(arr)
    if not results.pose_landmarks:
        return None
    lms = results.pose_landmarks.landmark

    # Head top: topmost y across nose/eyes/ears
    head_idxs = [0, 1, 2, 3, 4, 7, 8]
    head_ys = [lms[i].y * h for i in head_idxs if lms[i].visibility > 0.3]
    head_xs = [lms[i].x * w for i in head_idxs if lms[i].visibility > 0.3]
    if not head_ys:
        return None
    head_y = min(head_ys)
    head_x = sum(head_xs) / len(head_xs)

    # Hips
    hip_idxs = [23, 24]
    hip_ys = [lms[i].y * h for i in hip_idxs if lms[i].visibility > 0.2]
    hip_xs = [lms[i].x * w for i in hip_idxs if lms[i].visibility > 0.2]
    if not hip_ys:
        return None
    hip_y = sum(hip_ys) / len(hip_ys)
    hip_x = sum(hip_xs) / len(hip_xs)

    # Knees (may be cropped in some photos)
    knee_idxs = [25, 26]
    knee_ys = [lms[i].y * h for i in knee_idxs if lms[i].visibility > 0.2]

    knees_seen = bool(knee_ys)
    if knees_seen:
        knee_y = sum(knee_ys) / len(knee_ys)
        mid_thigh_y = (hip_y + knee_y) / 2
    else:
        # Estimate mid-thigh from torso length (head->hip ~ 1.0, hip->mid-thigh ~ 0.35)
        torso = hip_y - head_y
        mid_thigh_y = hip_y + torso * 0.35

    # Body center X: average of shoulders and hips (more stable than just one)
    sh_idxs = [11, 12]
    sh_xs = [lms[i].x * w for i in sh_idxs if lms[i].visibility > 0.2]
    if sh_xs:
        body_cx = (sum(sh_xs) / len(sh_xs) + hip_x) / 2
    else:
        body_cx = hip_x

    # Facing direction (for mirror matching on side shots)
    facing = "right" if head_x >= hip_x else "left"

    return {
        "head_top": (head_x, head_y),
        "mid_thigh_y": mid_thigh_y,
        "body_cx": body_cx,
        "img_size": (w, h),
        "body_h_src": mid_thigh_y - head_y,
        "facing": facing,
        "knees_seen": knees_seen,
        "hip_y": hip_y,
    }


def is_usable_photo(info):
    """Return True if the photo has head_top clearly visible inside the
    frame and hip visible."""
    if not info or info["body_h_src"] <= 0:
        return False
    head_y = info["head_top"][1]
    img_w, img_h = info["img_size"]
    if img_w < MIN_PHOTO_W:
        return False
    if head_y < 0.05 * img_h:           # head touching top edge -> unusable
        return False
    if info["hip_y"] > img_h * 0.97:    # hip below image -> not usable
        return False
    return True


def min_scale_for_full_coverage(info):
    """Smallest scale we can apply to this source such that the panel
    fills completely with NO padding when head_top sits at y=HEAD_OFFSET
    and the body is centered horizontally."""
    head_x, head_y = info["head_top"]
    cx = info["body_cx"]
    src_w, src_h = info["img_size"]

    s_above = HEAD_OFFSET / max(head_y, 1)
    s_left = (PANEL_W / 2) / max(cx, 1)
    s_right = (PANEL_W / 2) / max(src_w - cx, 1)
    s_below = (PANEL_H - HEAD_OFFSET) / max(src_h - head_y, 1)
    return max(s_above, s_left, s_right, s_below)


def match_pair_target_body_h(b_info, a_info):
    """Pick the smallest body_height_in_panel such that BOTH photos can
    fill the panel without padding when scaled to that body height."""
    b_scale_min = min_scale_for_full_coverage(b_info)
    a_scale_min = min_scale_for_full_coverage(a_info)
    b_body_min = b_scale_min * b_info["body_h_src"]
    a_body_min = a_scale_min * a_info["body_h_src"]
    return max(b_body_min, a_body_min)


def pick_bottom_landmark(b_info, a_info):
    """Choose the lowest body landmark that's REAL (not extrapolated) in
    both photos. Falls back from mid-thigh -> hip if either photo doesn't
    show the legs."""
    b_img_h = b_info["img_size"][1]
    a_img_h = a_info["img_size"][1]
    b_thigh_ok = b_info["knees_seen"] and b_info["mid_thigh_y"] < b_img_h * 0.99
    a_thigh_ok = a_info["knees_seen"] and a_info["mid_thigh_y"] < a_img_h * 0.99
    if b_thigh_ok and a_thigh_ok:
        return "mid_thigh"
    return "hip"


def body_h_src_for(info, landmark):
    """Return head_top -> landmark distance in source pixels."""
    if landmark == "mid_thigh":
        return info["mid_thigh_y"] - info["head_top"][1]
    return info["hip_y"] - info["head_top"][1]


def render_panel(pil_img, info, target_body_h):
    """Scale + crop to fit (PANEL_W x PANEL_H) with head at HEAD_OFFSET
    and body centered horizontally. Body height in panel = target_body_h."""
    head_x, head_y = info["head_top"]
    cx = info["body_cx"]
    src_w, src_h = info["img_size"]
    body_h_src = info["body_h_src"]

    scale = target_body_h / body_h_src
    new_w = int(round(src_w * scale))
    new_h = int(round(src_h * scale))
    scaled = pil_img.resize((new_w, new_h), Image.LANCZOS)

    head_y_s = head_y * scale
    cx_s = cx * scale

    crop_left = int(round(cx_s - PANEL_W / 2))
    crop_top = int(round(head_y_s - HEAD_OFFSET))
    crop_right = crop_left + PANEL_W
    crop_bottom = crop_top + PANEL_H

    # Nudge crop back inside if rounding pushed it past an edge (~1px corrections)
    if crop_left < 0:
        crop_right -= crop_left
        crop_left = 0
    if crop_top < 0:
        crop_bottom -= crop_top
        crop_top = 0
    if crop_right > new_w:
        crop_left -= (crop_right - new_w)
        crop_right = new_w
    if crop_bottom > new_h:
        crop_top -= (crop_bottom - new_h)
        crop_bottom = new_h
    crop_left = max(0, crop_left)
    crop_top = max(0, crop_top)

    return scaled.crop((crop_left, crop_top, crop_right, crop_bottom))


def build_composite(before_path, after_path, pose, out_path):
    """Build one BEFORE | AFTER composite from two photo files.

    Args:
        before_path: path to the older photo
        after_path:  path to the newer photo
        pose:        "front", "side", or "back" (informational tag; logic
                     is the same for all poses, MediaPipe handles facing)
        out_path:    where the composite PNG will be written

    Returns:
        out_path (str)

    Raises:
        CompositeError if either photo can't be processed (pose landmarks
        unreadable, head above frame, hip out of frame, etc.).
    """
    pose = (pose or "front").lower()
    if pose not in {"front", "side", "back"}:
        raise CompositeError(f"unknown pose '{pose}' (expected front|side|back)")

    b_img = Image.open(before_path).convert("RGB")
    a_img = Image.open(after_path).convert("RGB")

    b_info = get_landmarks(b_img)
    if not is_usable_photo(b_info):
        raise CompositeError("before: pose landmarks unreadable or body not in frame")
    a_info = get_landmarks(a_img)
    if not is_usable_photo(a_info):
        raise CompositeError("after: pose landmarks unreadable or body not in frame")

    # Mirror BEFORE if facing direction differs from AFTER
    if b_info["facing"] != a_info["facing"]:
        b_img = b_img.transpose(Image.FLIP_LEFT_RIGHT)
        b_info = get_landmarks(b_img)
        if not is_usable_photo(b_info):
            raise CompositeError("before: pose landmarks lost after mirror")

    # Pick the lowest landmark visible in BOTH photos
    landmark = pick_bottom_landmark(b_info, a_info)
    b_body_h_src = body_h_src_for(b_info, landmark)
    a_body_h_src = body_h_src_for(a_info, landmark)
    b_info = {**b_info, "body_h_src": b_body_h_src}
    a_info = {**a_info, "body_h_src": a_body_h_src}

    target_body_h = match_pair_target_body_h(b_info, a_info)

    b_panel = render_panel(b_img, b_info, target_body_h)
    a_panel = render_panel(a_img, a_info, target_body_h)

    canvas = Image.new("RGB", (PANEL_W * 2 + SEAM_PX, PANEL_H), (255, 255, 255))
    canvas.paste(b_panel, (0, 0))
    canvas.paste(a_panel, (PANEL_W + SEAM_PX, 0))

    out_path = str(out_path)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, "PNG", optimize=True)
    return out_path
