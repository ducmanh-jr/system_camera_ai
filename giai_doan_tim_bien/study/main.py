import argparse
import csv
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"
PLATES_ALL_OUTPUT_DIR = OUTPUTS_DIR / "plates_all"
PLATES_FILTERED_OUTPUT_DIR = OUTPUTS_DIR / "plates_filtered"
PLATES_OUTPUT_DIR = PLATES_FILTERED_OUTPUT_DIR
VEHICLES_OUTPUT_DIR = OUTPUTS_DIR / "vehicles"
REVIEWS_OUTPUT_DIR = OUTPUTS_DIR / "reviews"
DB_PATH = OUTPUTS_DIR / "violations.db"
CSV_PATH = OUTPUTS_DIR / "violators.csv"
TRACKS_CSV_PATH = OUTPUTS_DIR / "tracks.csv"
OUTPUT_VIDEO_PATH = OUTPUTS_DIR / "output.mp4"
DEFAULT_VIDEO_PATH = DATA_DIR / "video.mp4"

VEHICLE_MODEL_PATH = MODELS_DIR / "yolov8m.pt"
LICENSE_PLATE_MODEL_PATH = MODELS_DIR / "license_plate_detector.pt"
VEHICLE_LABELS = {2: "Car", 3: "Motorcycle", 5: "Bus", 7: "Truck"}
"""
DEFAULT_VEHICLE_CONF = 0.2
DEFAULT_PLATE_CONF = 0.3
DEFAULT_IOU = 0.5
DEFAULT_PLAYBACK_SPEED = 0.5
DEFAULT_ROI = "0.0,0.55,1.0,1.0"
DEFAULT_PLATE_INTERVAL = 1
DEFAULT_MIN_TRACK_FRAMES = 5
DEFAULT_CROP_EXPAND_RATIO = 0.18
DEFAULT_MIN_PLATE_ASPECT = 1.1
DEFAULT_MAX_PLATE_ASPECT = 6.5
DEFAULT_MIN_PLATE_AREA_RATIO = 0.003
DEFAULT_MAX_PLATE_AREA_RATIO = 0.10
DEFAULT_MIN_VEHICLE_AREA_RATIO = 0.006
DEFAULT_MIN_PLATE_WIDTH = 50
DEFAULT_MIN_PLATE_HEIGHT = 18
DEFAULT_MIN_PLATE_SHARPNESS = 2800.0
DEFAULT_MIN_PLATE_SCORE = 0.52
DEFAULT_PLATE_PADDING_RATIO = 0.08
DEFAULT_DUPLICATE_PLATE_SIMILARITY = 0.72
DEFAULT_DUPLICATE_FRAME_WINDOW = 35
DEFAULT_DUPLICATE_POSITION_DISTANCE = 90
SHARPNESS_NORMALIZER = 12000.0
PREVIEW_PLATE_WIDTH = 120
PREVIEW_PLATE_HEIGHT = 55
"""
DEFAULT_VEHICLE_CONF = 0.18
DEFAULT_PLATE_CONF = 0.25
DEFAULT_IOU = 0.5
DEFAULT_PLAYBACK_SPEED = 0.5
DEFAULT_ROI = "0.0,0.40,1.0,1.0"
DEFAULT_PLATE_INTERVAL = 1
DEFAULT_MIN_TRACK_FRAMES = 4
DEFAULT_CROP_EXPAND_RATIO = 0.24
DEFAULT_MIN_PLATE_ASPECT = 0.70
DEFAULT_MAX_PLATE_ASPECT = 6.5
DEFAULT_MIN_PLATE_AREA_RATIO = 0.0015
DEFAULT_MAX_PLATE_AREA_RATIO = 0.14
DEFAULT_MIN_VEHICLE_AREA_RATIO = 0.003
DEFAULT_MIN_PLATE_WIDTH = 32
DEFAULT_MIN_PLATE_HEIGHT = 12
DEFAULT_MIN_PLATE_SHARPNESS = 1800.0
DEFAULT_MIN_PLATE_SCORE = 0.46
DEFAULT_PLATE_PADDING_RATIO = 0.12
DEFAULT_DUPLICATE_PLATE_SIMILARITY = 0.70
DEFAULT_DUPLICATE_FRAME_WINDOW = 45
DEFAULT_DUPLICATE_POSITION_DISTANCE = 110
SHARPNESS_NORMALIZER = 12000.0
PREVIEW_PLATE_WIDTH = 120
PREVIEW_PLATE_HEIGHT = 55

@dataclass
class ViolationRecord:
    violation_id: int
    track_id: int
    vehicle_type: str
    frame_id: int
    timestamp: str
    plate_path: str
    vehicle_path: str
    review_path: str = ""
    plate_frame_id: int = -1
    plate_confidence: float = 0.0
    plate_sharpness: float = 0.0
    plate_score: float = 0.0


@dataclass
class PlateCandidate:
    crop: object
    box: tuple
    relative_box: tuple
    confidence: float
    sharpness: float
    area_score: float
    contrast: float
    score: float
    frame_id: int


@dataclass
class TrackStats:
    track_id: int
    vehicle_type: str
    first_frame: int
    last_frame: int
    seen_frames: int = 0
    record_frame: int = -1
    recorded: bool = False


def ensure_output_dirs(clear_plate_images=True):
    OUTPUTS_DIR.mkdir(exist_ok=True)
    PLATES_ALL_OUTPUT_DIR.mkdir(exist_ok=True)
    PLATES_FILTERED_OUTPUT_DIR.mkdir(exist_ok=True)
    VEHICLES_OUTPUT_DIR.mkdir(exist_ok=True)
    REVIEWS_OUTPUT_DIR.mkdir(exist_ok=True)
    if clear_plate_images:
        for output_dir in (
            PLATES_ALL_OUTPUT_DIR,
            PLATES_FILTERED_OUTPUT_DIR,
            VEHICLES_OUTPUT_DIR,
            REVIEWS_OUTPUT_DIR,
        ):
            for image_path in output_dir.glob("*.png"):
                image_path.unlink()


def create_connection(db_path, append=False):
    conn = sqlite3.connect(db_path)
    if not append:
        conn.execute("DROP TABLE IF EXISTS violations")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS violations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            violation_id INTEGER,
            track_id INTEGER,
            vehicle_type TEXT NOT NULL,
            frame_id INTEGER NOT NULL,
            timestamp TEXT NOT NULL,
            plate_path TEXT,
            vehicle_path TEXT,
            review_path TEXT,
            plate_frame_id INTEGER,
            plate_confidence REAL,
            plate_sharpness REAL,
            plate_score REAL
        )
        """
    )
    conn.commit()
    return conn


def save_violation_to_db(conn, record):
    conn.execute(
        """
        INSERT INTO violations (
            violation_id, track_id, vehicle_type, frame_id, timestamp,
            plate_path, vehicle_path, review_path, plate_frame_id,
            plate_confidence, plate_sharpness, plate_score
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            record.violation_id,
            record.track_id,
            record.vehicle_type,
            record.frame_id,
            record.timestamp,
            record.plate_path,
            record.vehicle_path,
            record.review_path,
            record.plate_frame_id,
            record.plate_confidence,
            record.plate_sharpness,
            record.plate_score,
        ),
    )
    conn.commit()


def clamp_box(box, width, height):
    x1, y1, x2, y2 = map(int, box)
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(0, min(x2, width))
    y2 = max(0, min(y2, height))
    return x1, y1, x2, y2


def expand_box(box, width, height, ratio):
    x1, y1, x2, y2 = clamp_box(box, width, height)
    box_width = x2 - x1
    box_height = y2 - y1
    pad_x = int(box_width * ratio)
    pad_y = int(box_height * ratio)
    return clamp_box((x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y), width, height)


def is_valid_crop(x1, y1, x2, y2):
    return x2 > x1 and y2 > y1


def calculate_sharpness(image):
    if image is None or image.size == 0:
        return 0.0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def calculate_contrast(image):
    if image is None or image.size == 0:
        return 0.0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(gray.std())


def calculate_plate_score(confidence, sharpness, contrast, plate_box, crop_width, crop_height):
    px1, py1, px2, py2 = plate_box
    plate_width = px2 - px1
    plate_height = py2 - py1
    plate_area = max(1, (px2 - px1) * (py2 - py1))
    crop_area = max(1, crop_width * crop_height)
    sharpness_score = min(sharpness / SHARPNESS_NORMALIZER, 1.0)
    area_score = min(plate_area / (crop_area * 0.08), 1.0)
    contrast_score = min(contrast / 65.0, 1.0)
    size_score = min(min(plate_width / 95.0, plate_height / 32.0), 1.0)
    combined_score = (
        confidence * 0.35
        + sharpness_score * 0.25
        + area_score * 0.10
        + contrast_score * 0.10
        + size_score * 0.20
    )
    return combined_score, area_score


def is_plausible_plate_box(
    plate_box,
    crop_width,
    crop_height,
    min_aspect,
    max_aspect,
    min_area_ratio,
    max_area_ratio,
    min_plate_width,
    min_plate_height,
):
    px1, py1, px2, py2 = plate_box
    plate_width = px2 - px1
    plate_height = py2 - py1
    if plate_width < min_plate_width or plate_height < min_plate_height:
        return False

    aspect = plate_width / plate_height
    area_ratio = (plate_width * plate_height) / max(1, crop_width * crop_height)
    return (
        min_aspect <= aspect <= max_aspect
        and min_area_ratio <= area_ratio <= max_area_ratio
    )


def is_vehicle_large_enough(vehicle_box, frame_width, frame_height, min_area_ratio):
    x1, y1, x2, y2 = vehicle_box
    area = max(0, x2 - x1) * max(0, y2 - y1)
    frame_area = max(1, frame_width * frame_height)
    return area / frame_area >= min_area_ratio


def detect_license_plate(
    frame,
    vehicle_box,
    license_plate_detector,
    conf,
    iou,
    crop_expand_ratio,
    min_plate_aspect,
    max_plate_aspect,
    min_plate_area_ratio,
    max_plate_area_ratio,
    min_plate_width,
    min_plate_height,
    min_plate_sharpness,
    plate_padding_ratio,
):
    frame_height, frame_width = frame.shape[:2]
    x1, y1, x2, y2 = expand_box(
        vehicle_box, frame_width, frame_height, crop_expand_ratio
    )
    if not is_valid_crop(x1, y1, x2, y2):
        return None

    vehicle_crop = frame[y1:y2, x1:x2]
    crop_height, crop_width = vehicle_crop.shape[:2]
    plate_results = license_plate_detector(vehicle_crop, conf=conf, iou=iou, verbose=False)

    best_candidate = None
    for result in plate_results:
        for box in result.boxes:
            confidence = float(box.conf[0]) if box.conf is not None else 0.0
            px1, py1, px2, py2 = clamp_box(
                box.xyxy[0], crop_width, crop_height
            )
            if not is_valid_crop(px1, py1, px2, py2):
                continue
            if not is_plausible_plate_box(
                (px1, py1, px2, py2),
                crop_width,
                crop_height,
                min_plate_aspect,
                max_plate_aspect,
                min_plate_area_ratio,
                max_plate_area_ratio,
                min_plate_width,
                min_plate_height,
            ):
                continue
            padded_px1, padded_py1, padded_px2, padded_py2 = expand_box(
                (px1, py1, px2, py2), crop_width, crop_height, plate_padding_ratio
            )
            plate_crop = vehicle_crop[padded_py1:padded_py2, padded_px1:padded_px2]
            sharpness = calculate_sharpness(plate_crop)
            if sharpness < min_plate_sharpness:
                continue
            contrast = calculate_contrast(plate_crop)
            score, area_score = calculate_plate_score(
                confidence, sharpness, contrast, (px1, py1, px2, py2), crop_width, crop_height
            )
            if best_candidate is None or score > best_candidate.score:
                plate_box = (
                    x1 + padded_px1,
                    y1 + padded_py1,
                    x1 + padded_px2,
                    y1 + padded_py2,
                )
                relative_box = (
                    padded_px1 / crop_width,
                    padded_py1 / crop_height,
                    padded_px2 / crop_width,
                    padded_py2 / crop_height,
                )
                best_candidate = PlateCandidate(
                    crop=plate_crop.copy(),
                    box=plate_box,
                    relative_box=relative_box,
                    confidence=confidence,
                    sharpness=sharpness,
                    area_score=area_score,
                    contrast=contrast,
                    score=score,
                    frame_id=-1,
                )

    return best_candidate


def draw_label(frame, text, origin, color):
    x, y = origin
    y = max(20, y)
    cv2.putText(
        frame,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
        cv2.LINE_AA,
    )


def draw_plate_preview(frame, plate_crop, x, y):
    if plate_crop is None or plate_crop.size == 0:
        return

    frame_height, frame_width = frame.shape[:2]
    preview_width = min(PREVIEW_PLATE_WIDTH, frame_width - x)
    preview_height = min(PREVIEW_PLATE_HEIGHT, frame_height - y)
    if preview_width <= 0 or preview_height <= 0:
        return

    plate_resized = cv2.resize(plate_crop, (preview_width, preview_height))
    frame[y : y + preview_height, x : x + preview_width] = plate_resized


def parse_roi(roi_text):
    if not roi_text:
        return None
    values = [float(value.strip()) for value in roi_text.split(",")]
    if len(values) != 4:
        raise ValueError("--roi can co dang x1,y1,x2,y2")
    x1, y1, x2, y2 = values
    if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
        raise ValueError("--roi dung toa do ti le trong khoang 0..1 va x1<x2, y1<y2")
    return x1, y1, x2, y2


def scale_roi(roi, width, height):
    if roi is None:
        return None
    x1, y1, x2, y2 = roi
    return int(x1 * width), int(y1 * height), int(x2 * width), int(y2 * height)


def point_in_roi(x, y, roi_box):
    if roi_box is None:
        return True
    x1, y1, x2, y2 = roi_box
    return x1 <= x <= x2 and y1 <= y <= y2


def save_plate_crop(track_id, candidate):
    if candidate is None:
        return ""

    plate_path = PLATES_OUTPUT_DIR / (
        f"cropped_plate_track_{track_id}_frame_{candidate.frame_id}.png"
    )
    cv2.imwrite(str(plate_path), candidate.crop)
    return str(plate_path)


def save_all_plate_crop(track_id, candidate):
    if candidate is None:
        return ""

    plate_path = PLATES_ALL_OUTPUT_DIR / (
        f"plate_track_{track_id}_frame_{candidate.frame_id}_score_{candidate.score:.4f}.png"
    )
    cv2.imwrite(str(plate_path), candidate.crop)
    return str(plate_path)


def save_vehicle_crop(violation_id, frame, vehicle_box):
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = expand_box(vehicle_box, width, height, 0.08)
    if not is_valid_crop(x1, y1, x2, y2):
        return ""
    vehicle_crop = frame[y1:y2, x1:x2]
    vehicle_path = VEHICLES_OUTPUT_DIR / f"vehicle_V{violation_id:03d}.png"
    cv2.imwrite(str(vehicle_path), vehicle_crop)
    return str(vehicle_path)


def save_vehicle_crop_image(violation_id, vehicle_crop):
    if vehicle_crop is None or vehicle_crop.size == 0:
        return ""
    vehicle_path = VEHICLES_OUTPUT_DIR / f"vehicle_V{violation_id:03d}.png"
    cv2.imwrite(str(vehicle_path), vehicle_crop)
    return str(vehicle_path)


def crop_vehicle_snapshot(frame, vehicle_box):
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = expand_box(vehicle_box, width, height, 0.08)
    if not is_valid_crop(x1, y1, x2, y2):
        return None
    return frame[y1:y2, x1:x2].copy()


def project_plate_box(vehicle_box, relative_box):
    x1, y1, x2, y2 = vehicle_box
    rel_x1, rel_y1, rel_x2, rel_y2 = relative_box
    width = x2 - x1
    height = y2 - y1
    return (
        int(x1 + rel_x1 * width),
        int(y1 + rel_y1 * height),
        int(x1 + rel_x2 * width),
        int(y1 + rel_y2 * height),
    )


def create_review_image(record):
    if not record.vehicle_path:
        return ""

    vehicle = cv2.imread(record.vehicle_path)
    if vehicle is None:
        return ""

    plate = cv2.imread(record.plate_path) if record.plate_path else None
    canvas_width = 900
    left_width = 620
    right_width = canvas_width - left_width
    canvas_height = 420
    canvas = np.full((canvas_height, canvas_width, 3), 255, dtype=np.uint8)

    vehicle_h, vehicle_w = vehicle.shape[:2]
    vehicle_scale = min(left_width / vehicle_w, 330 / vehicle_h)
    vehicle_resized = cv2.resize(
        vehicle, (int(vehicle_w * vehicle_scale), int(vehicle_h * vehicle_scale))
    )
    canvas[70 : 70 + vehicle_resized.shape[0], 20 : 20 + vehicle_resized.shape[1]] = vehicle_resized

    title = f"V{record.violation_id:03d} | Track {record.track_id} | {record.vehicle_type}"
    cv2.putText(canvas, title, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.putText(
        canvas,
        f"Record frame: {record.frame_id}",
        (20, 395),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (60, 60, 60),
        2,
        cv2.LINE_AA,
    )

    if plate is not None:
        plate_h, plate_w = plate.shape[:2]
        plate_scale = min((right_width - 40) / plate_w, 160 / plate_h)
        plate_resized = cv2.resize(
            plate, (int(plate_w * plate_scale), int(plate_h * plate_scale))
        )
        px = left_width + 20
        py = 90
        canvas[py : py + plate_resized.shape[0], px : px + plate_resized.shape[1]] = plate_resized
        cv2.rectangle(
            canvas,
            (px, py),
            (px + plate_resized.shape[1], py + plate_resized.shape[0]),
            (255, 0, 0),
            2,
        )
        info_lines = [
            f"Plate frame: {record.plate_frame_id}",
            f"Conf: {record.plate_confidence:.3f}",
            f"Sharp: {record.plate_sharpness:.1f}",
            f"Score: {record.plate_score:.3f}",
        ]
    else:
        info_lines = ["Plate: None"]

    for index, line in enumerate(info_lines):
        cv2.putText(
            canvas,
            line,
            (left_width + 20, 290 + index * 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (50, 50, 50),
            2,
            cv2.LINE_AA,
        )

    review_path = REVIEWS_OUTPUT_DIR / f"review_V{record.violation_id:03d}.png"
    cv2.imwrite(str(review_path), canvas)
    return str(review_path)


def replace_record_plate_path(record, new_plate_path):
    old_plate_path = Path(record.plate_path) if record.plate_path else None
    if old_plate_path and old_plate_path.exists() and old_plate_path != Path(new_plate_path):
        old_plate_path.unlink()
    record.plate_path = new_plate_path


def update_record_plate(record, candidate):
    plate_path = save_plate_crop(record.track_id, candidate)
    replace_record_plate_path(record, plate_path)
    record.plate_frame_id = candidate.frame_id
    record.plate_confidence = candidate.confidence
    record.plate_sharpness = candidate.sharpness
    record.plate_score = candidate.score
    return plate_path


def plate_signature_from_crop(crop):
    if crop is None or crop.size == 0:
        return None
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (96, 32), interpolation=cv2.INTER_AREA)
    gray = cv2.equalizeHist(gray)
    vector = gray.astype(np.float32).reshape(-1)
    vector -= float(vector.mean())
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-6:
        return None
    return vector / norm


def plate_center(candidate):
    x1, y1, x2, y2 = candidate.box
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def plate_position_distance(candidate, kept_candidate):
    cx1, cy1 = plate_center(candidate)
    cx2, cy2 = plate_center(kept_candidate)
    return float(np.hypot(cx1 - cx2, cy1 - cy2))


def plate_size_ratio(candidate, kept_candidate):
    x1, y1, x2, y2 = candidate.box
    kx1, ky1, kx2, ky2 = kept_candidate.box
    area = max(1, (x2 - x1) * (y2 - y1))
    kept_area = max(1, (kx2 - kx1) * (ky2 - ky1))
    return min(area, kept_area) / max(area, kept_area)


def is_duplicate_plate(candidate, kept_candidates, similarity_threshold):
    signature = plate_signature_from_crop(candidate.crop)
    if signature is None:
        return False
    for kept_candidate in kept_candidates:
        kept_signature = plate_signature_from_crop(kept_candidate.crop)
        if kept_signature is None:
            continue
        similarity = float(np.dot(signature, kept_signature))
        nearby_frame_duplicate = (
            abs(candidate.frame_id - kept_candidate.frame_id) <= 30
            and similarity >= 0.50
        )
        candidate_h, candidate_w = candidate.crop.shape[:2]
        kept_h, kept_w = kept_candidate.crop.shape[:2]
        candidate_aspect = candidate_w / max(1, candidate_h)
        kept_aspect = kept_w / max(1, kept_h)
        frame_gap = abs(candidate.frame_id - kept_candidate.frame_id)
        position_distance = plate_position_distance(candidate, kept_candidate)
        size_ratio = plate_size_ratio(candidate, kept_candidate)
        nearby_wide_duplicate = (
            frame_gap <= 5
            and similarity >= 0.35
            and candidate_aspect >= 2.2
            and kept_aspect >= 2.2
            and abs(candidate_aspect - kept_aspect) <= 0.8
        )
        nearby_position_duplicate = (
            frame_gap <= DEFAULT_DUPLICATE_FRAME_WINDOW
            and position_distance <= DEFAULT_DUPLICATE_POSITION_DISTANCE
            and size_ratio >= 0.45
            and similarity >= 0.22
        )
        if (
            similarity >= similarity_threshold
            or nearby_frame_duplicate
            or nearby_wide_duplicate
            or nearby_position_duplicate
        ):
            return True
    return False


def build_records_from_best_tracks(best_plate_by_track, best_vehicle_by_track, track_stats, min_track_frames, min_plate_score):
    kept = []
    kept_candidates = []
    candidates = sorted(
        best_plate_by_track.items(),
        key=lambda item: item[1].score,
        reverse=True,
    )
    for track_id, candidate in candidates:
        stats = track_stats.get(track_id)
        if stats is None or candidate is None:
            continue
        if stats.seen_frames < min_track_frames or candidate.score < min_plate_score:
            continue
        if is_duplicate_plate(
            candidate, kept_candidates, DEFAULT_DUPLICATE_PLATE_SIMILARITY
        ):
            continue
        kept.append((track_id, stats, candidate))
        kept_candidates.append(candidate)

    records = []
    kept.sort(key=lambda item: item[2].frame_id)
    for track_id, stats, candidate in kept:
        violation_id = len(records) + 1
        plate_path = save_plate_crop(track_id, candidate)
        vehicle_path = save_vehicle_crop_image(
            violation_id, best_vehicle_by_track.get(track_id)
        )
        record = ViolationRecord(
            violation_id=violation_id,
            track_id=track_id,
            vehicle_type=stats.vehicle_type,
            frame_id=candidate.frame_id,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            plate_path=plate_path,
            vehicle_path=vehicle_path,
            plate_frame_id=candidate.frame_id,
            plate_confidence=candidate.confidence,
            plate_sharpness=candidate.sharpness,
            plate_score=candidate.score,
        )
        stats.recorded = True
        stats.record_frame = candidate.frame_id
        records.append(record)
    return records


def get_track_stats(track_stats, track_id, vehicle_type, frame_id):
    if track_id not in track_stats:
        track_stats[track_id] = TrackStats(
            track_id=track_id,
            vehicle_type=vehicle_type,
            first_frame=frame_id,
            last_frame=frame_id,
        )

    stats = track_stats[track_id]
    stats.last_frame = frame_id
    stats.seen_frames += 1
    return stats


def write_csv(csv_path, records):
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "Ma Vi Pham",
                "Track ID",
                "Loai Phuong Tien",
                "Frame Ghi Nhan",
                "Thoi Gian",
                "Anh Xe",
                "Anh Vung Bien So",
                "Anh Review",
                "Frame Bien So Tot Nhat",
                "Do Tin Cay Bien So",
                "Do Net Bien So",
                "Diem Bien So",
            ]
        )
        for record in records:
            writer.writerow(
                [
                    f"V{record.violation_id:03d}",
                    record.track_id,
                    record.vehicle_type,
                    record.frame_id,
                    record.timestamp,
                    record.vehicle_path or "None",
                    record.plate_path or "None",
                    record.review_path or "None",
                    record.plate_frame_id if record.plate_frame_id >= 0 else "None",
                    f"{record.plate_confidence:.4f}" if record.plate_path else "None",
                    f"{record.plate_sharpness:.1f}" if record.plate_path else "None",
                    f"{record.plate_score:.4f}" if record.plate_path else "None",
                ]
            )


def write_tracks_csv(csv_path, track_stats, best_plate_by_track):
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "Track ID",
                "Loai Phuong Tien",
                "Frame Dau",
                "Frame Cuoi",
                "So Frame Xuat Hien",
                "Frame Ghi Nhan",
                "Da Ghi Nhan",
                "Co Bien So",
                "Frame Bien So Tot Nhat",
                "Do Tin Cay Bien So",
            ]
        )
        for track_id in sorted(track_stats):
            stats = track_stats[track_id]
            plate = best_plate_by_track.get(track_id)
            writer.writerow(
                [
                    stats.track_id,
                    stats.vehicle_type,
                    stats.first_frame,
                    stats.last_frame,
                    stats.seen_frames,
                    stats.record_frame if stats.record_frame >= 0 else "None",
                    "Yes" if stats.recorded else "No",
                    "Yes" if plate is not None else "No",
                    plate.frame_id if plate is not None else "None",
                    f"{plate.score:.4f}" if plate is not None else "None",
                ]
            )


def create_video_writer(video_path, width, height, fps):
    video_path.parent.mkdir(exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output_fps = fps if fps and fps > 0 else 30
    return cv2.VideoWriter(str(video_path), fourcc, output_fps, (width, height))


def detect_traffic_violation(
    video_path,
    vehicle_model,
    license_plate_detector,
    conn,
    vehicle_conf=DEFAULT_VEHICLE_CONF,
    plate_conf=DEFAULT_PLATE_CONF,
    iou=DEFAULT_IOU,
    playback_speed=DEFAULT_PLAYBACK_SPEED,
    plate_interval=DEFAULT_PLATE_INTERVAL,
    min_track_frames=DEFAULT_MIN_TRACK_FRAMES,
    crop_expand_ratio=DEFAULT_CROP_EXPAND_RATIO,
    min_plate_aspect=DEFAULT_MIN_PLATE_ASPECT,
    max_plate_aspect=DEFAULT_MAX_PLATE_ASPECT,
    min_plate_area_ratio=DEFAULT_MIN_PLATE_AREA_RATIO,
    max_plate_area_ratio=DEFAULT_MAX_PLATE_AREA_RATIO,
    min_vehicle_area_ratio=DEFAULT_MIN_VEHICLE_AREA_RATIO,
    min_plate_width=DEFAULT_MIN_PLATE_WIDTH,
    min_plate_height=DEFAULT_MIN_PLATE_HEIGHT,
    min_plate_sharpness=DEFAULT_MIN_PLATE_SHARPNESS,
    min_plate_score=DEFAULT_MIN_PLATE_SCORE,
    plate_padding_ratio=DEFAULT_PLATE_PADDING_RATIO,
    roi=None,
    save_video=True,
    output_video_path=OUTPUT_VIDEO_PATH,
    show=False,
):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Khong mo duoc video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    delay = int(1000 / fps / playback_speed) if fps > 0 and playback_speed > 0 else 1
    roi_box = scale_roi(roi, width, height)
    video_writer = create_video_writer(output_video_path, width, height, fps) if save_video else None

    best_plate_by_track = {}
    best_vehicle_by_track = {}
    track_stats = {}
    frame_id = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        clean_frame = frame.copy()

        results = vehicle_model.track(
            clean_frame,
            conf=vehicle_conf,
            iou=iou,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False,
        )

        for result in results:
            for box in result.boxes:
                if box.id is None:
                    continue

                cls = int(box.cls[0])
                if cls not in VEHICLE_LABELS:
                    continue

                track_id = int(box.id[0])
                x1, y1, x2, y2 = clamp_box(box.xyxy[0], width, height)
                if not is_valid_crop(x1, y1, x2, y2):
                    continue
                if not is_vehicle_large_enough(
                    (x1, y1, x2, y2), width, height, min_vehicle_area_ratio
                ):
                    continue

                vehicle_type = VEHICLE_LABELS[cls]
                center_y = (y1 + y2) // 2
                center_x = (x1 + x2) // 2
                if not point_in_roi(center_x, center_y, roi_box):
                    continue

                stats = get_track_stats(track_stats, track_id, vehicle_type, frame_id)

                should_try_plate = frame_id % plate_interval == 0
                if should_try_plate:
                    plate_candidate = detect_license_plate(
                        clean_frame,
                        (x1, y1, x2, y2),
                        license_plate_detector,
                        plate_conf,
                        iou,
                        crop_expand_ratio,
                        min_plate_aspect,
                        max_plate_aspect,
                        min_plate_area_ratio,
                        max_plate_area_ratio,
                        min_plate_width,
                        min_plate_height,
                        min_plate_sharpness,
                        plate_padding_ratio,
                    )
                    if plate_candidate is not None:
                        plate_candidate.frame_id = frame_id
                        save_all_plate_crop(track_id, plate_candidate)
                    current_best = best_plate_by_track.get(track_id)
                    if plate_candidate is not None and (
                        current_best is None or plate_candidate.score > current_best.score
                    ):
                        best_plate_by_track[track_id] = PlateCandidate(
                            crop=plate_candidate.crop.copy(),
                            box=plate_candidate.box,
                            relative_box=plate_candidate.relative_box,
                            confidence=plate_candidate.confidence,
                            sharpness=plate_candidate.sharpness,
                            area_score=plate_candidate.area_score,
                            contrast=plate_candidate.contrast,
                            score=plate_candidate.score,
                            frame_id=plate_candidate.frame_id,
                        )
                        best_vehicle_by_track[track_id] = crop_vehicle_snapshot(
                            clean_frame, (x1, y1, x2, y2)
                        )

                plate_candidate_for_draw = best_plate_by_track.get(track_id)
                if plate_candidate_for_draw is not None:
                    expanded_current_box = expand_box(
                        (x1, y1, x2, y2), width, height, crop_expand_ratio
                    )
                    px1, py1, px2, py2 = project_plate_box(
                        expanded_current_box, plate_candidate_for_draw.relative_box
                    )
                    cv2.rectangle(frame, (px1, py1), (px2, py2), (255, 0, 0), 2)

                color = (0, 0, 255) if track_id in best_plate_by_track else (0, 180, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                draw_label(
                    frame,
                    f"ID {track_id} - {vehicle_type}",
                    (x1, y1 - 10),
                    color,
                )

        if roi_box is not None:
            rx1, ry1, rx2, ry2 = roi_box
            cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (255, 255, 0), 2)

        if video_writer is not None:
            video_writer.write(frame)

        if show:
            cv2.imshow("Video - tracking", frame)
            if cv2.waitKey(delay) & 0xFF == ord("q"):
                break

        frame_id += 1

    cap.release()
    if video_writer is not None:
        video_writer.release()
    if show:
        cv2.destroyAllWindows()

    records = build_records_from_best_tracks(
        best_plate_by_track,
        best_vehicle_by_track,
        track_stats,
        min_track_frames,
        min_plate_score,
    )
    for record in records:
        record.review_path = create_review_image(record)
        save_violation_to_db(conn, record)
    write_csv(CSV_PATH, records)
    write_tracks_csv(TRACKS_CSV_PATH, track_stats, best_plate_by_track)
    print(f"[INFO] Da luu CSV: {CSV_PATH}")
    print(f"[INFO] Da luu thong ke track: {TRACKS_CSV_PATH}")
    if save_video:
        print(f"[INFO] Da luu video ket qua: {output_video_path}")
    print(f"[INFO] Da luu DB: {DB_PATH}")
    print(f"[INFO] Tong so luot ghi nhan: {len(records)}")
    print(f"[INFO] Tong so track phuong tien: {len(track_stats)}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Detect va chon vung bien so tot nhat trong ROI gan camera bang YOLO tracking."
    )
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO_PATH)
    parser.add_argument(
        "--vehicle-conf",
        type=float,
        default=DEFAULT_VEHICLE_CONF,
        help="Nguong confidence cho model phuong tien.",
    )
    parser.add_argument(
        "--plate-conf",
        type=float,
        default=DEFAULT_PLATE_CONF,
        help="Nguong confidence cho model bien so.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        help="Tuong thich lenh cu; ap dung cho ca phuong tien va bien so.",
    )
    parser.add_argument("--iou", type=float, default=DEFAULT_IOU)
    parser.add_argument("--speed", type=float, default=DEFAULT_PLAYBACK_SPEED)
    parser.add_argument(
        "--plate-interval",
        type=int,
        default=DEFAULT_PLATE_INTERVAL,
        help="Chi detect bien so moi N frame de giam thoi gian chay.",
    )
    parser.add_argument(
        "--min-track-frames",
        type=int,
        default=DEFAULT_MIN_TRACK_FRAMES,
        help="Chi ghi nhan xe sau khi track xuat hien it nhat N frame.",
    )
    parser.add_argument(
        "--crop-expand",
        type=float,
        default=DEFAULT_CROP_EXPAND_RATIO,
        help="Mo rong box xe truoc khi detect bien so.",
    )
    parser.add_argument(
        "--min-plate-aspect",
        type=float,
        default=DEFAULT_MIN_PLATE_ASPECT,
        help="Ti le rong/cao nho nhat cua box bien so hop le.",
    )
    parser.add_argument(
        "--max-plate-aspect",
        type=float,
        default=DEFAULT_MAX_PLATE_ASPECT,
        help="Ti le rong/cao lon nhat cua box bien so hop le.",
    )
    parser.add_argument(
        "--min-plate-area-ratio",
        type=float,
        default=DEFAULT_MIN_PLATE_AREA_RATIO,
        help="Dien tich box bien so nho nhat so voi crop xe.",
    )
    parser.add_argument(
        "--max-plate-area-ratio",
        type=float,
        default=DEFAULT_MAX_PLATE_AREA_RATIO,
        help="Dien tich box bien so lon nhat so voi crop xe.",
    )
    parser.add_argument(
        "--min-vehicle-area-ratio",
        type=float,
        default=DEFAULT_MIN_VEHICLE_AREA_RATIO,
        help="Bo qua xe qua nho/qua xa theo ti le dien tich tren frame.",
    )
    parser.add_argument(
        "--min-plate-width",
        type=int,
        default=DEFAULT_MIN_PLATE_WIDTH,
        help="Bo qua crop bien so co chieu rong pixel qua nho.",
    )
    parser.add_argument(
        "--min-plate-height",
        type=int,
        default=DEFAULT_MIN_PLATE_HEIGHT,
        help="Bo qua crop bien so co chieu cao pixel qua nho.",
    )
    parser.add_argument(
        "--min-plate-sharpness",
        type=float,
        default=DEFAULT_MIN_PLATE_SHARPNESS,
        help="Bo qua crop bien so mo theo variance of Laplacian.",
    )
    parser.add_argument(
        "--min-plate-score",
        type=float,
        default=DEFAULT_MIN_PLATE_SCORE,
        help="Chi xuat crop bien so co diem chat luong toi thieu.",
    )
    parser.add_argument(
        "--plate-padding",
        type=float,
        default=DEFAULT_PLATE_PADDING_RATIO,
        help="Mo rong bbox bien so truoc khi luu de tranh cat sat ky tu.",
    )
    parser.add_argument(
        "--roi",
        default=DEFAULT_ROI,
        help="Vung gan camera can xu ly theo ti le x1,y1,x2,y2. Mac dinh la nua duoi frame.",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Giu du lieu cu trong database thay vi xoa truoc khi chay.",
    )
    parser.add_argument(
        "--output-video",
        type=Path,
        default=OUTPUT_VIDEO_PATH,
        help="Duong dan file video MP4 ket qua.",
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Khong xuat video ket qua.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Hien cua so realtime khi dang xu ly.",
    )
    parser.add_argument(
        "--save-video",
        action="store_true",
        help="Tuong thich lenh cu; video hien duoc xuat mac dinh.",
    )
    parser.add_argument(
        "--debug-video",
        type=Path,
        help="Tuong thich lenh cu; nen dung --output-video.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Tuong thich lenh cu; mac dinh da khong hien realtime.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    ensure_output_dirs(clear_plate_images=not args.append)

    if not args.video.exists():
        print(f"[ERROR] Khong tim thay video: {args.video}")
        return
    if not VEHICLE_MODEL_PATH.exists():
        print(f"[ERROR] Khong tim thay model phuong tien: {VEHICLE_MODEL_PATH}")
        return
    if not LICENSE_PLATE_MODEL_PATH.exists():
        print(f"[ERROR] Khong tim thay model bien so: {LICENSE_PLATE_MODEL_PATH}")
        return

    vehicle_model = YOLO(str(VEHICLE_MODEL_PATH))
    license_plate_detector = YOLO(str(LICENSE_PLATE_MODEL_PATH))
    output_video_path = args.debug_video if args.debug_video else args.output_video
    vehicle_conf = args.conf if args.conf is not None else args.vehicle_conf
    plate_conf = args.conf if args.conf is not None else args.plate_conf
    try:
        roi = parse_roi(args.roi)
    except ValueError as exc:
        print(f"[ERROR] {exc}")
        return

    with create_connection(DB_PATH, append=args.append) as conn:
        detect_traffic_violation(
            video_path=args.video,
            vehicle_model=vehicle_model,
            license_plate_detector=license_plate_detector,
            conn=conn,
            vehicle_conf=vehicle_conf,
            plate_conf=plate_conf,
            iou=args.iou,
            playback_speed=args.speed,
            plate_interval=max(1, args.plate_interval),
            min_track_frames=max(1, args.min_track_frames),
            crop_expand_ratio=max(0.0, args.crop_expand),
            min_plate_aspect=max(0.1, args.min_plate_aspect),
            max_plate_aspect=max(args.min_plate_aspect, args.max_plate_aspect),
            min_plate_area_ratio=max(0.0, args.min_plate_area_ratio),
            max_plate_area_ratio=max(args.min_plate_area_ratio, args.max_plate_area_ratio),
            min_vehicle_area_ratio=max(0.0, args.min_vehicle_area_ratio),
            min_plate_width=max(1, args.min_plate_width),
            min_plate_height=max(1, args.min_plate_height),
            min_plate_sharpness=max(0.0, args.min_plate_sharpness),
            min_plate_score=max(0.0, args.min_plate_score),
            plate_padding_ratio=max(0.0, args.plate_padding),
            roi=roi,
            save_video=not args.no_video,
            output_video_path=output_video_path,
            show=args.show and not args.no_show,
        )


if __name__ == "__main__":
    main()
