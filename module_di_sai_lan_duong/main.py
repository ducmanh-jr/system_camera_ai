import argparse
import csv
import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
DEFAULT_VIDEO_PATH = ROOT_DIR / "giai_doan_tim_bien" / "data" / "video.mp4"
DEFAULT_MODEL_PATH = ROOT_DIR / "giai_doan_tim_bien" / "models" / "yolov8m.pt"
FALLBACK_MODEL_PATH = ROOT_DIR / "giai_doan_tien_xu_ly" / "yolov8m.pt"
DEFAULT_CONFIG_PATH = BASE_DIR / "lane_config.json"
OUTPUTS_DIR = BASE_DIR / "outputs"
VIOLATING_VEHICLES_DIR = OUTPUTS_DIR / "violating_vehicles"
REVIEWS_DIR = OUTPUTS_DIR / "reviews"
CSV_PATH = OUTPUTS_DIR / "violations.csv"
DB_PATH = OUTPUTS_DIR / "violations.db"
SUMMARY_PATH = OUTPUTS_DIR / "summary.txt"
TRACK_HISTORY_PATH = OUTPUTS_DIR / "track_zone_history.csv"
DEBUG_VIDEO_PATH = OUTPUTS_DIR / "debug_wrong_lane.mp4"

VEHICLE_LABELS = {2: "Car", 3: "Motorcycle", 5: "Bus", 7: "Truck"}
VIOLATION_TITLE = "Khong chap hanh hieu lenh, chi dan cua bien bao hieu, vach ke duong"
LAW_SOURCE = "Nghi dinh 168/2024/ND-CP"
FINE_BY_GROUP = {
    "car": "400.000 - 600.000 VND",
    "motorcycle": "100.000 - 200.000 VND",
}
VIOLATION_RULES = {
    ("left", "straight"): {
        "case_id": "CASE_3",
        "behavior": "Di thang tren lan duong bat buoc re trai",
        "legal_name": VIOLATION_TITLE,
    },
    ("straight", "turn_left"): {
        "case_id": "CASE_4",
        "behavior": "Re trai tren lan duong bat buoc di thang",
        "legal_name": VIOLATION_TITLE,
    },
}
ZONE_NUMBER = {
    "left": "1",
    "straight": "2",
    "left_exit": "3",
    "straight_exit": "4",
    "far_straight_exit": "5",
}


def configure_output_dir(output_dir):
    global OUTPUTS_DIR
    global VIOLATING_VEHICLES_DIR
    global REVIEWS_DIR
    global CSV_PATH
    global DB_PATH
    global SUMMARY_PATH
    global TRACK_HISTORY_PATH
    global DEBUG_VIDEO_PATH

    OUTPUTS_DIR = Path(output_dir)
    VIOLATING_VEHICLES_DIR = OUTPUTS_DIR / "violating_vehicles"
    REVIEWS_DIR = OUTPUTS_DIR / "reviews"
    CSV_PATH = OUTPUTS_DIR / "violations.csv"
    DB_PATH = OUTPUTS_DIR / "violations.db"
    SUMMARY_PATH = OUTPUTS_DIR / "summary.txt"
    TRACK_HISTORY_PATH = OUTPUTS_DIR / "track_zone_history.csv"
    DEBUG_VIDEO_PATH = OUTPUTS_DIR / "debug_wrong_lane.mp4"


@dataclass
class TrackPoint:
    frame_id: int
    center: tuple
    bottom_center: tuple
    bbox: tuple
    lane: str = ""


@dataclass
class TrackState:
    track_id: int
    vehicle_type: str
    points: list = field(default_factory=list)
    best_crop: object = None
    best_frame: object = None
    best_bbox: tuple = None
    best_score: float = 0.0
    best_frame_id: int = -1
    best_sharpness: float = 0.0
    best_area_ratio: float = 0.0
    last_frame_image: object = None
    last_bbox: tuple = None


@dataclass
class ViolationRecord:
    violation_id: int
    track_id: int
    vehicle_type: str
    case_id: str
    start_lane: str
    direction: str
    legal_name: str
    behavior: str
    fine_range: str
    law_source: str
    reason: str
    evidence: str
    vehicle_image_frame: int
    vehicle_image_score: float
    vehicle_image_sharpness: float
    first_frame: int
    last_frame: int
    vehicle_path: str
    review_path: str
    timestamp: str


def ensure_output_dirs(clear=True):
    OUTPUTS_DIR.mkdir(exist_ok=True)
    VIOLATING_VEHICLES_DIR.mkdir(exist_ok=True)
    REVIEWS_DIR.mkdir(exist_ok=True)
    if clear:
        for folder in (VIOLATING_VEHICLES_DIR, REVIEWS_DIR):
            for image_path in folder.glob("*.png"):
                image_path.unlink()
        for text_path in REVIEWS_DIR.glob("*.txt"):
            text_path.unlink()


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def scale_polygon(points, width, height):
    return np.array(
        [[int(x * width), int(y * height)] for x, y in points],
        dtype=np.int32,
    )


def build_lane_polygons(config, width, height):
    return {
        lane_id: scale_polygon(lane["polygon"], width, height)
        for lane_id, lane in config["lanes"].items()
    }


def build_direction_zones(config, width, height):
    return {
        zone_id: scale_polygon(points, width, height)
        for zone_id, points in config.get("direction_zones", {}).items()
    }


def point_in_polygon(point, polygon):
    return cv2.pointPolygonTest(polygon, point, False) >= 0


def lane_for_point(point, lane_polygons):
    for lane_id, polygon in lane_polygons.items():
        if point_in_polygon(point, polygon):
            return lane_id
    return ""


def direction_for_zone(zone_id):
    if zone_id == "left_exit":
        return "turn_left"
    if zone_id == "far_straight_exit":
        return "straight"
    if zone_id == "right_exit":
        return "turn_right"
    return ""


def zone_for_point(point, direction_zones):
    for zone_id, polygon in direction_zones.items():
        if point_in_polygon(point.center, polygon):
            return zone_id
    return ""


def zones_for_point(point, direction_zones):
    return [
        zone_id
        for zone_id, polygon in direction_zones.items()
        if point_in_polygon(point.center, polygon)
    ]


def final_exit_zone(points, direction_zones, min_terminal_points=3, terminal_window=12):
    terminal_points = points[-terminal_window:]
    zone_counts = {}
    zone_last_frame = {}
    for point in terminal_points:
        for zone_id, polygon in direction_zones.items():
            if point_in_polygon(point.center, polygon):
                zone_counts[zone_id] = zone_counts.get(zone_id, 0) + 1
                zone_last_frame[zone_id] = point.frame_id
    if not zone_counts:
        return "", -1, 0, len(terminal_points)
    zone_id, count = max(zone_counts.items(), key=lambda item: item[1])
    if count < min_terminal_points:
        return "", -1, count, len(terminal_points)
    return zone_id, zone_last_frame[zone_id], count, len(terminal_points)


def direction_from_exit_zone(points, direction_zones):
    zone_id, _, _, _ = final_exit_zone(points, direction_zones)
    return direction_for_zone(zone_id)


def exit_zone_for_points(points, direction_zones):
    zone_id, frame_id, _, _ = final_exit_zone(points, direction_zones)
    return zone_id, frame_id


def compact_sequence(values):
    sequence = []
    for value in values:
        if value and (not sequence or sequence[-1] != value):
            sequence.append(value)
    return sequence


def zone_sequence(points, direction_zones):
    return compact_sequence(zone_for_point(point, direction_zones) for point in points)


def direct_zone_sequence(points, direction_zones):
    sequence = []
    previous_zones = set()
    for point in points:
        current_zones = set(zones_for_point(point, direction_zones))
        for zone_id in direction_zones:
            if zone_id not in current_zones or zone_id in previous_zones:
                continue
            if not sequence or sequence[-1] != zone_id:
                sequence.append(zone_id)
        previous_zones = current_zones
    return sequence


def touched_zone_counts(points, direction_zones):
    counts = {}
    for point in points:
        for zone_id in zones_for_point(point, direction_zones):
            counts[zone_id] = counts.get(zone_id, 0) + 1
    return counts


def zone_frame_ranges(points, direction_zones):
    ranges = {}
    active = {}
    for point in points:
        touched = set(zones_for_point(point, direction_zones))
        for zone_id in touched:
            active.setdefault(zone_id, point.frame_id)
        for zone_id in list(active):
            if zone_id not in touched:
                ranges.setdefault(zone_id, []).append((active.pop(zone_id), point.frame_id - 1))
    for zone_id, start_frame in active.items():
        ranges.setdefault(zone_id, []).append((start_frame, points[-1].frame_id))
    return ranges


def track_touches_zone(points, direction_zones, zone_id):
    polygon = direction_zones.get(zone_id)
    if polygon is None:
        return False
    return any(point_in_polygon(point.center, polygon) for point in points)


def lane_sequence(points):
    return compact_sequence(point.lane for point in points)


def numbered_sequence(lanes, zones):
    sequence = []
    for value in lanes + zones:
        number = ZONE_NUMBER.get(value)
        if number and (not sequence or sequence[-1] != number):
            sequence.append(number)
    return sequence


def sequence_has_order(sequence, ordered_values):
    if not ordered_values:
        return True
    index = 0
    for value in sequence:
        if value == ordered_values[index]:
            index += 1
            if index == len(ordered_values):
                return True
    return False


def clamp_box(box, width, height):
    x1, y1, x2, y2 = map(int, box)
    return (
        max(0, min(x1, width - 1)),
        max(0, min(y1, height - 1)),
        max(0, min(x2, width)),
        max(0, min(y2, height)),
    )


def expand_box(box, width, height, ratio=0.10):
    x1, y1, x2, y2 = clamp_box(box, width, height)
    pad_x = int((x2 - x1) * ratio)
    pad_y = int((y2 - y1) * ratio)
    return clamp_box((x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y), width, height)


def crop_vehicle(frame, bbox):
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = expand_box(bbox, width, height)
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2].copy()


def image_sharpness(image):
    if image is None or image.size == 0:
        return 0.0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def vehicle_image_score(crop, bbox, frame_width, frame_height):
    if crop is None or crop.size == 0:
        return 0.0, 0.0, 0.0
    x1, y1, x2, y2 = bbox
    area = max(0, x2 - x1) * max(0, y2 - y1)
    area_ratio = area / max(1, frame_width * frame_height)
    sharpness = image_sharpness(crop)
    area_score = min(area_ratio / 0.16, 1.0)
    sharpness_score = min(sharpness / 900.0, 1.0)
    score = area_score * 0.65 + sharpness_score * 0.35
    return score, sharpness, area_ratio


def classify_direction(points, frame_width, config, direction_zones=None):
    if len(points) < 2:
        return "unknown"
    if direction_zones:
        zone_direction = direction_from_exit_zone(points, direction_zones)
        if zone_direction:
            return zone_direction
        if config.get("direction", {}).get("require_exit_zone", False):
            return "unknown"
    first = points[0].center
    last = points[-1].center
    dx = last[0] - first[0]
    dy = last[1] - first[1]
    direction_cfg = config.get("direction", {})
    min_move = frame_width * direction_cfg.get("min_track_displacement_ratio", 0.04)
    if abs(dx) + abs(dy) < min_move:
        return "unknown"

    turn_dx = frame_width * direction_cfg.get("turn_dx_ratio", 0.08)
    straight_dx = frame_width * direction_cfg.get("straight_dx_ratio", 0.06)
    left_turn_dx_negative = direction_cfg.get("left_turn_dx_negative", True)
    left_turn = dx <= -turn_dx if left_turn_dx_negative else dx >= turn_dx

    if left_turn:
        return "turn_left"
    if abs(dx) <= straight_dx:
        return "straight"
    return "turn_right"


def dominant_start_lane(points):
    lane_counts = {}
    first_points = points[: max(3, min(10, len(points) // 3))]
    for point in first_points:
        if point.lane:
            lane_counts[point.lane] = lane_counts.get(point.lane, 0) + 1
    if not lane_counts:
        return ""
    return max(lane_counts.items(), key=lambda item: item[1])[0]


def start_lane_evidence(points):
    first_points = points[: max(3, min(10, len(points) // 3))]
    lane_counts = {}
    for point in first_points:
        if point.lane:
            lane_counts[point.lane] = lane_counts.get(point.lane, 0) + 1
    if not lane_counts:
        return "", 0, len(first_points)
    lane, count = max(lane_counts.items(), key=lambda item: item[1])
    return lane, count, len(first_points)


def vehicle_group(vehicle_type):
    if vehicle_type == "Motorcycle":
        return "motorcycle"
    return "car"


def fine_for_vehicle(vehicle_type):
    return FINE_BY_GROUP[vehicle_group(vehicle_type)]


def violation_rule_for(start_lane, direction):
    return VIOLATION_RULES.get((start_lane, direction))


def lane_near_stop_line(points, config, frame_height):
    stop_y = config.get("stop_line_y", 0.58) * frame_height
    tolerance = config.get("lane_change", {}).get("near_stop_line_ratio", 0.08) * frame_height
    candidates = [
        point for point in points
        if point.lane and abs(point.bottom_center[1] - stop_y) <= tolerance
    ]
    if not candidates:
        return ""
    return candidates[-1].lane


def evaluate_track(track, config, frame_width, frame_height, direction_zones=None):
    if len(track.points) < 2:
        return None
    start_lane, start_votes, start_total = start_lane_evidence(track.points)
    if not start_lane or start_lane not in config["lanes"]:
        return None

    priority_zones = zone_sequence(track.points, direction_zones or {})
    zones = direct_zone_sequence(track.points, direction_zones or {})
    lanes = lane_sequence(track.points)
    numbers = numbered_sequence(lanes, zones)
    direction = classify_direction(track.points, frame_width, config, direction_zones)
    exit_zone, exit_frame, exit_votes, exit_total = final_exit_zone(track.points, direction_zones or {})

    touched_far_straight = track_touches_zone(track.points, direction_zones or {}, "far_straight_exit")
    if start_lane == "left" and touched_far_straight:
        rule = violation_rule_for("left", "straight")
        evidence = (
            f"start_lane={start_lane} ({start_votes}/{start_total} diem dau); "
            f"sequence={'>'.join(numbers) or 'unknown'}; "
            f"zone_sequence={'>'.join(zones) or 'unknown'}; "
            f"priority_zone_sequence={'>'.join(priority_zones) or 'unknown'}; "
            "touched_zone=far_straight_exit"
        )
        return {
            "start_lane": "left",
            "direction": "straight",
            "case_id": rule["case_id"],
            "legal_name": rule["legal_name"],
            "behavior": rule["behavior"],
            "reason": rule["behavior"],
            "evidence": evidence,
        }

    if (
        start_lane == "left"
        and direction == "turn_left"
        and (
            "straight" in lanes[1:]
            or "straight_exit" in zones
            or sequence_has_order(zones, ["straight_exit", "left_exit"])
        )
    ):
        evidence = (
            f"start_lane={start_lane} ({start_votes}/{start_total} diem dau); "
            f"sequence={'>'.join(numbers) or 'unknown'}; "
            f"lane_sequence={'>'.join(lanes) or 'unknown'}; "
            f"zone_sequence={'>'.join(zones) or 'unknown'}; "
            f"final_exit_zone={exit_zone or 'unknown'} ({exit_votes}/{exit_total} diem cuoi)"
        )
        return {
            "start_lane": start_lane,
            "direction": direction,
            "case_id": "CASE_6",
            "legal_name": VIOLATION_TITLE,
            "behavior": "Doi sang lan di thang hoac vung di thang roi moi re trai",
            "reason": "Doi sang lan di thang hoac vung di thang roi moi re trai",
            "evidence": evidence,
        }

    allowed = set(config["lanes"][start_lane].get("allowed", []))
    if direction != "unknown" and direction not in allowed:
        rule = violation_rule_for(start_lane, direction)
        if rule is None:
            return None
        evidence = (
            f"start_lane={start_lane} ({start_votes}/{start_total} diem dau); "
            f"sequence={'>'.join(numbers) or 'unknown'}; "
            f"direction={direction}; final_exit_zone={exit_zone or 'unknown'} "
            f"({exit_votes}/{exit_total} diem cuoi); "
            f"exit_frame={exit_frame if exit_frame >= 0 else 'unknown'}; "
            f"zone_sequence={'>'.join(zones) or 'unknown'}"
        )
        return {
            "start_lane": start_lane,
            "direction": direction,
            "case_id": rule["case_id"],
            "legal_name": rule["legal_name"],
            "behavior": rule["behavior"],
            "reason": rule["behavior"],
            "evidence": evidence,
        }

    return None


def draw_lanes(frame, lane_polygons, config):
    overlay = frame.copy()
    colors = {"left": (0, 180, 255), "straight": (0, 200, 0)}
    for lane_id, polygon in lane_polygons.items():
        color = colors.get(lane_id, (255, 255, 0))
        cv2.fillPoly(overlay, [polygon], color)
        cv2.polylines(frame, [polygon], True, color, 3)
        label_point = tuple(polygon.mean(axis=0).astype(int))
        cv2.putText(
            frame,
            config["lanes"][lane_id].get("name", lane_id),
            label_point,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
            cv2.LINE_AA,
        )
    cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, frame)
    stop_y = int(config.get("stop_line_y", 0.58) * frame.shape[0])
    cv2.line(frame, (0, stop_y), (frame.shape[1], stop_y), (255, 255, 255), 2)


def draw_direction_zones(frame, direction_zones):
    colors = {
        "left_exit": (255, 180, 0),
        "straight_exit": (80, 220, 80),
        "far_straight_exit": (120, 255, 255),
        "right_exit": (255, 120, 255),
    }
    for zone_id, polygon in direction_zones.items():
        color = colors.get(zone_id, (180, 180, 180))
        cv2.polylines(frame, [polygon], True, color, 2)
        label_point = tuple(polygon.mean(axis=0).astype(int))
        cv2.putText(frame, zone_id, label_point, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)


def draw_track_path(frame, points, color=(0, 0, 255)):
    if len(points) < 2:
        return
    centers = [point.center for point in points]
    for p1, p2 in zip(centers, centers[1:]):
        cv2.line(frame, p1, p2, color, 3)
    cv2.circle(frame, centers[0], 7, (255, 255, 255), -1)
    cv2.circle(frame, centers[-1], 7, color, -1)


def save_vehicle_image(record_id, track):
    if track.best_crop is None:
        return ""
    path = VIOLATING_VEHICLES_DIR / f"vehicle_V{record_id:03d}_track_{track.track_id}.png"
    cv2.imwrite(str(path), track.best_crop)
    return str(path)


def create_review_image(record, track, lane_polygons, direction_zones, config):
    review_source = track.last_frame_image if track.last_frame_image is not None else track.best_frame
    review_bbox = track.last_bbox if track.last_bbox is not None else track.best_bbox
    if review_source is None:
        return ""
    frame = review_source.copy()
    draw_lanes(frame, lane_polygons, config)
    draw_direction_zones(frame, direction_zones)
    draw_track_path(frame, track.points)
    if review_bbox:
        x1, y1, x2, y2 = review_bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
    path = REVIEWS_DIR / f"review_V{record.violation_id:03d}_track_{track.track_id}.png"
    cv2.imwrite(str(path), frame)

    detail_path = REVIEWS_DIR / f"review_V{record.violation_id:03d}_track_{track.track_id}.txt"
    lines = [
        f"V{record.violation_id:03d} | {record.case_id} | Track {track.track_id} | {track.vehicle_type}",
        f"Lan xuat phat: {record.start_lane} | Huong: {record.direction}",
        f"Frame review: frame cuoi track {record.last_frame}",
        f"Hanh vi: {record.behavior}",
        f"Muc phat: {record.fine_range}",
        f"Anh xe ro nhat: frame {record.vehicle_image_frame} | score {record.vehicle_image_score:.3f} | sharp {record.vehicle_image_sharpness:.1f}",
        f"Bang chung: {record.evidence}",
    ]
    with open(detail_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return str(path)


def create_connection(db_path):
    conn = sqlite3.connect(db_path)
    conn.execute("DROP TABLE IF EXISTS violations")
    conn.execute(
        """
        CREATE TABLE violations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            violation_id INTEGER,
            track_id INTEGER,
            vehicle_type TEXT,
            case_id TEXT,
            start_lane TEXT,
            direction TEXT,
            legal_name TEXT,
            behavior TEXT,
            fine_range TEXT,
            law_source TEXT,
            reason TEXT,
            evidence TEXT,
            vehicle_image_frame INTEGER,
            vehicle_image_score REAL,
            vehicle_image_sharpness REAL,
            first_frame INTEGER,
            last_frame INTEGER,
            vehicle_path TEXT,
            review_path TEXT,
            timestamp TEXT
        )
        """
    )
    conn.commit()
    return conn


def save_records(records):
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Ma Vi Pham", "Track ID", "Loai Phuong Tien", "Ma Truong Hop",
            "Lan Xuat Phat", "Huong Di Chuyen", "Ten Loi Phap Luat",
            "Hanh Vi Chi Tiet", "Muc Phat", "Can Cu", "Bang Chung",
            "Frame Anh Xe Ro Nhat", "Diem Anh Xe", "Do Net Anh Xe",
            "Frame Dau", "Frame Cuoi",
            "Anh Xe Vi Pham", "Anh Review", "Thoi Gian",
        ])
        for record in records:
            writer.writerow([
                f"V{record.violation_id:03d}", record.track_id, record.vehicle_type,
                record.case_id, record.start_lane, record.direction,
                record.legal_name, record.behavior, record.fine_range,
                record.law_source, record.evidence,
                record.vehicle_image_frame,
                f"{record.vehicle_image_score:.4f}",
                f"{record.vehicle_image_sharpness:.1f}",
                record.first_frame, record.last_frame, record.vehicle_path,
                record.review_path, record.timestamp,
            ])

    with create_connection(DB_PATH) as conn:
        for record in records:
            conn.execute(
                """
                INSERT INTO violations (
                    violation_id, track_id, vehicle_type, case_id, start_lane,
                    direction, legal_name, behavior, fine_range, law_source,
                    reason, evidence, vehicle_image_frame, vehicle_image_score,
                    vehicle_image_sharpness, first_frame, last_frame, vehicle_path,
                    review_path, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.violation_id, record.track_id, record.vehicle_type,
                    record.case_id, record.start_lane, record.direction,
                    record.legal_name, record.behavior, record.fine_range,
                    record.law_source, record.reason, record.evidence,
                    record.vehicle_image_frame, record.vehicle_image_score,
                    record.vehicle_image_sharpness,
                    record.first_frame, record.last_frame,
                    record.vehicle_path, record.review_path, record.timestamp,
                ),
            )
        conn.commit()


def write_track_zone_history(tracks, direction_zones, min_track_frames):
    with open(TRACK_HISTORY_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Track ID", "Loai Phuong Tien", "Hop Le", "So Diem",
            "Frame Dau", "Frame Cuoi",
            "Lan Xuat Phat", "Bang Chung Lan Xuat Phat",
            "Chuoi Lan", "Chuoi Vung Uu Tien",
            "Chuoi Vung Quet Truc Tiep", "Chuoi So",
            "So Diem Tung Vung", "Khoang Frame Tung Vung",
            "Vung Dich Cuoi", "Bang Chung Vung Dich Cuoi",
        ])
        for track_id in sorted(tracks):
            track = tracks[track_id]
            if not track.points:
                continue
            lanes = lane_sequence(track.points)
            priority_zones = zone_sequence(track.points, direction_zones)
            direct_zones = direct_zone_sequence(track.points, direction_zones)
            numbers = numbered_sequence(lanes, direct_zones)
            start_lane, start_votes, start_total = start_lane_evidence(track.points)
            exit_zone, exit_frame, exit_votes, exit_total = final_exit_zone(track.points, direction_zones)
            zone_counts = touched_zone_counts(track.points, direction_zones)
            frame_ranges = zone_frame_ranges(track.points, direction_zones)
            zone_count_text = "; ".join(
                f"{ZONE_NUMBER.get(zone_id, zone_id)}:{zone_id}={zone_counts[zone_id]}"
                for zone_id in direction_zones
                if zone_id in zone_counts
            )
            frame_range_text = "; ".join(
                f"{ZONE_NUMBER.get(zone_id, zone_id)}:{zone_id}="
                + "|".join(f"{start}-{end}" for start, end in frame_ranges[zone_id])
                for zone_id in direction_zones
                if zone_id in frame_ranges
            )
            writer.writerow([
                track.track_id,
                track.vehicle_type,
                "yes" if len(track.points) >= min_track_frames else "no",
                len(track.points),
                track.points[0].frame_id,
                track.points[-1].frame_id,
                start_lane or "unknown",
                f"{start_votes}/{start_total}",
                ">".join(lanes) or "unknown",
                ">".join(priority_zones) or "unknown",
                ">".join(direct_zones) or "unknown",
                ">".join(numbers) or "unknown",
                zone_count_text or "none",
                frame_range_text or "none",
                exit_zone or "unknown",
                f"{exit_votes}/{exit_total}; frame={exit_frame if exit_frame >= 0 else 'unknown'}",
            ])


def write_summary(records, total_tracks):
    case_counts = {}
    for record in records:
        case_counts[record.case_id] = case_counts.get(record.case_id, 0) + 1

    lines = [
        "BAO CAO PHAT HIEN XE DI SAI LAN",
        f"Thoi gian chay: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Tong track hop le: {total_tracks}",
        f"Tong phuong tien vi pham: {len(records)}",
        "",
        "Thong ke theo truong hop:",
    ]
    if case_counts:
        for case_id in sorted(case_counts):
            lines.append(f"- {case_id}: {case_counts[case_id]}")
    else:
        lines.append("- Khong co vi pham")

    lines.extend(
        [
            "",
            "Luat ap dung:",
            f"- Can cu: {LAW_SOURCE}",
            f"- Ten loi: {VIOLATION_TITLE}",
            f"- O to/xe tai/xe khach: {FINE_BY_GROUP['car']}",
            f"- Xe may: {FINE_BY_GROUP['motorcycle']}",
            "",
            "File ket qua:",
            f"- CSV: {CSV_PATH}",
            f"- DB: {DB_PATH}",
            f"- Lich su vung theo track: {TRACK_HISTORY_PATH}",
            f"- Anh xe vi pham: {VIOLATING_VEHICLES_DIR}",
            f"- Anh review: {REVIEWS_DIR}",
        ]
    )
    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def process_video(args):
    ensure_output_dirs(clear=not args.append)
    config = load_config(args.config)
    model_path = args.model
    if not model_path.exists() and FALLBACK_MODEL_PATH.exists():
        model_path = FALLBACK_MODEL_PATH
    if not args.video.exists():
        raise FileNotFoundError(f"Khong tim thay video: {args.video}")
    if not model_path.exists():
        raise FileNotFoundError(f"Khong tim thay model YOLO: {model_path}")

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise FileNotFoundError(f"Khong mo duoc video: {args.video}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    lane_polygons = build_lane_polygons(config, width, height)
    direction_zones = build_direction_zones(config, width, height)
    model = YOLO(str(model_path))
    writer = None
    if not args.no_video:
        DEBUG_VIDEO_PATH.parent.mkdir(exist_ok=True)
        writer = cv2.VideoWriter(
            str(DEBUG_VIDEO_PATH),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )

    tracks = {}
    frame_id = 0
    while cap.isOpened():
        if args.max_frames and frame_id >= args.max_frames:
            break
        ret, frame = cap.read()
        if not ret:
            break
        clean_frame = frame.copy()
        results = model.track(
            clean_frame,
            conf=args.conf,
            iou=args.iou,
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
                x1, y1, x2, y2 = clamp_box(box.xyxy[0], width, height)
                if x2 <= x1 or y2 <= y1:
                    continue
                track_id = int(box.id[0])
                center = ((x1 + x2) // 2, (y1 + y2) // 2)
                bottom_center = ((x1 + x2) // 2, y2)
                vehicle_type = VEHICLE_LABELS[cls]
                lane = lane_for_point(center, lane_polygons)
                if track_id not in tracks and not lane:
                    continue
                track = tracks.setdefault(track_id, TrackState(track_id, vehicle_type))
                track.vehicle_type = vehicle_type
                track.points.append(TrackPoint(frame_id, center, bottom_center, (x1, y1, x2, y2), lane))
                track.last_bbox = (x1, y1, x2, y2)
                track.last_frame_image = clean_frame.copy()

                vehicle_crop = crop_vehicle(clean_frame, (x1, y1, x2, y2))
                score, sharpness, area_ratio = vehicle_image_score(
                    vehicle_crop, (x1, y1, x2, y2), width, height
                )
                if score > track.best_score:
                    track.best_score = score
                    track.best_frame_id = frame_id
                    track.best_sharpness = sharpness
                    track.best_area_ratio = area_ratio
                    track.best_bbox = (x1, y1, x2, y2)
                    track.best_crop = vehicle_crop
                    track.best_frame = clean_frame.copy()

                color = (0, 0, 255) if lane == "straight" else (0, 180, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"ID {track_id} {vehicle_type} {lane}", (x1, max(24, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)

        if writer is not None:
            draw_lanes(frame, lane_polygons, config)
            draw_direction_zones(frame, direction_zones)
            writer.write(frame)
        frame_id += 1

    cap.release()
    if writer is not None:
        writer.release()

    records = []
    for track in tracks.values():
        if len(track.points) < args.min_track_frames:
            continue
        violation = evaluate_track(track, config, width, height, direction_zones)
        if violation is None:
            continue
        violation_id = len(records) + 1
        vehicle_path = save_vehicle_image(violation_id, track)
        record = ViolationRecord(
            violation_id=violation_id,
            track_id=track.track_id,
            vehicle_type=track.vehicle_type,
            case_id=violation["case_id"],
            start_lane=violation["start_lane"],
            direction=violation["direction"],
            legal_name=violation["legal_name"],
            behavior=violation["behavior"],
            fine_range=fine_for_vehicle(track.vehicle_type),
            law_source=LAW_SOURCE,
            reason=f'{violation["legal_name"]}: {violation["reason"]}',
            evidence=violation["evidence"],
            vehicle_image_frame=track.best_frame_id,
            vehicle_image_score=track.best_score,
            vehicle_image_sharpness=track.best_sharpness,
            first_frame=track.points[0].frame_id,
            last_frame=track.points[-1].frame_id,
            vehicle_path=vehicle_path,
            review_path="",
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )
        record.review_path = create_review_image(record, track, lane_polygons, direction_zones, config)
        records.append(record)

    save_records(records)
    write_track_zone_history(tracks, direction_zones, args.min_track_frames)
    write_summary(records, len(tracks))
    print(f"[INFO] Tong track hop le: {len(tracks)}")
    print(f"[INFO] Tong phuong tien vi pham: {len(records)}")
    print(f"[INFO] Anh xe vi pham: {VIOLATING_VEHICLES_DIR}")
    print(f"[INFO] Anh review: {REVIEWS_DIR}")
    print(f"[INFO] CSV: {CSV_PATH}")
    print(f"[INFO] DB: {DB_PATH}")
    print(f"[INFO] Lich su vung theo track: {TRACK_HISTORY_PATH}")
    print(f"[INFO] Bao cao: {SUMMARY_PATH}")
    if not args.no_video:
        print(f"[INFO] Video debug: {DEBUG_VIDEO_PATH}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Phat hien xe di sai lan dua tren lane polygon, tracking va vector quy dao."
    )
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO_PATH)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--conf", type=float, default=0.20)
    parser.add_argument("--iou", type=float, default=0.50)
    parser.add_argument("--min-track-frames", type=int, default=6)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUTS_DIR,
        help="Thu muc luu CSV, DB, anh bang chung va video debug.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Gioi han so frame de chay nhanh khi test/demo. 0 la xu ly het video.",
    )
    parser.add_argument("--no-video", action="store_true", help="Khong xuat video debug.")
    parser.add_argument("--append", action="store_true", help="Khong xoa anh output cu truoc khi chay.")
    return parser.parse_args()


def main():
    args = parse_args()
    configure_output_dir(args.output_dir)
    try:
        process_video(args)
    except Exception as exc:
        print(f"[ERROR] {exc}")


if __name__ == "__main__":
    main()
