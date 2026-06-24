import argparse
import base64
import json
import math
import shutil
import tkinter as tk
from datetime import datetime
from pathlib import Path

import cv2

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
DEFAULT_VIDEO_PATH = ROOT_DIR / "find_license_plate" / "data" / "video.mp4"
DEFAULT_CONFIG_PATH = BASE_DIR / "lane_config.json"
DEFAULT_FRAME_ID = 100
WINDOW_NAME = "Wrong lane config calibrator"

REGION_ORDER = ["left", "straight", "left_exit", "straight_exit", "far_straight_exit"]
REGION_LABELS = {
    "left": "1 - lan re trai",
    "straight": "2 - lan di thang",
    "left_exit": "3 - vung dich re trai",
    "straight_exit": "4 - vung dich di thang",
    "far_straight_exit": "5 - vung di thang phia xa",
}
REGION_COLORS = {
    "left": (0, 180, 255),
    "straight": (0, 220, 0),
    "left_exit": (255, 180, 0),
    "straight_exit": (80, 220, 80),
    "far_straight_exit": (120, 255, 255),
}


class Calibrator:
    def __init__(self, frame, config, config_path, max_display_width):
        self.frame = frame
        self.config = config
        self.config_path = config_path
        self.height, self.width = frame.shape[:2]
        self.scale = min(1.0, max_display_width / self.width)
        self.display_size = (int(self.width * self.scale), int(self.height * self.scale))
        self.selected = "left"
        self.points = self.load_points_from_config()
        self.mouse_xy = None
        self.status = "Ctrl+1/2/3/4 de chon vung. Keo dinh/canh de sua polygon."
        self.root = None
        self.canvas = None
        self.photo = None
        self.drag_index = None
        self.drag_threshold = 14

    def load_points_from_config(self):
        points = {region: [] for region in REGION_ORDER}
        lanes = self.config.get("lanes", {})
        zones = self.config.get("direction_zones", {})
        for region in ("left", "straight"):
            for x, y in lanes.get(region, {}).get("polygon", []):
                points[region].append((int(x * self.width), int(y * self.height)))
        for region in ("left_exit", "straight_exit", "far_straight_exit"):
            for x, y in zones.get(region, []):
                points[region].append((int(x * self.width), int(y * self.height)))
        return points

    def display_to_original(self, x, y):
        return int(x / self.scale), int(y / self.scale)

    def original_to_display(self, point):
        x, y = point
        return int(x * self.scale), int(y * self.scale)

    def select_region(self, region):
        self.selected = region
        self.drag_index = None
        self.status = f"Dang chon {self.selected}. Keo dinh hoac keo canh de them diem."
        self.redraw()

    def clear_selected(self):
        self.points[self.selected] = []
        self.status = f"Da xoa {self.selected}. Click de tao cac diem moi."
        self.redraw()

    def clear_all(self):
        for region in REGION_ORDER:
            self.points[region] = []
        self.selected = REGION_ORDER[0]
        self.status = "Da xoa tat ca. Bat dau lai tu vung 1."
        self.redraw()

    def normalized_points(self, region):
        normalized = []
        for x, y in self.points[region]:
            nx = round(max(0.0, min(1.0, x / self.width)), 4)
            ny = round(max(0.0, min(1.0, y / self.height)), 4)
            normalized.append([nx, ny])
        return normalized

    def save(self):
        missing = [region for region in REGION_ORDER if len(self.points[region]) < 3]
        if missing:
            self.status = "Chua luu: moi vung can toi thieu 3 diem. Kiem tra: " + ", ".join(missing)
            print("[WARN]", self.status)
            return False

        backup_path = self.config_path.with_suffix(
            f".backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        shutil.copy2(self.config_path, backup_path)

        self.config.setdefault("lanes", {})
        self.config.setdefault("direction_zones", {})
        self.config["lanes"].setdefault("left", {"name": "Lan re trai", "allowed": ["turn_left", "u_turn"]})
        self.config["lanes"].setdefault("straight", {"name": "Lan di thang", "allowed": ["straight"]})
        self.config["lanes"]["left"]["polygon"] = self.normalized_points("left")
        self.config["lanes"]["straight"]["polygon"] = self.normalized_points("straight")
        self.config["direction_zones"]["left_exit"] = self.normalized_points("left_exit")
        self.config["direction_zones"]["straight_exit"] = self.normalized_points("straight_exit")
        self.config["direction_zones"]["far_straight_exit"] = self.normalized_points("far_straight_exit")

        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)
            f.write("\n")
        print(f"[INFO] Da luu config: {self.config_path}")
        print(f"[INFO] Backup config cu: {backup_path}")
        self.status = "Da luu config thanh cong."
        self.redraw()
        return True

    def run(self):
        self.root = tk.Tk()
        self.root.title(WINDOW_NAME)
        self.canvas = tk.Canvas(self.root, width=self.display_size[0], height=self.display_size[1])
        self.canvas.pack()

        self.canvas.bind("<ButtonPress-1>", self.on_left_press)
        self.canvas.bind("<B1-Motion>", self.on_left_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_left_release)
        self.canvas.bind("<Button-3>", self.on_right_click)
        self.root.bind("<Control-KeyPress-1>", lambda event: self.select_region("left"))
        self.root.bind("<Control-KeyPress-2>", lambda event: self.select_region("straight"))
        self.root.bind("<Control-KeyPress-3>", lambda event: self.select_region("left_exit"))
        self.root.bind("<Control-KeyPress-4>", lambda event: self.select_region("straight_exit"))
        self.root.bind("<Control-KeyPress-5>", lambda event: self.select_region("far_straight_exit"))
        self.root.bind("<KeyPress-c>", lambda event: self.clear_selected())
        self.root.bind("<KeyPress-r>", lambda event: self.clear_all())
        self.root.bind("<KeyPress-s>", lambda event: self.save())
        self.root.bind("<KeyPress-q>", lambda event: self.root.destroy())
        self.root.bind("<Escape>", lambda event: self.root.destroy())

        self.redraw()
        self.root.mainloop()

    def on_left_press(self, event):
        original = self.display_to_original(event.x, event.y)
        self.mouse_xy = original
        vertex_index = self.nearest_vertex_index(event.x, event.y)
        if vertex_index is not None:
            self.drag_index = vertex_index
            self.status = f"Dang keo diem {vertex_index + 1} cua {self.selected}."
            self.update_drag_point(event.x, event.y)
            return

        edge_index = self.nearest_edge_insert_index(event.x, event.y)
        if edge_index is not None:
            self.points[self.selected].insert(edge_index, original)
            self.drag_index = edge_index
            self.status = f"Da chen diem moi vao canh cua {self.selected}. Keo de dat vi tri."
            self.update_drag_point(event.x, event.y)
            return

        self.points[self.selected].append(original)
        self.drag_index = len(self.points[self.selected]) - 1
        self.status = f"Da them diem {self.drag_index + 1} cho {self.selected}."
        self.redraw()

    def on_left_drag(self, event):
        if self.drag_index is None:
            return
        self.update_drag_point(event.x, event.y)

    def on_left_release(self, event):
        if self.drag_index is not None:
            self.update_drag_point(event.x, event.y)
            self.status = f"{self.selected}: {len(self.points[self.selected])} diem. Bam s de luu."
        self.drag_index = None
        self.redraw()

    def update_drag_point(self, x, y):
        if self.drag_index is None:
            return
        original = self.display_to_original(x, y)
        ox = max(0, min(self.width - 1, original[0]))
        oy = max(0, min(self.height - 1, original[1]))
        self.points[self.selected][self.drag_index] = (ox, oy)
        self.mouse_xy = (ox, oy)
        self.redraw()

    def on_right_click(self, event):
        vertex_index = self.nearest_vertex_index(event.x, event.y)
        if vertex_index is not None:
            self.points[self.selected].pop(vertex_index)
            self.status = f"Da xoa diem {vertex_index + 1} cua {self.selected}."
            self.redraw()
        elif self.points[self.selected]:
            self.points[self.selected].pop()
            self.status = f"Da xoa diem cuoi cua {self.selected}."
            self.redraw()

    def nearest_vertex_index(self, x, y):
        best_index = None
        best_distance = self.drag_threshold
        for index, point in enumerate(self.points[self.selected]):
            px, py = self.original_to_display(point)
            distance = math.hypot(px - x, py - y)
            if distance <= best_distance:
                best_distance = distance
                best_index = index
        return best_index

    def nearest_edge_insert_index(self, x, y):
        points = [self.original_to_display(point) for point in self.points[self.selected]]
        if len(points) < 2:
            return None
        best_insert_index = None
        best_distance = self.drag_threshold
        edge_count = len(points) if len(points) >= 3 else len(points) - 1
        for index in range(edge_count):
            p1 = points[index]
            p2 = points[(index + 1) % len(points)]
            distance = self.distance_to_segment((x, y), p1, p2)
            if distance <= best_distance:
                best_distance = distance
                best_insert_index = index + 1
        return best_insert_index

    @staticmethod
    def distance_to_segment(point, p1, p2):
        px, py = point
        x1, y1 = p1
        x2, y2 = p2
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0 and dy == 0:
            return math.hypot(px - x1, py - y1)
        t = ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)
        t = max(0.0, min(1.0, t))
        proj_x = x1 + t * dx
        proj_y = y1 + t * dy
        return math.hypot(px - proj_x, py - proj_y)

    def redraw(self):
        if self.canvas is None:
            return
        self.canvas.delete("all")
        display_frame = cv2.resize(self.frame, self.display_size)
        rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        ok, encoded = cv2.imencode(".png", rgb)
        if not ok:
            return
        self.photo = tk.PhotoImage(data=base64.b64encode(encoded).decode("ascii"))
        self.canvas.create_image(0, 0, image=self.photo, anchor="nw")
        for region in REGION_ORDER:
            self.draw_region(region, region == self.selected)
        self.draw_help()

    def draw_region(self, region, active):
        points = [self.original_to_display(point) for point in self.points[region]]
        color = self.tk_color(REGION_COLORS[region])
        if len(points) >= 2:
            flat = [coord for point in points for coord in point]
            if len(points) >= 3:
                self.canvas.create_polygon(flat, outline=color, fill=color, stipple="gray25", width=3 if active else 2)
            else:
                self.canvas.create_line(flat, fill=color, width=3 if active else 2)
        for index, (x, y) in enumerate(points, start=1):
            radius = 6 if active else 4
            self.canvas.create_oval(x - radius, y - radius, x + radius, y + radius, fill=color, outline=color)
            self.canvas.create_text(x + 12, y - 12, text=str(index), fill=color, font=("Arial", 12, "bold"))

    def draw_help(self):
        lines = [
            "Ctrl+1 left | Ctrl+2 straight | Ctrl+3 left_exit | Ctrl+4 straight_exit | Ctrl+5 far_straight",
            "Drag vertex: move | Drag edge: insert point | Right click vertex: delete | c clear | s save | q quit",
            f"Editing: {REGION_LABELS[self.selected]} | {len(self.points[self.selected])} diem",
            self.status,
        ]
        if self.mouse_xy:
            x, y = self.mouse_xy
            lines.append(f"Pixel ({x}, {y}) | ratio ({x / self.width:.4f}, {y / self.height:.4f})")
        y = 18
        for line in lines:
            self.canvas.create_text(12, y, text=line, anchor="w", fill="black", font=("Arial", 11, "bold"))
            self.canvas.create_text(11, y - 1, text=line, anchor="w", fill="white", font=("Arial", 11, "bold"))
            y += 20

    def tk_color(self, bgr):
        b, g, r = bgr
        return f"#{r:02x}{g:02x}{b:02x}"


def read_frame(video_path, frame_id):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Khong mo duoc video: {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise ValueError(f"Khong doc duoc frame {frame_id} tu video: {video_path}")
    return frame


def parse_args():
    parser = argparse.ArgumentParser(description="Click de ve lane_config.json tren frame video.")
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO_PATH)
    parser.add_argument("--frame", type=int, default=DEFAULT_FRAME_ID)
    parser.add_argument("--image", type=Path, help="Dung anh rieng thay vi doc frame tu video.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--max-display-width", type=int, default=1400)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.image:
        frame = cv2.imread(str(args.image))
        if frame is None:
            raise FileNotFoundError(f"Khong doc duoc anh: {args.image}")
    else:
        frame = read_frame(args.video, args.frame)

    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)

    calibrator = Calibrator(frame, config, args.config, args.max_display_width)
    calibrator.run()


if __name__ == "__main__":
    main()
