from __future__ import annotations

import csv
import json
import shutil
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from demo.apps.api.ocr_adapter import VietnamesePlateOCR


ROOT_DIR = Path(__file__).resolve().parents[3]   # demo/apps/api -> demo/apps -> demo -> project root
DEMO_DIR = ROOT_DIR / "demo"
APP_DIR = DEMO_DIR / "apps" / "api"
WEB_DIR = DEMO_DIR / "web" / "dashboard"
RUNTIME_DIR = DEMO_DIR / "runtime" / "viettraffic"
UPLOAD_DIR = RUNTIME_DIR / "uploads"
JOB_DIR = RUNTIME_DIR / "jobs"
PARKING_DIR = RUNTIME_DIR / "parking"
PARKING_DB_PATH = PARKING_DIR / "parking.json"
TICKET_DIR = PARKING_DIR / "tickets"
CAPTURE_DIR = PARKING_DIR / "captures"

PLATE_PIPELINE_DIR = ROOT_DIR / "find_license_plate"
PLATE_OUTPUT_DIR = PLATE_PIPELINE_DIR / "outputs"
WRONG_LANE_DIR = ROOT_DIR / "find_wrong_lane"
WRONG_LANE_OUTPUT_DIR = WRONG_LANE_DIR / "outputs"

_pipeline_lock = threading.Lock()
_jobs: dict[str, dict[str, Any]] = {}
_active_processes: dict[str, subprocess.Popen] = {}

# Singleton cache — models are loaded once and reused across all requests
_ocr_singleton: VietnamesePlateOCR | None = None
_plate_detector_singleton: Any | None = None
_model_load_lock = threading.Lock()


def _get_ocr() -> VietnamesePlateOCR:
    global _ocr_singleton
    if _ocr_singleton is None:
        with _model_load_lock:
            if _ocr_singleton is None:
                _ocr_singleton = VietnamesePlateOCR(use_gpu=False)
    return _ocr_singleton


def _get_plate_detector() -> Any:
    global _plate_detector_singleton
    if _plate_detector_singleton is None:
        with _model_load_lock:
            if _plate_detector_singleton is None:
                from ultralytics import YOLO
                plate_detector_path = ROOT_DIR / "find_license_plate" / "models" / "license_plate_detector.pt"
                _plate_detector_singleton = YOLO(str(plate_detector_path))
    return _plate_detector_singleton

STAGE_LABELS = {
    "queued": "Xếp hàng",
    "detecting_plates": "Cắt biển số",
    "wrong_lane": "Phân tích sai làn",
    "ocr": "Đọc ký tự (train_ocr)",
    "done": "Hoàn thành",
}


app = FastAPI(title="VietTraffic AI", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _ensure_dirs() -> None:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    JOB_DIR.mkdir(parents=True, exist_ok=True)
    WEB_DIR.mkdir(parents=True, exist_ok=True)
    PARKING_DIR.mkdir(parents=True, exist_ok=True)
    TICKET_DIR.mkdir(parents=True, exist_ok=True)
    CAPTURE_DIR.mkdir(parents=True, exist_ok=True)


_ensure_dirs()
app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")
app.mount("/files", StaticFiles(directory=str(RUNTIME_DIR)), name="files")
app.mount("/data_images", StaticFiles(directory=str(ROOT_DIR / "data")), name="data_images")


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _seconds_since(value: str | None) -> int:
    if not value:
        return 0
    try:
        return max(0, int((datetime.now() - datetime.fromisoformat(value)).total_seconds()))
    except ValueError:
        return 0


def _format_seconds(seconds: int | float) -> str:
    seconds = max(0, int(seconds))
    minutes, rest = divmod(seconds, 60)
    if minutes:
        return f"{minutes}m {rest:02d}s"
    return f"{rest}s"


def _stage(stage_id: str, seconds: int, progress_from: int, progress_to: int, module: str) -> dict[str, Any]:
    return {
        "id": stage_id,
        "label": STAGE_LABELS.get(stage_id, stage_id),
        "module": module,
        "estimated_seconds": seconds,
        "progress_from": progress_from,
        "progress_to": progress_to,
    }


def _build_runtime_plan(mode: str, max_frames: int) -> list[dict[str, Any]]:
    frame_budget = max_frames if max_frames > 0 else 900
    scale = max(0.65, min(3.2, frame_budget / 220))
    plate_seconds = int(85 * scale)
    wrong_lane_seconds = int(80 * scale)
    ocr_seconds = int(18 + min(60, frame_budget / 18))
    if mode == "plate":
        return [
            _stage("queued", 4, 0, 8, "api"),
            _stage("detecting_plates", plate_seconds, 8, 70, "find_license_plate"),
            _stage("ocr", ocr_seconds, 70, 98, "train_ocr"),
            _stage("done", 2, 98, 100, "api"),
        ]
    if mode == "wrong_lane":
        return [
            _stage("queued", 4, 0, 10, "api"),
            _stage("wrong_lane", wrong_lane_seconds, 10, 96, "find_wrong_lane"),
            _stage("done", 2, 96, 100, "api"),
        ]
    return [
        _stage("queued", 4, 0, 6, "api"),
        _stage("wrong_lane", wrong_lane_seconds, 6, 75, "find_wrong_lane"),
        _stage("ocr", ocr_seconds, 75, 98, "train_ocr"),
        _stage("done", 2, 98, 100, "api"),
    ]



def _runtime_stage(job: dict[str, Any]) -> dict[str, Any] | None:
    stage_id = job.get("stage")
    for item in job.get("runtime_plan") or []:
        if item.get("id") == stage_id:
            return item
    return None


def _hydrate_runtime(job: dict[str, Any]) -> dict[str, Any]:
    is_done = job.get("status") in ("completed", "failed")

    # --- Fix legacy module names & labels in saved runtime_plan ---
    for item in job.get("runtime_plan") or []:
        if item.get("module") == "giai_doan_doc_ki_tu":
            item["module"] = "train_ocr"
        if item.get("id") in STAGE_LABELS:
            item["label"] = STAGE_LABELS[item["id"]]

    # --- Fix legacy/non-accented messages ---
    for key in ("message", "runtime_message"):
        val = job.get(key)
        if isinstance(val, str):
            if "giai_doan_doc_ki_tu" in val:
                val = val.replace("giai_doan_doc_ki_tu", "train_ocr")
            if "Dang doc ky tu" in val:
                val = val.replace("Dang doc ky tu", "Đang đọc ký tự")
            if "Cat bien so" in val:
                val = val.replace("Cat bien so", "Cắt biển số")
            if "Phan tich sai lan" in val:
                val = val.replace("Phan tich sai lan", "Phân tích sai làn")
            if "Hoan thanh" in val:
                val = val.replace("Hoan thanh", "Hoàn thành")
            if "phan tich bien so" in val:
                val = val.replace("phan tich bien so", "phân tích biển số")
            if "phan tich sai lan" in val:
                val = val.replace("phan tich sai lan", "phân tích sai làn")
            if "phan tich giao thong" in val:
                val = val.replace("phan tich giao thong", "phân tích giao thông")
            if "Da nhan file, dang cho xu ly" in val:
                val = val.replace("Da nhan file, dang cho xu ly", "Đã nhận file, đang chờ xử lý")
            job[key] = val

    # --- Fix legacy OCR source names in results ---
    res = job.get("result")
    if isinstance(res, dict):
        for p in res.get("plates") or []:
            if p.get("ocr_source") == "giai_doan_doc_ki_tu":
                p["ocr_source"] = "train_ocr"
        for v in res.get("violations") or []:
            pr = v.get("plate_result")
            if isinstance(pr, dict) and pr.get("ocr_source") == "giai_doan_doc_ki_tu":
                pr["ocr_source"] = "train_ocr"

    stage = _runtime_stage(job)
    total_estimate = sum(int(item.get("estimated_seconds") or 0) for item in job.get("runtime_plan") or [])

    # --- Elapsed time: freeze on completion, keep ticking while running ---
    if is_done:
        started = job.get("started_at") or job.get("created_at")
        ended = job.get("finished_at")
        if not ended:
            upd = job.get("updated_at")
            if upd:
                try:
                    if " " in upd:
                        ended = datetime.strptime(upd, "%Y-%m-%d %H:%M:%S").isoformat()
                    else:
                        ended = datetime.fromisoformat(upd).isoformat()
                except Exception:
                    pass
        if started and ended:
            try:
                start_dt = datetime.fromisoformat(started.replace(" ", "T"))
                end_dt = datetime.fromisoformat(ended.replace(" ", "T"))
                job["elapsed_seconds"] = max(0, int((end_dt - start_dt).total_seconds()))
            except ValueError:
                job["elapsed_seconds"] = _seconds_since(started)
        else:
            job["elapsed_seconds"] = _seconds_since(started)
        job["stage_elapsed_seconds"] = 0
        
        msg = job.get("message", "")
        if job.get("status") == "completed" and ("đọc ký tự" in msg.lower() or "doc ky tu" in msg.lower() or "train_ocr" in msg.lower()):
            msg = "Hoàn thành phân tích"
        job["runtime_message"] = msg
    else:
        job["elapsed_seconds"] = _seconds_since(job.get("started_at") or job.get("created_at"))
        job["stage_elapsed_seconds"] = _seconds_since(job.get("stage_started_at"))

    job["elapsed_text"] = _format_seconds(job["elapsed_seconds"])
    job["estimated_total_seconds"] = total_estimate
    job["estimated_total_text"] = _format_seconds(total_estimate)
    job["stage_elapsed_text"] = _format_seconds(job["stage_elapsed_seconds"])

    if stage and not is_done:
        estimate = int(stage.get("estimated_seconds") or 1)
        job["stage_estimated_seconds"] = estimate
        job["stage_estimated_text"] = _format_seconds(estimate)
        job["stage_overdue"] = job.get("status") == "running" and job["stage_elapsed_seconds"] > estimate
        if job.get("status") == "running":
            if job["stage_overdue"]:
                job["runtime_message"] = f"{job.get('message', '')} - đang quá dự kiến {job['stage_elapsed_text']}/{job['stage_estimated_text']}"
            else:
                job["runtime_message"] = job.get("message", "")
    return job


def _url_for_runtime_path(path: Path) -> str:
    try:
        rel = path.resolve().relative_to(RUNTIME_DIR.resolve()).as_posix()
        return f"/files/{rel}"
    except ValueError:
        return ""


def _job_path(job_id: str) -> Path:
    return JOB_DIR / job_id


def _write_job(job_id: str) -> None:
    path = _job_path(job_id) / "job.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jobs[job_id], ensure_ascii=False, indent=2), encoding="utf-8")


def _set_job(job_id: str, **updates: Any) -> None:
    current = _jobs[job_id]
    old_stage = current.get("stage")
    if "stage" in updates and updates.get("stage") != old_stage:
        updates.setdefault("stage_started_at", _now_iso())
    if updates.get("status") == "running" and not current.get("started_at"):
        updates.setdefault("started_at", _now_iso())
    if updates.get("status") in ("completed", "failed") and not current.get("finished_at"):
        updates.setdefault("finished_at", _now_iso())
    _jobs[job_id].update(updates)
    _jobs[job_id]["artifacts"] = _scan_job_artifacts(job_id)
    _jobs[job_id]["updated_at"] = _now()
    _hydrate_runtime(_jobs[job_id])
    _write_job(job_id)


def _scan_job_artifacts(job_id: str) -> dict[str, Any]:
    root = _job_path(job_id)
    plate_dir = root / "plate_outputs"
    wrong_dir = root / "wrong_lane_outputs"
    return {
        "plate_crops": len(list((plate_dir / "plates_filtered").glob("*.png"))),
        "plate_reviews": len(list((plate_dir / "reviews").glob("*.png"))),
        "vehicle_crops": len(list((plate_dir / "vehicles").glob("*.png"))),
        "wrong_lane_reviews": len(list((wrong_dir / "reviews").glob("*.png"))),
        "wrong_lane_vehicles": len(list((wrong_dir / "violating_vehicles").glob("*.png"))),
        "plate_csv": _url_for_runtime_path(plate_dir / "violators.csv") if (plate_dir / "violators.csv").exists() else "",
        "wrong_lane_csv": _url_for_runtime_path(wrong_dir / "violations.csv") if (wrong_dir / "violations.csv").exists() else "",
        "tracks_csv": _url_for_runtime_path(plate_dir / "tracks.csv") if (plate_dir / "tracks.csv").exists() else "",
        "debug_video": _url_for_runtime_path(wrong_dir / "debug_wrong_lane.mp4") if (wrong_dir / "debug_wrong_lane.mp4").exists() else "",
    }


def _job_log_tail(job_id: str, limit: int = 5000) -> str:
    chunks: list[str] = []
    for name in ("plate_pipeline.log", "wrong_lane_pipeline.log"):
        path = _job_path(job_id) / name
        if path.exists():
            text = path.read_text(encoding="utf-8", errors="replace")
            chunks.append(f"[{name}]\n{text[-limit:]}")
    return "\n\n".join(chunks)[-limit:]


def _copy_tree_contents(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    if not src.exists():
        return
    for child in src.iterdir():
        target = dst / child.name
        if child.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(child, target)
        else:
            shutil.copy2(child, target)


def _run_command(command: list[str], cwd: Path, job_id: str | None = None, timeout_seconds: int = 900, log_path: Path | None = None) -> tuple[int, str]:
    try:
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("w", encoding="utf-8", errors="replace") as log_file:
                process = subprocess.Popen(
                    command,
                    cwd=str(cwd),
                    text=True,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    encoding="utf-8",
                    errors="replace",
                )
                if job_id:
                    _active_processes[job_id] = process
                started = time.monotonic()
                while process.poll() is None:
                    if time.monotonic() - started > timeout_seconds:
                        process.kill()
                        process.wait(timeout=5)
                        log_file.write(f"\n[TIMEOUT] Qua {timeout_seconds} giay, job bi dung de bao ve may.\n")
                        break
                    time.sleep(1)
                if job_id and job_id in _active_processes:
                    del _active_processes[job_id]
            return process.returncode if process.returncode is not None else 124, log_path.read_text(encoding="utf-8", errors="replace")
        completed = subprocess.run(
            command,
            cwd=str(cwd),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
        )
        return completed.returncode, completed.stdout
    except subprocess.TimeoutExpired as exc:
        output = exc.stdout or ""
        if isinstance(output, bytes):
            output = output.decode("utf-8", errors="replace")
        return 124, output + f"\n[TIMEOUT] Qua {timeout_seconds} giay, job bi dung de bao ve may.\n"


def _progress(job_id: str, value: int, stage: str, message: str, **extra: Any) -> None:
    _set_job(
        job_id,
        progress=max(0, min(100, value)),
        stage=stage,
        message=message,
        log_tail=_job_log_tail(job_id),
        **extra,
    )


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _collect_plate_results(job_id: str, copied_output_dir: Path) -> list[dict[str, Any]]:
    ocr = VietnamesePlateOCR(use_gpu=False)
    plate_dir = copied_output_dir / "plates_filtered"
    rows = _read_csv_rows(copied_output_dir / "violators.csv")
    row_by_path = {Path(row.get("Anh Vung Bien So", "")).name: row for row in rows}

    plates: list[dict[str, Any]] = []
    for plate_path in sorted(plate_dir.glob("*.png")):
        row = row_by_path.get(plate_path.name, {})
        try:
            ocr_result = ocr.read_image(plate_path)
            ocr_error = ""
        except Exception as exc:
            ocr_result = None
            ocr_error = str(exc)
        plates.append(
            {
                "id": plate_path.stem,
                "track_id": row.get("Track ID") or _infer_track_id(plate_path.name),
                "vehicle_type": row.get("Loai Phuong Tien", "Vehicle"),
                "frame": row.get("Frame Bien So Tot Nhat") or row.get("Frame Ghi Nhan", ""),
                "plate_image": _url_for_runtime_path(plate_path),
                "vehicle_image": _url_for_runtime_path(copied_output_dir / "vehicles" / f"vehicle_V{len(plates)+1:03d}.png"),
                "review_image": _url_for_runtime_path(copied_output_dir / "reviews" / f"review_V{len(plates)+1:03d}.png"),
                "detector_confidence": row.get("Do Tin Cay Bien So", ""),
                "quality_score": row.get("Diem Bien So", ""),
                "ocr_raw": ocr_result.raw if ocr_result else "",
                "plate": ocr_result.plate if ocr_result else "",
                "ocr_confidence": ocr_result.confidence if ocr_result else 0.0,
                "valid": ocr_result.valid if ocr_result else False,
                "ocr_source": ocr_result.source if ocr_result else "train_ocr",
                "ocr_error": ocr_error,
            }
        )
    return plates


def _infer_track_id(filename: str) -> str:
    marker = "track_"
    if marker not in filename:
        return ""
    rest = filename.split(marker, 1)[1]
    return rest.split("_", 1)[0]


def _run_plate_job(job_id: str, input_path: Path, save_video: bool, max_frames: int, params: dict[str, Any]) -> None:
    job_root = _job_path(job_id)
    copied_output = job_root / "plate_outputs"
    with _pipeline_lock:
        _progress(job_id, 12, "detecting_plates", "Đang phát hiện phương tiện và cắt biển số", status="running")
        command = [
            sys.executable,
            "-u",
            "main.py",
            "--video",
            str(input_path),
            "--output-dir",
            str(copied_output),
            "--max-frames",
            str(max(0, max_frames)),
        ]
        if not save_video:
            command.append("--no-video")
        
        # Add plate detection parameters
        if params.get("roi"):
            command.extend(["--roi", params["roi"]])
        if params.get("plate_conf"):
            command.extend(["--plate-conf", str(params["plate_conf"])])
        if params.get("min_plate_score"):
            command.extend(["--min-plate-score", str(params["min_plate_score"])])
        if params.get("plate_interval"):
            command.extend(["--plate-interval", str(params["plate_interval"])])

        code, log = _run_command(command, PLATE_PIPELINE_DIR, timeout_seconds=1200, log_path=job_root / "plate_pipeline.log")
        if code != 0:
            _set_job(job_id, status="failed", stage="detecting_plates", progress=100, message="Pipeline cắt biển số bị lỗi", log=log[-4000:], log_tail=log[-4000:])
            return

    _progress(job_id, 70, "ocr", "Đang đọc ký tự bằng model train_ocr")
    plates = _collect_plate_results(job_id, copied_output)
    video_url = _url_for_runtime_path(copied_output / "output.mp4")
    _set_job(
        job_id,
        status="completed",
        stage="done",
        progress=100,
        message="Hoàn thành phân tích biển số",
        log_tail=_job_log_tail(job_id),
        result={
            "mode": "plate",
            "total_plates": len(plates),
            "valid_plates": sum(1 for p in plates if p["valid"]),
            "plates": plates,
            "output_video": video_url,
            "csv": _url_for_runtime_path(copied_output / "violators.csv"),
            "tracks_csv": _url_for_runtime_path(copied_output / "tracks.csv"),
        },
    )


def _collect_wrong_lane_results(copied_output_dir: Path) -> dict[str, Any]:
    rows = _read_csv_rows(copied_output_dir / "violations.csv")
    violations = []
    for index, row in enumerate(rows, start=1):
        review_name = f"review_V{index:03d}_track_{row.get('Track ID', '')}.png"
        vehicle_name = f"vehicle_V{index:03d}_track_{row.get('Track ID', '')}.png"
        review_candidates = list((copied_output_dir / "reviews").glob(f"review_V{index:03d}*.png"))
        vehicle_candidates = list((copied_output_dir / "violating_vehicles").glob(f"vehicle_V{index:03d}*.png"))
        violations.append(
            {
                "id": row.get("Ma Vi Pham", f"V{index:03d}"),
                "track_id": row.get("Track ID", ""),
                "vehicle_type": row.get("Loai Phuong Tien", ""),
                "frame": row.get("Frame Anh Xe Ro Nhat", row.get("Frame Ket Luan", row.get("Frame Ghi Nhan", ""))),
                "case_id": row.get("Ma Truong Hop", ""),
                "start_lane": row.get("Lan Xuat Phat", ""),
                "direction": row.get("Huong Di Chuyen", ""),
                "legal_name": row.get("Ten Loi Phap Luat", ""),
                "reason": row.get("Hanh Vi Chi Tiet", row.get("Ket Luan", row.get("Ly Do", "Sai lan"))),
                "fine": row.get("Muc Phat", ""),
                "law_source": row.get("Can Cu", ""),
                "evidence": row.get("Bang Chung", ""),
                "review_image": _url_for_runtime_path(review_candidates[0]) if review_candidates else review_name,
                "vehicle_image": _url_for_runtime_path(vehicle_candidates[0]) if vehicle_candidates else vehicle_name,
            }
        )
    return {
        "mode": "wrong_lane",
        "total_violations": len(violations),
        "violations": violations,
        "debug_video": _url_for_runtime_path(copied_output_dir / "debug_wrong_lane.mp4") if (copied_output_dir / "debug_wrong_lane.mp4").exists() else "",
        "csv": _url_for_runtime_path(copied_output_dir / "violations.csv"),
    }


def _load_parking() -> dict[str, Any]:
    if PARKING_DB_PATH.exists():
        try:
            return json.loads(PARKING_DB_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"sessions": [], "fee_per_hour": 10000}


def _save_parking(data: dict[str, Any]) -> None:
    PARKING_DB_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _normalize_plate_text(plate: str) -> str:
    return "".join(ch for ch in plate.upper() if ch.isalnum())


def _read_plate_from_capture(image: UploadFile | None, prefix: str) -> tuple[str, dict[str, Any], str]:
    if image is None:
        return "", {}, ""
    suffix = Path(image.filename or "capture.jpg").suffix or ".jpg"
    capture_id = f"{prefix}_{uuid.uuid4().hex[:10]}{suffix}"
    capture_path = CAPTURE_DIR / capture_id
    with capture_path.open("wb") as handle:
        shutil.copyfileobj(image.file, handle)
    try:
        import cv2
        from ultralytics import YOLO
        
        img = cv2.imread(str(capture_path))
        if img is None or img.size == 0:
            raise ValueError("Invalid image file")
            
        plate_detector_path = ROOT_DIR / "find_license_plate" / "models" / "license_plate_detector.pt"
        plate_detector = YOLO(str(plate_detector_path))
        results = plate_detector(img, conf=0.15, verbose=False)
        
        best_box = None
        best_conf = 0.0
        for r in results:
            for box in r.boxes:
                conf = float(box.conf[0]) if box.conf is not None else 0.0
                if conf > best_conf:
                    best_conf = conf
                    best_box = box.xyxy[0].tolist()
                    
        if best_box is not None:
            h, w = img.shape[:2]
            x1, y1, x2, y2 = map(int, best_box)
            pad_x = int((x2 - x1) * 0.1)
            pad_y = int((y2 - y1) * 0.1)
            x1 = max(0, x1 - pad_x)
            y1 = max(0, y1 - pad_y)
            x2 = min(w, x2 + pad_x)
            y2 = min(h, y2 + pad_y)
            
            plate_crop = img[y1:y2, x1:x2]
            if plate_crop.size > 0:
                temp_crop_path = CAPTURE_DIR / f"crop_{capture_id}"
                cv2.imwrite(str(temp_crop_path), plate_crop)
                try:
                    ocr_result = VietnamesePlateOCR(use_gpu=False).read_image(temp_crop_path)
                finally:
                    if temp_crop_path.exists():
                        try:
                            temp_crop_path.unlink()
                        except Exception:
                            pass
            else:
                ocr_result = VietnamesePlateOCR(use_gpu=False).read_image(capture_path)
        else:
            ocr_result = VietnamesePlateOCR(use_gpu=False).read_image(capture_path)
            
        plate = ocr_result.plate
        payload = {
            "raw": ocr_result.raw,
            "plate": ocr_result.plate,
            "confidence": ocr_result.confidence,
            "valid": ocr_result.valid,
            "source": ocr_result.source,
        }
    except Exception as exc:
        try:
            ocr_result = VietnamesePlateOCR(use_gpu=False).read_image(capture_path)
            plate = ocr_result.plate
            payload = {
                "raw": ocr_result.raw,
                "plate": ocr_result.plate,
                "confidence": ocr_result.confidence,
                "valid": ocr_result.valid,
                "source": ocr_result.source,
            }
        except Exception as exc_inner:
            plate = ""
            payload = {"error": f"{exc} | {exc_inner}", "source": "train_ocr"}
            
    return plate, payload, _url_for_runtime_path(capture_path)


def _parking_fee(entry_at: str, exit_at: str, fee_per_hour: int) -> dict[str, Any]:
    start = datetime.fromisoformat(entry_at)
    end = datetime.fromisoformat(exit_at)
    minutes = max(1, int((end - start).total_seconds() // 60))
    billable_hours = max(1, (minutes + 59) // 60)
    return {
        "duration_minutes": minutes,
        "billable_hours": billable_hours,
        "fee": billable_hours * fee_per_hour,
    }


def _write_ticket(session: dict[str, Any]) -> str:
    ticket_id = session["ticket_id"]
    ticket_path = TICKET_DIR / f"{ticket_id}.html"
    html = f"""<!doctype html>
<html lang="vi">
<head>
  <meta charset="utf-8">
  <title>Vé xe {session.get('plate', '')}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 28px; color: #111; }}
    .ticket {{ width: 360px; border: 2px solid #111; padding: 18px; }}
    h1 {{ font-size: 20px; margin: 0 0 12px; }}
    .plate {{ font-size: 28px; font-weight: 800; border: 1px solid #111; padding: 8px; text-align: center; margin: 12px 0; }}
    .row {{ display: flex; justify-content: space-between; border-top: 1px solid #ddd; padding: 8px 0; gap: 12px; }}
    .fee {{ font-size: 22px; font-weight: 800; }}
    @media print {{ body {{ margin: 0; }} .ticket {{ border: 0; width: auto; }} }}
  </style>
</head>
<body>
  <section class="ticket">
    <h1>Vé xe trung tâm thương mại</h1>
    <div>Mã vé: <strong>{ticket_id}</strong></div>
    <div class="plate">{session.get('plate', '')}</div>
    <div class="row"><span>Giờ vào</span><strong>{session.get('entry_at', '')}</strong></div>
    <div class="row"><span>Giờ ra</span><strong>{session.get('exit_at', '')}</strong></div>
    <div class="row"><span>Thời gian</span><strong>{session.get('duration_minutes', 0)} phút</strong></div>
    <div class="row"><span>Số giờ tính phí</span><strong>{session.get('billable_hours', 0)}</strong></div>
    <div class="row fee"><span>Phí</span><strong>{session.get('fee', 0):,} VND</strong></div>
  </section>
</body>
</html>"""
    ticket_path.write_text(html, encoding="utf-8")
    return _url_for_runtime_path(ticket_path)


def _run_wrong_lane_job(job_id: str, input_path: Path, save_video: bool, max_frames: int) -> None:
    job_root = _job_path(job_id)
    copied_output = job_root / "wrong_lane_outputs"
    with _pipeline_lock:
        _progress(job_id, 20, "wrong_lane", "Đang phân tích sai làn đường", status="running")
        command = [
            sys.executable,
            "-u",
            "main.py",
            "--video",
            str(input_path),
            "--output-dir",
            str(copied_output),
            "--max-frames",
            str(max(0, max_frames)),
        ]
        if not save_video:
            command.append("--no-video")
        code, log = _run_command(command, WRONG_LANE_DIR, job_id=job_id, timeout_seconds=1200, log_path=job_root / "wrong_lane_pipeline.log")
        if code != 0:
            _set_job(job_id, status="failed", stage="wrong_lane", progress=100, message="Pipeline phân tích sai làn bị lỗi", log=log[-4000:], log_tail=log[-4000:])
            return
    _set_job(job_id, status="completed", stage="done", progress=100, message="Hoàn thành phân tích sai làn đường", log_tail=_job_log_tail(job_id), result=_collect_wrong_lane_results(copied_output))


def _detect_and_ocr_plate_for_vehicle(vehicle_path: Path, plate_detector: Any, ocr_engine: VietnamesePlateOCR, plate_output_dir: Path, track_id: str) -> dict[str, Any] | None:
    import cv2
    if not vehicle_path.exists():
        return None
    img = cv2.imread(str(vehicle_path))
    if img is None or img.size == 0:
        return None
        
    h, w = img.shape[:2]
    results = plate_detector(img, conf=0.15, verbose=False)
    
    best_box = None
    best_conf = 0.0
    for r in results:
        for box in r.boxes:
            conf = float(box.conf[0]) if box.conf is not None else 0.0
            if conf > best_conf:
                best_conf = conf
                best_box = box.xyxy[0].tolist()
                
    if best_box is None:
        return None
        
    x1, y1, x2, y2 = map(int, best_box)
    pad_x = int((x2 - x1) * 0.1)
    pad_y = int((y2 - y1) * 0.1)
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w, x2 + pad_x)
    y2 = min(h, y2 + pad_y)
    
    plate_crop = img[y1:y2, x1:x2]
    if plate_crop.size == 0:
        return None
        
    plate_filename = f"plate_track_{track_id}.png"
    plate_path = plate_output_dir / plate_filename
    cv2.imwrite(str(plate_path), plate_crop)
    
    try:
        ocr_result = ocr_engine.read_image(plate_path)
        ocr_error = ""
    except Exception as exc:
        ocr_result = None
        ocr_error = str(exc)
        
    return {
        "plate_image": _url_for_runtime_path(plate_path),
        "ocr_raw": ocr_result.raw if ocr_result else "",
        "plate": ocr_result.plate if ocr_result else "",
        "ocr_confidence": ocr_result.confidence if ocr_result else 0.0,
        "valid": ocr_result.valid if ocr_result else False,
        "ocr_source": ocr_result.source if ocr_result else "train_ocr",
        "ocr_error": ocr_error,
    }


def _run_traffic_job(job_id: str, input_path: Path, save_video: bool, max_frames: int, params: dict[str, Any]) -> None:
    job_root = _job_path(job_id)
    plate_output = job_root / "plate_outputs"
    plate_output.mkdir(parents=True, exist_ok=True)
    wrong_lane_output = job_root / "wrong_lane_outputs"
    
    with _pipeline_lock:
        # 1. Run wrong lane analysis first
        _progress(job_id, 15, "wrong_lane", "Đang phân tích đi sai làn đường", status="running")
        wrong_command = [
            sys.executable,
            "-u",
            "main.py",
            "--video",
            str(input_path),
            "--output-dir",
            str(wrong_lane_output),
            "--max-frames",
            str(max(0, max_frames)),
        ]
        if not save_video:
            wrong_command.append("--no-video")
        code_w, log_w = _run_command(wrong_command, WRONG_LANE_DIR, job_id=job_id, timeout_seconds=1200, log_path=job_root / "wrong_lane_pipeline.log")
        if code_w != 0:
            _set_job(job_id, status="failed", stage="wrong_lane", progress=100, message="Pipeline phân tích sai làn bị lỗi", log=log_w[-4000:], log_tail=log_w[-4000:])
            return

    # 2. Run plate detection & OCR ONLY on violating vehicles
    _progress(job_id, 75, "ocr", "Đang nhận diện biển số và đọc ký tự (train_ocr) cho xe vi phạm")
    
    wrong_lane = _collect_wrong_lane_results(wrong_lane_output)
    
    from ultralytics import YOLO
    plate_detector_path = ROOT_DIR / "find_license_plate" / "models" / "license_plate_detector.pt"
    plate_detector = YOLO(str(plate_detector_path))
    ocr_engine = VietnamesePlateOCR(use_gpu=False)
    
    enriched = []
    plates = []
    
    for violation in wrong_lane["violations"]:
        track_id = violation.get("track_id", "")
        # Find the vehicle crop image
        vehicle_candidates = list((wrong_lane_output / "violating_vehicles").glob(f"vehicle_*_track_{track_id}.png"))
        plate_result = None
        if vehicle_candidates:
            vehicle_path = vehicle_candidates[0]
            try:
                ocr_info = _detect_and_ocr_plate_for_vehicle(
                    vehicle_path=vehicle_path,
                    plate_detector=plate_detector,
                    ocr_engine=ocr_engine,
                    plate_output_dir=plate_output,
                    track_id=track_id
                )
                if ocr_info:
                    plate_result = {
                        "id": f"plate_{track_id}",
                        "track_id": track_id,
                        "vehicle_type": violation.get("vehicle_type", "Vehicle"),
                        "frame": violation.get("frame", ""),
                        "plate_image": ocr_info["plate_image"],
                        "vehicle_image": violation.get("vehicle_image"),
                        "review_image": violation.get("review_image"),
                        "detector_confidence": "1.0",
                        "quality_score": "1.0",
                        "ocr_raw": ocr_info["ocr_raw"],
                        "plate": ocr_info["plate"],
                        "ocr_confidence": ocr_info["ocr_confidence"],
                        "valid": ocr_info["valid"],
                        "ocr_source": ocr_info["ocr_source"],
                        "ocr_error": ocr_info["ocr_error"],
                    }
                    plates.append(plate_result)
            except Exception as e:
                print(f"Error processing plate for track {track_id}: {e}")
                
        enriched.append({**violation, "plate_result": plate_result})
        
    _set_job(
        job_id,
        status="completed",
        stage="done",
        progress=100,
        message="Hoàn thành phân tích giao thông",
        log_tail=_job_log_tail(job_id),
        result={
            "mode": "traffic",
            "total_plates": len(plates),
            "valid_plates": sum(1 for p in plates if p["valid"]),
            "total_violations": len(enriched),
            "plates": plates,
            "violations": enriched,
            "csv": wrong_lane.get("csv", ""),
            "tracks_csv": "",
            "debug_video": wrong_lane.get("debug_video", ""),
        },
    )


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "root": str(ROOT_DIR),
        "ocr_model": str(VietnamesePlateOCR().model_dir),
        "time": _now(),
    }


@app.get("/api/parking")
def parking_state() -> dict[str, Any]:
    data = _load_parking()
    sessions = data.get("sessions", [])
    # Clean up legacy sources in parking sessions dynamically
    for s in sessions:
        if s.get("source") == "giai_doan_doc_ki_tu":
            s["source"] = "train_ocr"
        for ocr_key in ("entry_ocr", "exit_ocr"):
            ocr = s.get(ocr_key)
            if isinstance(ocr, dict) and ocr.get("source") == "giai_doan_doc_ki_tu":
                ocr["source"] = "train_ocr"
    return {
        "fee_per_hour": data.get("fee_per_hour", 10000),
        "active": [item for item in sessions if item.get("status") == "active"],
        "completed": [item for item in sessions if item.get("status") == "completed"],
        "sessions": sessions,
    }


@app.get("/api/parking/images")
def list_parking_images() -> dict[str, Any]:
    images_dir = ROOT_DIR / "data"
    if not images_dir.exists():
        return {"images": []}
    valid_exts = {".jpg", ".jpeg", ".png", ".webp"}
    files = [f.name for f in images_dir.iterdir() if f.is_file() and f.suffix.lower() in valid_exts]
    return {"images": sorted(files)}


@app.post("/api/ocr/test")
async def ocr_test(
    image: UploadFile = File(...),
    model: str = Form("train"),
) -> dict[str, Any]:
    """Upload a single image, run YOLO plate detection + OCR, return detailed results."""
    suffix = Path(image.filename or "test.jpg").suffix or ".jpg"
    test_id = f"ocrtest_{uuid.uuid4().hex[:10]}{suffix}"
    save_path = CAPTURE_DIR / test_id
    with save_path.open("wb") as fh:
        shutil.copyfileobj(image.file, fh)

    result: dict[str, Any] = {
        "filename": image.filename,
        "saved_as": test_id,
        "image_url": _url_for_runtime_path(save_path),
        "detection": None,
        "ocr": None,
        "plate": "",
        "error": None,
    }

    try:
        import cv2

        img = cv2.imread(str(save_path))
        if img is None or img.size == 0:
            raise ValueError("Không đọc được file ảnh — định dạng không hỗ trợ hoặc file lỗi.")

        h_orig, w_orig = img.shape[:2]
        plate_detector = _get_plate_detector()  # reuse cached model
        det_results = plate_detector(img, conf=0.10, verbose=False)

        detections = []
        best_box = None
        best_conf = 0.0
        for r in det_results:
            for box in r.boxes:
                conf = float(box.conf[0]) if box.conf is not None else 0.0
                xyxy = box.xyxy[0].tolist()
                detections.append({"box": xyxy, "conf": round(conf, 4)})
                if conf > best_conf:
                    best_conf = conf
                    best_box = xyxy

        result["detection"] = {
            "count": len(detections),
            "detections": detections,
            "best_conf": round(best_conf, 4) if best_box else None,
        }

        crop_url = None
        if best_box is not None:
            x1, y1, x2, y2 = map(int, best_box)
            pad_x = max(4, int((x2 - x1) * 0.10))
            pad_y = max(4, int((y2 - y1) * 0.10))
            x1 = max(0, x1 - pad_x); y1 = max(0, y1 - pad_y)
            x2 = min(w_orig, x2 + pad_x); y2 = min(h_orig, y2 + pad_y)
            plate_crop = img[y1:y2, x1:x2]
            crop_id = f"ocrtest_crop_{uuid.uuid4().hex[:8]}{suffix}"
            crop_path = CAPTURE_DIR / crop_id
            cv2.imwrite(str(crop_path), plate_crop)
            crop_url = _url_for_runtime_path(crop_path)
            result["detection"]["crop_url"] = crop_url
            result["detection"]["crop_box"] = [x1, y1, x2, y2]
            ocr_input = crop_path
        else:
            result["detection"]["crop_url"] = None
            ocr_input = save_path

        ocr = _get_ocr().read_image(ocr_input, model_name=model)  # reuse cached model
        result["ocr"] = {
            "raw": ocr.raw,
            "plate": ocr.plate,
            "confidence": round(float(ocr.confidence or 0), 4),
            "valid": ocr.valid,
            "source": ocr.source,
        }
        result["plate"] = ocr.plate

    except Exception as exc:
        result["error"] = str(exc)
        try:
            ocr = _get_ocr().read_image(save_path, model_name=model)  # reuse cached model
            result["ocr"] = {
                "raw": ocr.raw,
                "plate": ocr.plate,
                "confidence": round(float(ocr.confidence or 0), 4),
                "valid": ocr.valid,
                "source": ocr.source,
            }
            result["plate"] = ocr.plate
        except Exception as exc2:
            result["ocr"] = {"error": str(exc2)}

    return result


@app.post("/api/parking/config")
def parking_config(fee_per_hour: int = Form(10000)) -> dict[str, Any]:
    data = _load_parking()
    data["fee_per_hour"] = max(1000, int(fee_per_hour))
    _save_parking(data)
    return parking_state()


@app.post("/api/parking/entry")
async def parking_entry(plate: str = Form(""), image: UploadFile | None = File(None)) -> dict[str, Any]:
    detected_plate, ocr_payload, capture_url = _read_plate_from_capture(image, "entry")
    final_plate = _normalize_plate_text(plate or detected_plate)
    if not final_plate:
        raise HTTPException(status_code=400, detail="Can bien so hoac anh crop bien so")
    data = _load_parking()
    sessions = data.setdefault("sessions", [])
    active = next((item for item in sessions if item.get("plate") == final_plate and item.get("status") == "active"), None)
    if active:
        return {"session": active, "parking": parking_state(), "message": "Xe nay dang o trong bai"}
    session = {
        "id": uuid.uuid4().hex[:12],
        "plate": final_plate,
        "status": "active",
        "entry_at": _now_iso(),
        "entry_image": capture_url,
        "entry_ocr": ocr_payload,
        "source": "train_ocr" if ocr_payload else "manual",
    }
    sessions.insert(0, session)
    _save_parking(data)
    return {"session": session, "parking": parking_state(), "message": "Da ghi xe vao"}


@app.post("/api/parking/exit")
async def parking_exit(plate: str = Form(""), image: UploadFile | None = File(None)) -> dict[str, Any]:
    detected_plate, ocr_payload, capture_url = _read_plate_from_capture(image, "exit")
    final_plate = _normalize_plate_text(plate or detected_plate)
    if not final_plate:
        raise HTTPException(status_code=400, detail="Can bien so hoac anh crop bien so")
    data = _load_parking()
    sessions = data.setdefault("sessions", [])
    session = next((item for item in sessions if item.get("plate") == final_plate and item.get("status") == "active"), None)
    if not session:
        raise HTTPException(status_code=404, detail="Khong co luot xe dang gui voi bien so nay")
    exit_at = _now_iso()
    fee = _parking_fee(session["entry_at"], exit_at, int(data.get("fee_per_hour", 10000)))
    session.update(
        {
            "status": "completed",
            "exit_at": exit_at,
            "exit_image": capture_url,
            "exit_ocr": ocr_payload,
            "ticket_id": uuid.uuid4().hex[:10].upper(),
            **fee,
        }
    )
    session["ticket_url"] = _write_ticket(session)
    _save_parking(data)
    return {"session": session, "parking": parking_state(), "message": "Da ghi xe ra va xuat ve"}


@app.get("/api/parking/tickets/{ticket_id}")
def parking_ticket(ticket_id: str) -> FileResponse:
    path = TICKET_DIR / f"{ticket_id}.html"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Khong tim thay ve")
    return FileResponse(path, media_type="text/html")


@app.post("/api/jobs")
async def create_job(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    mode: str = "plate",
    save_video: bool = False,
    max_frames: int = 220,
    roi: str = "",
    plate_conf: float = 0.25,
    min_plate_score: float = 0.5,
    plate_interval: int = 2,
) -> dict[str, Any]:
    if mode not in {"plate", "wrong_lane", "traffic"}:
        raise HTTPException(status_code=400, detail="mode phai la plate, wrong_lane hoac traffic")
    suffix = Path(file.filename or "input.mp4").suffix or ".mp4"
    job_id = uuid.uuid4().hex[:12]
    job_root = _job_path(job_id)
    job_root.mkdir(parents=True, exist_ok=True)
    upload_path = UPLOAD_DIR / f"{job_id}{suffix}"
    with upload_path.open("wb") as handle:
        shutil.copyfileobj(file.file, handle)
    runtime_plan = _build_runtime_plan(mode, max_frames)
    
    params = {
        "roi": roi,
        "plate_conf": plate_conf,
        "min_plate_score": min_plate_score,
        "plate_interval": plate_interval,
    }
    
    _jobs[job_id] = {
        "id": job_id,
        "status": "queued",
        "stage": "queued",
        "mode": mode,
        "filename": file.filename,
        "input_url": _url_for_runtime_path(upload_path),
        "message": "Da nhan file, dang cho xu ly",
        "progress": 0,
        "runtime_plan": runtime_plan,
        "stage_started_at": _now_iso(),
        "started_at": None,
        "max_frames": max_frames,
        "save_video": save_video,
        "params": params,
        "artifacts": {},
        "log_tail": "",
        "created_at": _now(),
        "updated_at": _now(),
        "result": None,
    }
    _write_job(job_id)
    if mode == "plate":
        background_tasks.add_task(_run_plate_job, job_id, upload_path, save_video, max_frames, params)
    elif mode == "traffic":
        background_tasks.add_task(_run_traffic_job, job_id, upload_path, save_video, max_frames, params)
    else:
        background_tasks.add_task(_run_wrong_lane_job, job_id, upload_path, save_video, max_frames)
    return _jobs[job_id]


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str) -> dict[str, Any]:
    killed = False
    if job_id in _active_processes:
        try:
            _active_processes[job_id].kill()
            killed = True
        except Exception:
            pass
        del _active_processes[job_id]
        
    if job_id in _jobs and _jobs[job_id]["status"] not in ["completed", "failed"]:
        _set_job(job_id, status="failed", message="Người dùng đã huỷ tác vụ.", progress=100, stage="done")
        return {"message": "Đã huỷ tác vụ thành công", "killed": killed}
        
    raise HTTPException(status_code=400, detail="Tác vụ không còn chạy.")


@app.delete("/api/jobs/{job_id}")
def delete_job(job_id: str) -> dict[str, Any]:
    if job_id in _active_processes:
        try:
            _active_processes[job_id].kill()
        except Exception:
            pass
        del _active_processes[job_id]

    if job_id in _jobs:
        del _jobs[job_id]

    job_root = _job_path(job_id)
    if job_root.exists():
        try:
            shutil.rmtree(job_root)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Không thể xóa thư mục job: {str(e)}")

    return {"message": "Đã xóa tác vụ thành công"}



@app.get("/api/jobs")
def list_jobs() -> dict[str, Any]:
    for job_file in JOB_DIR.glob("*/job.json"):
        try:
            data = json.loads(job_file.read_text(encoding="utf-8"))
            _jobs.setdefault(data["id"], data)
        except Exception:
            continue
    for job_id, job in list(_jobs.items()):
        job["artifacts"] = _scan_job_artifacts(job_id)
        job["log_tail"] = _job_log_tail(job_id)
        if "runtime_plan" not in job:
            job["runtime_plan"] = _build_runtime_plan(job.get("mode", "plate"), int(job.get("max_frames") or 220))
        if "progress" not in job:
            job["progress"] = 100 if job.get("status") in {"completed", "failed"} else 0
        _hydrate_runtime(job)
    return {"jobs": sorted(_jobs.values(), key=lambda item: item["created_at"], reverse=True)}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    if job_id in _jobs:
        _jobs[job_id]["artifacts"] = _scan_job_artifacts(job_id)
        _jobs[job_id]["log_tail"] = _job_log_tail(job_id)
        if "runtime_plan" not in _jobs[job_id]:
            _jobs[job_id]["runtime_plan"] = _build_runtime_plan(_jobs[job_id].get("mode", "plate"), int(_jobs[job_id].get("max_frames") or 220))
        _hydrate_runtime(_jobs[job_id])
        return _jobs[job_id]
    job_file = _job_path(job_id) / "job.json"
    if job_file.exists():
        data = json.loads(job_file.read_text(encoding="utf-8"))
        data["artifacts"] = _scan_job_artifacts(job_id)
        data["log_tail"] = _job_log_tail(job_id)
        if "runtime_plan" not in data:
            data["runtime_plan"] = _build_runtime_plan(data.get("mode", "plate"), int(data.get("max_frames") or 220))
        _hydrate_runtime(data)
        _jobs[job_id] = data
        return data
    raise HTTPException(status_code=404, detail="Khong tim thay job")
