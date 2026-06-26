"""
So sánh hiệu năng nhận diện biển số của 3 model OCR:
  - Model 1: CRNN (VGG-Lite + BiLSTM + CTC) — train riêng
  - Model 2: CRNN (VGG-Lite + BiLSTM + CTC) — train riêng, hyperparams khác
  - Model 3: Ensemble (PaddleOCR + PyTorch Transformer CTC + merge rules + plate normalization)
"""
import csv
import re
import json
from pathlib import Path
from typing import Dict, List, Tuple


def normalize_plate_text(text: str) -> str:
    """Chuẩn hóa text biển số: uppercase, chỉ giữ 0-9 A-Z Đ."""
    text = str(text or "").upper().strip()
    return re.sub(r"[^0-9A-Z\u0110]", "", text)


def levenshtein_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i]
        for j, cb in enumerate(b, start=1):
            curr.append(min(curr[-1] + 1, prev[j] + 1, prev[j - 1] + (ca != cb)))
        prev = curr
    return prev[-1]


def char_accuracy(pred: str, gt: str) -> float:
    """Character accuracy = 1 - CER (based on Levenshtein distance)."""
    pred_norm = normalize_plate_text(pred)
    gt_norm = normalize_plate_text(gt)
    max_len = max(len(pred_norm), len(gt_norm))
    if max_len == 0:
        return 1.0
    return max(0.0, 1.0 - levenshtein_distance(pred_norm, gt_norm) / max_len)


def cer(pred: str, gt: str) -> float:
    """Character Error Rate."""
    pred_norm = normalize_plate_text(pred)
    gt_norm = normalize_plate_text(gt)
    if len(gt_norm) == 0:
        return 0.0 if len(pred_norm) == 0 else 1.0
    return levenshtein_distance(pred_norm, gt_norm) / len(gt_norm)


def load_labels(path: str) -> Dict[str, str]:
    """Load ground truth labels từ test_labels.txt (format: filename<TAB>label)."""
    labels = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if "\t" in line:
                fname, text = line.split("\t", 1)
            else:
                parts = line.split(maxsplit=1)
                if len(parts) == 2:
                    fname, text = parts
                else:
                    continue
            labels[fname.strip()] = text.strip()
    return labels


def load_model12_results(path: str) -> Dict[str, dict]:
    """Load CSV results từ Model 1 hoặc Model 2 (columns: image, prediction, latency_ms)."""
    results = {}
    if not Path(path).exists():
        print(f"  [WARN] File not found: {path}")
        return results
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            results[row["image"]] = {
                "pred": row.get("prediction", ""),
                "latency_ms": float(row.get("latency_ms", 0.0)),
            }
    return results


def load_model3_results(path: str) -> Dict[str, dict]:
    """Load CSV results từ Model 3 ensemble (pipeline.py output).
    Lấy cột 'prediction' (ensemble merged) và 'total_time_ms'."""
    results = {}
    if not Path(path).exists():
        print(f"  [WARN] File not found: {path}")
        return results
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            results[row["image"]] = {
                "pred": row.get("prediction", ""),
                "latency_ms": float(row.get("total_time_ms", 0.0)),
            }
    return results


def evaluate_model(
    model_name: str, results: Dict[str, dict], ground_truth: Dict[str, str]
) -> dict:
    """Tính metrics cho 1 model. Trả về dict chứa thống kê."""
    if not results:
        print(f"  [WARN] No results for {model_name}")
        return {}

    char_accs = []
    cers = []
    plate_correct = 0
    plate_total = 0
    latencies = []
    mismatches = []

    for fname, gt in ground_truth.items():
        if fname not in results:
            continue
        pred = results[fname]["pred"]
        latency = results[fname]["latency_ms"]

        ca = char_accuracy(pred, gt)
        cr = cer(pred, gt)
        char_accs.append(ca)
        cers.append(cr)

        gt_norm = normalize_plate_text(gt)
        pred_norm = normalize_plate_text(pred)
        correct = gt_norm == pred_norm
        plate_correct += int(correct)
        plate_total += 1
        latencies.append(latency)

        if not correct:
            mismatches.append((fname, gt, pred))

    if plate_total == 0:
        print(f"  [WARN] No matching images for {model_name}")
        return {}

    stats = {
        "model": model_name,
        "num_images": plate_total,
        "char_acc": sum(char_accs) / len(char_accs) * 100,
        "cer": sum(cers) / len(cers) * 100,
        "plate_acc": plate_correct / plate_total * 100,
        "plate_correct": plate_correct,
        "plate_wrong": plate_total - plate_correct,
        "avg_latency_ms": sum(latencies) / len(latencies),
        "total_time_s": sum(latencies) / 1000.0,
        "mismatches": mismatches,
    }
    return stats


def print_comparison_table(all_stats: List[dict]):
    """In bảng so sánh đẹp."""
    print("\n" + "=" * 90)
    print("  SO SANH HIEU NANG NHAN DIEN BIEN SO - 3 MODEL OCR")
    print("=" * 90)
    print(
        f"  {'Model':<28} {'Char Acc':>10} {'CER':>8} {'Plate Acc':>11} "
        f"{'Correct/Total':>14} {'Avg Time':>10}"
    )
    print("-" * 90)
    for s in all_stats:
        print(
            f"  {s['model']:<28} {s['char_acc']:>9.2f}% {s['cer']:>7.2f}% "
            f"{s['plate_acc']:>10.2f}% "
            f"{s['plate_correct']:>5}/{s['num_images']:<5} "
            f"{s['avg_latency_ms']:>8.1f}ms"
        )
    print("=" * 90)

    # Best model
    best = max(all_stats, key=lambda x: x["plate_acc"])
    print(f"\n  >>> BEST MODEL: {best['model']}")
    print(f"      Plate Accuracy: {best['plate_acc']:.2f}%")
    print(f"      Character Accuracy: {best['char_acc']:.2f}%")
    print(f"      CER: {best['cer']:.2f}%")


def save_detailed_report(all_stats: List[dict], output_path: str):
    """Lưu báo cáo chi tiết ra file CSV."""
    rows = []
    for s in all_stats:
        rows.append({
            "Model": s["model"],
            "Num Images": s["num_images"],
            "Character Accuracy (%)": f"{s['char_acc']:.2f}",
            "CER (%)": f"{s['cer']:.2f}",
            "Plate Accuracy (%)": f"{s['plate_acc']:.2f}",
            "Plates Correct": s["plate_correct"],
            "Plates Wrong": s["plate_wrong"],
            "Avg Latency (ms)": f"{s['avg_latency_ms']:.1f}",
            "Total Time (s)": f"{s['total_time_s']:.1f}",
        })

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n  [REPORT] Saved: {output_path}")


def save_mismatches(all_stats: List[dict], output_path: str):
    """Lưu danh sách các biển số nhận sai của từng model."""
    rows = []
    for s in all_stats:
        for fname, gt, pred in s.get("mismatches", []):
            rows.append({
                "model": s["model"],
                "image": fname,
                "ground_truth": gt,
                "prediction": pred,
                "gt_normalized": normalize_plate_text(gt),
                "pred_normalized": normalize_plate_text(pred),
            })
    if rows:
        with open(output_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"  [MISMATCHES] Saved: {output_path}")


def main():
    root_dir = Path(r"C:\Users\Admin\ducmanhjr\sosanh")
    labels_path = root_dir / "ocr_dataset" / "test_labels.txt"

    print("Loading ground truth labels...")
    gt = load_labels(str(labels_path))
    print(f"  Loaded {len(gt)} ground truth labels.\n")

    all_stats = []

    # --- Model 1: CRNN standalone ---
    print("[1/3] Loading Model 1 results...")
    m1_res = load_model12_results(str(root_dir / "model1_results.csv"))
    s1 = evaluate_model("Model 1 (CRNN v1)", m1_res, gt)
    if s1:
        all_stats.append(s1)

    # --- Model 2: CRNN standalone ---
    print("[2/3] Loading Model 2 results...")
    m2_res = load_model12_results(str(root_dir / "model2_results.csv"))
    s2 = evaluate_model("Model 2 (CRNN v2)", m2_res, gt)
    if s2:
        all_stats.append(s2)

    # --- Model 3: Ensemble (PaddleOCR + PyTorch + merge rules) ---
    print("[3/3] Loading Model 3 results (Ensemble)...")
    m3_res = load_model3_results(str(root_dir / "model3_results_fixed.csv"))
    s3 = evaluate_model("Model 3 (Ensemble)", m3_res, gt)
    if s3:
        all_stats.append(s3)

    # --- In kết quả ---
    if all_stats:
        print_comparison_table(all_stats)
        save_detailed_report(all_stats, str(root_dir / "comparison_report.csv"))
        save_mismatches(all_stats, str(root_dir / "all_mismatches.csv"))
    else:
        print("No results to compare.")


if __name__ == "__main__":
    main()
