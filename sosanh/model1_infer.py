import os
import sys
import time
import csv
from pathlib import Path

# Add model 1 to path
sys.path.insert(0, r"C:\Users\Admin\ducmanhjr\sosanh\model 1")

from src.inference.predictor import Predictor

def main():
    test_dir = Path(r"C:\Users\Admin\ducmanhjr\sosanh\ocr_dataset\test")
    ckpt_path = r"C:\Users\Admin\ducmanhjr\sosanh\model 1\runs\crnn_base\best.pt"
    output_csv = r"C:\Users\Admin\ducmanhjr\sosanh\model1_results.csv"
    
    print("Loading Model 1...")
    predictor = Predictor.from_checkpoint(ckpt_path)
    print("Model 1 loaded successfully. Running inference...")
    
    # Get all jpg files
    images = sorted([p for p in test_dir.iterdir() if p.is_file() and p.suffix.lower() == ".jpg"])
    print(f"Found {len(images)} images in test set.")
    
    results = []
    for idx, img_path in enumerate(images):
        start_time = time.perf_counter()
        try:
            pred_text = predictor.predict_image(str(img_path))
        except Exception as e:
            pred_text = ""
            print(f"Error on {img_path.name}: {e}")
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        
        results.append({
            "image": img_path.name,
            "prediction": pred_text,
            "latency_ms": latency_ms
        })
        
        if (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1}/{len(images)} images...")
            
    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "prediction", "latency_ms"])
        writer.writeheader()
        writer.writerows(results)
        
    print(f"Done Model 1! Results saved to {output_csv}")

if __name__ == "__main__":
    main()
