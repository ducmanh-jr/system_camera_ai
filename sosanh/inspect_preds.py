import csv
target_images = ['mb_30.jpg', 'mb_306.jpg', 'mb_505.jpg', 'type1_450.jpg']
with open('model3_results_fixed.csv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row['image'] in target_images:
            print(f"{row['image']}:")
            print(f"  GT:        {row.get('ground_truth')}")
            print(f"  PaddleOCR: {row.get('train1_paddleocr_pred')}")
            print(f"  PyTorch:   {row.get('train2_best_ema_pred')}")
            print(f"  Ensemble:  {row.get('prediction')}")
