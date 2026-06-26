import csv
import re
from collections import Counter

def normalize(text):
    text = str(text or "").upper().strip()
    return re.sub(r"[^0-9A-Z\u0110]", "", text)

# Load mismatches
mismatches = []
with open("all_mismatches.csv", "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        mismatches.append(row)

m1_wrong = [r for r in mismatches if r["model"] == "Model 1 (CRNN v1)"]
m3_wrong = [r for r in mismatches if r["model"] == "Model 3 (Ensemble)"]

print(f"Model 1 sai: {len(m1_wrong)} bien")
print(f"Model 3 sai: {len(m3_wrong)} bien")

m1_wrong_imgs = set(r["image"] for r in m1_wrong)
m3_wrong_imgs = set(r["image"] for r in m3_wrong)

m3_only_wrong = m3_wrong_imgs - m1_wrong_imgs
m1_only_wrong = m1_wrong_imgs - m3_wrong_imgs
both_wrong = m1_wrong_imgs & m3_wrong_imgs

print(f"\nChi Model 3 sai (Model 1 dung): {len(m3_only_wrong)}")
print(f"Chi Model 1 sai (Model 3 dung): {len(m1_only_wrong)}")
print(f"Ca 2 deu sai: {len(both_wrong)}")

# Show examples where only Model 3 is wrong
m3_dict = {r["image"]: r for r in m3_wrong}
m1_dict = {r["image"]: r for r in m1_wrong}

print("\n--- 15 VD: Model 3 SAI, Model 1 DUNG ---")
for i, img in enumerate(sorted(m3_only_wrong)):
    r = m3_dict[img]
    gt_n = normalize(r["ground_truth"])
    pred_n = normalize(r["prediction"])
    print(f"  {img}: GT=[{r['ground_truth']}] M3=[{r['prediction']}] | norm: GT={gt_n} M3={pred_n}")
    if i >= 14:
        break

print("\n--- 10 VD: Model 1 SAI, Model 3 DUNG ---")
for i, img in enumerate(sorted(m1_only_wrong)):
    r = m1_dict[img]
    gt_n = normalize(r["ground_truth"])
    pred_n = normalize(r["prediction"])
    print(f"  {img}: GT=[{r['ground_truth']}] M1=[{r['prediction']}] | norm: GT={gt_n} M1={pred_n}")
    if i >= 9:
        break

# Error pattern analysis for Model 3
print("\n--- PHAN TICH LOI MODEL 3 ---")
len_diff_count = 0
digit_err = 0
letter_err = 0
space_err = 0
for r in m3_wrong:
    gt_n = normalize(r["ground_truth"])
    pred_n = normalize(r["prediction"])
    if len(gt_n) != len(pred_n):
        len_diff_count += 1
    for gc, pc in zip(gt_n, pred_n):
        if gc != pc:
            if gc.isdigit() and pc.isdigit():
                digit_err += 1
            elif gc.isalpha() and pc.isalpha():
                letter_err += 1
            else:
                space_err += 1

print(f"  Sai do khac do dai (them/thieu ky tu): {len_diff_count}/{len(m3_wrong)}")
print(f"  Nham so voi so (0<->8, 1<->7...): {digit_err}")
print(f"  Nham chu voi chu (A<->4, D<->0...): {letter_err}")
print(f"  Nham loai khac (chu<->so): {space_err}")

# Same analysis for Model 1
print("\n--- PHAN TICH LOI MODEL 1 ---")
len_diff_count = 0
digit_err = 0
letter_err = 0
space_err = 0
for r in m1_wrong:
    gt_n = normalize(r["ground_truth"])
    pred_n = normalize(r["prediction"])
    if len(gt_n) != len(pred_n):
        len_diff_count += 1
    for gc, pc in zip(gt_n, pred_n):
        if gc != pc:
            if gc.isdigit() and pc.isdigit():
                digit_err += 1
            elif gc.isalpha() and pc.isalpha():
                letter_err += 1
            else:
                space_err += 1

print(f"  Sai do khac do dai (them/thieu ky tu): {len_diff_count}/{len(m1_wrong)}")
print(f"  Nham so voi so (0<->8, 1<->7...): {digit_err}")
print(f"  Nham chu voi chu (A<->4, D<->0...): {letter_err}")
print(f"  Nham loai khac (chu<->so): {space_err}")
