# Giai doan tim vung bien so

## Muc tieu

Du an nay dung de tim va cat vung anh bien so xe trong video. Ket qua mong muon la anh crop bien so ro, gan camera, it bi mo, it bi cat sat ky tu.

Giai doan nay chi lam:

- Phat hien va track phuong tien.
- Tim vung bien so trong tung xe.
- Luu tat ca vung bien so ung vien.
- Loc ra cac vung bien so tot nhat.

Giai doan nay chua lam:

- Chua OCR.
- Chua doc chuoi ky tu tren bien.
- Chua kiem tra dung/sai noi dung bien so.

File chay chinh hien tai la `main.py`.

## Y tuong chinh

Code khong con dua vao vach/cat vach. Thay vao do, video duoc xu ly theo ROI gan camera. ROI la vung trong khung hinh noi bien so thuong ro hon, lon hon va de cat hon.

Voi moi frame:

1. YOLO phat hien va track xe.
2. Chi giu xe co tam nam trong ROI.
3. Cat vung xe tu frame goc sach, khong co khung ve debug.
4. Chay model bien so trong crop xe.
5. Neu tim thay bien so, tinh diem chat luong.
6. Luu tat ca candidate vao `outputs/plates_all/`.
7. Moi track xe chi giu candidate co diem tot nhat.
8. Cuoi video, loc bot ket qua trung/lap va luu ket qua cuoi vao `outputs/plates_filtered/`.

## Cach cham diem bien so

Moi candidate bien so duoc cham diem tu nhieu yeu to:

- Do tin cay cua model bien so.
- Do net cua crop.
- Do tuong phan cua crop.
- Kich thuoc bien so.
- Ti le dien tich bien so so voi crop xe.

Muc tieu la uu tien crop bien so vua ro, vua du lon, vua it bi cat sat.

## ROI

ROI co dang:

```text
x1,y1,x2,y2
```

Tat ca gia tri la ti le tu `0.0` den `1.0`.

Vi du mac dinh:

```text
0.0,0.55,1.0,1.0
```

Nghia la:

- `0.0`: bat dau tu mep trai.
- `0.55`: bat dau tu 55% chieu cao frame.
- `1.0`: ket thuc o mep phai.
- `1.0`: ket thuc o day frame.

Tuc la lay toan bo chieu ngang, nhung chi xu ly vung duoi frame tu 55% tro xuong.

Mot so ROI co the thu:

```bash
# Vung duoi mac dinh
py main.py --no-video --roi 0.0,0.55,1.0,1.0

# Lay rong hon, tu nua frame tro xuong
py main.py --no-video --roi 0.0,0.5,1.0,1.0

# Lay 3/4 duoi frame
py main.py --no-video --roi 0.0,0.25,1.0,1.0

# Bo bot hai mep trai/phai, chi lay giua-duoi
py main.py --no-video --roi 0.15,0.45,0.85,1.0
```

## Dau vao

```text
data/video.mp4
```

Model can co:

```text
models/yolov8m.pt
models/license_plate_detector.pt
```

## Dau ra

Sau khi chay, ket qua nam trong `outputs/`.

```text
outputs/
|-- plates_all/
|-- plates_filtered/
|-- vehicles/
|-- reviews/
|-- violators.csv
|-- tracks.csv
|-- violations.db
`-- output.mp4
```

Y nghia:

- `plates_all/`: tat ca anh bien so ung vien tim duoc trong qua trinh quet video.
- `plates_filtered/`: anh bien so da loc, moi track giu candidate tot hon va da qua buoc gop trung.
- `vehicles/`: crop xe tu frame co bien so tot nhat.
- `reviews/`: anh review gom xe, bien so va thong tin diem.
- `violators.csv`: bang tong hop ket qua da loc.
- `tracks.csv`: thong ke cac track xe.
- `violations.db`: database SQLite luu ket qua.
- `output.mp4`: video debug co ve khung ROI, xe va bien so.

## Cach chay

Chay nhanh, khong xuat video:

```bash
py main.py --no-video
```

Chay day du va xuat video debug:

```bash
py main.py
```

Dung video khac:

```bash
py main.py --video path/to/video.mp4 --no-video
```

## Tham so hay chinh

```bash
--roi
```

Vung xu ly gan camera. Day la tham so nen chinh dau tien neu ket qua qua nhieu, qua it, hoac bien so chua o goc tot.

```bash
--plate-conf
```

Nguong confidence cua model bien so. Tang len de loc chat hon, giam xuong de bat duoc nhieu hon.

```bash
--min-plate-score
```

Nguong diem chat luong de dua vao ket qua filtered. Tang len neu co nhieu crop xau, giam xuong neu bi thieu bien.

```bash
--min-plate-width
--min-plate-height
```

Kich thuoc toi thieu cua crop bien so. Tang len de bo bien qua nho.

```bash
--plate-padding
```

Mo rong bbox bien so truoc khi luu. Tang len neu crop hay bi cat sat ky tu.

```bash
--plate-interval
```

Tan suat detect bien so. Gia tri `1` la detect moi frame, chinh xac hon nhung cham hon. Gia tri lon hon chay nhanh hon nhung co the bo lo frame dep.

## Vi du lenh thuc te

Bat nhieu bien hon:

```bash
py main.py --no-video --min-plate-score 0.48 --plate-conf 0.25
```

Loc chat hon:

```bash
py main.py --no-video --min-plate-score 0.6 --plate-conf 0.4
```

Tim trong vung gan camera hon:

```bash
py main.py --no-video --roi 0.0,0.65,1.0,1.0
```

## Luu y

- Neu `plates_all/` co bien dung nhung `plates_filtered/` khong co, hay giam `--min-plate-score`.
- Neu `plates_filtered/` co nhieu anh xau, hay tang `--min-plate-score`, `--plate-conf`, hoac thu hep ROI.
- Neu crop bi cat sat ky tu, hay tang `--plate-padding`.
- Neu van bi lap cung mot bien, can them OCR hoac gop theo noi dung bien so de chinh xac hon.
