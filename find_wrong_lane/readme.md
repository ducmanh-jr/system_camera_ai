# Module phat hien xe di sai lan

Module nay dung YOLO tracking de theo doi phuong tien, xac dinh lan xuat phat bang polygon cau hinh san, phan loai huong di chuyen bang vector quy dao, sau do xuat anh cac phuong tien vi pham.

## Dau vao

- Video mac dinh: `../giai_doan_tim_bien/data/video.mp4`
- Model mac dinh: `../giai_doan_tim_bien/models/yolov8m.pt`
- Cau hinh lan: `lane_config.json`

## Dau ra

Sau khi chay, ket qua nam trong `outputs/`:

```text
outputs/
|-- violating_vehicles/   # anh crop phuong tien vi pham
|-- reviews/              # anh frame co polygon lan, quy dao va ly do vi pham
|-- violations.csv
|-- violations.db
`-- debug_wrong_lane.mp4
```

Thu muc quan trong nhat theo yeu cau la:

```text
outputs/violating_vehicles/
```

Anh trong `violating_vehicles/` duoc chon theo diem chat luong:

- Xe cang gan camera thi bbox cang lon, diem cang cao.
- Anh cang net thi Laplacian variance cang cao, diem cang cao.
- Diem cuoi cung ket hop do gan va do net, giup uu tien anh xe vi pham ro nhat thay vi chi lay frame cuoi.

## Cach chay

Chay nhanh, chi xuat anh va bang ket qua:

```bash
py main.py --no-video
```

Chay day du kem video debug:

```bash
py main.py
```

Dung video khac:

```bash
py main.py --video path/to/video.mp4 --no-video
```

## Luat dang ap dung

Module chi xuat anh phuong tien khi co du bang chung theo cac truong hop sau:

| Ma | Lan xuat phat | Huong thuc te | Ket luan | Muc phat |
| --- | --- | --- | --- | --- |
| `CASE_3` | `left` | `straight` | Di thang tren lan bat buoc re trai | O to/xe tai/xe khach: 400.000 - 600.000 VND; xe may: 100.000 - 200.000 VND |
| `CASE_4` | `straight` | `turn_left` | Re trai tren lan bat buoc di thang | O to/xe tai/xe khach: 400.000 - 600.000 VND; xe may: 100.000 - 200.000 VND |
| `CASE_6` | `left` | `turn_left` | Doi sang lan/vung di thang roi moi re trai | O to/xe tai/xe khach: 400.000 - 600.000 VND; xe may: 100.000 - 200.000 VND |

Ten loi phap luat ghi vao ket qua:

```text
Khong chap hanh hieu lenh, chi dan cua bien bao hieu, vach ke duong
```

Can cu hien thi trong ket qua: `Nghi dinh 168/2024/ND-CP`.

De giam bat oan, module chi ket luan khi:

- Xe co lan xuat phat ro trong vung `left` hoac `straight`.
- Xe di toi vung dich ro trong `left_exit`, `straight_exit` hoac `far_straight_exit`.
- Huong thuc te khong thuoc danh sach `allowed` cua lan xuat phat.
- Rieng `CASE_6`: xe xuat phat tu `left`, co di qua `straight`/`straight_exit`, sau do ket thuc o `left_exit`.

## Bang chuoi hanh vi

Quy uoc vung:

| So | Vung | Y nghia |
| --- | --- | --- |
| `1` | `left` | Lan re trai truoc vach dung |
| `2` | `straight` | Lan di thang truoc vach dung |
| `3` | `left_exit` | Vung dich re trai |
| `4` | `straight_exit` | Vung trung gian tren truc di thang, chua ket luan di thang |
| `5` | `far_straight_exit` | Vung xac nhan xe di thang |

Quy tac ket luan:

| Chuoi hanh vi | Ket luan |
| --- | --- |
| `1 -> 3` | Hop le |
| `1 -> 3 -> 3` | Hop le |
| `2 -> 5` | Hop le |
| `2 -> 4 -> 5` | Hop le |
| `2 -> 5 -> 5` | Hop le |
| `1 -> 5` | Vi pham `CASE_3`: di thang tren lan bat buoc re trai |
| `1 -> 3 -> 5` | Vi pham `CASE_3`: di thang tren lan bat buoc re trai |
| `1 -> 4 -> 5` | Vi pham `CASE_3`: di thang tren lan bat buoc re trai |
| `1 -> 2 -> 5` | Vi pham `CASE_3`: di thang tren lan bat buoc re trai |
| `2 -> 3` | Vi pham `CASE_4`: re trai tren lan bat buoc di thang |
| `2 -> 4 -> 3` | Vi pham `CASE_4`: re trai tren lan bat buoc di thang |
| `2 -> 5 -> 3` | Vi pham `CASE_4`: re trai tren lan bat buoc di thang |
| `2 -> 1 -> 3` | Vi pham `CASE_4`: re trai tren lan bat buoc di thang |
| `1 -> 2 -> 3` | Vi pham `CASE_6`: doi sang lan/vung di thang roi moi re trai |
| `1 -> 2 -> 4 -> 3` | Vi pham `CASE_6`: doi sang lan/vung di thang roi moi re trai |
| `1 -> 4 -> 3` | Vi pham `CASE_6`: doi sang lan/vung di thang roi moi re trai |
| `1 -> 2 -> 4 -> 5 -> 3` | Vi pham `CASE_6`: doi sang lan/vung di thang roi moi re trai |
| `1`, `2`, `3`, `4`, `5`, `1 -> 2`, `2 -> 1`, `1 -> 4`, `1 -> 2 -> 4`, `2 -> 4` | Chua du bang chung, khong ket luan |

Voi cac chuoi khong nam nguyen van trong bang, module ap dung quy tac tong quat:

- Bat dau tu `1`, ket thuc o `3`, va khong di qua `2`/`4`/`5`: hop le.
- Bat dau tu `2`, ket thuc o `5`: hop le.
- Bat dau tu `1`, chi can cham/di qua `5`: `CASE_3`.
- Bat dau tu `2`, ket thuc o `3`: `CASE_4`.
- Bat dau tu `1`, co di qua `2` hoac `4`, sau do ket thuc o `3`: `CASE_6`.
- Vung `4` chi la trung gian; neu xe moi cham `4` ma chua vao `5` hoac `3` thi khong ket luan.
- Rieng vung `5`: xe da xuat hien o `1` ma cham `5` thi ket luan `CASE_3` ngay, khong can doi track ket thuc o `5`.

## Chinh lan duong

Co 2 cach chinh lan: click truc tiep bang tool hieu chuan, hoac sua JSON thu cong.

### Cach 1: click de ve config

Chay:

```bash
py calibrate_config.py
```

Tool se mo frame video va ve cac polygon hien tai. Polygon co the co nhieu diem, khong gioi han 4 diem.

- `Ctrl+1`: chon `left` - lan re trai truoc vach dung.
- `Ctrl+2`: chon `straight` - lan di thang truoc vach dung.
- `Ctrl+3`: chon `left_exit` - vung dich neu xe re trai.
- `Ctrl+4`: chon `straight_exit` - vung dich neu xe di thang.
- `Ctrl+5`: chon `far_straight_exit` - vung di thang phia xa.
- Keo mot diem co san: di chuyen diem do.
- Keo tren mot canh polygon: chen them diem moi vao canh do va di chuyen diem moi.
- Click ngoai polygon: them diem moi vao cuoi danh sach diem.
- Chuot phai vao mot diem: xoa diem do.
- `c`: xoa toan bo diem cua vung dang chon.
- `r`: xoa tat ca vung va ve lai tu dau.
- `s`: luu vao `lane_config.json`.
- `q` hoac `ESC`: thoat.

Moi lan luu, tool tao backup config cu dang `lane_config.backup_YYYYMMDD_HHMMSS.json`.

Neu muon chon frame khac de ve:

```bash
py calibrate_config.py --frame 180
```

Neu muon ve tren anh co san:

```bash
py calibrate_config.py --image calibration_frames/frame_0100.jpg
```

### Cach 2: sua JSON thu cong

Mo `lane_config.json` va chinh cac polygon theo toa do ti le `x,y` trong khoang `0.0..1.0`.

Vi du:

```json
"left": {
  "name": "Lan re trai",
  "allowed": ["turn_left", "u_turn"],
  "polygon": [[0.02, 0.50], [0.45, 0.50], [0.50, 1.00], [0.00, 1.00]]
}
```

Logic mac dinh duoc dat theo huong uu tien do chinh xac:

- Xe bat dau tu `left` nhung di `straight` -> vi pham.
- Xe bat dau tu `straight` nhung `turn_left` -> vi pham.
- Chi ket luan khi track di toi vung dich ro rang trong `direction_zones`.
- Rule doi lan gan vach dung duoc tat mac dinh vi de gay bat nham khi xe nam sat vach chia lan.

Huong re trai duoc suy ra tu chuyen dong ngang cua tam xe. Neu camera bi nguoc huong, doi:

```json
"left_turn_dx_negative": false
```

## Tham so hay dung

```bash
py main.py --no-video --conf 0.25 --min-track-frames 8
```

- `--conf`: nguong confidence YOLO cho phuong tien.
- `--min-track-frames`: so frame toi thieu de mot track duoc danh gia.
- `--config`: dung file cau hinh lan khac.
- `--no-video`: bo qua video debug de chay nhanh hon.

## Luu y

`lane_config.json` hien la cau hinh mau. De ket qua chinh xac, can can polygon theo goc camera cua tung video. Cach nhanh nhat la chay co video debug, xem `outputs/debug_wrong_lane.mp4`, roi dieu chinh polygon cho khop vach/lan that.
