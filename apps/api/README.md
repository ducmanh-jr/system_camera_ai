# VietTraffic AI Web/API

Backend FastAPI + dashboard web cho du an `thi_giac`.

## Chay server

Tu thu muc goc:

```powershell
py -m uvicorn apps.api.main:app --host 127.0.0.1 --port 8000
```

Mo:

```text
http://127.0.0.1:8000/
```

## API chinh

- `GET /api/health`: kiem tra backend va duong dan OCR model.
- `POST /api/jobs?mode=plate`: upload video/anh de cat bien so va doc OCR.
- `POST /api/jobs?mode=wrong_lane`: upload video de phan tich sai lan.
- `GET /api/jobs`: danh sach job.
- `GET /api/jobs/{job_id}`: trang thai va ket qua job.

## Ràng buộc OCR

Phần đọc ký tự biển số sử dụng adapter trong `apps/api/ocr_adapter.py`, nạp model `PlateOCREnsemble` từ:

```text
train_ocr/
```

Cụ thể:
- Model ensemble được khai báo trong `train_ocr/src/pipeline.py`.
- Logic hậu xử lý (post-processing) và kiểm chuẩn chữ cái tiếng Việt (bao gồm chữ 'Đ') đã được tích hợp trực tiếp (in-line) trong adapter `apps/api/ocr_adapter.py` nhằm tối ưu độ chính xác và tránh phụ thuộc vào module cũ.

## Luu y hien tai

- Cac pipeline cu dang ghi ra thu muc `outputs` rieng cua tung module. Backend co lock de moi lan chi chay mot pipeline, sau do copy ket qua sang `runtime/viettraffic/jobs/{job_id}`.
- Neu muon chay nhieu job song song that su, buoc tiep theo nen refactor pipeline cu de nhan `output_dir` rieng cho tung job.
- Che do `plate` hien dung `giai_doan_tim_bien/main.py` de cat crop bien so, sau do moi doc OCR bang `giai_doan_doc_ki_tu`.
- Che do `wrong_lane` dung `module_di_sai_lan_duong/main.py` va doc ket qua CSV/anh bang chung.
