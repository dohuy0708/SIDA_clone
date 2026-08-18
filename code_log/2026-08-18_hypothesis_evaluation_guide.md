# 📑 Hướng Dẫn Tổng Hợp: Quy Trình Đánh Giá & Kiểm Chứng Giả Thuyết Nghiên Cứu

> **Đề tài:** Đánh giá tính dễ tổn thương của SIDA trước nén ảnh mạng xã hội (*Fragility of Localization in Deepfake Detection under Social Media Compression*).  
> **Mục tiêu:** Chứng minh định lượng rằng thuật toán nén ảnh mạng xã hội (JPEG Multi-pass / Downscaling) làm suy giảm nghiêm trọng độ chính xác phân loại và khả năng khoanh vùng vùng giả mạo (mIoU / Dice) của mô hình SIDA.

---

## 1. Cấu Trúc Bộ Dữ Liệu Thử Nghiệm

Bộ dữ liệu gồm **300 ảnh chuẩn** từ tập `validation` của tác giả (`saberzl/SID_Set`):
- 🟢 **100 ảnh Real:** Ảnh thật chụp tự nhiên.
- 🔵 **100 ảnh Synthetic:** Ảnh tạo 100% bằng AI (Diffusion/GAN).
- 🔴 **100 ảnh Tampered:** Ảnh bị chỉnh sửa/cắt ghép (kèm 100 mặt nạ Ground Truth Mask chuẩn).

Được chia thành 2 tập để so sánh đối chứng:
1. **`./examples/`**: 300 ảnh gốc chất lượng cao + Ground Truth Mask (`gt_masks/`) + `ground_truth.csv`.
2. **`./examples_compressed/`**: 300 ảnh tương ứng đã qua mô phỏng nén mạng xã hội (JPEG Quality=60, 2 passes).

---

## 2. Các Bước Thực Hiện Trên Google Colab

Toàn bộ quá trình chỉ gồm **4 ô code** sạch sẽ:

### Ô 1: Khởi tạo môi trường & Đồng bộ dữ liệu
```python
# ============================================================
# BƯỚC 1: ĐỒNG BỘ SOURCE CODE & DATASET TỪ GITHUB
# ============================================================
%cd /content
!rm -rf /content/SIDA
!git clone https://github.com/dohuy0708/SIDA_clone.git /content/SIDA
%cd /content/SIDA
!pip install -r requirements.txt
!pip install -U bitsandbytes>=0.46.1 accelerate torchviz scikit-learn matplotlib
```

---

### Ô 2: Chạy Đánh Giá Trên 300 Ảnh Gốc (Original)
```python
# ============================================================
# BƯỚC 2: CHẠY SIDA TRÊN 300 ẢNH GỐC
# Thời gian ước tính: ~5-7 phút trên GPU T4
# ============================================================
%cd /content/SIDA
!python colab_eval.py --version='saberzl/SIDA-7B' --load_in_4bit --image_dir='./examples' --vis_save_path='./vis_output_original'
```

---

### Ô 3: Chạy Đánh Giá Trên 300 Ảnh Bị Nén (Compressed)
```python
# ============================================================
# BƯỚC 3: CHẠY SIDA TRÊN 300 ẢNH BỊ NÉN MẠNG XÃ HỘI
# Thời gian ước tính: ~5-7 phút trên GPU T4
# ============================================================
%cd /content/SIDA
!python colab_eval.py --version='saberzl/SIDA-7B' --load_in_4bit --image_dir='./examples_compressed' --vis_save_path='./vis_output_compressed'
```

---

### Ô 4: Tự Động Tính Toán Chỉ Số & Xuất Báo Cáo Đối Chứng
```python
# ============================================================
# BƯỚC 4: TÍNH TOÁN METRICS (Accuracy, F1, mIoU, Dice) & XUẤT REPORT
# ============================================================
%cd /content/SIDA
!python compare_benchmark.py
```

---

## 3. Các Chỉ Số Đánh Giá Khoa Học (Metrics)

| Tiêu chí | Chỉ số | Ý nghĩa khoa học |
| :--- | :--- | :--- |
| **Phân loại (Classification)** | **Accuracy (%)** | Tỷ lệ nhận diện đúng chung trên cả 3 nhãn (Real, Synthetic, Tampered). |
| | **Macro F1-Score (%)** | Đo độ cân bằng chính xác trên từng nhãn, tránh thiên lệch. |
| **Khoanh vùng (Localization)** | **mIoU (%)** *(mean Intersection over Union)* | Độ khớp giữa vùng khoanh của SIDA so với vùng giả mạo thật. |
| | **mDice (%)** *(Dice Similarity Coefficient)* | Độ chính xác và độ nét ranh giới vùng giả mạo ở cấp độ pixel. |

---

## 4. Cách Đọc Kết Quả Để Báo Cáo Thầy Hướng Dẫn

Sau khi chạy xong Bước 4, bạn sẽ nhận được bảng tổng kết:
- **Nếu `mIoU` sụt giảm (ví dụ: từ 65% xuống còn 35%)**: Chứng minh thuật toán nén làm mất chi tiết tần số cao, khiến SAM cắt nhầm ranh giới hoặc không thể định vị vùng bị sửa.
- **Nếu `Accuracy / F1` sụt giảm**: Chứng minh nén JPEG tạo ra các artifact khối làm SIDA nhận nhầm ảnh thật thành ảnh do AI vẽ (`full synthetic`).

➔ **Kết luận bài nghiên cứu:** Đã chứng minh được điểm yếu của mô hình SIDA trước nén mạng xã hội, từ đó mở ra hướng giải quyết: *Tăng cường tiền xử lý khôi phục ảnh (Restoration Pre-processing)* hoặc *Tập huấn lại với kỹ thuật nén mô phỏng (Data Augmentation)*.
