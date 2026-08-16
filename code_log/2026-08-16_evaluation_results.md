# 📊 Báo cáo Kết quả: So sánh SIDA trên Ảnh Gốc vs Ảnh Nén MXH

> **Ngày chạy:** 16/08/2026  
> **Mô hình:** SIDA-7B (4-bit quantization)  
> **GPU:** Google Colab T4 (16GB VRAM)  
> **Dataset:** 51 ảnh gốc + 51 ảnh nén mô phỏng mạng xã hội

---

## 1. Tổng hợp kết quả phân loại

### Ảnh GỐC (`./examples`)

| Phân loại | Số lượng | Tỷ lệ |
|-----------|----------|--------|
| **Tampered** (giả mạo) | 1 (image 50) | 2.0% |
| **Real** (thật) | 1 (image 14) | 2.0% |
| **Full Synthetic** (nhân tạo) | 49 | 96.0% |

### Ảnh ĐÃ NÉN (`./examples_compressed`)

| Phân loại | Số lượng | Tỷ lệ |
|-----------|----------|--------|
| **Tampered** (giả mạo) | 1 (image 50) | 2.0% |
| **Real** (thật) | 1 (image 8) | 2.0% |
| **Full Synthetic** (nhân tạo) | 49 | 96.0% |

---

## 2. ⚡ Phát hiện quan trọng: Nén ảnh làm thay đổi phân loại

Mặc dù **tỷ lệ tổng thể** giống nhau (1 tampered, 1 real, 49 synthetic), nhưng **danh tính các ảnh bị phân loại sai đã thay đổi**:

| Ảnh | Kết quả GỐC | Kết quả SAU NÉN | Nhận xét |
|-----|-------------|-----------------|----------|
| `image (14).jpg` | ✅ **Real** | ❌ **Synthetic** | Nén làm mất dấu hiệu "thật" → bị hiểu nhầm thành ảnh nhân tạo |
| `image (8).jpg` | **Synthetic** | ❌ **Real** | Nén làm mờ dấu hiệu "giả" → bị hiểu nhầm thành ảnh thật |
| `image (50).jpg` | **Tampered** ✅ | **Tampered** ✅ | Kết quả giữ nguyên (mask vẫn được tạo) |

> [!WARNING]
> **Kết luận:** Nén ảnh mạng xã hội đã khiến SIDA **phân loại sai ít nhất 2/51 ảnh (3.9%)**. Đặc biệt nguy hiểm:
> - Ảnh thật bị phân loại thành giả → **False Positive**  
> - Ảnh giả bị phân loại thành thật → **False Negative** (bỏ lọt deepfake!)

---

## 3. ⚠️ Vấn đề cần lưu ý: Thiên lệch phân loại (Classification Bias)

Mô hình phân loại **96% ảnh là "full synthetic"**, điều này gợi ý một số khả năng:

### Khả năng 1: Dataset thực sự chứa nhiều ảnh synthetic
- Nếu 51 ảnh test đều là ảnh deepfake/AI-generated → kết quả này là hợp lý
- **Cần kiểm tra:** Ground truth label của từng ảnh trong bộ test

### Khả năng 2: 4-bit quantization gây suy giảm chất lượng phân loại
- Nén mô hình từ FP16 xuống 4-bit có thể làm mất precision
- **Kiểm chứng:** So sánh với kết quả chạy ở FP16/8-bit (cần GPU mạnh hơn)

### Khả năng 3: Hàm `evaluate()` đã được viết lại
- Do bất tương thích transformers, hàm evaluate đã phải viết lại
- Lần forward pass thứ 2 (để lấy hidden states) có thể cho kết quả hơi khác so với code gốc
- **Kiểm chứng:** Thử chạy với vài ảnh chắc chắn là "real" (ảnh chụp camera thật)

---

## 4. 🔬 Đề xuất bước tiếp theo

### Ngắn hạn (có thể làm ngay):
1. [ ] **Kiểm tra ground truth:** Xác định xem 51 ảnh test thuộc class nào (real/tampered/synthetic)
2. [ ] **Thử với ảnh real chắc chắn:** Upload vài bức ảnh chụp từ camera thật vào `examples/` và chạy lại
3. [ ] **So sánh mask quality:** So sánh mask của `image (50)` giữa bản gốc và bản nén (IoU)

### Trung hạn (cần chuẩn bị):
4. [ ] **Dùng dataset chuẩn:** Chạy trên CASIA v2 hoặc Columbia Uncompressed (có ground truth rõ ràng)
5. [ ] **Tăng số lượng ảnh test:** 51 ảnh quá ít để kết luận thống kê
6. [ ] **Thử nhiều mức nén:** Q=80, Q=60, Q=40, Q=20 để vẽ đường cong suy giảm

### Dài hạn (cho bài nghiên cứu):
7. [ ] Triển khai 1 trong 3 giải pháp (A/B/C) từ implementation_plan
8. [ ] Viết báo cáo với biểu đồ so sánh

---

## 5. File kết quả đầu ra

```
/content/SIDA/
├── vis_output_original/        ← Kết quả 51 ảnh gốc
│   ├── image (50)_mask_0.jpg   ← Mask vùng giả mạo
│   ├── image (50)_masked_img_0.jpg
│   └── results.csv             ← Bảng tổng hợp phân loại
├── vis_output_compressed/      ← Kết quả 51 ảnh nén
│   ├── image (50)_mask_0.jpg
│   ├── image (50)_masked_img_0.jpg
│   └── results.csv
```
