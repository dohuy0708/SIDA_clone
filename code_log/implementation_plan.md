# Kế hoạch Nghiên cứu: Đánh giá và Cải thiện Độ Robust của SIDA trước nén ảnh mạng xã hội

Kế hoạch này vạch ra lộ trình chi tiết để bạn có thể thực hiện bài nghiên cứu khoa học của mình, giải quyết cả vấn đề về phần cứng (máy yếu) và yêu cầu chuyên môn từ Thầy hướng dẫn (tìm dataset chứng minh điểm yếu và đề xuất giải pháp).

## 1. Giải quyết vấn đề Phần cứng (Card 4GB VRAM)

Với cấu hình hiện tại, bạn không thể chạy trực tiếp SIDA trên laptop. Để làm nghiên cứu, bạn cần sử dụng Điện toán đám mây (Cloud Computing).

**Giải pháp đề xuất:**
*   **Sử dụng Google Colab (Miễn phí):** Bạn có thể đưa code này lên Google Colab. Colab cấp miễn phí GPU T4 (16GB VRAM), đủ để bạn chạy bản **SIDA-7B** (với tùy chọn `--load_in_4bit` hoặc `--load_in_8bit`).
*   **Kaggle (Miễn phí):** Tương tự Colab, cấp GPU P100 hoặc T4x2 (16GB VRAM) mỗi tuần 30 tiếng.
*   **Thuê GPU (Có phí nhưng cực rẻ):** Nếu cần train lại mô hình (Fine-tuning), bạn có thể thuê GPU trên Vast.ai hoặc RunPod (khoảng $0.2 - $0.4 / giờ cho card RTX 3090 24GB).

## 2. Tìm/Tạo Tập dữ liệu (Dataset) để kiểm nghiệm điểm yếu

Thay vì mất công tìm kiếm một bộ dữ liệu có sẵn đã bị nén (rất khó để biết chính xác nó đã bị nén qua các bước nào), **giải pháp chuẩn nhất trong nghiên cứu khoa học là Tự mô phỏng (Simulate) quá trình nén của mạng xã hội trên một bộ dữ liệu chuẩn**.

**Phương pháp thực hiện:**
1.  **Lấy tập dữ liệu gốc:** Lấy ngay một phần của tập **SID-Set** (tập Test) hoặc các tập chuẩn như **CASIA v2**, **Columbia Uncompressed**.
2.  **Viết Script mô phỏng nén (Compression Simulator):** Chúng ta sẽ viết một đoạn code Python (dùng thư viện `OpenCV` hoặc `Pillow`, `Albumentations`) để tạo ra một "Facebook/Messenger Filter". Pipeline này sẽ thực hiện:
    *   *Downscaling:* Resize ảnh nếu cạnh dài > 2048px (chuẩn Facebook).
    *   *JPEG Compression:* Nén ảnh với Quality thấp (ví dụ: Q=60, 40, 20) để tạo Block artifacts.
    *   *Chroma Subsampling:* Ép hệ màu về 4:2:0.
    *   *Multi-pass Compression:* Lưu đi lưu lại file JPEG 3-5 lần để mô phỏng việc ảnh bị đăng lại nhiều lần.
3.  **So sánh:** Chạy mô hình SIDA trên tập "Ảnh Gốc" và tập "Ảnh bị Nén". Tính chỉ số IoU (Intersection over Union) và so sánh sự sụt giảm để đưa vào báo cáo, chính thức chứng minh điểm yếu mà bạn đã lập luận.

## 3. Giải pháp khắc phục (Đề xuất cho Thầy hướng dẫn)

Sau khi chứng minh được SIDA nhận diện kém trên ảnh bị nén, đây là 3 hướng giải quyết bạn có thể chọn làm trọng tâm cho bài nghiên cứu:

### Giải pháp A: Data Augmentation (Tăng cường dữ liệu) - Dễ thực hiện nhất
*   **Cách làm:** Train/Fine-tune lại mô hình SIDA. Trong quá trình train, thay vì chỉ đưa ảnh rõ nét vào, ta sẽ chủ động áp dụng các phép biến đổi mô phỏng nén mạng xã hội (JPEG, WebP compression, Blur, Downscale) vào dữ liệu huấn luyện.
*   **Kỳ vọng:** SIDA sẽ "học" được cách bỏ qua nhiễu nén và tập trung vào các đặc trưng giả mạo cốt lõi ở tần số thấp hơn.

### Giải pháp B: Image Restoration Pre-processing (Khôi phục ảnh tiền xử lý) - Khả thi cao, không cần train lại SIDA
*   **Cách làm:** Trước khi đưa ảnh vào SIDA, ta dùng một mạng Neural nhỏ chuyên làm nhiệm vụ khử nhiễu nén JPEG (JPEG Artifact Removal) hoặc siêu phân giải (như SwinIR, Real-ESRGAN).
*   **Kỳ vọng:** Hình ảnh được khôi phục lại các chi tiết vùng biên (edge) bị mất do nén, giúp SAM (mô hình tạo mask của SIDA) cắt ranh giới chính xác hơn.

### Giải pháp C: Multi-Frequency Feature Fusion (Can thiệp vào kiến trúc) - Khó, dành cho đồ án tốt nghiệp xuất sắc
*   **Cách làm:** Sửa đổi kiến trúc của mô hình. Nén JPEG thường chỉ làm mất thông tin ở dải tần số cao (High-frequency) nhưng giữ lại dải tần số thấp (Low-frequency). Ta sẽ trích xuất thêm đặc trưng trong không gian miền tần số (DCT coefficients) thay vì chỉ nhìn vào điểm ảnh (pixel/RGB) rồi cung cấp cho SIDA.

---

## User Review Required

> [!IMPORTANT]  
> Xin bạn hãy xem xét Kế hoạch này. Để bắt đầu, bạn muốn đi theo hướng nào?
> 1. Tôi sẽ hướng dẫn bạn cách thiết lập để chạy trên **Google Colab** hoặc **Kaggle** miễn phí.
> 2. Tôi sẽ viết giúp bạn **Script Python mô phỏng nén ảnh mạng xã hội** để bạn tự tạo Dataset kiểm chứng điểm yếu.
> 3. Bạn muốn thảo luận sâu hơn về 3 giải pháp khắc phục (A, B, C) để báo cáo với Thầy hướng dẫn.

Hãy cho tôi biết lựa chọn của bạn!
