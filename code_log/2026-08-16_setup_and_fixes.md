# 📋 Change Log — 2026-08-16: Tương thích SIDA với Google Colab (Python 3.12 + Transformers mới)

> **Ngày:** 16/08/2026  
> **Mục tiêu:** Chạy SIDA-7B gốc trên Google Colab mà KHÔNG thay đổi logic của tác giả  
> **Nguyên tắc:** Chỉ sửa import, version, và compatibility wrapper

---

## Tổng kết thay đổi so với code gốc

| File | Thay đổi | Loại |
|------|---------|------|
| `model/SIDA.py` | Comment out `import deepspeed` | Import |
| `model/SIDA.py` | Dùng `forward()` thay vì `generate()` để lấy hidden_states + dynamic alignment | Compatibility |
| `model/llava/model/__init__.py` | Comment out import MPT | Import |
| `model/llava/model/language_model/llava_llama.py` | Thêm `exist_ok=True` cho `AutoConfig.register` | Compatibility |
| `model/llava/model/llava_arch.py` | Thêm `if attention_mask is not None:` guard | Null safety |
| `requirements.txt` | Bỏ packages không cần (ray, deepspeed, openai...), bỏ version lock cũ | Version |
| `colab_eval.py` | **MỚI** — Script batch inference cho Colab | New file |
| `simulate_compression.py` | **MỚI** — Mô phỏng nén ảnh mạng xã hội | New file |

---

## Chi tiết lý do từng thay đổi

### 1. `model/SIDA.py` — `import deepspeed` → comment out
- `deepspeed` không cài được trên Colab miễn phí (cần CUDA dev toolkit)
- SIDA không dùng deepspeed khi inference

### 2. `model/SIDA.py` — hidden_states extraction
- **Vấn đề:** `generate()` trả `hidden_states` format khác nhau giữa transformers 4.31 và transformers mới
- **Giải pháp:** Dùng `generate()` CHỈ lấy `output_ids`, sau đó gọi `forward()` để lấy `hidden_states` (format ổn định trên mọi phiên bản)
- **Alignment:** Thêm dynamic alignment giữa mask và hidden_states (lệch 1 token do BOS)
- **Logic gốc** (cls_head, mask, segmentation, attention, SAM): **100% giữ nguyên**

### 3. `model/llava/model/__init__.py` — comment out MPT import
- `_expand_mask` bị xóa trong transformers 4.40+
- SIDA không sử dụng MPT architecture

### 4. `model/llava/model/language_model/llava_llama.py` — `exist_ok=True`
- Transformers mới đã tích hợp sẵn model type "llava"
- Gọi `register()` trùng sẽ raise `ValueError`

### 5. `model/llava/model/llava_arch.py` — attention_mask guard
- Transformers mới không luôn truyền `attention_mask` vào `prepare_inputs_labels_for_multimodal()`
- Truy cập `.shape` trên `None` gây `AttributeError`

### 6. `requirements.txt` — dọn sạch
- Bỏ: `ray`, `deepspeed`, `openai`, `fastapi`, `gradio`, `uvicorn` (không cần cho inference)
- Bỏ version lock: `torch`, `transformers`, `numpy` (xung đột Python 3.12)
- Nâng: `bitsandbytes>=0.46.1` (yêu cầu của transformers mới)

---

## Hướng dẫn chạy trên Colab

```
Ô 1: Mount Drive + giải nén SIDA.zip
Ô 2: pip install -r requirements.txt + bitsandbytes accelerate torchviz
Ô 3: python colab_eval.py --version='saberzl/SIDA-7B' --load_in_4bit --image_dir='./examples' --vis_save_path='./vis_output_original'
Ô 4: Test từng ảnh riêng lẻ (tùy chọn)
Ô 5: Tải kết quả về máy (tùy chọn)
```
