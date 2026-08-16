# 📋 Tổng kết thay đổi — CHỈ tương thích thư viện (không thay đổi logic)

> Khôi phục từ git + chỉ giữ lại các thay đổi tối thiểu

## Danh sách thay đổi

| File | Thay đổi | Loại |
|------|---------|------|
| `model/SIDA.py` | Comment out `import deepspeed` | Import |
| `model/SIDA.py` | Wrapper xử lý `hidden_states` format khác nhau giữa các phiên bản transformers | Compatibility |
| `model/llava/model/__init__.py` | Comment out import MPT (`_expand_mask` bị xóa trong transformers 4.40+) | Import |
| `model/llava/model/language_model/llava_llama.py` | Thêm `exist_ok=True` cho `AutoConfig.register` | Compatibility |
| `model/llava/model/llava_arch.py` | Thêm `if attention_mask is not None:` guard | Null safety |
| `requirements.txt` | Comment out version lock cũ (torch, transformers, deepspeed, numpy, ray), nâng bitsandbytes | Version |

## Nguyên tắc
- ✅ KHÔNG thay đổi bất kỳ logic xử lý nào (cls_head, mask, segmentation, attention)
- ✅ Chỉ sửa import, version, và compatibility wrapper
- ✅ Code gốc của tác giả được giữ nguyên 100%
