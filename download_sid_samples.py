import os
import argparse
import pandas as pd
from PIL import Image
from tqdm import tqdm

try:
    from datasets import load_dataset
except ImportError:
    print("Vui lòng cài đặt thư viện datasets: pip install datasets")
    exit(1)


def parse_args():
    parser = argparse.ArgumentParser(description="Download balanced samples from SID_Set on Hugging Face")
    parser.add_argument("--num_per_class", type=int, default=100, help="Số lượng ảnh cho mỗi nhãn (mặc định: 100)")
    parser.add_argument("--split", type=str, default="validation", help="Split dataset (validation hoặc test)")
    parser.add_argument("--output_dir", type=str, default="./examples", help="Thư mục lưu ảnh đầu ra")
    parser.add_argument("--save_masks", action="store_true", default=True, help="Lưu ground truth mask nếu có")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    mask_dir = os.path.join(args.output_dir, "gt_masks")
    if args.save_masks:
        os.makedirs(mask_dir, exist_ok=True)

    label_names = {
        0: "real",
        1: "synthetic",
        2: "tampered"
    }

    counts = {0: 0, 1: 0, 2: 0}
    target_count = args.num_per_class
    total_needed = target_count * 3

    print(f">>> Đang kết nối tới Hugging Face: saberzl/SID_Set (split='{args.split}')...")
    print(f">>> Mục tiêu: Tải {target_count} ảnh/nhãn x 3 nhãn = {total_needed} ảnh vào '{args.output_dir}'")

    # Sử dụng streaming=True để tải từng ảnh nhanh chóng mà KHÔNG cần tải toàn bộ 30.000 ảnh
    dataset = load_dataset("saberzl/SID_Set", split=args.split, streaming=True)

    metadata = []
    pbar = tqdm(total=total_needed, desc="Đang tải ảnh")

    for item in dataset:
        label = item.get("label", None)
        if label is None or label not in counts:
            continue

        if counts[label] >= target_count:
            # Nếu đã đủ số lượng cho nhãn này thì bỏ qua
            if all(c >= target_count for c in counts.values()):
                break
            continue

        counts[label] += 1
        idx = counts[label]
        cls_name = label_names[label]
        img_filename = f"{cls_name}_{idx:03d}.jpg"
        img_path = os.path.join(args.output_dir, img_filename)

        # Lưu ảnh
        img = item.get("image")
        if img is not None:
            if isinstance(img, Image.Image):
                img = img.convert("RGB")
                img.save(img_path, "JPEG", quality=95)
            else:
                # Trường hợp định dạng khác
                Image.fromarray(img).convert("RGB").save(img_path, "JPEG", quality=95)

        # Lưu mask (nếu là tampered và có mask)
        mask_filename = ""
        if args.save_masks and label == 2 and "mask" in item and item["mask"] is not None:
            mask = item["mask"]
            mask_filename = f"{cls_name}_{idx:03d}_mask.png"
            mask_path = os.path.join(mask_dir, mask_filename)
            if isinstance(mask, Image.Image):
                mask.save(mask_path)
            else:
                Image.fromarray(mask).save(mask_path)

        metadata.append({
            "filename": img_filename,
            "label_id": label,
            "label_name": cls_name,
            "ground_truth_mask": mask_filename
        })

        pbar.update(1)
        pbar.set_postfix({"real": counts[0], "synthetic": counts[1], "tampered": counts[2]})

        if all(c >= target_count for c in counts.values()):
            break

    pbar.close()

    # Lưu file CSV ground truth để sau này đối chiếu và tính IoU / Accuracy
    csv_path = os.path.join(args.output_dir, "ground_truth.csv")
    df = pd.DataFrame(metadata)
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    print("\n" + "="*50)
    print(f"✅ ĐÃ TẢI HOÀN TẤT BỘ DỮ LIỆU CHUẨN:")
    print(f" - Real (Thật): {counts[0]} ảnh")
    print(f" - Synthetic (Nhân tạo): {counts[1]} ảnh")
    print(f" - Tampered (Cắt ghép): {counts[2]} ảnh")
    print(f" - Tổng cộng: {sum(counts.values())} ảnh")
    print(f" - File danh sách nhãn chuẩn: {csv_path}")
    print("="*50)


if __name__ == "__main__":
    main()
