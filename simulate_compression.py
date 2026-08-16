import cv2
import os
import argparse
from pathlib import Path

def simulate_social_media_compression(image_path, output_path, quality=60, max_size=2048, passes=1):
    """
    Mô phỏng quá trình nén của mạng xã hội: Downscale, JPEG Compression (lossy).
    """
    # 1. Đọc ảnh
    img = cv2.imread(image_path)
    if img is None:
        print(f"Không thể đọc ảnh: {image_path}")
        return False
        
    # 2. Downscaling (Mô phỏng resize của Facebook/Messenger)
    height, width = img.shape[:2]
    if max(height, width) > max_size:
        scale = max_size / max(height, width)
        new_width = int(width * scale)
        new_height = int(height * scale)
        img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
    # 3. Multi-pass JPEG Compression (Mô phỏng việc upload/download nhiều lần)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    
    current_img = img
    for i in range(passes):
        # Mã hóa thành JPEG trong bộ nhớ
        result, encimg = cv2.imencode('.jpg', current_img, encode_param)
        if not result:
            return False
        # Giải mã lại thành ảnh (đã bị nén, mất dữ liệu)
        current_img = cv2.imdecode(encimg, cv2.IMREAD_COLOR)
        
    # 4. Lưu ảnh cuối cùng ra output
    cv2.imwrite(output_path, current_img, encode_param)
    return True

def process_directory(input_dir, output_dir, quality=60, passes=1):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    valid_exts = ['.jpg', '.jpeg', '.png', '.webp']
    count = 0
    
    for img_file in input_path.rglob('*'):
        if img_file.suffix.lower() in valid_exts:
            # Tạo cấu trúc thư mục tương ứng ở output
            rel_path = img_file.relative_to(input_path)
            out_file = output_path / rel_path
            out_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Đổi đuôi thành .jpg vì mạng xã hội thường dùng JPEG
            out_file = out_file.with_suffix('.jpg')
            
            success = simulate_social_media_compression(
                str(img_file), 
                str(out_file), 
                quality=quality, 
                passes=passes
            )
            if success:
                count += 1
                if count % 10 == 0:
                    print(f"Đã xử lý {count} ảnh...")
                    
    print(f"Hoàn tất! Đã nén {count} ảnh và lưu vào {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate Social Media Image Compression")
    parser.add_argument("--input", type=str, default="./examples", help="Thư mục chứa ảnh gốc")
    parser.add_argument("--output", type=str, default="./examples_compressed", help="Thư mục lưu ảnh đã nén")
    parser.add_argument("--quality", type=int, default=50, help="Chất lượng JPEG (1-100), càng thấp càng mờ")
    parser.add_argument("--passes", type=int, default=2, help="Số lần nén qua lại (mô phỏng tải lên tải xuống nhiều lần)")
    
    args = parser.parse_args()
    print(f"Bắt đầu nén ảnh từ '{args.input}' sang '{args.output}' với chất lượng {args.quality} và {args.passes} vòng nén...")
    process_directory(args.input, args.output, args.quality, args.passes)
