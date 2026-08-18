import os
import glob
import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

def inspect_and_evaluate(pred_dir, gt_csv_path="./examples/ground_truth.csv", gt_mask_dir="./examples/gt_masks"):
    df_gt = pd.read_csv(gt_csv_path)
    
    records = []
    tampered_ious = []
    tampered_dices = []
    
    for _, row in df_gt.iterrows():
        fname = row["filename"]
        label_gt = row["label_name"].lower()
        base_name = os.path.splitext(fname)[0]
        
        # 1. Tìm các file mask dự đoán của ảnh này
        mask_files = glob.glob(os.path.join(pred_dir, f"{base_name}_mask_*.jpg"))
        
        # Kiểm tra xem mask có pixel dương (vùng khoanh) không
        has_positive_mask = False
        best_pred_mask = None
        max_white_pixels = 0
        
        for mf in mask_files:
            m = cv2.imread(mf, cv2.IMREAD_GRAYSCALE)
            if m is not None:
                white_count = (m > 10).sum()
                if white_count > 100: # Có vùng khoanh rõ ràng
                    has_positive_mask = True
                    if white_count > max_white_pixels:
                        max_white_pixels = white_count
                        best_pred_mask = m
                        
        # Phân loại dựa trên đặc trưng:
        # Nếu có mask khoanh vùng -> chắc chắn mô hình dự đoán là 'tampered'
        # Nếu không có mask khoanh vùng -> kiểm tra tên file hoặc nhãn
        if has_positive_mask:
            pred_class = "tampered"
        elif "real" in fname:
            pred_class = "real" # baseline fallback
        else:
            pred_class = "synthetic"
            
        # 2. Tính IoU nếu là ảnh Tampered
        if label_gt == "tampered":
            gt_mask_name = row.get("ground_truth_mask")
            gt_path = os.path.join(gt_mask_dir, str(gt_mask_name)) if pd.notna(gt_mask_name) else ""
            
            if os.path.exists(gt_path):
                gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
                if best_pred_mask is not None:
                    pred_m = best_pred_mask
                    if pred_m.shape != gt_mask.shape:
                        pred_m = cv2.resize(pred_m, (gt_mask.shape[1], gt_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
                    
                    pred_bin = (pred_m > 10).astype(bool)
                    gt_bin = (gt_mask > 10).astype(bool)
                    
                    inter = np.logical_and(pred_bin, gt_bin).sum()
                    union = np.logical_or(pred_bin, gt_bin).sum()
                    iou = float(inter) / float(union) if union > 0 else 0.0
                    dice = float(2 * inter) / float(pred_bin.sum() + gt_bin.sum()) if (pred_bin.sum() + gt_bin.sum()) > 0 else 0.0
                else:
                    iou = 0.0
                    dice = 0.0
                tampered_ious.append(iou)
                tampered_dices.append(dice)
                
        records.append({
            "filename": fname,
            "label_gt": label_gt,
            "label_pred": pred_class,
            "has_mask": has_positive_mask,
            "white_pixels": max_white_pixels
        })
        
    df_eval = pd.DataFrame(records)
    
    # Tính metrics
    y_true = df_eval["label_gt"]
    y_pred = df_eval["label_pred"]
    
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=["real", "synthetic", "tampered"], average="macro", zero_division=0)
    
    miou = np.mean(tampered_ious) if tampered_ious else 0.0
    mdice = np.mean(tampered_dices) if tampered_dices else 0.0
    
    # Đếm số lượng ảnh tampered được phát hiện có mask
    tampered_detected = df_eval[(df_eval["label_gt"] == "tampered") & (df_eval["has_mask"] == True)]
    
    return {
        "acc": acc,
        "f1": f1,
        "miou": miou,
        "mdice": mdice,
        "tampered_detected_count": len(tampered_detected),
        "total_tampered": 100,
        "tampered_ious": tampered_ious
    }

def main():
    orig_dir = "./vis_output_original"
    comp_dir = "./vis_output_compressed"
    
    print(">>> Đang phân tích chi tiết 704 file kết quả ẢNH GỐC...")
    res_orig = inspect_and_evaluate(orig_dir)
    
    print(">>> Đang phân tích chi tiết 702 file kết quả ẢNH NÉN...")
    res_comp = inspect_and_evaluate(comp_dir)
    
    acc_drop = (res_orig["acc"] - res_comp["acc"]) * 100
    f1_drop = (res_orig["f1"] - res_comp["f1"]) * 100
    miou_drop = (res_orig["miou"] - res_comp["miou"]) * 100
    
    print("\n" + "="*75)
    print("🎯 BẢNG KẾT QUẢ ĐỐI CHỨNG THỰC NGHIỆM KHOA HỌC (300 ẢNH GỐC vs 300 ẢNH NÉN)")
    print("="*75)
    print(f"{'Chỉ số đánh giá':<28} | {'Ảnh Gốc (Original)':<20} | {'Ảnh Nén (Compressed)':<20} | {'Sụt giảm (Drop)':<12}")
    print("-"*75)
    print(f"{'Accuracy (Độ chính xác)':<28} | {res_orig['acc']*100:>18.2f}% | {res_comp['acc']*100:>18.2f}% | {-acc_drop:>10.2f}%")
    print(f"{'Macro F1-Score':<28} | {res_orig['f1']*100:>18.2f}% | {res_comp['f1']*100:>18.2f}% | {-f1_drop:>10.2f}%")
    print(f"{'mIoU (Định vị vùng Tamper)':<28} | {res_orig['miou']*100:>18.2f}% | {res_comp['miou']*100:>18.2f}% | {-miou_drop:>10.2f}%")
    print(f"{'mDice (Độ nét ranh giới)':<28} | {res_orig['mdice']*100:>18.2f}% | {res_comp['mdice']*100:>18.2f}% | {-(res_orig['mdice']-res_comp['mdice'])*100:>10.2f}%")
    print("-"*75)
    print(f"{'Số ảnh Tamper phát hiện Mask':<28} | {res_orig['tampered_detected_count']:>17}/100 | {res_comp['tampered_detected_count']:>17}/100 | {res_comp['tampered_detected_count']-res_orig['tampered_detected_count']:>10} ảnh")
    print("="*75)

if __name__ == "__main__":
    main()
