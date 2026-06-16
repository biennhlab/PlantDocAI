"""
scripts/createSplits.py
━━━━━━━━━━━━━━━━━━━━━━
Script độc lập để chia dữ liệu ảnh thành các tập train/val/test CSV.
Sử dụng dữ liệu nằm sẵn trên bộ nhớ máy (local).

Cách chạy cơ bản theo baseline.yaml:
  python scripts/createSplits.py --dataDir data/extended/train --outDir data/splits

Cách chạy tùy chỉnh tỷ lệ:
  python scripts/createSplits.py --dataDir data/extended/train --outDir data/splits --train_ratio 0.8 --val_ratio 0.1 --test_ratio 0.1
"""
import argparse
import sys
from pathlib import Path

# Thêm gốc project vào module path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.dataSplit import createSplits, SplitConfig

def parseArgs():
    parser = argparse.ArgumentParser(description="Tạo các file Split CSV (train.csv, val.csv, test.csv) từ thư mục gốc chứa ảnh.")
    parser.add_argument("--dataDir", type=str, required=True, 
                        help="Đường dẫn đến thư mục chứa dữ liệu local (VD: data/extended/train)")
    parser.add_argument("--outDir", type=str, required=True, 
                        help="Đường dẫn lưu file CSV (VD: data/splits)")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Tỷ lệ tập Train (mặc định: 0.8)")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Tỷ lệ tập Validation (mặc định: 0.1)")
    parser.add_argument("--test_ratio", type=float, default=0.1, help="Tỷ lệ tập Test (mặc định: 0.1)")
    parser.add_argument("--seed", type=int, default=42, help="Seed tạo random (mặc định: 42)")
    return parser.parse_args()

def main():
    args = parseArgs()
    
    # Kiểm tra tỷ lệ
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 1e-5:
        print(f"[LỖI] Tổng tỷ lệ phải bằng 1.0. Hiện tại là {total_ratio:.3f}")
        sys.exit(1)
        
    data_path = Path(args.dataDir)
    if not data_path.exists() or not data_path.is_dir():
        print(f"[LỖI] Thư mục dữ liệu (dataDir) không tồn tại hoặc sai đường dẫn: {args.dataDir}")
        print(r"Vui lòng tải ảnh dataset về máy, ví dụ đặt tại \PlantDocAI\data\extended\train")
        sys.exit(1)
        
    out_path = Path(args.outDir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    config = SplitConfig(
        trainRatio=args.train_ratio,
        valRatio=args.val_ratio,
        testRatio=args.test_ratio,
        seed=args.seed
    )
    
    print(f"="*60)
    print(f"  PlantDocAI - Bắt đầu chia Splits từ Data Local")
    print(f"  Data Input : {args.dataDir}")
    print(f"  Output Dir : {args.outDir}")
    print(f"  Tỷ lệ      : Train ({args.train_ratio}) | Val ({args.val_ratio}) | Test ({args.test_ratio})")
    print(f"="*60)
    
    try:
        saved_paths = createSplits(
            dataDir=args.dataDir,
            outDir=args.outDir,
            splitConfig=config
        )
        print("\n[OK] Đã tạo thành công các file sau:")
        for k, v in saved_paths.items():
            print(f"  - {k.capitalize():<7}: {v}")
    except Exception as e:
        print(f"\n[LỖI] Quá trình chia split thất bại: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
