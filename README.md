# PlantDoc AI 🌿🩺

**Hệ thống nhận diện bệnh cây trồng từ ảnh lá cây, tích hợp AI giải thích được (Explainable AI - Grad-CAM) và khuyến nghị xử lý chuyên môn.**

Dự án phục vụ học phần **"Thực tập cơ sở"** (GVHD: **Thầy Nguyễn Xuân Đức**).

---

## 1. Tổng quan (Overview)

**PlantDoc AI** là một hệ thống Deep Learning đầu cuối (end-to-end) giải quyết bài toán phân loại hình ảnh bệnh trên lá cây. Không chỉ dừng lại ở việc đưa ra dự đoán "hộp đen" (black-box), dự án còn tập trung vào **tính minh bạch** thông qua Grad-CAM và cung cấp các **khuyến nghị xử lý nông nghiệp** cụ thể.

*   **Vấn đề giải quyết:** Nhu cầu nhận diện nhanh chóng và chính xác các loại bệnh trên cây trồng từ ảnh chụp thực địa, hỗ trợ người nông dân và kỹ sư nông nghiệp ra quyết định kịp thời.
*   **Tại sao dự án đáng làm?** Khác với các mô hình phân loại thông thường, PlantDoc AI tích hợp kiểm soát đầu vào (cảnh báo ảnh ngoài phân phối - OOD) và Grad-CAM (giải thích vùng mô hình tập trung). Điều này giúp tăng độ tin cậy và ngăn chặn việc hệ thống đưa ra kết quả sai lệch khi người dùng tải lên những bức ảnh không phải là lá cây.

---

## 2. Tính năng nổi bật (Key Features)

*   ✅ **Phân loại đa lớp (Multi-class Classification):** Nhận diện chính xác loại cây và bệnh lý dựa trên tập dữ liệu PlantVillage mở rộng.
*   ✅ **Giải thích mô hình (Grad-CAM):** Trực quan hóa vùng ảnh (heatmap) mà mô hình chú ý nhất khi đưa ra dự đoán.
*   ✅ **Kiểm soát đầu vào & Cảnh báo OOD:** Sử dụng Entropy-based Detection và Confidence Thresholding để cảnh báo nếu ảnh đầu vào không hợp lệ hoặc có độ tin cậy thấp.
*   ✅ **Khuyến nghị xử lý (Recommendations):** Cung cấp nguyên nhân, triệu chứng và hướng dẫn điều trị/phòng ngừa chi tiết cho từng loại bệnh.
*   ✅ **Huấn luyện linh hoạt (Config-driven Training):** Hỗ trợ Staged Fine-tuning, Layer-wise Learning Rate, Weighted Random Sampler và Augmentation đa dạng thông qua file YAML.
*   ✅ **Giao diện Web chuyên nghiệp:** Ứng dụng Streamlit hiển thị rõ ràng thông tin mô hình, kết quả dự đoán Top-K, Grad-CAM và khuyến nghị.

---

## 3. Kiến trúc dự án (Project Architecture)

```text
[Dataset] → scripts/createSplits.py → [Train/Val/Test CSV]
                                             ↓
[configs/*.yaml] ─────────────────────→ scripts/train.py → [artifacts/checkpoints]
                                             ↓
                                      scripts/evaluate.py (Metrics, Confusion Matrix)
                                             ↓
[User Image] → app.py (Streamlit) / scripts/predict.py
                     ↓
             Inference Pipeline
             (Image Preprocessing → Feature Extraction → Prediction + Grad-CAM)
                     ↓
[Top-K Results] + [Grad-CAM Overlay] + [Agricultural Recommendations] + [OOD Warnings]
```

---

## 4. Cấu trúc thư mục (Repository Structure)

```text
PlantDocAI/
├── app.py                    # Giao diện chính của Streamlit App
├── app/                      # Các module phụ trợ cho Streamlit
│   └── recommendations.py    # Logic hiển thị khuyến nghị bệnh
├── configs/                  # Các file cấu hình YAML (baseline, extended,...)
├── data/                     # Thư mục chứa dữ liệu thô và splits (CSV)
├── scripts/                  # Các kịch bản chạy độc lập (entrypoints)
│   ├── train.py              # Script huấn luyện mô hình
│   ├── evaluate.py           # Script đánh giá mô hình trên tập test
│   ├── predict.py            # Script chạy inference trên một ảnh
│   └── createSplits.py       # Script chia tập dữ liệu train/val/test
├── src/                      # Source code core của hệ thống
│   ├── data/                 # DataLoader, Augmentations
│   ├── evaluation/           # Metrics, Inference Pipeline
│   ├── explain/              # Thuật toán Grad-CAM
│   ├── models/               # Factory khởi tạo model (timm)
│   ├── training/             # Trainer, Loss, Optimizer, Scheduler
│   └── utils/                # Logging, Seed, Config parsing
├── artifacts/                # (Sinh ra khi train) Chứa model weights, config.json
└── requirements.txt          # Các thư viện phụ thuộc
```

---

## 5. Hướng dẫn cài đặt (Installation)

Yêu cầu: **Python 3.9+**

```bash
# 1. Clone repository
git clone https://github.com/biennhlab/PlantDocAI.git
cd PlantDocAI

# 2. Tạo môi trường ảo (khuyến nghị)
python -m venv .venv

# Kích hoạt môi trường (Windows)
.venv\Scripts\activate
# Kích hoạt môi trường (macOS/Linux)
source .venv/bin/activate

# 3. Cài đặt dependencies
pip install -r requirements.txt
```

---

## 6. Chuẩn bị dữ liệu (Dataset Preparation)

1. Tải bộ dữ liệu gốc [extended dataset](https://drive.google.com/file/d/1l4EuesCfAA3NyNTQV5gC7Ua1oi-1Y_Mp/view?usp=drive_link) và giải nén vào thư mục `data/extended/` (hoặc thư mục tương ứng trong config).
   Cấu trúc mong đợi:
   ```
   data/extended/
   ├── Apple___Apple_scab/
   ├── Apple___Black_rot/
   └── ...
   ```
2. Chạy script để tạo các file phân chia dữ liệu (`train.csv`, `val.csv`, `test.csv`):
   ```bash
   python scripts/createSplits.py --dataDir data/extended --outDir data/splits
   ```

---

## 7. Cấu hình mô hình (Configuration)

Mọi thay đổi về siêu tham số (hyperparameters), đường dẫn dữ liệu hay augmentation đều nằm trong `configs/`.
Ví dụ một số cài đặt quan trọng trong `configs/extended.yaml`:

*   `modelName`: Kiến trúc backbone (vd: `mobilenetv2_100`, `efficientnet_b0`).
*   `batchSize`, `numEpochs`, `learningRate`: Cấu hình train cơ bản.
*   `useStagedFinetuning`: Bật/tắt huấn luyện 2 giai đoạn (freeze head trước, sau đó unfreeze toàn bộ).
*   `augmentation`: Các phép biến đổi dữ liệu (Rotate, Blur, Sharpness, Elastic, CoarseDropout).

---

## 8. Huấn luyện (Training)

Để bắt đầu huấn luyện mô hình với cấu hình `extended.yaml`:

```bash
python scripts/train.py --config configs/extended.yaml
```

**Workflow Checkpoint (Quản lý Artifacts):**
*   Mô hình, cấu hình và logs sẽ được lưu tự động vào `artifacts/<experimentName>/`.
*   Trọng số mô hình tốt nhất được lưu tại: `artifacts/<experimentName>/checkpoints/best.pt`.
*   Nếu quá trình huấn luyện bị gián đoạn, bạn có thể tiếp tục bằng cờ `--resume`:
    ```bash
    python scripts/train.py --config configs/extended.yaml --resume
    ```

---

## 9. Đánh giá (Evaluation)

Để đánh giá mô hình đã huấn luyện trên tập Test (tính toán Accuracy, F1-macro, xuất Confusion Matrix):

```bash
python scripts/evaluate.py --modelDir artifacts/mobilenetV2_extended
```
*Lưu ý: Thay `mobilenetV2_extended` bằng tên experiment thực tế của bạn.*

---

## 10. Chạy dự đoán ảnh đơn (Inference)

Sử dụng CLI để dự đoán một ảnh bất kỳ (không cần bật Web App):

```bash
python scripts/predict.py --image path/to/leaf.jpg --modelDir artifacts/mobilenetV2_extended --topK 3
```
*Output sẽ in ra terminal các nhãn có xác suất cao nhất kèm theo độ tin cậy (confidence).*

---

## 11. Giao diện Web (Streamlit App)

Dự án cung cấp một giao diện web chuyên nghiệp để demo trực tiếp.

```bash
streamlit run app.py
```

*   **Chức năng:** Tải ảnh lên, ứng dụng sẽ tự động chạy Inference, sinh ảnh Grad-CAM, hiển thị phân bố Top-K và đề xuất xử lý bệnh tật.
*   **Safety Warning:** Ứng dụng sẽ hiện cảnh báo nếu độ tin cậy thấp hoặc có dấu hiệu ảnh tải lên nằm ngoài phân phối dữ liệu (không phải lá cây).

---

## 12. Hạn chế và Hướng phát triển (Limitations & Future Work)

### Hạn chế hiện tại:
*   **Domain Gap:** Mô hình được huấn luyện chủ yếu trên ảnh chụp trong điều kiện phòng thí nghiệm (PlantVillage) với nền đơn giản. Độ chính xác có thể giảm khi áp dụng trên ảnh thực địa với nền phức tạp.
*   **Input Validation:** Module cảnh báo ảnh không phải lá (OOD) hiện dựa vào mức độ Entropy của Softmax, chưa phải là một mô hình phân loại Leaf vs. Non-leaf độc lập hoàn toàn.
*   **Tính chất tham khảo:** Grad-CAM chỉ là công cụ trực quan hóa hậu nghiệm (post-hoc) giúp giải thích quyết định của mô hình, không phải bằng chứng y học thực vật tuyệt đối.

### Hướng phát triển:
*   **Bổ sung dữ liệu thực địa:** Tích hợp thêm các bộ dữ liệu chụp từ nông trại thực tế để tăng khả năng tổng quát hóa.
*   **Cải tiến kiến trúc:** Thử nghiệm các kiến trúc ViT (Vision Transformer) để so sánh hiệu năng khai phá đặc trưng cục bộ và toàn cục.
*   **Deployment:** Đóng gói bằng Docker và triển khai lên các nền tảng đám mây (AWS/GCP) hoặc tối ưu hóa ONNX/TFLite để đưa xuống thiết bị di động (Edge AI).

---
 