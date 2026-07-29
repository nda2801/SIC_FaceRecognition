# Đồ án nhận diện khuôn mặt nhiều người

## Chạy project trên máy khác

Dataset, virtual environment, cache, checkpoint và output được loại khỏi GitHub bằng `.gitignore`. Sau khi clone project:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Tải Pins Face Recognition rồi đặt các thư mục `pins_*` trực tiếp vào:

```text
src/dataset/
```

Kiểm tra PyTorch có nhận GPU không:

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Chạy từ thư mục `src` để dùng đường dẫn dataset mặc định là `dataset`:

```powershell
cd src
python train.py --experiment_name sic_vit_4
```

Thử biến thể 12 Transformer blocks:

```powershell
python train.py --experiment_name vit_tiny_12 --depth 12 --mlp_ratio 4
```

Kết quả nằm trong `outputs/<experiment_name>/`; checkpoint nằm trong `src/checkpoints/`. Các thư mục này không được commit.

Project được xây theo pipeline:

`Ảnh/video -> Face Detection -> Face Alignment -> Face Embedding -> So khớp danh tính -> Visualization`

## Mục tiêu

- Phát hiện đồng thời nhiều khuôn mặt trong một khung hình.
- Nhận diện danh tính bằng vector đặc trưng (embedding), không chỉ dùng softmax classifier.
- Dùng ResNeSt làm backbone chính và so sánh với một backbone Transformer.
- Đánh giá riêng khuôn mặt nhỏ/gần/xa.
- Có Early Stopping, data augmentation, biểu đồ loss/accuracy và confusion matrix.
- Demo thời gian thực bằng Web App; Mobile/WinForms là phần mở rộng.

## Các phase

### Phase 0 - Hiểu bài toán

Phân biệt:

- Detection: mặt nằm ở đâu?
- Recognition: mặt là ai?
- Verification: hai ảnh có phải cùng người không?
- Identification: một ảnh thuộc người nào trong cơ sở dữ liệu?

Đầu ra phase: sơ đồ pipeline và tiêu chí đánh giá.

### Phase 1 - Chuẩn bị dữ liệu

- Chọn dataset danh tính và dataset detection.
- Khảo sát số ảnh/người, ảnh lỗi, mất cân bằng lớp.
- Chia train/validation/test theo đúng nguyên tắc, tránh rò rỉ dữ liệu.
- Tạo biến thể khuôn mặt nhỏ để đánh giá theo kích thước.

Đầu ra phase: dữ liệu sạch, file thống kê và ảnh minh họa.

### Phase 2 - Baseline nhận diện một khuôn mặt

- Dùng ảnh mặt đã crop/aligned.
- Transfer learning ResNet18 trước để kiểm tra toàn bộ pipeline.
- Theo dõi train/validation loss và accuracy.
- Early Stopping và lưu checkpoint tốt nhất.

Đầu ra phase: baseline chạy đúng và có biểu đồ.

### Phase 3 - ResNeSt và face embedding

- Thay backbone bằng ResNeSt50 từ `timm`.
- Huấn luyện embedding với ArcFace/CosFace.
- So khớp bằng cosine similarity và chọn ngưỡng cho lớp `Unknown`.

Đầu ra phase: nhận diện người đã biết và từ chối người lạ.

### Phase 4 - Face Detection nhiều khuôn mặt

- Baseline bằng RetinaFace hoặc SCRFD pretrained.
- Phát hiện từng bounding box, căn chỉnh bằng landmarks, rồi nhận diện từng mặt.
- Đánh giá ảnh có 1, 2, 3+ khuôn mặt.

Đầu ra phase: ảnh/video nhiều người có box, tên và confidence.

### Phase 5 - Khuôn mặt nhỏ và ở xa

- Chia kết quả theo diện tích bounding box: small, medium, large.
- Báo cáo precision, recall, AP cho detector và accuracy/TAR/FAR cho recognition.
- Thử multi-scale inference, tăng độ phân giải detector và augmentation mô phỏng ảnh xa.

Đầu ra phase: bảng so sánh định lượng theo kích thước khuôn mặt.

### Phase 6 - Transformer

- Giữ nguyên pipeline và thay backbone recognition bằng ViT/Swin Transformer.
- So sánh ResNeSt với Transformer trong cùng split dữ liệu và cùng metric.

Đầu ra phase: bảng accuracy, tốc độ, số tham số và bộ nhớ.

### Phase 7 - Sản phẩm và visualization

- Web App: ưu tiên Streamlit hoặc FastAPI + giao diện web.
- Luồng webcam thời gian thực: FPS, số mặt, tên, confidence.
- Dashboard: loss, accuracy, confusion matrix, ROC và thống kê theo kích thước mặt.
- Xuất ONNX để chuẩn bị cho WinForms hoặc Mobile.

Đầu ra phase: demo chạy được và video thuyết trình.

## Dataset đề xuất

Phương án phù hợp đồ án và GPU 4 GB:

1. Pins Face Recognition: dùng để học identification nhiều lớp; quy mô vừa, dễ bắt đầu.
2. LFW: dùng làm tập benchmark verification; không nên coi là tập train chính.
3. WIDER FACE: dùng cho detection và đặc biệt phù hợp đánh giá khuôn mặt nhỏ/khó.
4. CASIA-WebFace hoặc VGGFace2: chỉ dùng subset nếu cần nâng cấp vì toàn bộ dữ liệu khá lớn.
5. Dữ liệu thành viên trong lớp: dùng cho demo thực tế, phải có sự đồng ý của người được chụp.

Không trộn ngẫu nhiên ảnh gần như giống nhau của cùng một video vào cả train và test. Nếu tự quay video, phải chia theo buổi quay/video trước rồi mới trích frame.

## Cấu trúc dự kiến

```text
face_recognition_project/
  configs/
  data/
    raw/
    processed/
    splits/
  notebooks/
  src/
    datasets/
    detection/
    recognition/
    training/
    evaluation/
    app/
  outputs/
    checkpoints/
    figures/
    logs/
```
