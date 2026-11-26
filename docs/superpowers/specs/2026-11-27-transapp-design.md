# Thiết kế chi tiết `TransApp` (v1) — Công cụ tách Timeline & OCR Hardsub Tiếng Trung

**Ngày:** 2026-11-27  
**Trạng thái:** Đã hoàn thiện theo phản hồi người dùng & rà soát kỹ thuật — chờ phê duyệt trước khi lập kế hoạch triển khai  
**Workspace mục tiêu:** `d:\Code Project\TransApp` (hiện chỉ chứa tài liệu thiết kế này, chưa khởi tạo Git hay code)

---

## 1. Bối cảnh & Mục tiêu (Intent & Why)

Người dùng làm phụ đề phim/hoạt hình (anime/donghua) với áp lực deadline cao. Nguồn đầu vào là video **subraw** (có sẵn hardsub tiếng Trung trên khung hình), mỗi lần xử lý **từng tập một** rồi chuyển sang **Aegisub** (dịch/biên tập) và **Adobe Premiere Pro** (edit).

### Đặc điểm thực tế từ 2 ảnh mẫu người dùng cung cấp
- **Ảnh 1:** `不用亲眼看见即将开始的地狱` (chữ trắng, viền đen, font không chân, nền tối, hơi mềm do nén video).
- **Ảnh 2:** `大家都怎么样了` (chữ trắng, viền đen đậm, font nét tròn rõ, nền hoạt hình có các mảng màu cam/vàng/nâu sáng nằm sát sau chữ).
- **Hệ quả thiết kế:**
  1. Ngôn ngữ mục tiêu duy nhất: **tiếng Trung Giản thể (`简体中文`)**, kiểu trình bày **`白字黑边` (chữ trắng viền đen)**.
  2. Do nền hoạt hình phía sau chữ thay đổi màu và độ sáng liên tục (như mảng cam sáng ở ảnh 2), bước phát hiện đổi câu **không được so pixel RGB/grayscale thô**; phải tách mặt nạ nét chữ trắng viền tối trước khi tính biến đổi giữa các frame.
  3. Vị trí phụ đề **cố định một chỗ** trên khung hình (có thể khác nhau giữa các bộ phim), cần cho phép người dùng kéo chọn vùng ROI trực quan và lưu preset để tái sử dụng ở các tập sau.

### Môi trường & Dữ liệu kiểm thử đã xác nhận
- **Hệ điều hành:** Windows
- **Phần cứng:** Intel Core Ultra 7 255HX / 16 GB RAM / NVIDIA GeForce RTX 5060 Laptop GPU (driver `621.01`)
- **Python sẵn có:** `Python 3.11.9`
- **FFmpeg hệ thống:** Chưa có `ffmpeg` trong `PATH` → giải mã video bằng `PyAV` (đóng gói sẵn FFmpeg).
- **Dữ liệu kiểm thử thực tế của người dùng:** 2 ảnh mẫu `.png`, **1 clip mẫu ~2 phút** (để căn chỉnh ngưỡng & tốc độ), và **1 clip dài ~27 phút** (để kiểm chứng độ chuẩn, RAM và thời gian chạy thực tế cả tập).

---

## 2. Phạm vi đã chốt: Goals, Non-Goals & Anti-Goals

### Goals (v1)
1. **Luồng xử lý tập trung 1 tập / lần:** Mở 1 file video → chọn/tải preset vùng ROI → chạy tách timeline & OCR → rà soát nhanh dòng nghi ngờ → xuất `.ass` và `.srt`.
2. **Chọn vùng phụ đề (ROI) trực quan & lưu Preset:** Kéo/chỉnh khung trên frame mẫu của video theo tỉ lệ chuẩn hóa `[0.0, 1.0]` (dùng chung được cho các tập có độ phân giải khác nhau), lưu preset dùng lại cho tập sau.
3. **Bắt Timeline chính xác theo frame:** Đọc `PTS` thực tế từng frame từ stream video (hỗ trợ chuẩn video VFR), quét mặt nạ chữ trắng viền đen và tinh chỉnh biên tới từng frame (sai số biên mục tiêu ≤ 1–2 frame), không tách vụn 1 câu khi nền hoạt hình chuyển động.
4. **OCR Tiếng Trung chính xác & Gộp dòng thông minh:** Chọn frame rõ nét nhất ở khoảng giữa đoạn để chạy OCR tiếng Trung, chuẩn hóa chữ Hán và gộp các đoạn liền kề giống nội dung.
5. **Rà soát nhanh trong phiên làm việc:** Bảng review giữ trạng thái trong RAM suốt lúc mở app (hiển thị ảnh cắt ROI ngay cạnh dòng đang chọn, cho lọc riêng dòng độ tin cậy thấp để sửa text/time nhẹ rồi xuất lại tức thì); tắt app hoặc mở tập khác là tự dọn sạch, không sinh file rác vào thư mục phim.
6. **Xuất đồng thời 2 định dạng chuẩn khớp timestamp 100%:**
   - **`.ass`** chuẩn **Aegisub** (hỗ trợ Style mặc định theo độ phân giải video hoặc nạp Header/Style từ file `.ass` mẫu; đánh dấu dòng confidence thấp ở cột `Actor` để không làm bẩn lời thoại).
   - **`.srt`** UTF-8 sạch để import trực tiếp vào **Adobe Premiere Pro** (Text → Captions).

### Non-Goals (Ngoài phạm vi v1)
- Không chạy hàng loạt (batch queue nhiều tập cùng lúc) hay lưu/resume project trên ổ đĩa qua các lần tắt mở app.
- Không xử lý video *raw* không có chữ (không dùng Whisper / Speech-to-Text).
- Không tích hợp dịch AI Trung → Việt hay sửa chính tả bằng LLM/cloud API trong v1 (người dùng tự dịch sau tại Aegisub/bên ngoài).
- Không nhận diện chữ nhảy tự do nhiều vị trí ngoài vùng ROI đã chọn, không điều khiển GUI Aegisub hay chạy MCP server.

### Anti-Goals
- Không nạp toàn bộ video vào RAM (tránh tràn 16 GB RAM trên tập 27+ phút).
- Không phụ thuộc PyTorch/Paddle CUDA đời cũ dễ lỗi trên GPU RTX 5060 (Blackwell).
- Không tự tạo thêm file cache/tạm vào thư mục chứa video của người dùng ngoài `.ass` và `.srt`.

---

## 3. Lựa chọn Công nghệ (Author Calls)

| Thành phần | Lựa chọn | Lý do |
| :--- | :--- | :--- |
| **Ngôn ngữ & Môi trường** | Python 3.11 (`pyproject.toml` + `.venv`) | Khớp sẵn `Python 3.11.9` trên máy đích |
| **Desktop GUI** | **PySide6** (Qt for Python) | Mở file Windows tự nhiên, canvas vẽ/kéo ROI mượt trên frame video, bảng review + ảnh cắt ROI trực tiếp, không cần bật trình duyệt/server phụ |
| **Giải mã Video** | **PyAV (`av`)** | Đóng gói sẵn FFmpeg trong wheel (không phụ thuộc `ffmpeg` trong `PATH`), đọc chính xác timestamp `PTS` kể cả video VFR |
| **Xử lý ảnh & Mask** | **OpenCV (`opencv-python-headless`) + NumPy** | Thao tác nhanh trên vùng ảnh ROI nhỏ |
| **OCR Tiếng Trung** | **RapidOCR (`rapidocr-onnxruntime`)** dùng model `PP-OCRv4` tiếng Trung | Gọn, chính xác cao cho chữ Hán Giản thể in trên video; chạy mặc định bằng ONNX Runtime CPU (`U7-255HX`) ổn định và kiểm thử đối chiếu thêm `DirectML` (`RTX 5060`) trên clip 2 phút để chọn backend nhanh/ổn định nhất |

---

## 4. Kiến trúc & Luồng xử lý 1 tập

```text
1 File Video (.mp4/.mkv/...) + Vùng ROI (chuẩn hóa 0..1)
   │
   ▼
[1. VideoReader (PyAV)] ──► Decode tuần tự, crop ngay vùng ROI + lấy frame PTS (ms)
   │
   ▼
[2. StrokeMask (OpenCV)] ──► Mặt nạ chữ trắng viền đen (HSV + kiểm tra viền tối bao quanh)
   │
   ▼
[3. Segmenter] ──► Quét thô (step=2 frame) + Ring buffer tinh chỉnh biên từng frame (±step)
   │                Sinh các đoạn thô: (start_ms, end_ms, các frame ứng viên giữa đoạn)
   ▼
[4. RapidOCR Engine] ──► Chọn frame rõ nhất mỗi đoạn → OCR tiếng Trung
   │                     Trả về: text + confidence + ảnh cắt ROI (PNG bytes trong RAM)
   ▼
[5. PostProcessor] ──► Chuẩn hóa chữ Hán + Gộp đoạn kề nhau giống text (similarity ≥ 0.85)
   │                   Lọc nhiễu < min_duration_ms + Gắn cờ dòng confidence thấp (< 0.85)
   ▼
[6. Review UI (In-Memory Session)] ──► Lọc dòng nghi ngờ, đối chiếu ảnh crop & sửa tay nếu cần
   │
   ├──► [ASS Exporter] ──► <ten_video>.ass (Aegisub, Style mặc định hoặc từ file .ass mẫu)
   └──► [SRT Exporter] ──► <ten_video>.srt (UTF-8, Premiere Pro)
```

### Chi tiết các khâu đảm bảo độ chuẩn Timeline & Text

#### A. Mặt nạ chữ trắng viền đen (`subtitle_mask.py`)
1. Chuyển ảnh crop ROI từ BGR sang HSV.
2. **Lọc lõi chữ trắng:** `V >= white_v_min` (mặc định `185`) và `S <= white_s_max` (mặc định `65`) — loại bỏ các mảng màu sáng như cam/vàng trong ảnh mẫu 2.
3. **Kiểm tra viền tối (`Outline Check`):** Dilate vùng trắng một bán kính nhỏ (2–4 px theo chiều cao ROI), kiểm tra sự hiện diện của pixel tối (`V <= outline_v_max`, mặc định `85`) bao quanh viền. Loại các mảng trắng lớn không có viền đen (mảng trời/áo trắng) và nhiễu hạt quá nhỏ.

#### B. Phân đoạn & Tinh chỉnh biên (`segmenter.py`)
1. **Quét thô + Ring buffer:** Quét theo bước `coarse_step` (mặc định `2` frame) đồng thời giữ một vòng đệm `ring buffer` nhỏ (`coarse_step + 2` crop frames trong RAM) để **không phải seek ngược video chậm**.
2. **Phát hiện chuyển trạng thái:** Theo dõi mật độ nét chữ (`fg_ratio`) và độ tương đồng mặt nạ giữa các bước (`IoU` sau dilate nhẹ 1 px để chịu sai lệch nén video). Khi chuyển `trống ↔ có chữ` hoặc `câu cũ → câu mới`, duyệt từng frame trong ring buffer để chốt `start_pts_ms` và `end_pts_ms` sát frame thực tế (áp dụng hysteresis nhẹ cho đoạn fade-in/out).
3. **Chọn frame gửi OCR:** Giữ tối đa 3 frame ứng viên quanh 35%, 50%, 65% thời lượng đoạn, chọn frame có độ sắc nét cạnh cao nhất để OCR (tránh frame đầu/cuối bị mờ).

#### C. OCR Tiếng Trung (`ocr_engine.py`)
1. Đưa **ảnh màu/grayscale gốc vùng ROI** vào RapidOCR (không đưa ảnh nhị phân `mask` để giữ nguyên nét anti-aliasing chữ Hán).
2. Nếu chiều cao chữ/ROI nhỏ (`< 48 px`), tự đệm viền tối xung quanh và phóng to nhẹ (`inter_cubic`, 1.5x–2x) trước khi đưa vào RapidOCR.
3. Ghép nhiều dòng trong vùng ROI theo thứ tự từ trên xuống (`\N` trong ASS, `\n` trong SRT); `confidence` của event lấy theo `min` độ tự tin các dòng để không bỏ sót dòng bị mờ.

#### D. Hậu xử lý gộp dòng (`postprocess.py`)
1. **Làm sạch:** Bỏ khoảng trắng dư giữa ký tự Hán, giữ dấu câu, loại nhiễu ký tự đơn lẻ confidence cực thấp.
2. **Gộp đoạn trùng nội dung:** Nếu 2 đoạn kề nhau có khoảng cách `<= 160 ms` **VÀ** độ tương đồng ký tự `>= 0.85` (hoặc giống hệt khi bỏ dấu câu nhiễu), gộp liền thành 1 event `[start_1, end_2]` và giữ text có `confidence` cao hơn.
3. **Lọc nhiễu & Gắn cờ:** Loại đoạn `< 120 ms`; gắn `flagged = True` nếu `confidence < 0.85` hoặc xuất hiện ký tự bất thường.

---

## 5. Thiết kế Xuất File & Giao diện Desktop (`PySide6`)

### 5.1. Xuất `.ass` & `.srt`
- **Thư mục xuất mặc định:** Cùng thư mục và cùng tên gốc với file video (`<ten_video>.ass` và `<ten_video>.srt`), kèm tùy chọn đổi thư mục xuất và cảnh báo/tùy chọn trước khi ghi đè file đã tồn tại.
- **`.ass` (`ass_exporter.py`):**
  - Tự gán `PlayResX`/`PlayResY` khớp kích thước video (hoặc nạp nguyên `[Script Info]` và `[V4+ Styles]` từ file `.ass` mẫu nếu người dùng chọn).
  - Timestamp định dạng `H:MM:SS.cc` (centisecond, đảm bảo `end > start`).
  - Dòng bị gắn cờ `flagged` được ghi vào cột `Actor` (ví dụ `CHECK_0.74`) để dễ lọc trong Aegisub mà nội dung text thoại vẫn sạch hoàn toàn.
- **`.srt` (`srt_exporter.py`):**
  - Chuẩn SubRip `UTF-8` (`HH:MM:SS,mmm --> HH:MM:SS,mmm`), text sạch không chứa thẻ ASS hay nhãn `CHECK_`.

### 5.2. Bố cục Giao diện Desktop (1 tập / lần)

```text
+----------------------------------------------------------------------------------+
| TransApp - Tách Sub Raw Tiếng Trung                              [Preset ROI v]  |
+--------------------------------------+-------------------------------------------+
| [1. XEM TRƯỚC & CHỌN VÙNG ROI]       | [2. CẤU HÌNH & CHẠY TẬP HIỆN TẠI]         |
|                                      |                                           |
|  +--------------------------------+  | File video: [D:\Phim\Tap_01.mp4] [Chọn..] |
|  |                                |  | Preset ROI: [Donghua_Duoi v] [Lưu] [Xóa]  |
|  |      Video Frame Preview       |  | Thư mục xuất: (o) Cùng thư mục video      |
|  |   +------------------------+   |  |               ( ) Chọn thư mục khác [...] |
|  |   |   Kéo chọn khung ROI   |   |  | Style ASS:    (o) Mặc định  ( ) File mẫu  |
|  |   +------------------------+   |  | Xuất ra:      [x] .ass      [x] .srt      |
|  +--------------------------------+  | Ngưỡng nghi ngờ OCR: [0.85]               |
|  [|<] [-1f] [=====|======] [+1f] [>|]| [Test thử 1 frame hiện tại]               |
|  ROI: x=10% y=80% w=80% h=14%        | [ ▶ Bắt đầu tách sub ]    [ ■ Dừng ]      |
|                                      | Tiến độ: [████████░░░░░] 62% (00:16:40)   |
+--------------------------------------+-------------------------------------------+
| [3. RÀ SOÁT & SỬA NHANH]  Lọc: (o) Tất cả (412)  ( ) Dòng nghi ngờ (14)          |
|------------------------------------------------------+---------------------------|
| STT | Start      | End        | Conf | Text (sửa)   | Ảnh crop ROI dòng đang chọn|
| 12  | 00:01:12.4 | 00:01:14.8 | 0.96 | 不用亲眼看见… | [不用亲眼看见即将开始的地狱] |
| 13! | 00:01:15.0 | 00:01:16.9 | 0.74 | 大家都怎么样了 | (phóng to rõ để đối chiếu) |
|------------------------------------------------------+---------------------------|
|                                         [ Xuất .ass + .srt ]  (Tự xuất khi xong) |
+----------------------------------------------------------------------------------+
```

- **Bộ nhớ review:** Ảnh crop ROI của từng dòng lưu dưới dạng PNG nén trong RAM (`bytes` trên `SubtitleEvent`, ước tính ~15–30 MB cho tập 27 phút có ~500–900 câu), tự động giải phóng hoàn toàn khi mở video khác hoặc đóng app.
- **Lưu Preset ROI:** Lưu tại thư mục cấu hình của app (`presets.json` trong app/user config), không ghi vào thư mục chứa video.

---

## 6. Cấu trúc thư mục dự án (`d:\Code Project\TransApp`)

```text
TransApp/
├── pyproject.toml
├── README.md
├── docs/
│   └── superpowers/
│       └── specs/
│           └── 2026-11-27-transapp-design.md
├── src/
│   └── transapp/
│       ├── __init__.py
│       ├── app.py                  # Entry point khởi chạy PySide6
│       ├── core/
│       │   ├── models.py           # RoiRect, SubtitleEvent, ExtractConfig, ExtractResult
│       │   ├── video_reader.py     # PyAV stream decoder, PTS (ms), seek frame preview
│       │   ├── subtitle_mask.py    # Mặt nạ chữ trắng viền đen (HSV + outline verify)
│       │   ├── segmenter.py        # Quét thô + ring buffer tinh chỉnh biên từng frame
│       │   ├── ocr_engine.py       # RapidOCR wrapper (tiền xử lý crop, confidence, CPU/DML)
│       │   ├── postprocess.py      # Chuẩn hóa text, gộp đoạn giống nhau, lọc nhiễu, gắn cờ
│       │   ├── pipeline.py         # Điều phối chạy 1 tập, progress callback, cancel flag
│       │   └── presets.py          # Lưu/tải ROI preset (JSON trong thư mục cấu hình app)
│       ├── exporters/
│       │   ├── ass_exporter.py     # Tạo .ass chuẩn Aegisub (mặc định hoặc từ file mẫu)
│       │   └── srt_exporter.py     # Tạo .srt chuẩn UTF-8 cho Premiere Pro
│       └── ui/
│           ├── main_window.py      # Cửa sổ chính, luồng QThread worker 1 tập
│           ├── roi_canvas.py       # Widget xem frame + kéo/resize ROI + xem thử mask/OCR
│           └── review_table.py     # Bảng rà soát + bộ lọc nghi ngờ + khung ảnh crop đối chiếu
└── tests/
    ├── test_subtitle_mask.py       # Unit test mask trên ảnh tổng hợp & crop từ 2 ảnh mẫu
    ├── test_segmenter.py           # Unit test bắt biên, ring buffer, tránh tách vụn
    ├── test_postprocess.py         # Unit test gộp dòng giống text, lọc nhiễu, gắn cờ
    ├── test_ass_exporter.py        # Unit test format ASS, template style, Actor flag
    └── test_srt_exporter.py        # Unit test format SRT & newline handling
```

---

## 7. Những điểm để lại cho Builder khi lập kế hoạch & Kiểm thử

### Phần để lại cho Builder
- Hiệu chỉnh chi tiết các ngưỡng mặc định (`white_v_min`, `white_s_max`, `outline_v_max`, ngưỡng `IoU` tách câu, hệ số phóng to ảnh crop) dựa trên clip mẫu 2 phút thực tế của người dùng.
- Đo đối chiếu tốc độ/độ ổn định thực tế giữa `onnxruntime` (CPU `U7-255HX`) và `onnxruntime-directml` (`RTX 5060`) trên clip 2 phút và 27 phút để chọn cấu hình mặc định tối ưu.
- Thiết kế chi tiết tín hiệu `QThread` (tiến trình, hủy ngang an toàn, test nhanh 1 frame) và hộp thoại xác nhận ghi đè file `.ass`/`.srt` nếu trùng tên.

### Chiến lược kiểm thử (Verification Strategy)
1. **Unit tests (`pytest`):** Test mặt nạ chữ trắng viền đen (kèm fixture cắt từ 2 ảnh mẫu `不用亲眼看见即将开始的地狱` và `大家都怎么样了`), thuật toán phân đoạn biên, gộp đoạn giống text, và xuất `.ass` / `.srt`.
2. **Hiệu chỉnh trên clip mẫu 2 phút:** Chạy kiểm tra độ khớp biên frame và tỉ lệ nhận diện đúng chữ tiếng Trung.
3. **Kiểm chứng toàn tập trên clip 27 phút:** Đo thời gian chạy tổng, đỉnh RAM (< 1.5 GB), mở `.ass` trên Aegisub kiểm tra timeline + cột `Actor` đánh dấu dòng nghi ngờ, và import `.srt` vào Adobe Premiere Pro xác nhận không lỗi font/encoding.
