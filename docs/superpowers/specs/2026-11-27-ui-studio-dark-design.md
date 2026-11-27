# Thiết kế nâng cấp Giao diện `TransApp` — Hướng 1: Pro Video Studio Dark

**Ngày:** 2026-11-27  
**Phong cách chọn:** Hướng 1 — Pro Video Studio Dark (đồng bộ cảm giác dựng phim với Adobe Premiere Pro / DaVinci Resolve)

## Mục tiêu nâng cấp
- Thay thế hoàn toàn giao diện Qt Fusion mặc định xám phẳng bằng bộ theme **Studio Dark QSS** đồng nhất (nền than chì `#0E1014`, surface `#15181F`, viền mảnh `#252A34`, nhấn xanh dương Premiere `#2563EB`, vùng ROI xanh ngọc `#10B981`, cảnh báo vàng hổ phách `#F59E0B`, hủy/xóa đỏ dịu `#EF4444`).
- Xóa bỏ toàn bộ `QGroupBox` lồng nhau gây rối ở cột phải; đưa chọn file video + badge độ phân giải/FPS + Preset ROI lên **Top Header Bar** 1 dòng gọn gàng.
- Thay các nút chữ thô (`|<`, `-1f`, `+1f`, `>|`, `...`) bằng **icon vector SVG** (vẽ qua `PySide6.QtSvg`) sắc nét trên màn hình Windows HiDPI 125%–150%.
- Tăng diện tích khung Preview video (~65% chiều ngang) với lớp làm mờ nhẹ ngoài khung ROI để nổi bật dải sub cần bắt; dùng font monospace (`Cascadia Mono`/`Consolas`) cho timecode & bảng số `Start`/`End`/`Conf`.
- Giữ nguyên 100% các thuộc tính widget, tín hiệu và API hiện có để toàn bộ 80 bài test hoạt động nguyên vẹn.

## Thành phần thêm/cập nhật
- `src/transapp/ui/icons.py`: Bộ icon SVG nhúng tối giản (`film`, `folder`, `save`, `trash`, `skip_back`, `step_back`, `step_fwd`, `skip_fwd`, `scan`, `play`, `stop`, `export`).
- `src/transapp/ui/theme.py`: Token màu, font stack và hàm `apply_studio_theme(app)` áp dụng QSS + `QPalette` tối nhất quán.
- `src/transapp/ui/roi_canvas.py`: Lớp dim nhẹ ngoài khung ROI, góc khung sắc nét, empty-state rõ ràng hơn.
- `src/transapp/ui/review_table.py`: Header toolbar gọn, bộ lọc pill rõ ràng, bỏ lưới dọc rườm rà, số monospace, highlight vàng hổ phách dịu cho dòng nghi ngờ, hỗ trợ nút hành động bên phải thanh tiêu đề.
- `src/transapp/ui/main_window.py` & `src/transapp/app.py`: Bố cục Top Bar + Preview/Transport Bar + Inspector phẳng + Review Dock tích hợp nút `Xuất .ass + .srt`.
