# TransApp v2 — Frame-Accurate Precision & Pro-Subber Upgrade Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan phase-by-phase, task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. **Do NOT begin Phase N+1 until Phase N's Verification Gate passes and is approved.**

**Goal:** Upgrade TransApp into a frame-accurate (`±0–1 frame`), high-throughput (`~60–110s` per 24m episode), near-100%-accurate Simplified Chinese hardsub workstation with Aegisub-locked `.ass`/`.srt` export, OP/ED `In/Out` trim, and pro-subber `ReviewTable` verification tools.

**Architecture:** Single-pass streaming decode (`VideoReader` with signed `PTS[0]` origin & exact `frame_pts_ms[]` table) feeds a native 4D ONNX-batched `ch_PP-OCRv5 Det Mobile` scanner (`S=2..3`) paired with tight-box visual signatures and a streaming ring buffer that refines every onset, offset, and `0ms` back-to-back cut in-memory to exact frame indices (`±0 frame`, zero re-seeks). Stage 2 runs borderless `ch_PP-OCRv5 Server` with interior-first early-exit, confidence-weighted + character-level consensus voting, and targeted binary-search split safety, followed by Aegisub midpoint centisecond snapping and a unified Dark Studio `In/Out` + `ReviewTable` QA workflow.

**Tech Stack:** Python 3.11, PySide6 6.12, PyAV 18.1, OpenCV (`opencv-python-headless`), RapidOCR 3.10 (`ch_PP-OCRv5`), ONNX Runtime DirectML 1.24, pytest + pytest-qt.

**Spec:** [2026-11-27-transapp-v2-precision-upgrade-design.md](file:///d:/Code%20Project/TransApp/docs/superpowers/specs/2026-11-27-transapp-v2-precision-upgrade-design.md)

## Global Constraints
- Single-episode in-memory workflow per run: NO batch queue, NO disk project cache, NO Whisper/audio ASR, NO LLM translation.
- Preserve existing PySide6 Dark Studio theme tokens (`#0E1014`, `#15181F`, `#2563EB`, `#10B981`, `#2B2110`/`#FCD34D`) and compact fit (`1160×680` / `1280×672 @ 150% DPI`).
- Preserve existing `ReviewTable` column indices `0..4` (`STT, Start, End, Conf, Text`) and `delete_selected_row()` alias for UI backward compatibility.
- Keep `EX/`, `assets/*.mp4`, `.venv/`, and `*.lnk` out of Git.
- Every task follows TDD (failing test → verify fail → minimal implementation → verify pass → commit) and runs `$env:QT_QPA_PLATFORM='offscreen'; .venv\Scripts\pytest -q -W error` with 0 warnings.

## Review Focus (Top Failure Modes Pinned to Task Tests)
1. **Timeline Origin Desync (non-zero, negative B-frame edit-list, or `None` `stream.start_time` + 23.976fps rational drift)** → pinned in **Task 1** (`test_video_reader.py`).
2. **NTSC (`23.976` / `29.97fps`) & VFR ASS Centisecond Rounding Bleed (`Start <= pts[f] < End`)** → pinned in **Task 7** (`test_exporters.py` libass visibility oracle across 1,000 cues).
3. **`0ms` Back-to-Back Equal-Width Dialogue Cuts & 1-Char Minimal Pairs (`"放过你的"` → `"放过他的"`)** → pinned in **Task 3 & Task 5** (`test_deep_extractor.py`, `test_postprocess.py`).
4. **Short `160ms` Cues vs Isolated `40ms` Noise Flashes + Alpha Fade-In/Out + Zero Re-Seeks** → pinned in **Task 3 & Task 4** (`test_deep_extractor.py`).
5. **ASS `{...}` Override Tag Swallowing, ASS/SRT Empty-Line Parity, Post-Quantization Overlaps & DirectML Concurrent Access** → pinned in **Task 5, Task 6 & Task 7** (`test_ocr_engine.py`, `test_exporters.py`).
6. **OP/ED `In/Out` Trim Absolute Timestamp Invariance & `flagged_only` In-Place Edit Contract** → pinned in **Task 8 & Task 9** (`test_pipeline.py`, `test_ui.py`).

---

## File Map

| Phase | Files Created | Files Modified |
| :--- | :--- | :--- |
| **Phase 1** | — | `src/transapp/core/video_reader.py`, `src/transapp/core/ocr_engine.py`, `src/transapp/core/deep_extractor.py`, `src/transapp/core/models.py`, `src/transapp/core/pipeline.py`, `tests/test_video_reader.py`, `tests/test_ocr_engine.py`, `tests/test_deep_extractor.py`, `tests/test_pipeline.py` |
| **Phase 2** | — | `src/transapp/core/deep_extractor.py`, `src/transapp/core/ocr_engine.py`, `src/transapp/core/postprocess.py`, `src/transapp/core/pipeline.py`, `tests/test_deep_extractor.py`, `tests/test_ocr_engine.py`, `tests/test_postprocess.py`, `tests/test_pipeline.py` |
| **Phase 3** | `src/transapp/exporters/sanitize.py` | `src/transapp/exporters/ass_exporter.py`, `src/transapp/exporters/srt_exporter.py`, `src/transapp/core/models.py`, `src/transapp/core/video_reader.py`, `src/transapp/core/pipeline.py`, `src/transapp/ui/main_window.py`, `src/transapp/ui/theme.py`, `src/transapp/ui/icons.py`, `tests/test_exporters.py`, `tests/test_pipeline.py`, `tests/test_ui.py` |
| **Phase 4** | — | `src/transapp/ui/review_table.py`, `src/transapp/ui/main_window.py`, `src/transapp/ui/roi_canvas.py`, `src/transapp/ui/icons.py`, `tests/test_ui.py` |

---

# PHASE 1: PTS Foundation + Native 4D Neural Det + Frame-Accurate Ring-Buffer Engine

### Task 1: Exact Rational PTS Origin, `frame_pts_ms[]` & Fast Duration Probe in `VideoReader`

**Files:**
- Modify: `src/transapp/core/video_reader.py`
- Modify: `src/transapp/core/models.py` (add `frame_pts_ms: list[int] = field(default_factory=list)` on `ExtractResult`)
- Test: `tests/test_video_reader.py`

**Interfaces:**
- Produces:
  - `VideoMetadata.start_pts_ms: int` (signed first decoded frame PTS when available, never prematurely clamped to `0`)
  - `VideoReader.iter_roi_frames(roi: RoiRect, *, start_ms: int | None = None, end_ms: int | None = None) -> Iterator[FrameSlice]` where `FrameSlice.pts_ms` is exact raw ms (monotonic, rational `Fraction(pts) * time_base * 1000`, never `frame.dts` fallback on B-frame videos) and `FrameSlice.rel_ms: int` (`max(0, raw_pts_ms - origin_pts_ms)` computed *after* signed subtraction)
  - Fast packet-demux fallback in `_probe_stream` (reads `packet.pts` without decoding frames when container duration is missing)

- [ ] **Step 1: Write failing tests in `tests/test_video_reader.py`**
  - `test_video_reader_signed_origin_pts_negative_edit_list_and_missing_stream_start_time`: verify negative first-frame PTS (`-83ms`) and `stream.start_time=None` with frame 0 at `12000ms` both yield `rel_ms == 0` on frame 0 and exact `0ms` drift at frame 100.
  - `test_video_reader_ntsc_23976_rational_pts_zero_drift`: verify 1,000 ticks at `24000/1001` timebase produce zero cumulative rounding drift.
  - `test_video_reader_packet_demux_duration_probe_without_full_decode`: verify missing container duration resolves via packet demux rather than decoding every video frame.
- [ ] **Step 2: Run failing tests**
  - Run: `.venv\Scripts\pytest tests/test_video_reader.py -v -W error`
  - Expected: FAIL on new tests.
- [ ] **Step 3: Implement in `src/transapp/core/video_reader.py` and `src/transapp/core/models.py`**
  - Probe first decoded frame PTS cleanly for `origin_pts_ms` (keeping signed value); compute rational `round(Fraction(pts) * time_base * 1000)`; avoid `frame.dts` fallback on B-frames (use monotonic `prev_pts + frame_step` instead); add `rel_ms` property/field on `FrameSlice`; switch missing-duration probe to packet demux; add `frame_pts_ms: list[int]` on `ExtractResult`.
- [ ] **Step 4: Verify tests pass**
  - Run: `.venv\Scripts\pytest tests/test_video_reader.py -v -W error`
  - Expected: PASS.
- [ ] **Step 5: Commit**
  - `git add src/transapp/core/video_reader.py src/transapp/core/models.py tests/test_video_reader.py && git commit -m "feat(core): add signed PTS origin and exact rational frame timeline in VideoReader"`

---

### Task 2: Native 4D ONNX Batching in `OcrEngine.detect_grid_batch` (Kill Black-Grid Spam)

**Files:**
- Modify: `src/transapp/core/ocr_engine.py`
- Test: `tests/test_ocr_engine.py`

**Interfaces:**
- Produces:
  - `OcrEngine.detect_grid_batch(rois: list[np.ndarray]) -> list[list[tuple[float, float, float, float]]]` using true 4D NCHW tensor batching `(N, 3, rh, rw)` (`N=16` DirectML, `N=8` CPU) with `"Det.box_thresh": 0.55`, `"Det.use_dilation": True`, `"Det.score_mode": "fast"`, empty-map short-circuit (`preds[i].max() < box_thresh`), and min-height threshold `> 0.02` (supporting tall user ROIs).
  - Keeps injected `det_callable` & `ocr_callable` test paths intact.

- [ ] **Step 1: Write failing tests in `tests/test_ocr_engine.py`**
  - `test_detect_batch_4d_native_tensor_and_empty_map_short_circuit`: verify real `OcrEngine(provider="cpu")` runs 4D batched `Det Mobile` on `[sample_dark_bg.png, blank_bright_orange, sample_bright_bg.png]`, detecting boxes on both subtitle fixtures and returning `[]` on bright colored blank anime ROI (0 synthetic-border false positives!).
  - `test_detect_batch_preserves_small_subtitle_in_tall_roi`: verify a 1-line subtitle occupying 3% of a tall ROI is not dropped.
- [ ] **Step 2: Run failing tests**
  - Run: `.venv\Scripts\pytest tests/test_ocr_engine.py -v -W error`
  - Expected: FAIL on new tests.
- [ ] **Step 3: Implement native 4D batching in `src/transapp/core/ocr_engine.py`**
  - Resize chunk to common `(rw, rh)` (`scale = min(1.0, 640 / w0)`, `rh >= 32`, rounded to multiple of 32), normalize `(batch / 255.0 - 0.5) / 0.5`, transpose to NCHW float32, run `det_model.infer([tensor])[0]`, short-circuit if `float(preds[i].max()) < box_thresh`, run `det_model.postprocess_op` only on non-empty maps, normalize coords to `[0, 1]`. Remove `_GRID_PAD_X`/`_GRID_PAD_Y` from production path (only used in injected mock `det_callable` path).
- [ ] **Step 4: Verify tests pass**
  - Run: `.venv\Scripts\pytest tests/test_ocr_engine.py -v -W error`
  - Expected: PASS.
- [ ] **Step 5: Commit**
  - `git add src/transapp/core/ocr_engine.py tests/test_ocr_engine.py && git commit -m "perf(ocr): replace black-padded grid with native 4D ONNX Det Mobile batching"`

---

### Task 3: Tight-Box Visual Signature + Streaming Ring-Buffer Exact Frame Refinement (`±0 frame`)

**Files:**
- Modify: `src/transapp/core/deep_extractor.py`
- Modify: `src/transapp/core/pipeline.py`
- Test: `tests/test_deep_extractor.py`, `tests/test_pipeline.py`

**Interfaces:**
- Consumes: `VideoReader.iter_roi_frames`, `OcrEngine.detect_grid_batch`
- Produces:
  - `TightSignature` (`boxes`, `union_rect`, `edge_mask`, `luma_patch`) and `_compute_tight_signature(bgr_roi, boxes)` / `_tight_signatures_same_cue(sig_a, sig_b) -> bool`
  - `_RefinedSpan(start_idx, end_idx, start_ms, end_ms, union_rect, samples: list[_IndexedSample])` with exact inclusive frame indices `[start_idx, end_idx]`, `end_ms = frame_pts_ms[end_idx + 1]` (never `+step_ms`!), up to `24` evenly distributed samples keeping full tail coverage, and **zero Stage 1 `flash_bridge_max_ms` blind gluing**.
  - `detect_subtitle_spans(...) -> tuple[list[_RefinedSpan], list[int]]` returning `(eligible_spans, frame_pts_ms)`.

- [ ] **Step 1: Write failing tests in `tests/test_deep_extractor.py`**
  - `test_ring_buffer_refines_onset_offset_to_exact_frame_zero_drift`: synthetic 25fps stream with cue at exact frames `[5..18]` and `[22..35]`, coarse `sample_step=2` and `sample_step=3` → assert exact `start_idx == 5, end_idx == 18` (`start_ms == 200, end_ms == 760`) and `[22..35]` (`±0 frames`).
  - `test_zero_ms_back_to_back_equal_width_cues_split_at_exact_cut_frame`: 3 consecutive equal-width cues `[4..15]`, `[16..27]`, `[28..40]` with `0ms` gap over moving background → assert 3 separate spans with exact boundaries (`end_ms[0] == start_ms[1]`, `±0 frames`).
  - `test_short_160ms_cue_all_phases_caught_and_40ms_flash_dropped`: sweep `160ms` (4-frame) cue across all phase offsets `0..sample_step-1` + isolated `40ms` (1-frame) flash → 100% recall on `160ms` cues, 0% FP on `40ms` flash.
  - `test_vfr_timeline_uses_actual_next_frame_pts_no_overlap`: mixed `12fps / 25fps / 60fps` PTS sequence → assert `end_ms == pts[end_idx + 1]` and zero overlaps.
- [ ] **Step 2: Run failing tests**
  - Run: `.venv\Scripts\pytest tests/test_deep_extractor.py -v -W error`
  - Expected: FAIL on new tests.
- [ ] **Step 3: Implement Stage 1 Ring-Buffer Engine in `src/transapp/core/deep_extractor.py` & wire `frame_pts_ms` in `src/transapp/core/pipeline.py`**
  - Default coarse step: `sample_step = 2` for `fps <= 30`, `max(2, round(fps / 12.5))` for higher fps, `1` if `config.coarse_step == 1`.
  - Keep a fixed-size deque ring buffer (`maxlen = sample_step * 2 + 4`) of `(frame_idx, rel_ms, bgr_roi)` and collect `frame_pts_ms` for every frame.
  - Batch coarse frames into `detect_grid_batch` (`batch_size = 16` DirectML / `8` CPU).
  - On every transition (`EMPTY→TEXT`, `TEXT→EMPTY`, `TEXT_A→TEXT_B` via `_tight_signatures_same_cue`), refine the `sample_step - 1` buffered frames in-memory (no re-seek!) to lock exact `[start_idx, end_idx]`.
  - Replace broken `idx % stride` sample decimation with reservoir/uniform replacement keeping `samples[0]`, `samples[-1]`, and up to `22` evenly spaced interior samples.
  - Throttle `scan_progress_cb` to ~12 Hz (`>= 80ms` wall-clock interval).
  - Gate isolated single-frame hits (`start_idx == end_idx`, `dur < min_duration_ms`) cleanly.
- [ ] **Step 4: Run full pytest suite**
  - Run: `$env:QT_QPA_PLATFORM='offscreen'; .venv\Scripts\pytest -q -W error`
  - Expected: All tests PASS.
- [ ] **Step 5: Phase 1 Verification Gate (Real Assets Benchmark)**
  - Run benchmark on `assets/...Episode 01 copy.mp4` (1080p) and `assets/...Episode 01.mp4` (480p 24m):
    - Verify 1080p clip: `21–24` spans, `< 15s`, `22/22` exact cues.
    - Verify 24m episode: `< 400` spans, `< 120s`, `0` glued 1–2s shifts, `01:50..02:15` exact cues & frame-accurate start/end verified against `read_frame_at_ms`.
- [ ] **Step 6: Commit Phase 1**
  - `git add src/transapp/core/ src/transapp/ tests/ && git commit -m "feat(core): frame-accurate ring-buffer Stage 1 with native 4D Det Mobile"`

---

# PHASE 2: Stage 2 Weighted Consensus + Targeted Binary-Search Split + Postprocess & Thread Safety

### Task 4: Interior-First Adaptive Early-Exit, Weighted/Char Consensus & Binary-Search Safety Split

**Files:**
- Modify: `src/transapp/core/deep_extractor.py`
- Test: `tests/test_deep_extractor.py`

**Interfaces:**
- Consumes: `_RefinedSpan`, `frame_pts_ms`, `OcrEngine.recognize_roi`
- Produces: `recognize_spans_with_voting(eligible, frame_pts_ms, ocr_engine, config, ...)` with:
  - Interior-first probing (`[18%, 50%, 82%]` + median-luma neighbor dodge for glare frames) & adaptive early-exit (`3` identical clean votes `>= 0.93`, `dur <= 2800ms`).
  - Targeted binary-search split over intermediate stored samples `[i..j]` whenever Stage 2 OCR detects an intra-span text disagreement (e.g. 1-char minimal pair `"他..."` → `"她..."`), setting exact `end_ms = frame_pts_ms[cut_idx]` and `start_ms = frame_pts_ms[cut_idx]` (`±1 frame`, never coarse `0.96s` jumps!).
  - Expanded `_EDGE_NOISE_RE` (`[A-Za-z0-9#|/\\_^~`]{1,3}`), confidence-weighted ranking `sum(conf) * (0.72 if noisy else 1.0)`, edge-noise pooling onto clean votes, character-level alignment consensus when no majority exists, clean fuller Hanzi containment (`max_conf >= 0.92`), and selective `brightness_filter=True` retry on up to 2 frames.

- [ ] **Step 1: Write failing tests in `tests/test_deep_extractor.py`**
  - `test_stage2_early_exit_saves_ocr_calls_on_clean_consensus`: 10-sample clean span (`dur=1800ms`) calls `recognize_roi` only **3 times** (skipping edge frames `0` and `9`).
  - `test_stage2_binary_search_splits_one_char_minimal_pair_to_exact_frame`: single span containing `"我不会放过你的"` on frames `0..11` and `"我不会放过他的"` on frames `12..23` → bisects to exact cut frame `12` (`±1 frame`), producing 2 distinct segments (NOT merged!).
  - `test_stage2_weighted_consensus_pools_varied_edge_noise_and_symbol_artifacts`: votes `["G我们是救世"(0.78), "Y我们是救世"(0.76), "我们是救世"(0.99)]` → winner `"我们是救世"`; votes `["#一人"(0.62)]` triggers brightness retry.
  - `test_stage2_middle_noise_vote_does_not_shatter_single_cue`: votes `[A, A, GARBAGE, A, A]` stay 1 cue `A` instead of splitting into `[A, GARBAGE, A]`.
- [ ] **Step 2: Run failing tests**
  - Run: `.venv\Scripts\pytest tests/test_deep_extractor.py -v -W error`
  - Expected: FAIL.
- [ ] **Step 3: Implement in `src/transapp/core/deep_extractor.py`**
  - Also remove loose `_is_same_cue_text` (`0.72`) cluster merging across `440ms`: require strict text equivalence (or delegate adjacent-span merging cleanly to `postprocess.py` using true unpadded `gap_ms = next.start_ms - prev.end_ms`!).
- [ ] **Step 4: Verify tests pass**
  - Run: `.venv\Scripts\pytest tests/test_deep_extractor.py -v -W error`
  - Expected: PASS.
- [ ] **Step 5: Commit**
  - `git add src/transapp/core/deep_extractor.py tests/test_deep_extractor.py && git commit -m "feat(ocr): interior early-exit, weighted consensus voting, and binary-search safety split"`

---

### Task 5: Postprocess Precision Guards, Spatial Outlier Suppression & DirectML Thread Safety

**Files:**
- Modify: `src/transapp/core/postprocess.py`
- Modify: `src/transapp/core/ocr_engine.py`
- Modify: `src/transapp/core/pipeline.py`
- Test: `tests/test_postprocess.py`, `tests/test_ocr_engine.py`, `tests/test_pipeline.py`

**Interfaces:**
- Produces:
  - `postprocess_segments(segments, config, *, duration_ms: int | None = None) -> list[SubtitleEvent]`:
    - Protects 1-char substitutions (`"我不会放过你的"` vs `"我不会放过他的"`) from blind Tier-1 merge even at `gap_ms <= 160ms`.
    - Drops 1-char ultra-low-conf ghosts (`len == 1, conf < 0.72, dur < 400ms`) and short disallowed-symbol ghosts (`len <= 3, suspicious_noise, conf < conf_threshold`) while keeping `len >= 3` low-conf CJK lines flagged.
    - Preserves selected winner's true confidence when `_pick_merged_winner` prefers a fuller text; clamps `end_ms <= duration_ms`.
  - `_group_and_format_boxes` in `ocr_engine.py`: drops detached tiny low-conf 1-char Hanzi outlier (`len==1, score < 0.75, off-band > 0.8*h or height < 0.55*h`) when strong Hanzi subtitle (`>= 0.90`) exists.
  - `OcrEngine._inference_lock = threading.RLock()` guarding `_ensure_engine`, `_ensure_det_engine`, `detect_grid_batch`, and `recognize_roi`.

- [ ] **Step 1: Write failing tests in `tests/test_postprocess.py` and `tests/test_ocr_engine.py`**
  - `test_postprocess_does_not_merge_0ms_one_char_substitution_and_clamps_duration`: verify `"我不会放过你的"` (`0..1000`) + `"我不会放过他的"` (`1000..2000`) stay 2 events, and `end_ms` clamps to `duration_ms`.
  - `test_postprocess_drops_short_low_conf_hanzi_and_symbol_ghosts_keeps_real_dialogue`: drops `生` (`0.52`, `240ms`), `#一人` (`0.56`, `240ms`), keeps `嗯` (`0.95`, `240ms`) and `大家都怎么样了` (`0.72`, flagged).
  - `test_ocr_engine_drops_offband_tiny_hanzi_and_serializes_concurrent_calls`: verifies spatial outlier filtering and `RLock` thread safety under 2 concurrent worker threads.
- [ ] **Step 2: Run failing tests**
  - Run: `.venv\Scripts\pytest tests/test_postprocess.py tests/test_ocr_engine.py -v -W error`
  - Expected: FAIL.
- [ ] **Step 3: Implement in `postprocess.py`, `ocr_engine.py`, `pipeline.py`**
- [ ] **Step 4: Run full pytest suite & Phase 2 Verification Gate**
  - Run: `$env:QT_QPA_PLATFORM='offscreen'; .venv\Scripts\pytest -q -W error`
  - Benchmark both `assets/` videos: verify `22/22` on 1080p, `16/16` checkpoints + `0` edge-Latin + `<= 3` flagged lines post-OP (excluding ED) on 24m episode.
- [ ] **Step 5: Commit Phase 2**
  - `git add src/transapp/core/ tests/ && git commit -m "feat(core): tighten consensus postprocess guards and DirectML thread safety"`

---

# PHASE 3: Exporter Hardening + Aegisub Midpoint Snapping + OP/ED Time-Range Trim (`In / Out`)

### Task 6: Shared Sanitizer, ASS `{}` Escaping, Midpoint Centisecond Snapping & Zero-Overlap Exporters

**Files:**
- Create: `src/transapp/exporters/sanitize.py`
- Modify: `src/transapp/exporters/ass_exporter.py`
- Modify: `src/transapp/exporters/srt_exporter.py`
- Test: `tests/test_exporters.py`

**Interfaces:**
- Produces:
  - `prepare_export_cues(events: Sequence[SubtitleEvent]) -> list[SubtitleEvent]` (clean blank lines, skip whitespace-only cues identically for ASS & SRT, stably sort by `(start_ms, end_ms, index)`).
  - `snap_ass_centiseconds(start_ms: int, end_ms: int, frame_pts_ms: Sequence[int] | None) -> tuple[int, int]` (Aegisub inter-frame midpoint rule when `frame_pts_ms` provided, deterministic half-up `(ms + 5) // 10` fallback).
  - `export_ass(events, output_path, *, width=1920, height=1080, template_path=None, style_name=None, frame_pts_ms=None)` — escapes `{`→`｛`, `}`→`｝`, literal `\N`/`\n`/`\h`, preserves `[Fonts]`/`[Graphics]` and valid template `PlayResX/Y` (injects video `PlayRes` only if missing/`<= 0`), enforces post-quantization non-overlap `prev.end_cs <= curr.start_cs`.
  - `list_ass_template_styles(template_path: Path) -> list[str]`.
  - `export_srt(events, output_path)` — uses `prepare_export_cues`, enforces non-overlap `prev.end_ms <= curr.start_ms`.

- [ ] **Step 1: Write failing tests in `tests/test_exporters.py`**
  - `test_ass_midpoint_snapping_100_percent_libass_visibility_oracle`: 1,000 cues across `23.976`, `25`, `29.97`, `59.94`, and VFR timelines → assert **100.0% exact frame-set match** under `start_cs*10 <= pts[f] < end_cs*10` and `0` overlaps.
  - `test_ass_escapes_braces_and_literal_backslash_n_and_preserves_fonts_section`: verifies `{note}` → `｛note｝`, literal `\N` escaped, `[Fonts]` preserved after `[Events]`, missing `PlayRes` injected while valid template `PlayRes` kept.
  - `test_ass_and_srt_empty_cue_parity_chronological_sort_and_overlap_clamp`: out-of-order + overlapping + whitespace-only cues produce identical cue counts, strictly sorted, zero overlaps in both `.ass` and `.srt`.
- [ ] **Step 2: Run failing tests**
  - Run: `.venv\Scripts\pytest tests/test_exporters.py -v -W error`
  - Expected: FAIL.
- [ ] **Step 3: Implement `sanitize.py`, update `ass_exporter.py` and `srt_exporter.py`**
- [ ] **Step 4: Verify tests pass**
  - Run: `.venv\Scripts\pytest tests/test_exporters.py -v -W error`
  - Expected: PASS.
- [ ] **Step 5: Commit**
  - `git add src/transapp/exporters/ tests/test_exporters.py && git commit -m "feat(exporters): add Aegisub midpoint snapping, brace escaping, and zero-overlap parity"`

---

### Task 7: OP/ED Time-Range Trim (`In / Out`) End-to-End (Core + `TrimRangeSlider` UI)

**Files:**
- Modify: `src/transapp/core/models.py` (`ExtractConfig.trim_start_ms: int | None = None`, `trim_end_ms: int | None = None`)
- Modify: `src/transapp/core/video_reader.py`, `src/transapp/core/deep_extractor.py`, `src/transapp/core/pipeline.py`
- Modify: `src/transapp/ui/main_window.py`, `src/transapp/ui/theme.py`, `src/transapp/ui/icons.py`
- Test: `tests/test_pipeline.py`, `tests/test_ui.py`

**Interfaces:**
- Consumes: `VideoReader.iter_roi_frames(roi, start_ms=..., end_ms=...)`, `export_ass(..., frame_pts_ms=result.frame_pts_ms)`
- Produces:
  - `ExtractConfig.trim_start_ms` / `trim_end_ms` (0-based video-relative ms; `None` = full video)
  - `TrimRangeSlider(QSlider)` painting `#10B981`/`#2563EB` active range + `In` / `Out` / `Clear` buttons + `I` / `O` shortcuts in `MainWindow` (reset on `load_video()`).
  - Exported timestamps **ALWAYS remain absolute video-relative timestamps** (`rel_ms`), never shifted by `trim_start_ms`!

- [ ] **Step 1: Write failing tests in `tests/test_pipeline.py` and `tests/test_ui.py`**
  - `test_pipeline_trim_in_out_skips_op_ed_and_preserves_absolute_timestamps`: synthetic 3-cue video (`0.2s`, `1.2s`, `2.4s`); set `trim_start_ms=900, trim_end_ms=2000` → returns ONLY the middle cue with exact absolute `start_ms ~ 1200` (NOT `~300`!), clamping active end to `<= 2000`.
  - `test_main_window_trim_in_out_controls_and_reset_on_video_switch`: verifies `In`/`Out`/`Clear` UI controls, `build_config()` trim propagation, guard against `In >= Out`, and reset on `load_video()`.
- [ ] **Step 2: Run failing tests**
  - Run: `$env:QT_QPA_PLATFORM='offscreen'; .venv\Scripts\pytest tests/test_pipeline.py tests/test_ui.py -v -W error`
  - Expected: FAIL.
- [ ] **Step 3: Implement end-to-end trim in core & `MainWindow` + wire `frame_pts_ms` into `export_ass`**
- [ ] **Step 4: Run full pytest suite & Phase 3 Verification Gate**
  - Run: `$env:QT_QPA_PLATFORM='offscreen'; .venv\Scripts\pytest -q -W error`
  - Verify on real 24m episode with `In = 01:50`, `Out = 23:00`: skips OP/ED completely, exports `.ass` + `.srt` with exact absolute timestamps and zero overlaps.
- [ ] **Step 5: Commit Phase 3**
  - `git add src/transapp/ tests/ && git commit -m "feat(trim): add OP/ED In/Out trim range with absolute timestamp preservation"`

---

# PHASE 4: Pro-Subber Dark Studio `ReviewTable` & Timing Verification Inspector

### Task 8: Video↔Table Timing Sync, `flag_reason` Banner, Auto-Clear `flagged`, Multi-Line Edit & Safety Guards

**Files:**
- Modify: `src/transapp/ui/review_table.py`
- Modify: `src/transapp/ui/main_window.py`
- Modify: `src/transapp/ui/roi_canvas.py`
- Modify: `src/transapp/ui/icons.py`
- Test: `tests/test_ui.py`

**Interfaces:**
- Produces:
  - `ReviewTable.jump_requested = Signal(int)` (emits `rel_ms` for `[⏮ Start]` `Alt+S` / `[End ⏭]` `Alt+E`, wired to `MainWindow.seek_to_ms(self._start_pts_ms + rel_ms)`)
  - `ReviewTable.snap_start_requested` / `snap_end_requested` (`Ctrl+[` / `Ctrl+]`), `[-1f]` / `[+1f]` nudge buttons (`Shift` = `±100ms`), `[✓ Duyệt]` (`Ctrl+D`)
  - Uniform `HH:MM:SS.mmm` display in Start/End cells; `\N` ↔ `\n` multi-line edit preservation; `ev.flag_reason` tooltip + amber side-panel banner; in-place auto-clear `flagged` on manual text edit (never evicting row from active `flagged_only` view!)
  - Drag-and-drop video onto `MainWindow`/`RoiCanvas`, `_has_unexported_edits` guard on close/load/re-extract, dynamic `btn_export` label synced to `chk_ass`/`chk_srt`, empty-export warning, and Open Output Folder button.

- [ ] **Step 1: Write failing tests in `tests/test_ui.py`**
  - `test_review_table_jump_sync_nudge_and_flag_reason_auto_clear_in_place`: verifies `jump_requested` signal emits `start_ms`/`end_ms`, `flag_reason` displays in side panel + tooltip, editing text clears `flagged` & decrements badge **without** evicting row in `flagged_only` mode, `\N` preserves multi-line `\n`.
  - `test_main_window_unsaved_edits_guard_and_dynamic_export_button`: verifies unsaved edit prompt on video switch/close, `chk_ass`/`chk_srt` syncing `btn_export`, and empty-cue export guard.
- [ ] **Step 2: Run failing tests**
  - Run: `$env:QT_QPA_PLATFORM='offscreen'; .venv\Scripts\pytest tests/test_ui.py -v -W error`
  - Expected: FAIL.
- [ ] **Step 3: Implement in `review_table.py`, `main_window.py`, `roi_canvas.py`, `icons.py`**
- [ ] **Step 4: Verify tests pass**
  - Run: `$env:QT_QPA_PLATFORM='offscreen'; .venv\Scripts\pytest tests/test_ui.py -v -W error`
  - Expected: PASS.
- [ ] **Step 5: Commit**
  - `git add src/transapp/ui/ tests/test_ui.py && git commit -m "feat(ui): add Video-Table timing sync, flag_reason QA inspector, and unsaved guards"`

---

### Task 9: Power-User `ReviewTable` Ops (Multi-Delete, Merge, Split, Undo/Redo, Find/Replace, Shortcuts)

**Files:**
- Modify: `src/transapp/ui/review_table.py`
- Modify: `src/transapp/ui/main_window.py`
- Modify: `src/transapp/ui/icons.py`
- Test: `tests/test_ui.py`

**Interfaces:**
- Produces:
  - `ExtendedSelection` + `delete_selected_rows()` (keeping `delete_selected_row()` alias)
  - `merge_selected_rows()` (`Ctrl+M`: contiguous rows, deduplicates `text_similarity >= 0.80` else joins `\n`, keeps highest-conf thumbnail)
  - `split_selected_row(playhead_rel_ms: int | None = None)` (`Ctrl+K`: splits at playhead if inside `(start_ms, end_ms)` else midpoint)
  - In-memory Undo/Redo stack (`undo()` / `redo()`, depth 50, zero `thumbnail_png` duplication, `Ctrl+Z` / `Ctrl+Y`)
  - Collapsible Find & Replace bar (`Ctrl+F` / `Ctrl+H`, Next/Prev, Replace, Replace All as single undoable step)
  - `DoubleClicked | EditKeyPressed` edit triggers (replacing `AnyKeyPressed`), plus optional template Style dropdown in Inspector when ASS template has `> 1` styles.

- [ ] **Step 1: Write failing tests in `tests/test_ui.py`**
  - `test_review_table_multi_delete_merge_split_and_undo_redo`: verifies multi-row delete, contiguous merge (both near-duplicate dedup and distinct `\n` join), split at playhead & midpoint, and full `Ctrl+Z`/`Ctrl+Y` round-trip preserving `thumbnail_png`.
  - `test_review_table_find_replace_all_undoable_and_template_style_selector`: verifies Find/Replace All + single-step Undo, and ASS template style dropdown selection.
- [ ] **Step 2: Run failing tests**
  - Run: `$env:QT_QPA_PLATFORM='offscreen'; .venv\Scripts\pytest tests/test_ui.py -v -W error`
  - Expected: FAIL.
- [ ] **Step 3: Implement in `review_table.py`, `main_window.py`, `icons.py`**
- [ ] **Step 4: Run full pytest suite & Phase 4 Final Verification Gate**
  - Run: `$env:QT_QPA_PLATFORM='offscreen'; .venv\Scripts\pytest -v -W error`
  - Verify all tests pass with 0 warnings, window fits cleanly inside `1160×680` / `1280×672 @ 150% DPI`, and real-episode export opens in Aegisub with 0 timeline drift & 0 overlaps.
- [ ] **Step 5: Commit Phase 4**
  - `git add src/transapp/ui/ tests/test_ui.py && git commit -m "feat(ui): add multi-delete, merge/split, undo/redo, and find/replace in ReviewTable"`
