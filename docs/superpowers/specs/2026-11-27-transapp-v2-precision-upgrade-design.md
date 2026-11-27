# TransApp v2 — Frame-Accurate Precision & Pro-Subber Upgrade Design

**Date:** 2026-11-27  
**Status:** Proposed for User Approval  
**Scope:** Single-episode Windows desktop hardsub extractor (PySide6 + PyAV + OpenCV + RapidOCR v3 `ch_PP-OCRv5` ONNX DirectML/CPU) exporting `.ass` (Aegisub) and `.srt` (Adobe Premiere Pro).

---

## 1. Intent & Why Now

TransApp's borderless `ch_PP-OCRv5 Server` text recognition is now near-100% accurate on real anime/donghua episodes, but a 4-way forensic audit across `TransApp` and 4 reference repositories (`VideOCR`, `VideoSubFinder`, `video-subtitle-extractor`, `RapidVideOCR`) uncovered **two systemic bottlenecks**:

1. **Severe Timeline Drift (`0.5s – 2.0s+`) & Glued `0.00s` Cues in Aegisub**, caused by 6 compounding defects:
   - Stage 1 merges consecutive cues by bounding-box geometry alone (`_rects_similar` + EMA drift) across gaps up to `440ms` (`flash_bridge_max_ms 320 + step_ms 120`).
   - `_DetectedSpan.add_sample()` decimates via `stride *= 2` (`0.96s – 1.92s` gaps), and Stage 2 decimates again to 5 `linspace` votes, snapping internal cue boundaries to coarse `~1–2s` timestamps and erasing short middle cues.
   - Stage 1 runs coarse `7.5fps` without boundary refinement, blindly padding `+step_ms` onto every `end_ms` (broken on VFR and NTSC).
   - `VideoReader` anchors to `stream.start_time` (failing when `None` or negative B-frame edit list) and lacks an exact per-frame `frame_pts_ms[]` table.
   - `ass_exporter.py` uses Python Banker's rounding `round(ms / 10)` instead of Aegisub's midpoint centisecond rule, causing `±1 frame` late starts and cut bleeds on `23.976` / `29.97fps`.
   - `postprocess.py` and `exporters/` lack non-overlap clamping, ASS `{...}` escaping, and ASS/SRT empty-line parity.
2. **Stage 1 Background Spam (~83% False-Positive Spans)**:
   - Stage 1 `detect_grid_batch` still embeds ROIs inside synthetic black bars `(0, 0, 0)`, causing `ch_PP-OCRv5 Det Mobile` to emit `1,762` spans for `~299` cues on a 24m episode (wasting ~180s in Stage 2 and producing 1–2 char background ghosts).
   - Verified directly on project assets: `ch_PP-OCRv5_det_mobile.onnx` natively accepts dynamic 4D tensors `[N, 3, H, W]` (`N=16` DirectML, `N=8` CPU), eliminating 100% of synthetic grid borders and dropping spans **5×–12×** (`118 → 21–23` on 1080p clip, `175 → 14–18` on 480p OP) with **0 dropped cues** and **3× speedup**.

---

## 2. Goals, Non-Goals & Anti-Goals

### Goals
- **Frame-Accurate Timeline (`±0–1 frame`, Aegisub-locked)**: Start, End, 1-frame gaps, and `0ms` back-to-back dialogue cuts accurate to `±1 frame` across `23.976`, `25`, `29.97`, `59.94fps`, and VFR containers, with **0-frame libass display error** in Aegisub.
- **Preserved / Higher Text Accuracy (`≥99.5%`)**: Keep borderless `ch_PP-OCRv5 Server`, eliminate Stage 1 black-grid spam, upgrade to interior-first confidence-weighted + character-level consensus voting, and protect against 1-char minimal-pair merges (`"放过你的"` vs `"放过他的"`).
- **Fast Practical Throughput**: Full 24-minute episode in **~60–110s** on RTX 5060 DirectML (down from ~300s), with bounded `O(1)` streaming RAM (`< 50 MB`) and zero video re-seeking during Stage 1.
- **Pro-Subber Verification & Export Hygiene**: OP/ED `In/Out` trim (preserving absolute timestamps), Video↔ReviewTable Start/End sync + `±1f` nudge, visible `flag_reason` + auto-clear `flagged` on edit, multi-delete/merge/split/undo/redo/find-replace, and standards-compliant non-overlapping `.ass` & `.srt` output.

### Non-Goals (Hard Scope Boundaries)
- No multi-file batch queue, no disk project/cache persistence, no Whisper/audio ASR, no LLM translation.

### Anti-Goals (Failure Conditions)
- Never pile new heuristics on top of coarse `stride *= 2` / `5-vote linspace` guessing.
- Never sacrifice text accuracy or recall on short cues (`>= 150ms`), fade-in/out subs, or stylized/colored subs just to shave seconds.
- Never break existing PySide6 Dark Studio compactness (`1160×680` / `1280×672 @ 150% DPI`) or ship without measurable automated pass/fail gates per phase.

---

## 3. Approaches Considered

| Approach | Summary | Timing Accuracy | 24m RTX 5060 Runtime | Trade-Offs |
| :--- | :--- | :--- | :--- | :--- |
| **A. Coarse Native-4D Neural Det (`S=2..3`) + Streaming Ring-Buffer Refinement + Interior Consensus OCR + Targeted Binary Search** *(Recommended)* | Stream video once, keep a 12-frame ring buffer (~5 MB) + exact `frame_pts_ms[]` table. Run native 4D-batched `Det Mobile` every `S` frames + tight-box edge/luma signature to detect onset/offset/0ms cuts. Refine every transition on the `S-1` buffered frames in-memory (`±0 frame`). Run borderless `ch_PP-OCRv5 Server` on interior frames with weighted consensus + targeted binary-search split only if OCR finds an unexpected intra-span text change. Snap ASS to inter-frame midpoints. | **`±0–1 frame` (`±0–20ms`, 0-frame libass error)** | **~60–95s** (`S=3`) / **~85–115s** (`S=2`) | Best accuracy + speed + CPU fallback; zero re-seeking in Stage 1; handles colored/stylized/fade subs natively; double-protected against 1-char changes. |
| **B. Pixel-Mask-Guided Detection + Per-Frame Tracking (VSF-Hybrid)** | Run CPU HSV/Sobel mask on 100% of frames (`S=1`), trigger neural Det only on mask changes. | `±0 frame` on white-outlined subs; degrades to `±160ms` on colored/glowing subs | ~90–140s | Re-introduces fragile HSV/edge heuristics into Stage 1 — exact root cause of past regressions. Rejected. |
| **C. Dense Full-Frame 4D Neural Det (`S=1`) + Tight-Box SSIM + Server Voting** | Run 4D-batched `Det Mobile` on every single frame (`35,000` frames). | `±0 frame` | ~130–170s GPU (**15–25m CPU**) | Simplest state machine, but wastes ~65% GPU Det compute on static interior frames and is unusable on CPU fallback. Keep as optional `coarse_step=1` exhaustive mode sharing the exact same pipeline. |

**Selected Architecture:** **Approach A** (with `coarse_step=1` naturally executing Approach C through the same code path).

---

## 4. Architecture & Key Invariants

### 4.1 Exact PTS Foundation (`VideoReader`)
- Decode first frame in `__init__()` (or fast packet-demux duration probe when container duration is missing — never full-video decode on open!).
- Anchor `origin_pts_ms` to the **signed first decoded frame's actual PTS** (`first_frame_pts_ms`), computing `rel_ms = raw_pts_ms - origin_pts_ms` *before* clamping to `>= 0` (fixes negative B-frame edit-list shift and `stream.start_time = None` multi-second shift).
- Record `frame_pts_ms: list[int]` for every decoded frame `0..N-1` during `iter_roi_frames()`, supporting optional `start_ms` / `end_ms` (keyframe pre-roll seek + exact filtering while keeping **absolute** `rel_ms` unchanged!).
- Crop ROI directly from decoded frame (avoiding unnecessary full-frame BGR conversion on skipped frames when `sample_step > 1`, or fast slice copy into ring buffer).

### 4.2 Stage 1: Native 4D `Det Mobile` + Tight-Box Visual Signature + Ring-Buffer Boundary Refinement
- **Native 4D Batching (`OcrEngine.detect_batch`)**:
  - Replace black-padded stitched canvas with true 4D NCHW batch tensor `(N, 3, rh, rw)` (`N=16` DirectML, `N=8` CPU), `"Det.box_thresh": 0.55`, `"Det.use_dilation": True`, `"Det.score_mode": "fast"`.
  - Fast empty-map short-circuit (`preds[i].max() < box_thresh`) skips contour postprocessing on blank frames.
  - Retain injected-callable compatibility for deterministic unit tests.
- **Tight-Box Visual Signature (`TightSignature`)**:
  - Computed **strictly inside `Det Mobile`'s union text box** (ignoring ~80% of moving anime background): normalized Sobel/stroke edge mask + downsampled luma patch (`64×16`).
  - Two consecutive coarse hits belong to the same cue **only if** box geometry is tightly aligned (`tol <= 0.08`) AND tight-box appearance matches (`edge_iou >= 0.58` and `luma_ncc >= 0.78`).
  - **No Stage 1 `flash_bridge_max_ms` blind gluing**: an `EMPTY` coarse hit closes the span immediately (same-cue glare flash bridging happens in Stage 2/postprocess *after* text is known).
- **Streaming Ring-Buffer Refinement (`±0 frame`, zero re-seeks)**:
  - Keep a sliding ring buffer of `W = sample_step + 4` recent `(frame_idx, rel_ms, bgr_roi)` entries (~5 MB RAM).
  - Default coarse step: `sample_step = 2` for `<= 30fps` (`67–83ms` stride, guaranteeing `>= 2` hits on every `>= 150ms` cue!), `3` for `50–60fps`; `1` when `config.coarse_step == 1`.
  - On `EMPTY → TEXT`, `TEXT → EMPTY`, or `TEXT_A → TEXT_B`, run `Det Mobile` + `TightSignature` on the `sample_step - 1` buffered intermediate frames to lock exact inclusive `[start_idx, end_idx]`.
  - Derive timestamps from `frame_pts_ms[]`: `start_ms = pts[start_idx]`, `end_ms = pts[end_idx + 1]` (or `pts[end_idx] + median_frame_ms` at EOF), clamped to `[0, duration_ms]` and `trim_end_ms`.

### 4.3 Stage 2: Interior-First Weighted Consensus OCR + Targeted Binary-Search Split
- Store up to `_MAX_SPAN_SAMPLES = 24` evenly distributed JPEG Q95 samples **with exact `(frame_idx, pts_ms)`** (always preserving clean interior coverage; never dropping the tail via broken `idx % stride` modulo!).
- **Interior-first probing**: Probe `[18%, 50%, 82%]` first (avoiding fade-in/out edge frames and selecting median-luma neighbor to dodge white flash frames). Early-exit at 3 calls if all 3 match identically with `min(conf) >= 0.93`, `dur <= 2800ms`, and zero edge/symbol noise.
- **Targeted Binary-Search Safety Split**: If any two adjacent evaluated samples disagree (`not exact/strict match`), bisect intermediate samples in `[i, j]` (and if needed, read exact frame via `VideoReader` only for the rare ~1% intra-span 1-char minimal-pair split) to pinpoint the exact cut frame — **never** snapping to coarse vote intervals.
- **Weighted + Edge-Noise-Pooling + Character-Level Consensus**:
  - Expand `_EDGE_NOISE_RE` to include stray edge symbols (`#|/\_^~` beside Hanzi).
  - Score candidates by `sum(conf) * (0.72 if noisy else 1.0)`.
  - Pool edge-stripped noisy votes onto matching clean votes; support clean fuller Hanzi containment (`>= 0.92`); run selective `brightness_filter=True` retry on up to 2 frames if winner is still noisy or `< conf_threshold`.
  - Drop isolated 1-frame low-conf glitches and 1–2 char ultra-low-conf background ghosts (`conf < 0.72, dur < 400ms`) while preserving all real short/low-conf dialogue.
- **DirectML Thread Safety**: Guard lazy init AND inference inside `OcrEngine` with an internal `threading.RLock()` so retiring cancel threads or `Test 1 frame` can never race DirectML sessions. Throttle progress callbacks to ~12 Hz.

### 4.4 Postprocess & Exporter Standards
- Tighten `postprocess.py`: protect 0–160ms back-to-back 1-char substitutions (`"放过你的"` vs `"放过他的"`) from blind Tier-1 merge; clamp `end_ms <= duration_ms`; preserve exact winner confidence.
- Shared `src/transapp/exporters/sanitize.py`:
  - Canonical line cleaner (strip blank lines, skip empty cues identically in both `.ass` and `.srt`).
  - Stably sort cues by `(start_ms, end_ms, index)`.
  - ASS sanitizer: `{` → `｛`, `}` → `｝`, escape literal `\N`/`\n`/`\h` before converting real newlines to `\N`.
  - **Aegisub Midpoint Centisecond Snapping**: when `frame_pts_ms` is available, snap `start_cs` / `end_cs` to inter-frame midpoints `(pts[k-1] + pts[k]) / 2` (fallback deterministic half-up `(ms + 5) // 10`), then enforce post-quantization non-overlap `prev.end_cs <= curr.start_cs` (and `prev.end_ms <= curr.start_ms` in SRT).
  - Preserve template `[Fonts]`/`[Graphics]` sections and valid `PlayResX/Y` (inject video `PlayResX/Y` only when missing/`<= 0`), plus optional `style_name`.

### 4.5 Pro-Subber Dark Studio UX (`MainWindow` + `ReviewTable`)
- **OP/ED Time-Range Trim (`In / Out`)**: `TrimRangeSlider` painting active `[In, Out]` band + compact `In / Out / Clear` strip + `I` / `O` shortcuts; resets on `load_video()`.
- **Timing Verification & QA Loop**: Upgrade right-hand preview panel into a 4-row mini-inspector showing duration/char count, overlap/short-cue warning, amber `flag_reason` banner + `[✓ Duyệt]` button, ROI thumbnail, and `[⏮ Start]` (`Alt+S`) / `[End ⏭]` (`Alt+E`) / `[← In]` (`Ctrl+[`) / `[Out →]` (`Ctrl+]`) / `[-1f]` / `[+1f]` controls synced to `MainWindow.seek_to_ms()`.
- **Auto-clear `flagged` on manual edit** in-place (keeping row visible in `flagged_only` until filter switch); uniform `HH:MM:SS.mmm` display; `\N` multi-line editing; `ExtendedSelection` multi-delete; Merge (`Ctrl+M`), Split (`Ctrl+K`), Undo/Redo (`Ctrl+Z`/`Ctrl+Y`), collapsible Find/Replace (`Ctrl+F`/`Ctrl+H`); drag-and-drop video; unsaved-edits guard; Open Output Folder button.

---

## 5. Strict 4-Phase Rollout & Acceptance Gates

Work proceeds **strictly one phase at a time** — each phase must pass its automated test suite AND real-video verification gate before starting the next:

1. **Phase 1 — PTS Foundation + Native 4D Neural Det + Frame-Accurate Ring-Buffer Engine**
   - *Gate:* `0ms` drift on synthetic 23.976/25/29.97/VFR + non-zero/negative `start_pts_ms`; `±1 frame` accuracy on 0ms back-to-back cuts & `160ms` short cues; 1080p clip spans `~22` in `< 15s`; 24m episode spans `< 400` in `< 120s` with zero glued 1–2s shifts.
2. **Phase 2 — Stage 2 Weighted Consensus Voting + Safety Binary Search + Postprocess & Thread Safety**
   - *Gate:* `16/16` checkpoints on 24m Episode 01, `22/22` on 1080p clip, `0` post-OP edge-Latin noise, `0` false merges on minimal pairs (`放过你的`/`放过他的`), thread-safe concurrent OCR & cancel tests pass.
3. **Phase 3 — Exporter Hardening + Aegisub Midpoint Snapping + OP/ED Time-Range Trim (`In / Out`)**
   - *Gate:* `100.0%` libass visibility oracle match across 1,000 cues on 23.976/25/29.97/VFR; `0` overlaps; `{}` escaping & ASS/SRT parity verified; OP/ED trim skips OP/ED cleanly while preserving exact absolute video timestamps.
4. **Phase 4 — Pro-Subber `ReviewTable`, Video↔Table Sync & Power-User Editing**
   - *Gate:* All PySide6 UI/shortcut/undo/merge/split/find-replace/unsaved-guard tests pass headless (`QT_QPA_PLATFORM=offscreen`) + user confirms Aegisub & UI workflow on real episode.
