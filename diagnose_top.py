"""
diagnose_top.py
===============
Script chẩn đoán 3 cấp cho top.mp3 (và bài phonk tương tự).

Cấp 1 — Segment Stability : sharpness curve cho full / seg A / seg B / A+B
Cấp 2 — Mountain Consensus: top-5 local maxima, so sánh height vs area vs persistence
Cấp 3 — Beat Participation : len(segs) tại mỗi offset → 335ms vs 465ms

Chạy:
    python diagnose_top.py top.mp3 122.998
"""

import sys
import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
PRE_MS   = 100
POST_MS  = 200
SMOOTH   = 5      # uniform_filter1d size — giữ nhất quán test_cases.py
TOP_N    = 5      # top N mountains cho cấp 2

# Segments của top.mp3 (giây) — lấy từ osu observation
SEGMENTS = {
    "full"  : None,                   # toàn bài
    "seg_A" : (15.587,  44.368),
    "seg_B" : (61.929,  94.125),
    "A+B"   : [(15.587, 44.368), (61.929, 94.125)],  # combined
}


# ─────────────────────────────────────────────────────────────────────────────
# CORE: build sharpness curve từ một tập interval
# ─────────────────────────────────────────────────────────────────────────────
def build_sharpness(y, sr, bpm, intervals=None):
    """
    intervals: None → toàn bài
               list of (start_s, end_s) → chỉ lấy beat trong các khoảng đó
    Trả về (offsets, sharpnesses, smoothed, beat_counts)
      beat_counts[i] = số beat thực tế đóng góp vào offset i
    """
    beat_len = 60.0 / bpm
    beat_ms  = beat_len * 1000
    seg_smp  = int((PRE_MS + POST_MS) / 1000 * sr)
    total_s  = len(y) / sr

    # Chuẩn hoá intervals thành list of (start, end)
    if intervals is None:
        ivs = [(0.0, total_s)]
    elif isinstance(intervals[0], tuple):
        ivs = list(intervals)
    else:
        ivs = [intervals]  # single tuple

    def stack_template(offset_ms):
        off_s = offset_ms / 1000.0
        segs  = []
        for (t0, t1) in ivs:
            for t in np.arange(t0 + off_s, t1 - beat_len, beat_len):
                s = int((t - PRE_MS / 1000) * sr)
                e = s + seg_smp
                if s >= 0 and e <= len(y):
                    chunk = y[s:e]
                    if np.abs(chunk).max() > 0.01:
                        segs.append(chunk)
        return np.mean(segs, axis=0) if segs else None, len(segs)

    def sharpness_val(tmpl):
        beat_idx = int(PRE_MS / (PRE_MS + POST_MS) * len(tmpl))
        win      = int(0.02 * sr)
        left     = tmpl[beat_idx - win : beat_idx]
        right    = tmpl[beat_idx      : beat_idx + win]
        return float(np.var(right) / (np.var(left) + 1e-10))

    offsets      = np.arange(0, int(beat_ms), 1)
    sharpnesses  = []
    beat_counts  = []
    for o in offsets:
        tmpl, cnt = stack_template(o)
        sharpnesses.append(sharpness_val(tmpl) if tmpl is not None else 0.0)
        beat_counts.append(cnt)

    sharpnesses = np.array(sharpnesses)
    smoothed    = uniform_filter1d(sharpnesses, size=SMOOTH)
    beat_counts = np.array(beat_counts)
    return offsets, sharpnesses, smoothed, beat_counts


# ─────────────────────────────────────────────────────────────────────────────
# CẤP 2 HELPER: top-N local maxima
# ─────────────────────────────────────────────────────────────────────────────
def top_mountains(offsets, smoothed, n=TOP_N):
    peaks, props = find_peaks(smoothed, prominence=0.1 * smoothed.max(), width=3)
    if len(peaks) == 0:
        return []
    # Tính area bằng tích phân hình thang quanh mỗi đỉnh (±width)
    results = []
    for i, pk in enumerate(peaks):
        h = smoothed[pk]
        w = int(props["widths"][i])
        lo = max(0, pk - w)
        hi = min(len(smoothed) - 1, pk + w)
        area = float(np.trapz(smoothed[lo:hi+1]))
        results.append({
            "rank"    : 0,
            "peak_ms" : int(offsets[pk]),
            "height"  : round(h, 3),
            "width_ms": w * 2,
            "area"    : round(area, 2),
        })
    results.sort(key=lambda x: x["height"], reverse=True)
    for i, r in enumerate(results):
        r["rank"] = i + 1
    return results[:n]


# ─────────────────────────────────────────────────────────────────────────────
# LOG CẤP 1 + 3 (in bảng)
# ─────────────────────────────────────────────────────────────────────────────
def log_level1(label, offsets, smoothed, beat_counts, gt_offset=None):
    peak_idx = int(np.argmax(smoothed))
    peak_ms  = int(offsets[peak_idx])
    peak_h   = round(smoothed[peak_idx], 3)
    # area: tích phân toàn bộ smoothed (đại diện tổng năng lượng coherent)
    area     = round(float(np.trapz(smoothed)), 2)
    beats_at_peak = int(beat_counts[peak_idx])

    print(f"  [{label}]")
    print(f"    peak_ms={peak_ms}ms  height={peak_h}  area={area}  beats@peak={beats_at_peak}", end="")
    if gt_offset is not None:
        err = peak_ms - gt_offset
        print(f"  err_vs_GT={err:+.0f}ms", end="")
    print()


def log_level3_compare(label, offsets, beat_counts, check_offsets):
    print(f"  [{label}] beat participation:")
    for ms in check_offsets:
        idx = np.argmin(np.abs(offsets - ms))
        print(f"    offset={ms}ms → {int(beat_counts[idx])} beats")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def diagnose(path, bpm, gt_offset=465.0):
    print(f"\n{'='*70}")
    print(f"DIAGNOSE: {path}  BPM={bpm}  GT={gt_offset}ms")
    print(f"{'='*70}")

    y, sr = librosa.load(path, mono=True)
    total_s = len(y) / sr
    print(f"Duration: {total_s:.2f}s\n")

    curves = {}

    # ── CẤP 1: build curves ──────────────────────────────────────────────────
    print("── CẤP 1: SEGMENT STABILITY ──")
    interval_map = {
        "full"  : None,
        "seg_A" : (15.587,  44.368),
        "seg_B" : (61.929,  94.125),
        "A+B"   : [(15.587, 44.368), (61.929, 94.125)],
    }
    for label, ivs in interval_map.items():
        offsets, sharp, smoothed, bcounts = build_sharpness(y, sr, bpm, intervals=ivs)
        curves[label] = (offsets, sharp, smoothed, bcounts)
        log_level1(label, offsets, smoothed, bcounts, gt_offset)
    print()

    # ── CẤP 2: top-5 mountains ───────────────────────────────────────────────
    print("── CẤP 2: MOUNTAIN CONSENSUS (top-5 by height) ──")
    for label, (offsets, _, smoothed, _) in curves.items():
        mountains = top_mountains(offsets, smoothed)
        print(f"  [{label}]")
        print(f"    {'rank':>4}  {'peak_ms':>8}  {'height':>8}  {'width_ms':>9}  {'area':>8}")
        for m in mountains:
            flag = " ← GT" if abs(m["peak_ms"] - gt_offset) < 15 else ""
            print(f"    #{m['rank']:>3}  {m['peak_ms']:>8}ms  {m['height']:>8.3f}  {m['width_ms']:>9}ms  {m['area']:>8.2f}{flag}")
        print()

    # ── CẤP 3: beat participation ─────────────────────────────────────────────
    print("── CẤP 3: BEAT PARTICIPATION ──")
    check_ms = [312, 335, 465]  # 312=current pred, 335=mid candidate, 465=GT
    for label, (offsets, _, _, bcounts) in curves.items():
        log_level3_compare(label, offsets, bcounts, check_ms)
    print()

    # ── PLOT ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(16, 8), sharex=True, sharey=False)
    fig.patch.set_facecolor('#0d0d12')
    fig.suptitle(f"Segment Stability — {path}", color='#ddd', fontsize=13)

    label_list  = ["full", "seg_A", "seg_B", "A+B"]
    colors      = ['white', '#7ecfff', '#ff8888', '#f0c040']

    for ax, label, col in zip(axes.flat, label_list, colors):
        offsets, _, smoothed, bcounts = curves[label]
        ax.set_facecolor('#0d0d12')
        ax.grid(alpha=0.12)
        ax.tick_params(colors='#aaa')
        for sp in ax.spines.values():
            sp.set_color('#333')

        ax.plot(offsets, smoothed, color=col, lw=1.5, label=label)
        ax.axvline(x=gt_offset, color='#00ff88', lw=1.5, ls='--', label=f'GT {gt_offset:.0f}ms')

        # Mark top-2 mountains
        mountains = top_mountains(offsets, smoothed, n=2)
        for m in mountains:
            ax.axvline(x=m["peak_ms"], color=col, lw=1.0, ls=':', alpha=0.6,
                       label=f'peak {m["peak_ms"]}ms h={m["height"]:.2f}')

        ax.set_title(label, color='#ddd', fontsize=11)
        ax.set_ylabel('Sharpness', color='#aaa')
        ax.set_xlabel('Offset (ms)', color='#aaa')
        ax.legend(fontsize=8, facecolor='#1a1a22', labelcolor='#aaa')

    plt.tight_layout()
    out = path.rsplit(".", 1)[0] + "_segments.png"
    plt.savefig(out, dpi=120, facecolor='#0d0d12')
    plt.close('all')
    print(f"Plot saved → {out}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python diagnose_top.py <file.mp3> <bpm> [gt_offset_ms]")
        sys.exit(1)
    path       = sys.argv[1]
    bpm        = float(sys.argv[2])
    gt_offset  = float(sys.argv[3]) if len(sys.argv) > 3 else 465.0
    diagnose(path, bpm, gt_offset)
