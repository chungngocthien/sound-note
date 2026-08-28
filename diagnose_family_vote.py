"""
diagnose_family_vote.py
=======================
Script ngách: Subdivision Family Scoring + Positional Stability Matrix

Mục tiêu: Sau khi compare_templates.py và diagnose_low_freq.py chạy xong,
nếu low-freq peak vẫn không tự trỏ đúng GT, script này cung cấp tiebreaker
có cấu trúc dựa trên:

1. SUBDIVISION FAMILY MAPPING  : Nhóm các đỉnh theo family (mod beat/4 ≈ 122ms)
2. POSITIONAL STABILITY SCORE  : Mỗi đỉnh của full được score dựa trên
                                  drift so với peak tương ứng ở seg_A và seg_B
3. INTRA-FAMILY TIEBREAKER     : Trong cùng family, chọn member có stability
                                  cao nhất — không dùng height làm tiebreaker

Pipeline quyết định:
  full top-5 → group by family → score stability per member → pick winner

Tham số cấu hình:
  FAMILY_TOLERANCE_MS : Hai đỉnh cách nhau ≤ tolerance → cùng family member
  STABILITY_DRIFT_MAX : Drift giữa full-peak và segment-peak ≤ này → stable
  SUBDIV_PERIOD_MS    : Period của subdivision (beat_len / 4 ≈ 122ms)

Chạy:
    python diagnose_family_vote.py top.mp3 122.998 465
"""

import sys
import numpy as np
import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
PRE_MS              = 100
POST_MS             = 200
SMOOTH_SIZE         = 5
TOP_N               = 8           # Lấy nhiều đỉnh hơn để có đủ family member
FAMILY_TOLERANCE_MS = 15          # ±15ms → cùng subdivision slot
STABILITY_DRIFT_MAX = 20          # Drift ≤ 20ms giữa full và segment → stable
PROMINENCE_RATIO    = 0.08        # find_peaks prominence ≥ 8% của global max

SEGMENTS = {
    "seg_A": (15.587, 44.368),
    "seg_B": (61.929, 94.125),
}


# ─────────────────────────────────────────────────────────────────────────────
# CORE: build sharpness curve
# ─────────────────────────────────────────────────────────────────────────────
def build_sharpness(y, sr, bpm, intervals=None):
    beat_len = 60.0 / bpm
    beat_ms  = beat_len * 1000
    seg_smp  = int((PRE_MS + POST_MS) / 1000 * sr)
    total_s  = len(y) / sr

    if intervals is None:
        ivs = [(0.0, total_s)]
    elif isinstance(intervals[0], (int, float)):
        ivs = [intervals]
    else:
        ivs = list(intervals)

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
        return np.mean(segs, axis=0) if segs else None

    def sharpness_val(tmpl):
        beat_idx = int(PRE_MS / (PRE_MS + POST_MS) * len(tmpl))
        win      = int(0.02 * sr)
        left     = tmpl[beat_idx - win : beat_idx]
        right    = tmpl[beat_idx      : beat_idx + win]
        return float(np.var(right) / (np.var(left) + 1e-10))

    offsets     = np.arange(0, int(beat_ms), 1)
    sharpnesses = np.array([sharpness_val(stack_template(o)) or 0.0 for o in offsets])
    smoothed    = uniform_filter1d(sharpnesses, size=SMOOTH_SIZE)
    return offsets, smoothed


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: Extract top-N peaks từ full curve
# ─────────────────────────────────────────────────────────────────────────────
def extract_peaks(offsets, smoothed, n=TOP_N):
    peaks, props = find_peaks(
        smoothed,
        prominence=PROMINENCE_RATIO * smoothed.max(),
        width=3
    )
    if len(peaks) == 0:
        return []
    results = []
    for pk in peaks:
        results.append({
            "peak_ms": int(offsets[pk]),
            "height" : round(float(smoothed[pk]), 4),
        })
    results.sort(key=lambda x: x["height"], reverse=True)
    return results[:n]


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: Group peaks into subdivision families
# Hai đỉnh A và B cùng family nếu |A - B| mod subdiv_period ≤ FAMILY_TOLERANCE_MS
# ─────────────────────────────────────────────────────────────────────────────
def group_into_families(peaks, subdiv_period_ms):
    families = []
    assigned = [False] * len(peaks)

    for i, pk in enumerate(peaks):
        if assigned[i]:
            continue
        family = [pk]
        assigned[i] = True
        for j, pk2 in enumerate(peaks):
            if assigned[j]:
                continue
            diff = abs(pk["peak_ms"] - pk2["peak_ms"]) % subdiv_period_ms
            diff = min(diff, subdiv_period_ms - diff)   # wrap-around
            if diff <= FAMILY_TOLERANCE_MS:
                family.append(pk2)
                assigned[j] = True
        families.append(family)

    return families


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: Positional stability score
# Với mỗi đỉnh của full, tìm peak gần nhất trong mỗi segment curve,
# tính drift, score = 1.0 nếu drift ≤ STABILITY_DRIFT_MAX, giảm tuyến tính
# ─────────────────────────────────────────────────────────────────────────────
def nearest_peak_ms(seg_offsets, seg_smoothed, target_ms, search_window=60):
    lo = max(0, target_ms - search_window)
    hi = min(int(seg_offsets[-1]), target_ms + search_window)
    mask = (seg_offsets >= lo) & (seg_offsets <= hi)
    if not mask.any():
        return None, None
    sub_smooth = seg_smoothed.copy()
    sub_smooth[~mask] = 0
    idx = int(np.argmax(sub_smooth))
    return int(seg_offsets[idx]), float(seg_smoothed[idx])


def stability_score(full_peak_ms, seg_curves):
    """
    Trả về score [0.0, 1.0] và dict drift chi tiết.
    Score = mean(max(0, 1 - drift/STABILITY_DRIFT_MAX)) qua các segments.
    """
    drifts = {}
    scores = []
    for label, (seg_off, seg_sm) in seg_curves.items():
        nearest_ms, nearest_h = nearest_peak_ms(seg_off, seg_sm, full_peak_ms)
        if nearest_ms is None:
            drifts[label] = None
            scores.append(0.0)
        else:
            drift = abs(nearest_ms - full_peak_ms)
            drifts[label] = drift
            s = max(0.0, 1.0 - drift / STABILITY_DRIFT_MAX)
            scores.append(s)
    return float(np.mean(scores)) if scores else 0.0, drifts


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def diagnose(path, bpm, gt_offset=465.0):
    print(f"\n{'='*70}")
    print(f"FAMILY VOTE DIAGNOSE: {path}  BPM={bpm}  GT={gt_offset}ms")
    print(f"{'='*70}")

    y, sr = librosa.load(path, mono=True)
    subdiv_period_ms = (60.0 / bpm) * 1000 / 4   # ≈ 122ms cho 122.998 BPM
    print(f"Subdivision period: {subdiv_period_ms:.1f}ms\n")

    # Build curves
    print("Building sharpness curves...")
    full_off, full_sm = build_sharpness(y, sr, bpm, intervals=None)
    seg_curves = {}
    for label, (t0, t1) in SEGMENTS.items():
        off, sm = build_sharpness(y, sr, bpm, intervals=(t0, t1))
        seg_curves[label] = (off, sm)
    print("Done.\n")

    # Step 1: Extract peaks từ full
    peaks = extract_peaks(full_off, full_sm, n=TOP_N)
    print(f"── STEP 1: Top-{TOP_N} peaks from full curve ──")
    for p in peaks:
        flag = " ← GT" if abs(p["peak_ms"] - gt_offset) < 15 else ""
        print(f"  {p['peak_ms']:>5}ms  h={p['height']:.4f}{flag}")
    print()

    # Step 2: Family grouping
    families = group_into_families(peaks, subdiv_period_ms)
    print(f"── STEP 2: Subdivision Family Grouping (period={subdiv_period_ms:.0f}ms) ──")
    for i, fam in enumerate(families):
        members_str = ", ".join(f"{m['peak_ms']}ms(h={m['height']:.3f})" for m in fam)
        print(f"  Family {i+1}: [{members_str}]")
    print()

    # Step 3: Stability scoring per peak
    print("── STEP 3: Positional Stability Scores ──")
    print(f"  {'peak_ms':>8}  {'height':>8}  {'stability':>10}  {'drift_A':>8}  {'drift_B':>8}")
    print(f"  {'-'*55}")
    scored_peaks = []
    for p in peaks:
        score, drifts = stability_score(p["peak_ms"], seg_curves)
        p["stability"] = score
        p["drifts"] = drifts
        scored_peaks.append(p)
        drift_A = f"{drifts.get('seg_A', '?')}ms" if drifts.get('seg_A') is not None else "None"
        drift_B = f"{drifts.get('seg_B', '?')}ms" if drifts.get('seg_B') is not None else "None"
        flag = " ← GT" if abs(p["peak_ms"] - gt_offset) < 15 else ""
        print(f"  {p['peak_ms']:>8}ms  {p['height']:>8.4f}  {score:>10.3f}  {drift_A:>8}  {drift_B:>8}{flag}")
    print()

    # Step 4: Family winner — trong mỗi family, chọn member có stability cao nhất
    print("── STEP 4: Family Winners (tiebreaker = stability, not height) ──")
    family_winners = []
    for i, fam in enumerate(families):
        fam_with_score = [p for p in scored_peaks if p["peak_ms"] in [m["peak_ms"] for m in fam]]
        winner = max(fam_with_score, key=lambda x: (x["stability"], x["height"]))
        family_winners.append(winner)
        flag = " ← GT" if abs(winner["peak_ms"] - gt_offset) < 15 else ""
        print(f"  Family {i+1} winner: {winner['peak_ms']}ms  stability={winner['stability']:.3f}  h={winner['height']:.4f}{flag}")
    print()

    # Final decision: family winner với stability cao nhất toàn cục
    final = max(family_winners, key=lambda x: (x["stability"], x["height"]))
    err = final["peak_ms"] - gt_offset
    print(f"── FINAL DECISION ──")
    print(f"  Selected offset : {final['peak_ms']}ms")
    print(f"  Stability score : {final['stability']:.3f}")
    print(f"  Height          : {final['height']:.4f}")
    print(f"  Error vs GT     : {err:+.0f}ms  {'✓' if abs(err) < 10 else '✗'}")
    print()

    # ── PLOT ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True)
    fig.patch.set_facecolor('#0d0d12')
    fig.suptitle(f"Family Vote — {path}  |  Final: {final['peak_ms']}ms (err {err:+.0f}ms)", color='#ddd', fontsize=13)

    curve_data = [("full", full_off, full_sm, "white"), ("seg_A", *seg_curves["seg_A"], "#7ecfff"), ("seg_B", *seg_curves["seg_B"], "#ff8888")]
    for ax, (label, off, sm, col) in zip(axes, curve_data):
        ax.set_facecolor('#0d0d12')
        ax.grid(alpha=0.12)
        ax.tick_params(colors='#aaa')
        for sp in ax.spines.values(): sp.set_color('#333')

        ax.plot(off, sm, color=col, lw=1.5, label=label)
        ax.axvline(x=gt_offset, color='#00ff88', lw=1.5, ls='--', label=f'GT {gt_offset:.0f}ms')
        ax.axvline(x=final["peak_ms"], color='#f0c040', lw=1.5, ls=':', label=f'Vote {final["peak_ms"]}ms')

        # Đánh dấu các đỉnh đã score
        for p in scored_peaks:
            color = '#f0c040' if p["peak_ms"] == final["peak_ms"] else '#888'
            ax.axvline(x=p["peak_ms"], color=color, lw=0.8, ls='-', alpha=0.4)

        ax.set_title(label, color='#ddd', fontsize=11)
        ax.set_xlabel("Offset (ms)", color='#aaa')
        ax.set_ylabel("Sharpness", color='#aaa')
        ax.legend(fontsize=8, facecolor='#1a1a22', labelcolor='#aaa')

    plt.tight_layout()
    out = "diagnose_family_vote.png"
    plt.savefig(out, dpi=120, facecolor='#0d0d12')
    plt.close("all")
    print(f"Plot saved → {out}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python diagnose_family_vote.py <file.mp3> <bpm> [gt_offset_ms]")
        sys.exit(1)
    path      = sys.argv[1]
    bpm       = float(sys.argv[2])
    gt_offset = float(sys.argv[3]) if len(sys.argv) > 3 else 465.0
    diagnose(path, bpm, gt_offset)