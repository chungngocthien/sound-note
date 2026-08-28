"""
diagnose_phase.py
=================
Kiểm tra hai hypothesis:
  H1: Class A selection quá aggressive → dùng all onsets thì 110 BPM nổi lên
  H2: Onset selection đúng nhưng phase=0 assumption sai → cho offset tự do thì 110 nổi

Proxy rẻ: histogram phase alignment (không cần stack template)
  score(bpm, offset) = tỉ lệ onset rơi trong cửa sổ THRESHOLD_MS quanh lưới beat

Dùng:
    python diagnose_phase.py highscore.mp3 110.0 4.747
    python diagnose_phase.py stardust.mp3  110.0 0.0

Arg 3: thời điểm bắt đầu phân tích (giây) — skip intro lặng
"""

import sys
import numpy as np
import librosa
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ─── Args ────────────────────────────────────────────────────────────────────
path         = sys.argv[1] if len(sys.argv) > 1 else "highscore.mp3"
GROUND_TRUTH = float(sys.argv[2]) if len(sys.argv) > 2 else 110.0
SKIP_S       = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0   # skip intro

BPM_MIN, BPM_MAX, BPM_STEP = 80, 220, 0.5
THRESHOLD_MS = 50   # cửa sổ tolerance
OFFSET_STEP  = 2    # ms, bước quét offset (proxy nên để thô)

print(f"\nFile: {path} | GT: {GROUND_TRUTH} BPM | Skip intro: {SKIP_S}s")
print("="*60)

# ─── Load + skip intro ────────────────────────────────────────────────────────
y_full, sr = librosa.load(path, mono=True, duration=120)
skip_smp   = int(SKIP_S * sr)
y          = y_full[skip_smp:]
duration   = len(y) / sr
print(f"Analyzing {duration:.1f}s (after skipping {SKIP_S}s intro)")

# ─── Low-band filter ─────────────────────────────────────────────────────────
D     = librosa.stft(y, hop_length=512)
freqs = librosa.fft_frequencies(sr=sr)
mask  = (freqs >= 40) & (freqs <= 200)
D_low = D.copy(); D_low[~mask, :] = 0
y_low = librosa.istft(D_low, hop_length=512)

# ─── Onset detection ─────────────────────────────────────────────────────────
onset_frames = librosa.onset.onset_detect(
    y=y_low, sr=sr, hop_length=512, delta=0.05, wait=8
)
onset_times_raw = librosa.frames_to_time(onset_frames, sr=sr, hop_length=512)
# Offset lại theo thời gian thật (đã skip intro)
onset_times = onset_times_raw + SKIP_S

# Class A: top 10% energy rise
energies = []
for t in onset_times_raw:
    idx  = int(t * sr)
    pre  = y[max(0, idx - int(0.02*sr)):idx]
    post = y[idx:min(len(y), idx + int(0.02*sr))]
    energies.append(post.std() - pre.std())
energies = np.array(energies)
thr90    = np.percentile(energies, 90)
class_a  = onset_times[energies >= thr90]

print(f"All onsets: {len(onset_times)}  |  Class A (top 10%): {len(class_a)}")

# ─── Phase alignment score (proxy) ───────────────────────────────────────────
THRESHOLD_S = THRESHOLD_MS / 1000.0

def phase_score(bpm, onset_arr, offset_ms=0.0):
    """Score onset_arr với lưới bpm tại phase offset_ms."""
    if len(onset_arr) == 0:
        return 0.0
    beat_len = 60.0 / bpm
    off_s    = offset_ms / 1000.0
    # Shift onset về lưới: (t - offset) mod beat_len
    shifted  = (onset_arr - off_s) % beat_len
    # Khoảng cách tới lưới gần nhất
    dev      = np.minimum(shifted, beat_len - shifted)
    return float(np.mean(np.maximum(0, 1 - dev / THRESHOLD_S)))

def best_phase_score(bpm, onset_arr):
    """max score qua tất cả offset trong [0, beat_len)."""
    beat_ms  = 60_000.0 / bpm
    offsets  = np.arange(0, beat_ms, OFFSET_STEP)
    scores   = [phase_score(bpm, onset_arr, o) for o in offsets]
    best_o   = offsets[int(np.argmax(scores))]
    return float(np.max(scores)), float(best_o)

# ─── Scan BPM ────────────────────────────────────────────────────────────────
print("\nScanning BPM (phase=0, all onsets)...")
bpm_list    = np.arange(BPM_MIN, BPM_MAX, BPM_STEP)

scores_all_fixed  = np.array([phase_score(b, onset_times, 0.0) for b in bpm_list])
scores_ca_fixed   = np.array([phase_score(b, class_a,     0.0) for b in bpm_list])

print("Scanning BPM (phase=free, all onsets) — slower...")
scores_all_free   = []
best_offsets_all  = []
for b in bpm_list:
    s, o = best_phase_score(b, onset_times)
    scores_all_free.append(s)
    best_offsets_all.append(o)
scores_all_free  = np.array(scores_all_free)
best_offsets_all = np.array(best_offsets_all)

# ─── Report ──────────────────────────────────────────────────────────────────
def top5(scores, label):
    idx = np.argsort(scores)[::-1][:5]
    print(f"\n  {label}:")
    for i in idx:
        b  = bpm_list[i]
        diff = abs(b - GROUND_TRUTH)
        mark = " ★" if diff < 3 else ""
        print(f"    {b:.1f} BPM  score={scores[i]:.4f}  Δ{diff:.1f}{mark}")

print("\n" + "="*60)
print("RESULTS")
top5(scores_all_fixed, "All onsets, phase=0")
top5(scores_ca_fixed,  "Class A only, phase=0")
top5(scores_all_free,  "All onsets, phase=FREE")

# GT score
def gt_score_report(scores, label):
    idx_gt = np.argmin(np.abs(bpm_list - GROUND_TRUTH))
    print(f"  [{label}] GT {GROUND_TRUTH} BPM → score={scores[idx_gt]:.4f}")

print()
gt_score_report(scores_all_fixed, "all+fixed ")
gt_score_report(scores_ca_fixed,  "classA+fix")
gt_score_report(scores_all_free,  "all+free  ")

# Best offset tại GT BPM (phase-free)
idx_gt = np.argmin(np.abs(bpm_list - GROUND_TRUTH))
print(f"\n  Best offset for GT {GROUND_TRUTH} BPM (phase-free): {best_offsets_all[idx_gt]:.0f} ms")
print(f"  (Known ground truth offset: Highscore=280ms, stardust=210ms)")

# ─── Plot ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 10))
fig.patch.set_facecolor('#0d0d12')
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.3)

def style_ax(ax):
    ax.set_facecolor('#0d0d12')
    ax.grid(alpha=0.12, color='#555')
    ax.tick_params(colors='#aaa')
    for sp in ax.spines.values(): sp.set_color('#333')

def vlines(ax, gt):
    ax.axvline(x=gt, color='#00ff88', linewidth=2, linestyle='--', label=f'GT {gt} BPM')
    for mult in [0.5, 1.5, 2.0, 3.0]:
        h = gt * mult
        if BPM_MIN <= h <= BPM_MAX:
            ax.axvline(x=h, color='#7ecfff', linewidth=0.7, linestyle=':', alpha=0.5)

# Panel 1: All onsets, phase=0
ax1 = fig.add_subplot(gs[0, 0])
style_ax(ax1)
ax1.plot(bpm_list, scores_all_fixed, color='white', linewidth=1)
vlines(ax1, GROUND_TRUTH)
ax1.set_title('All onsets — phase=0', color='#ddd', fontsize=10)
ax1.set_ylabel('Score', color='#aaa'); ax1.set_xlabel('BPM', color='#aaa')
ax1.legend(fontsize=8, facecolor='#1a1a22', labelcolor='#aaa')

# Panel 2: Class A, phase=0
ax2 = fig.add_subplot(gs[0, 1])
style_ax(ax2)
ax2.plot(bpm_list, scores_ca_fixed, color='#ff8888', linewidth=1)
vlines(ax2, GROUND_TRUTH)
ax2.set_title('Class A only — phase=0', color='#ddd', fontsize=10)
ax2.set_ylabel('Score', color='#aaa'); ax2.set_xlabel('BPM', color='#aaa')
ax2.legend(fontsize=8, facecolor='#1a1a22', labelcolor='#aaa')

# Panel 3: All onsets, phase=free
ax3 = fig.add_subplot(gs[1, 0])
style_ax(ax3)
ax3.plot(bpm_list, scores_all_free, color='#7ecfff', linewidth=1)
vlines(ax3, GROUND_TRUTH)
ax3.set_title('All onsets — phase FREE (max over offsets)', color='#ddd', fontsize=10)
ax3.set_ylabel('Score', color='#aaa'); ax3.set_xlabel('BPM', color='#aaa')
ax3.legend(fontsize=8, facecolor='#1a1a22', labelcolor='#aaa')

# Panel 4: Best offset per BPM (phase-free)
ax4 = fig.add_subplot(gs[1, 1])
style_ax(ax4)
ax4.scatter(bpm_list, best_offsets_all, s=1, color='#f0c040', alpha=0.5)
vlines(ax4, GROUND_TRUTH)
# Đánh dấu offset GT
gt_off = {"highscore": 280, "stardust": 210}
for name, off in gt_off.items():
    ax4.axhline(y=off, color='#00ff88', linewidth=0.8, linestyle='--', alpha=0.6,
                label=f'{name} GT offset {off}ms')
ax4.set_title('Best offset per BPM (phase-free scan)', color='#ddd', fontsize=10)
ax4.set_ylabel('Best offset (ms)', color='#aaa'); ax4.set_xlabel('BPM', color='#aaa')
ax4.legend(fontsize=8, facecolor='#1a1a22', labelcolor='#aaa')

import os
base = os.path.splitext(os.path.basename(path))[0]
out  = f"diagnose_phase_{base}.png"
plt.suptitle(f'Phase Alignment Diagnostic — {path}', color='#eee', fontsize=13, y=1.01)
plt.savefig(out, dpi=120, facecolor='#0d0d12', bbox_inches='tight')
print(f"\nSaved: {out}")
