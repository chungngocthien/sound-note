"""
diagnose_bpm.py
===============
Không sửa gì. Chỉ quan sát.

Câu hỏi cần trả lời:
  1. Onset detector đang bắt vào cái gì trên Highscore?
  2. Class A (top 10% energy) có thực sự là kick không?
  3. Grid scan cho BPM nào điểm cao nhất, và tại sao?

Dùng:
    python diagnose_bpm.py highscore.mp3 110.0
    python diagnose_bpm.py stardust.mp3  110.0

Output:
  - diagnose_bpm_<filename>.png  (3 panel: spectrogram + onsets, score curve, class A timing)
  - in ra terminal: top 5 BPM candidates và score của chúng
"""

import sys
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ─── Args ────────────────────────────────────────────────────────────────────
path         = sys.argv[1] if len(sys.argv) > 1 else "highscore.mp3"
GROUND_TRUTH = float(sys.argv[2]) if len(sys.argv) > 2 else 110.0
BPM_MIN, BPM_MAX, STEP = 80, 220, 0.5
THRESHOLD_MS = 50  # 50ms tolerance khi score onset

print(f"\nFile: {path} | Ground truth BPM: {GROUND_TRUTH}")
print("="*60)

# ─── Load ─────────────────────────────────────────────────────────────────────
y, sr = librosa.load(path, mono=True, duration=60)   # chỉ 60s đầu để nhanh
duration = len(y) / sr
print(f"Loaded {duration:.1f}s @ {sr}Hz")

# ─── Low-band filter (giống beat_analyzer.py) ────────────────────────────────
D     = librosa.stft(y, hop_length=512)
freqs = librosa.fft_frequencies(sr=sr)
mask  = (freqs >= 40) & (freqs <= 200)
D_low = D.copy()
D_low[~mask, :] = 0
y_low = librosa.istft(D_low, hop_length=512)

# ─── Onset detection ─────────────────────────────────────────────────────────
onset_frames = librosa.onset.onset_detect(
    y=y_low, sr=sr, hop_length=512, delta=0.05, wait=8
)
onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=512)
print(f"Onsets detected (low-band 40-200Hz): {len(onset_times)}")

# ─── Class A: top 10% energy rise ────────────────────────────────────────────
energies = []
for t in onset_times:
    idx  = int(t * sr)
    pre  = y[max(0, idx - int(0.02*sr)):idx]
    post = y[idx:min(len(y), idx + int(0.02*sr))]
    energies.append(post.std() - pre.std())
energies = np.array(energies)

threshold_90 = np.percentile(energies, 90)
class_a_mask = energies >= threshold_90
class_a = onset_times[class_a_mask]
print(f"Class A (top 10% energy): {len(class_a)} onsets")

# Median interval của class A → implied BPM
if len(class_a) > 1:
    intervals = np.diff(class_a)
    median_interval = np.median(intervals)
    implied_bpm = 60.0 / median_interval
    print(f"Class A median interval: {median_interval*1000:.1f}ms → implied {implied_bpm:.1f} BPM")
    
    # Phân tích interval distribution
    print(f"Class A intervals — min:{intervals.min()*1000:.0f}ms  median:{median_interval*1000:.0f}ms  max:{intervals.max()*1000:.0f}ms")
    
    # Xem thử nếu chia đôi / nhân đôi implied BPM có khớp ground truth không
    for factor in [0.25, 0.5, 1.0, 2.0, 4.0]:
        candidate = implied_bpm * factor
        if 60 <= candidate <= 250:
            diff = abs(candidate - GROUND_TRUTH)
            marker = " ← MATCH" if diff < 5 else ""
            print(f"  × {factor:.2f} → {candidate:.1f} BPM  (Δ{diff:.1f} from GT){marker}")

# ─── Grid scan (giống beat_analyzer.py) ──────────────────────────────────────
print(f"\nGrid scan {BPM_MIN}-{BPM_MAX} BPM, step {STEP}...")
THRESHOLD_S = THRESHOLD_MS / 1000.0
results = []

for bpm in np.arange(BPM_MIN, BPM_MAX, STEP):
    beat_len   = 60.0 / bpm
    deviations = np.array([
        min(t % beat_len, beat_len - t % beat_len)
        for t in class_a
    ])
    score = float(np.mean(np.maximum(0, 1 - deviations / THRESHOLD_S)))
    results.append((round(float(bpm), 2), score))

results.sort(key=lambda x: -x[1])
print("\nTop 10 BPM candidates:")
for bpm, score in results[:10]:
    gt_diff = abs(bpm - GROUND_TRUTH)
    marker  = " ★ GROUND TRUTH" if gt_diff < 2 else ""
    print(f"  {bpm:6.1f} BPM  score={score:.4f}  Δ{gt_diff:.1f} from GT{marker}")

# Score của ground truth BPM
gt_key    = min(results, key=lambda x: abs(x[0] - GROUND_TRUTH))
print(f"\nGround truth {GROUND_TRUTH} BPM → closest in scan: {gt_key[0]} BPM, score={gt_key[1]:.4f}")
print(f"Best candidate: {results[0][0]} BPM, score={results[0][1]:.4f}")
print(f"Score ratio (GT / best): {gt_key[1] / results[0][1]:.3f}")

# Anti-harmonic check
best_bpm, best_score = results[0]
results_dict = dict(results)
half = round(best_bpm / 2 / STEP) * STEP
if half >= BPM_MIN:
    half_key   = min(results_dict.keys(), key=lambda k: abs(k - half))
    half_score = results_dict.get(half_key, 0)
    ratio      = half_score / best_score
    print(f"\nAnti-harmonic check: {best_bpm}/2 = {half:.1f} → nearest {half_key} BPM, score={half_score:.4f}, ratio={ratio:.3f}")
    fired = f"YES → output {half_key}" if ratio >= 0.65 else "NO"
    print(f"  Would anti-harmonic fire (ratio≥0.65)? {fired}")

# ─── Plot ─────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 12))
fig.patch.set_facecolor('#0d0d12')
gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

style = dict(facecolor='#0d0d12')
def style_ax(ax):
    ax.set_facecolor('#0d0d12')
    ax.grid(alpha=0.12, color='#555')
    ax.tick_params(colors='#aaa')
    for sp in ax.spines.values():
        sp.set_color('#333')

# Panel 1: Spectrogram full-band với onset markers
ax1 = fig.add_subplot(gs[0, :])
style_ax(ax1)
S_full = np.abs(librosa.stft(y, hop_length=512))
librosa.display.specshow(
    librosa.amplitude_to_db(S_full, ref=np.max),
    sr=sr, hop_length=512, x_axis='time', y_axis='log',
    ax=ax1, cmap='magma'
)
# Vẽ onset markers
for t in onset_times:
    ax1.axvline(x=t, color='#00ff88', linewidth=0.4, alpha=0.5)
for t in class_a:
    ax1.axvline(x=t, color='#ff4444', linewidth=0.8, alpha=0.8)
# Ground truth beat grid
beat_len_gt = 60.0 / GROUND_TRUTH
for beat_t in np.arange(0, duration, beat_len_gt):
    ax1.axvline(x=beat_t, color='#7ecfff', linewidth=0.6, alpha=0.4, linestyle='--')
ax1.set_ylim(20, 8000)
ax1.set_title(f'Spectrogram — green=all onsets, red=Class A, blue=GT beat grid ({GROUND_TRUTH} BPM)',
              color='#ddd', fontsize=10)
ax1.set_ylabel('Hz', color='#aaa')

# Panel 2: Low-band waveform (40-200Hz) với onset markers
ax2 = fig.add_subplot(gs[1, 0])
style_ax(ax2)
t_axis = np.linspace(0, duration, len(y_low))
ax2.plot(t_axis, y_low, color='#888', linewidth=0.3, alpha=0.8)
for t in onset_times:
    ax2.axvline(x=t, color='#00ff88', linewidth=0.5, alpha=0.6)
for t in class_a:
    ax2.axvline(x=t, color='#ff4444', linewidth=1.0, alpha=0.9)
ax2.set_title('Low-band waveform (40-200Hz)', color='#ddd', fontsize=10)
ax2.set_xlabel('Time (s)', color='#aaa')
ax2.set_ylabel('Amplitude', color='#aaa')
ax2.set_xlim(0, min(30, duration))  # zoom 30s đầu

# Panel 3: Energy distribution của onsets
ax3 = fig.add_subplot(gs[1, 1])
style_ax(ax3)
ax3.hist(energies, bins=40, color='#7ecfff', alpha=0.7, edgecolor='#555')
ax3.axvline(x=threshold_90, color='#ff4444', linewidth=1.5, linestyle='--',
            label=f'90th pct (Class A threshold)')
ax3.set_title('Onset energy distribution', color='#ddd', fontsize=10)
ax3.set_xlabel('Energy rise (post.std - pre.std)', color='#aaa')
ax3.set_ylabel('Count', color='#aaa')
ax3.legend(fontsize=8, facecolor='#1a1a22', labelcolor='#aaa')

# Panel 4: Grid scan score curve
ax4 = fig.add_subplot(gs[2, :])
style_ax(ax4)
bpms  = [r[0] for r in sorted(results, key=lambda x: x[0])]
scores= [r[1] for r in sorted(results, key=lambda x: x[0])]
ax4.plot(bpms, scores, color='white', linewidth=1.2)
ax4.axvline(x=GROUND_TRUTH, color='#00ff88', linewidth=2, linestyle='--',
            label=f'Ground truth {GROUND_TRUTH} BPM')
ax4.axvline(x=results[0][0], color='#ff4444', linewidth=2, linestyle='--',
            label=f'Best candidate {results[0][0]} BPM (score={results[0][1]:.3f})')
# Đánh dấu các harmonic của ground truth
for mult in [0.5, 1.0, 1.5, 2.0, 3.0, 4.0]:
    h = GROUND_TRUTH * mult
    if BPM_MIN <= h <= BPM_MAX:
        ax4.axvline(x=h, color='#7ecfff', linewidth=0.8, linestyle=':', alpha=0.6)
        ax4.text(h, max(scores)*0.95, f'×{mult}', color='#7ecfff', fontsize=7,
                 ha='center', va='top')
ax4.set_title(f'Grid scan score — blue dotted = harmonics of {GROUND_TRUTH} BPM',
              color='#ddd', fontsize=10)
ax4.set_xlabel('BPM', color='#aaa')
ax4.set_ylabel('Score', color='#aaa')
ax4.legend(fontsize=9, facecolor='#1a1a22', labelcolor='#aaa')
ax4.set_xlim(BPM_MIN, BPM_MAX)

# Save
import os
base = os.path.splitext(os.path.basename(path))[0]
out  = f"diagnose_bpm_{base}.png"
plt.suptitle(f'BPM Diagnostic — {path}', color='#eee', fontsize=13, y=1.01)
plt.savefig(out, dpi=120, facecolor='#0d0d12', bbox_inches='tight')
print(f"\nSaved: {out}")
