import numpy as np
import librosa
from scipy.ndimage import uniform_filter1d
import matplotlib.pyplot as plt
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "maps.mp3"
BPM  = float(sys.argv[2]) if len(sys.argv) > 2 else 120.0
GROUND_TRUTH = float(sys.argv[3]) if len(sys.argv) > 3 else 310.0

print(f"File: {path} | BPM: {BPM} | Ground truth: {GROUND_TRUTH}ms")

y, sr    = librosa.load(path, mono=True)
beat_len = 60.0 / BPM
PRE_MS, POST_MS = 100, 200
seg_smp  = int((PRE_MS + POST_MS) / 1000 * sr)

def stack_template(offset_ms):
    off  = offset_ms / 1000
    segs = []
    for t in np.arange(off, len(y)/sr - beat_len, beat_len):
        s = int((t - PRE_MS/1000) * sr)
        e = s + seg_smp
        if s >= 0 and e <= len(y):
            chunk = y[s:e]
            if np.abs(chunk).max() > 0.01:
                segs.append(chunk)
    return np.mean(segs, axis=0) if segs else None

def sharpness(tmpl):
    beat_idx = int(PRE_MS / (PRE_MS + POST_MS) * len(tmpl))
    win      = int(0.02 * sr)
    left     = tmpl[beat_idx - win : beat_idx]
    right    = tmpl[beat_idx : beat_idx + win]
    return float(np.var(right) / (np.var(left) + 1e-10))

offsets     = np.arange(0, int(beat_len * 1000), 1)
sharpnesses = np.array([sharpness(stack_template(o)) or 0 for o in offsets])
smoothed    = uniform_filter1d(sharpnesses, size=5)
d_sharp     = np.diff(smoothed)

peak_idx = int(np.argmax(smoothed))

# ── Method 1: Zero-crossing trước peak ──
zero_cross = None
for i in range(peak_idx, 0, -1):
    if d_sharp[i] >= 0 and d_sharp[i-1] < 0:
        zero_cross = i
        break

# Fallback sustain threshold
if zero_cross is None:
    threshold = 0.08 * d_sharp.max()
    for i in range(len(d_sharp) - 3):
        if d_sharp[i] > threshold and d_sharp[i+1] > threshold and d_sharp[i+2] > threshold:
            zero_cross = i
            break

ms_zerocross = offsets[zero_cross] if zero_cross is not None else 0

# ── Method 2: Valley (min slope) trước peak ──
search_start = max(0, peak_idx - 80)
region       = d_sharp[search_start:peak_idx]
valley_idx   = search_start + int(np.argmin(region)) if len(region) > 0 else 0
ms_valley    = offsets[valley_idx]

print(f"\nPeak sharpness at : {offsets[peak_idx]} ms  (score {smoothed[peak_idx]:.2f})")
print(f"Method 1 zero-cross : {ms_zerocross} ms  (lệch {ms_zerocross - GROUND_TRUTH:+.0f}ms)")
print(f"Method 2 valley     : {ms_valley} ms  (lệch {ms_valley - GROUND_TRUTH:+.0f}ms)")
print(f"Ground truth        : {GROUND_TRUTH} ms")

# ── Plot ──
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
fig.patch.set_facecolor('#0d0d12')

for ax in (ax1, ax2):
    ax.set_facecolor('#0d0d12'); ax.grid(alpha=0.12)
    ax.tick_params(colors='#aaa'); ax.spines[:].set_color('#333')
    ax.axvline(x=GROUND_TRUTH,  color='#00ff88', linestyle='--', linewidth=1.5,
               label=f'ground truth {GROUND_TRUTH:.0f}ms')
    ax.axvline(x=ms_zerocross,  color='#7ecfff', linestyle='--', linewidth=1.5,
               label=f'zero-cross {ms_zerocross}ms ({ms_zerocross-GROUND_TRUTH:+.0f}ms)')
    ax.axvline(x=ms_valley,     color='#f0c040', linestyle='--', linewidth=1.5,
               label=f'valley {ms_valley}ms ({ms_valley-GROUND_TRUTH:+.0f}ms)')

ax1.plot(offsets, smoothed, color='white', linewidth=1.5, label='sharpness (smoothed)')
ax1.set_ylabel('Sharpness', color='#aaa')
ax1.set_title(f'Offset detection — {path}', color='#ddd')
ax1.legend(fontsize=9)

ax2.plot(offsets[1:], d_sharp, color='#f0c040', linewidth=1.5, label='d(sharpness)/d(offset)')
ax2.axhline(y=0, color='#555', linewidth=1)
ax2.set_xlabel('Offset (ms)'); ax2.set_ylabel('Slope', color='#aaa')
ax2.legend(fontsize=9)

plt.tight_layout()
out = "diagnose_offset.png"
plt.savefig(out, dpi=130, facecolor='#0d0d12')
print(f"Saved {out}")
