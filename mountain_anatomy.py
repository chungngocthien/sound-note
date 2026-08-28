"""
mountain_anatomy.py
===================
Per-candidate anatomy table cho top.mp3.

Với mỗi candidate offset (334, 454, 467, 482ms), log:
  peak_ms               : offset candidate
  height                : sharpness(smoothed) tại offset đó
  area                  : tích phân smoothed curve ±50ms quanh peak
  template_peak_time    : thời điểm max amplitude trong raw template (ms từ beat)
  template_peak_amp     : giá trị max amplitude đó
  env_peak_time         : thời điểm max envelope trong |template| (ms từ beat)
  env_peak_amp          : giá trị max envelope đó
  energy_before_0ms     : RMS của template tại [-100ms, 0ms]
  energy_after_0ms      : RMS của template tại [0ms, +30ms]
  e_ratio               : energy_after / energy_before (contrast)
  cumul_slope           : slope của cumulative energy trong [0ms, +30ms]
                          (Gemini metric: đột biến sau beat → cao)
  centroid_frequency    : spectral centroid của right window [0, +30ms]
  spectral_rolloff      : 85% rolloff frequency của right window
  spectral_flux         : mean absolute diff của spectrum giữa left và right window

Chạy:
    python mountain_anatomy.py top.mp3 122.998
"""

import sys
import numpy as np
import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
PRE_MS       = 100
POST_MS      = 200
SMOOTH_SIZE  = 5
CANDIDATES   = [334, 454, 467, 482]   # ms — từ Step 1 của family vote
GT_OFFSET    = 465                     # ms

# ─────────────────────────────────────────────────────────────────────────────
# BUILD TEMPLATE
# ─────────────────────────────────────────────────────────────────────────────
def get_template(y, sr, bpm, offset_ms):
    beat_len = 60.0 / bpm
    seg_smp  = int((PRE_MS + POST_MS) / 1000 * sr)
    off_s    = offset_ms / 1000.0
    segs = []
    for t in np.arange(off_s, len(y)/sr - beat_len, beat_len):
        s = int((t - PRE_MS / 1000) * sr)
        e = s + seg_smp
        if s >= 0 and e <= len(y):
            chunk = y[s:e]
            if np.abs(chunk).max() > 0.01:
                segs.append(chunk)
    return np.mean(segs, axis=0) if segs else None, len(segs)


def sharpness_val(tmpl, sr):
    beat_idx = int(PRE_MS / (PRE_MS + POST_MS) * len(tmpl))
    win      = int(0.02 * sr)   # 20ms
    left     = tmpl[beat_idx - win : beat_idx]
    right    = tmpl[beat_idx      : beat_idx + win]
    return float(np.var(right) / (np.var(left) + 1e-10))


# ─────────────────────────────────────────────────────────────────────────────
# SHARPNESS CURVE (để lấy height và area)
# ─────────────────────────────────────────────────────────────────────────────
def build_sharpness_curve(y, sr, bpm):
    beat_len = 60.0 / bpm
    beat_ms  = beat_len * 1000
    seg_smp  = int((PRE_MS + POST_MS) / 1000 * sr)

    def stack(offset_ms):
        off_s = offset_ms / 1000.0
        segs  = []
        for t in np.arange(off_s, len(y)/sr - beat_len, beat_len):
            s = int((t - PRE_MS/1000) * sr)
            e = s + seg_smp
            if s >= 0 and e <= len(y):
                chunk = y[s:e]
                if np.abs(chunk).max() > 0.01:
                    segs.append(chunk)
        return np.mean(segs, axis=0) if segs else None

    offsets = np.arange(0, int(beat_ms), 1)
    sharp   = np.array([sharpness_val(stack(o), sr) if stack(o) is not None else 0.0
                        for o in offsets])
    smoothed = uniform_filter1d(sharp, size=SMOOTH_SIZE)
    return offsets, smoothed


# ─────────────────────────────────────────────────────────────────────────────
# ANATOMY của một candidate
# ─────────────────────────────────────────────────────────────────────────────
def anatomy(y, sr, bpm, offset_ms, offsets, smoothed):
    tmpl, n_beats = get_template(y, sr, bpm, offset_ms)
    if tmpl is None:
        return None

    beat_idx  = int(PRE_MS / (PRE_MS + POST_MS) * len(tmpl))
    time_axis = np.linspace(-PRE_MS, POST_MS, len(tmpl))   # ms
    ms_per_sample = (PRE_MS + POST_MS) / len(tmpl)

    # ── Height và Area từ smoothed curve ──────────────────────────────────
    idx = int(np.argmin(np.abs(offsets - offset_ms)))
    height = float(smoothed[idx])

    # area = tích phân smoothed ±50ms quanh candidate
    lo = max(0, idx - 50)
    hi = min(len(smoothed) - 1, idx + 50)
    area = float(np.trapz(smoothed[lo:hi+1]))

    # ── Template peak ──────────────────────────────────────────────────────
    tmpl_peak_idx  = int(np.argmax(np.abs(tmpl)))
    tmpl_peak_time = float(time_axis[tmpl_peak_idx])
    tmpl_peak_amp  = float(np.abs(tmpl[tmpl_peak_idx]))

    # ── Envelope peak ──────────────────────────────────────────────────────
    env = np.abs(tmpl)
    env_peak_idx  = int(np.argmax(env))
    env_peak_time = float(time_axis[env_peak_idx])
    env_peak_amp  = float(env[env_peak_idx])

    # ── Energy windows ────────────────────────────────────────────────────
    # before: [-100ms, 0ms] = [:beat_idx]
    # after:  [0ms, +30ms]
    after_30ms = beat_idx + int(30 / ms_per_sample)
    before_win = tmpl[:beat_idx]
    after_win  = tmpl[beat_idx : after_30ms]

    energy_before = float(np.sqrt(np.mean(before_win**2)))
    energy_after  = float(np.sqrt(np.mean(after_win**2)))
    e_ratio       = energy_after / (energy_before + 1e-10)

    # ── Cumulative energy slope (Gemini metric) ────────────────────────────
    # Tích lũy energy**2 từ beat_idx đến after_30ms, lấy slope tuyến tính
    after_energy = after_win**2
    cumul         = np.cumsum(after_energy)
    if len(cumul) > 1:
        x_norm    = np.arange(len(cumul)) / len(cumul)
        cumul_slope = float(np.polyfit(x_norm, cumul, 1)[0])
    else:
        cumul_slope = 0.0

    # ── Spectral features của right window [0, +30ms] ─────────────────────
    win_samples = max(len(after_win), 64)
    after_padded = np.pad(after_win, (0, max(0, win_samples - len(after_win))))

    # Left window tương đương để tính flux
    left_30ms = beat_idx - int(30 / ms_per_sample)
    left_win  = tmpl[max(0, left_30ms) : beat_idx]
    left_padded = np.pad(left_win, (0, max(0, win_samples - len(left_win))))

    n_fft = min(512, win_samples)
    freqs = np.fft.rfftfreq(n_fft, d=1/sr)

    spec_right = np.abs(np.fft.rfft(after_padded[:n_fft]))
    spec_left  = np.abs(np.fft.rfft(left_padded[:n_fft]))

    # Centroid
    spec_sum = spec_right.sum() + 1e-10
    centroid  = float(np.sum(freqs * spec_right) / spec_sum)

    # Rolloff 85%
    cumul_spec = np.cumsum(spec_right)
    rolloff_idx = np.searchsorted(cumul_spec, 0.85 * cumul_spec[-1])
    rolloff = float(freqs[min(rolloff_idx, len(freqs)-1)])

    # Spectral flux
    flux = float(np.mean(np.abs(spec_right - spec_left)))

    return {
        "peak_ms"           : offset_ms,
        "n_beats"           : n_beats,
        "height"            : round(height, 4),
        "area"              : round(area, 2),
        "template_peak_time": round(tmpl_peak_time, 1),
        "template_peak_amp" : round(tmpl_peak_amp, 4),
        "env_peak_time"     : round(env_peak_time, 1),
        "env_peak_amp"      : round(env_peak_amp, 4),
        "energy_before_0ms" : round(energy_before, 5),
        "energy_after_0ms"  : round(energy_after, 5),
        "e_ratio"           : round(e_ratio, 3),
        "cumul_slope"       : round(cumul_slope, 6),
        "centroid_hz"       : round(centroid, 1),
        "spectral_rolloff"  : round(rolloff, 1),
        "spectral_flux"     : round(flux, 5),
        "tmpl"              : tmpl,   # giữ lại để plot
    }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def run(path, bpm, gt_offset=GT_OFFSET):
    print(f"\n{'='*70}")
    print(f"MOUNTAIN ANATOMY: {path}  BPM={bpm}  GT={gt_offset}ms")
    print(f"{'='*70}\n")

    y, sr = librosa.load(path, mono=True)

    print("Building sharpness curve (full song)...")
    offsets, smoothed = build_sharpness_curve(y, sr, bpm)
    print("Done.\n")

    results = []
    for c in CANDIDATES:
        print(f"  Extracting anatomy @ {c}ms...")
        r = anatomy(y, sr, bpm, c, offsets, smoothed)
        if r:
            results.append(r)

    # ── PRINT TABLE ───────────────────────────────────────────────────────
    metrics = [
        ("peak_ms",            "peak_ms",           "ms"),
        ("height",             "height",             ""),
        ("area",               "area",               ""),
        ("template_peak_time", "tmpl_peak_t",        "ms"),
        ("template_peak_amp",  "tmpl_peak_amp",      ""),
        ("env_peak_time",      "env_peak_t",         "ms"),
        ("env_peak_amp",       "env_peak_amp",       ""),
        ("energy_before_0ms",  "E_before",           "RMS"),
        ("energy_after_0ms",   "E_after[0-30ms]",   "RMS"),
        ("e_ratio",            "E_ratio(after/bef)", ""),
        ("cumul_slope",        "cumul_slope[Gemini]",""),
        ("centroid_hz",        "centroid_hz",        "Hz"),
        ("spectral_rolloff",   "rolloff_85%",        "Hz"),
        ("spectral_flux",      "spectral_flux",      ""),
    ]

    header = f"\n{'Metric':<26}" + "".join(f"{r['peak_ms']:>12}ms" for r in results)
    print(header)
    print("-" * (26 + 14 * len(results)))
    for key, label, unit in metrics:
        row = f"{label:<26}"
        vals = [r[key] for r in results]
        # highlight max với dấu *
        try:
            max_val = max(vals)
            for v in vals:
                marker = "*" if v == max_val else " "
                row += f"{str(v)+unit:>13}{marker}"
        except Exception:
            for v in vals:
                row += f"{str(v)+unit:>14}"
        print(row)

    # Tóm tắt GT-relevant metrics
    print(f"\n{'='*50}")
    print("SUMMARY — Which candidate looks most like a downbeat?")
    print(f"{'='*50}")
    for r in results:
        flag = " ← closest to GT" if abs(r["peak_ms"] - gt_offset) < 15 else ""
        print(f"  {r['peak_ms']}ms{flag}")
        print(f"    E_ratio (after/before) : {r['e_ratio']} {'← high contrast' if r['e_ratio'] == max(x['e_ratio'] for x in results) else ''}")
        print(f"    cumul_slope [Gemini]   : {r['cumul_slope']} {'← steepest onset' if r['cumul_slope'] == max(x['cumul_slope'] for x in results) else ''}")
        print(f"    centroid_hz            : {r['centroid_hz']} Hz {'← lowest freq' if r['centroid_hz'] == min(x['centroid_hz'] for x in results) else ''}")
        print(f"    spectral_flux          : {r['spectral_flux']}")

    # ── PLOT: 4 templates + anatomy markers ───────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(16, 8))
    fig.patch.set_facecolor('#0d0d12')
    fig.suptitle(f"Mountain Anatomy — {path}", color='#ddd', fontsize=13)

    colors = ['#ff8888', '#00ff88', '#7ecfff', '#f0c040']
    time_axis_ref = np.linspace(-PRE_MS, POST_MS,
                                 int((PRE_MS + POST_MS) / 1000 * sr))

    for ax, r, col in zip(axes.flat, results, colors):
        tmpl = r["tmpl"]
        t    = np.linspace(-PRE_MS, POST_MS, len(tmpl))
        env  = np.abs(tmpl)

        ax.set_facecolor('#0d0d12')
        ax.grid(alpha=0.12)
        ax.tick_params(colors='#aaa')
        for sp in ax.spines.values(): sp.set_color('#333')

        ax.plot(t, tmpl, color=col, lw=1.0, alpha=0.7, label='waveform')
        ax.plot(t, env,  color='white', lw=1.2, alpha=0.6, label='envelope')
        ax.axvline(x=0,                    color='white',   lw=1.5, ls='--', alpha=0.5, label='beat marker')
        ax.axvline(x=r["env_peak_time"],   color='#ffaa00', lw=1.2, ls=':',  label=f"env_peak {r['env_peak_time']}ms")
        ax.axvline(x=30,                   color='#888',    lw=0.8, ls='-',  alpha=0.4, label='+30ms window')

        flag = "← GT" if abs(r["peak_ms"] - gt_offset) < 15 else ""
        ax.set_title(
            f"{r['peak_ms']}ms {flag} | E_ratio={r['e_ratio']} | slope={r['cumul_slope']:.4f} | cent={r['centroid_hz']}Hz",
            color='#ddd', fontsize=10
        )
        ax.set_xlabel("Time relative to beat (ms)", color='#aaa')
        ax.set_ylabel("Amplitude", color='#aaa')
        ax.legend(fontsize=8, facecolor='#1a1a22', labelcolor='#aaa')

    plt.tight_layout()
    out = "mountain_anatomy.png"
    plt.savefig(out, dpi=120, facecolor='#0d0d12')
    plt.close("all")
    print(f"\nPlot saved → {out}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python mountain_anatomy.py <file.mp3> <bpm> [gt_offset_ms]")
        sys.exit(1)
    path      = sys.argv[1]
    bpm       = float(sys.argv[2])
    gt_offset = float(sys.argv[3]) if len(sys.argv) > 3 else GT_OFFSET
    run(path, bpm, gt_offset)