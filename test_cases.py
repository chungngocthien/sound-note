import sys
import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d

# ─────────────────────────────────────────────────────────────────────────────
# DANH SÁCH TEST CASES
# ─────────────────────────────────────────────────────────────────────────────
SONGS = [
    {"path": "withyou.mp3", "gt_bpm": 102.001, "gt_offset": 178.0},
    {"path": "top.mp3",  "gt_bpm": 122.998, "gt_offset": 465.0},
    {"path": "mimimi.mp3",  "gt_bpm": 126.000, "gt_offset": 333},
    {"path": "lovedive.mp3",      "gt_bpm": 118.002, "gt_offset": 313},
    {"path": "boomboom.mp3",   "gt_bpm": 126.000, "gt_offset": 151},
    {"path": "1-800.mp3",       "gt_bpm": 132.0,   "gt_offset": 309},
]

# ─────────────────────────────────────────────────────────────────────────────
# BPM DETECTION — phase-free scan
# ─────────────────────────────────────────────────────────────────────────────
def detect_bpm_phase_free(y, sr, bpm_min=80, bpm_max=200, bpm_step=0.5,
                           threshold_ms=50, offset_step_ms=4):
    THRESHOLD_S = threshold_ms / 1000.0
    D = librosa.stft(y, hop_length=512)
    freqs = librosa.fft_frequencies(sr=sr)
    mask = (freqs >= 40) & (freqs <= 200)
    D_low = D.copy(); D_low[~mask, :] = 0
    y_low = librosa.istft(D_low, hop_length=512)

    onset_frames = librosa.onset.onset_detect(y=y_low, sr=sr, hop_length=512, delta=0.05, wait=8)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=512)

    if len(onset_times) < 4:
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        return float(tempo), 0.0, {}

    def phase_score(bpm, offset_ms):
        beat_len = 60.0 / bpm
        off_s = offset_ms / 1000.0
        shifted = (onset_times - off_s) % beat_len
        dev = np.minimum(shifted, beat_len - shifted)
        return float(np.mean(np.maximum(0, 1 - dev / THRESHOLD_S)))

    def best_phase(bpm):
        beat_ms = 60_000.0 / bpm
        offsets = np.arange(0, beat_ms, offset_step_ms)
        scores = [phase_score(bpm, o) for o in offsets]
        return float(scores[int(np.argmax(scores))]), float(offsets[int(np.argmax(scores))])

    results = {}
    for bpm in np.arange(bpm_min, bpm_max, bpm_step):
        s, o = best_phase(bpm)
        results[round(float(bpm), 2)] = {"score": s, "best_offset": o}

    ranked = sorted(results.items(), key=lambda x: x[1]["score"], reverse=True)
    best_bpm = ranked[0][0]

    half = round(best_bpm / 2 / bpm_step) * bpm_step
    if bpm_min <= half <= bpm_max:
        half_key = min(results.keys(), key=lambda k: abs(k - half))
        if results[half_key]["score"] >= results[best_bpm]["score"] * 0.65:
            best_bpm = half_key

    return best_bpm, results[best_bpm]["best_offset"], results

# ─────────────────────────────────────────────────────────────────────────────
# CORE SCAFFOLDING: Trích xuất Lưới Năng Lượng Phách (Dùng Chung)
# ─────────────────────────────────────────────────────────────────────────────
def _build_sharpness_curve(y, sr, bpm):
    beat_len = 60.0 / bpm
    beat_ms = beat_len * 1000
    PRE_MS, POST_MS = 100, 200
    seg_smp = int((PRE_MS + POST_MS) / 1000 * sr)

    def stack_template(offset_ms):
        off = offset_ms / 1000
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
        win = int(0.02 * sr)
        left = tmpl[beat_idx - win : beat_idx]
        right = tmpl[beat_idx : beat_idx + win]
        return float(np.var(right) / (np.var(left) + 1e-10))

    offsets = np.arange(0, int(beat_ms), 1)
    sharpnesses = np.array([sharpness(stack_template(o)) or 0.0 for o in offsets])
    smoothed = uniform_filter1d(sharpnesses, size=5)
    d_sharp = np.diff(smoothed)
    peak_idx = int(np.argmax(smoothed))
    
    return offsets, sharpnesses, smoothed, d_sharp, peak_idx

# ─────────────────────────────────────────────────────────────────────────────
# TWO DETECT OFFSET VARIATIONS
# ─────────────────────────────────────────────────────────────────────────────
def detect_offset_claude(smoothed, d_sharp, peak_idx):
    """Tolerance-based: dừng quét ngược khi chạm biên độ lồi 1% đỉnh"""
    mountain_threshold = 0.05 * smoothed[peak_idx]
    tolerance = 0.01 * smoothed[peak_idx]

    left = peak_idx
    while left > 0:
        if smoothed[left - 1] > smoothed[left] + tolerance:
            break
        if smoothed[left] < mountain_threshold:
            break
        left -= 1
    return left

def detect_offset_gemini(smoothed, d_sharp, peak_idx):
    """Lookahead-based: bộ đếm đà lùi 6ms để kiểm tra thung lũng thực"""
    mountain_threshold = 0.05 * smoothed[peak_idx]

    left = peak_idx
    best_valley_idx = peak_idx
    lowest_sharp_val = smoothed[peak_idx]
    LOOKAHEAD_MS = 6
    lookahead_counter = 0

    while left > 0:
        current_sharp = smoothed[left]
        if current_sharp < lowest_sharp_val:
            lowest_sharp_val = current_sharp
            best_valley_idx = left
            lookahead_counter = 0
        else:
            lookahead_counter += 1

        if current_sharp < mountain_threshold:
            break
        if lookahead_counter >= LOOKAHEAD_MS:
            left = best_valley_idx
            break
        left -= 1
    return left

# ─────────────────────────────────────────────────────────────────────────────
# RUN ONE CASE
# ─────────────────────────────────────────────────────────────────────────────
def run_one(song):
    path, gt_bpm, gt_offset = song["path"], song["gt_bpm"], song["gt_offset"]
    print(f"\n{'='*60}\nFile: {path} | GT: {gt_bpm} BPM | {gt_offset} ms\n{'='*60}")

    try:
        y, sr = librosa.load(path, mono=True)
    except Exception as e:
        print(f"  [ERROR] Không load được file: {e}"); return None

    print("  Detecting BPM (phase-free)...")
    pred_bpm, _, _ = detect_bpm_phase_free(y, sr)
    bpm_ok = abs(pred_bpm - gt_bpm) < 3.0
    print(f"  BPM predicted : {pred_bpm:.2f}  (error {pred_bpm - gt_bpm:+.2f})  {'✓' if bpm_ok else '✗ WRONG'}")

    bpm_for_offset = pred_bpm if bpm_ok else gt_bpm
    if not bpm_ok: print(f"  [fallback] Dùng GT BPM {gt_bpm} để quét offset")

    # Dựng đường cong đặc trưng (Chỉ chạy một lần duy nhất)
    offsets, sharpnesses, smoothed, d_sharp, peak_idx = _build_sharpness_curve(y, sr, bpm_for_offset)

    # Chạy song song 2 thuật toán
    zc_claude = detect_offset_claude(smoothed, d_sharp, peak_idx)
    zc_gemini = detect_offset_gemini(smoothed, d_sharp, peak_idx)

    offset_claude, offset_gemini = float(offsets[zc_claude]), float(offsets[zc_gemini])
    err_claude, err_gemini = offset_claude - gt_offset, offset_gemini - gt_offset

    print(f"  Offset Claude : {offset_claude:.1f} ms  (error {err_claude:+.1f} ms)  {'✓' if abs(err_claude)<10 else '✗'}")
    print(f"  Offset Gemini : {offset_gemini:.1f} ms  (error {err_gemini:+.1f} ms)  {'✓' if abs(err_gemini)<10 else '✗'}")

    # Vẽ đồ thị so sánh kép
    base = path.rsplit(".", 1)[0].replace(" ", "_")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    fig.patch.set_facecolor('#0d0d12')

    for ax in (ax1, ax2):
        ax.set_facecolor('#0d0d12')
        ax.grid(alpha=0.12); ax.tick_params(colors='#aaa')
        for sp in ax.spines.values(): sp.set_color('#333')
        ax.axvline(x=gt_offset, color='#00ff88', lw=1.5, ls='--', label=f'GT {gt_offset:.0f}ms')
        ax.axvline(x=offset_claude, color='#7ecfff', lw=1.5, ls=':', label=f'Claude {offset_claude:.0f}ms ({err_claude:+.1f}ms)')
        ax.axvline(x=offset_gemini, color='#ff8888', lw=1.5, ls='-.', label=f'Gemini {offset_gemini:.0f}ms ({err_gemini:+.1f}ms)')

    ax1.plot(offsets, smoothed, color='white', lw=1.5, label='sharpness (smoothed)')
    ax1.set_ylabel('Sharpness', color='#aaa')
    ax1.set_title(f'{path} — Claude: {offset_claude:.0f}ms | Gemini: {offset_gemini:.0f}ms', color='#ddd')
    ax1.legend(fontsize=9, facecolor='#1a1a22', labelcolor='#aaa')

    ax2.plot(offsets[1:], d_sharp, color='#f0c040', lw=1.5, label='d(sharpness)/d(offset)')
    ax2.axhline(y=0, color='#555', lw=1)
    ax2.set_xlabel('Offset (ms)'); ax2.set_ylabel('Slope', color='#aaa')
    ax2.legend(fontsize=9, facecolor='#1a1a22', labelcolor='#aaa')

    plt.tight_layout()
    plt.savefig(f"result_{base}.png", dpi=120, facecolor='#0d0d12')
    plt.clf(); plt.close('all')  # Giải phóng RAM triệt để chống leak khi stress-test
    return {
        "file": path, "gt_offset": gt_offset,
        "off_claude": offset_claude, "err_claude": err_claude,
        "off_gemini": offset_gemini, "err_gemini": err_gemini
    }

# ─────────────────────────────────────────────────────────────────────────────
# MAIN SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
def main():
    results = [run_one(s) for s in SONGS]
    results = [r for r in results if r is not None]

    print(f"\n{'='*85}\nSUMMARY COMPETING TABLE\n{'='*85}")
    print(f"{'File':<18} {'GT Off':>8} | {'Claude':>10} {'Err C':>9} | {'Gemini':>10} {'Err G':>9}")
    print('-'*85)
    for r in results:
        c_flag = '✓' if abs(r['err_claude']) < 10 else '✗'
        g_flag = '✓' if abs(r['err_gemini']) < 10 else '✗'
        print(f"{r['file']:<18} {r['gt_offset']:>8.1f} | {r['off_claude']:>10.1f} {r['err_claude']:>8.1f}ms {c_flag} | {r['off_gemini']:>10.1f} {r['err_gemini']:>8.1f}ms {g_flag}")
    print('='*85)

if __name__ == "__main__":
    main()