"""
beat_analyzer.py
================
Module phân tích nhịp nhạc — detect BPM và offset.

Dùng độc lập:
    python beat_analyzer.py Highscore.mp3

Gọi từ code khác:
    from beat_analyzer import analyze
    result = analyze("Highscore.mp3")
    print(result)  # {"bpm": 110.0, "offset_ms": 380.0, ...}
"""

import numpy as np
import librosa
from scipy.ndimage import uniform_filter1d


# ─────────────────────────────────────────────────────────────────────────────
# BPM DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def detect_bpm(y, sr, bpm_min=80, bpm_max=200, step=0.5):
    """
    Grid scan: thử từng BPM candidate, chấm điểm theo
    số onset rơi gần lưới nhất.
    Trả về float BPM.
    """
    # Low-band filter 40-200Hz để chỉ giữ kick/bass
    D     = librosa.stft(y, hop_length=512)
    freqs = librosa.fft_frequencies(sr=sr)
    mask  = (freqs >= 40) & (freqs <= 200)
    D_low = D.copy()
    D_low[~mask, :] = 0
    y_low = librosa.istft(D_low, hop_length=512)

    onset_frames = librosa.onset.onset_detect(
        y=y_low, sr=sr, hop_length=512, delta=0.05, wait=8
    )
    onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=512)

    if len(onset_times) < 4:
        # Fallback: librosa default
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        return float(tempo)

    # Phân loại Class A: top 10% energy rise
    energies = []
    for t in onset_times:
        idx  = int(t * sr)
        pre  = y[max(0, idx - int(0.02*sr)):idx]
        post = y[idx:min(len(y), idx + int(0.02*sr))]
        energies.append(post.std() - pre.std())
    energies = np.array(energies)
    class_a  = onset_times[energies >= np.percentile(energies, 90)]

    if len(class_a) < 4:
        class_a = onset_times  # fallback dùng tất cả

    # Grid scan với weighted score
    best_bpm, best_score = 80.0, -1.0
    THRESHOLD = 0.05  # 50ms
    results = []

    for bpm in np.arange(bpm_min, bpm_max, step):
        beat_len   = 60.0 / bpm
        deviations = np.array([
            min(t % beat_len, beat_len - t % beat_len)
            for t in class_a
        ])
        # Weighted: gần lưới hơn thì điểm cao hơn
        score = float(np.mean(np.maximum(0, 1 - deviations / THRESHOLD)))
        results.append((round(float(bpm), 2), score))
        if score > best_score:
            best_score = score
            best_bpm   = bpm

    # Anti-harmonic: nếu BPM/2 có score >= 65% thì chọn BPM/2
    # Tránh detect nhầm 220 thay vì 110
    results_dict = {r[0]: r[1] for r in results}
    half = round(best_bpm / 2 / step) * step
    if half >= bpm_min:
        half_key = min(results_dict.keys(), key=lambda k: abs(k - half))
        if results_dict.get(half_key, 0) >= best_score * 0.65:
            best_bpm = half_key

    return round(float(best_bpm), 2)


# ─────────────────────────────────────────────────────────────────────────────
# OFFSET DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def detect_offset(y, sr, bpm, scan_min_ms=None, scan_max_ms=None):
    """
    Stack template + sharpness curve + sustain threshold.
    Trả về float offset_ms.
    """
    beat_len = 60.0 / bpm
    beat_ms  = beat_len * 1000

    # Vùng quét: mặc định quét 1 chu kỳ beat
    if scan_min_ms is None:
        scan_min_ms = 0
    if scan_max_ms is None:
        scan_max_ms = int(beat_ms)

    PRE_MS, POST_MS = 100, 200
    seg_smp = int((PRE_MS + POST_MS) / 1000 * sr)

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

    # Bước 1ms trong vùng quét
    offsets     = np.arange(scan_min_ms, scan_max_ms, 1)
    sharpnesses = []
    for o in offsets:
        t = stack_template(o)
        sharpnesses.append(sharpness(t) if t is not None else 0.0)
    sharpnesses = np.array(sharpnesses)

    # Smooth + đạo hàm
    smoothed  = uniform_filter1d(sharpnesses, size=3)
    d_sharp   = np.diff(smoothed)
    peak_val  = d_sharp.max()
    threshold = 0.08 * peak_val

    # Dò xuôi từ trái — điểm đầu tiên vượt threshold giữ vững 3ms
    onset_idx = None
    for i in range(len(d_sharp) - 3):
        if (d_sharp[i]   > threshold and
            d_sharp[i+1] > threshold and
            d_sharp[i+2] > threshold):
            onset_idx = i
            break

    if onset_idx is None:
        # Fallback: argmax sharpness
        onset_idx = int(np.argmax(sharpnesses))

    return float(offsets[onset_idx])


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def analyze(path, bpm_hint=None):
    """
    Phân tích file nhạc, trả về dict:
    {
        "file"      : str,
        "duration_s": float,
        "bpm"       : float,
        "offset_ms" : float,
    }

    bpm_hint: nếu đã biết BPM trước thì truyền vào để bỏ qua bước detect BPM.
    """
    print(f"[analyzer] Loading {path} ...")
    y, sr = librosa.load(path, mono=True)
    duration = len(y) / sr

    print("[analyzer] Detecting BPM ...")
    bpm = bpm_hint if bpm_hint is not None else detect_bpm(y, sr)
    print(f"[analyzer] BPM = {bpm}")

    beat_ms = 60_000.0 / bpm
    print(f"[analyzer] Detecting offset (scanning 0 → {beat_ms:.0f} ms) ...")
    offset_ms = detect_offset(y, sr, bpm,
                              scan_min_ms=0,
                              scan_max_ms=int(beat_ms))
    print(f"[analyzer] Offset = {offset_ms} ms")

    return {
        "file"      : path,
        "duration_s": round(duration, 2),
        "bpm"       : bpm,
        "offset_ms" : offset_ms,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys, json

    if len(sys.argv) < 2:
        print("Usage: python beat_analyzer.py <file.mp3> [bpm]")
        sys.exit(1)

    path     = sys.argv[1]
    bpm_hint = float(sys.argv[2]) if len(sys.argv) > 2 else None
    result   = analyze(path, bpm_hint=bpm_hint)

    print("\n" + "="*40)
    print(json.dumps(result, indent=2, ensure_ascii=False))
