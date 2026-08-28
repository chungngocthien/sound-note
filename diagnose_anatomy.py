"""
diagnose_anatomy.py
===================
Trích xuất ma trận thông số vật lý chuyên sâu của các ứng viên (334, 454, 467, 482)
nhằm tìm ra Tiebreaker tối hậu dựa trên đặc tính sóng âm (Waveform Anatomy).
"""
import sys
import numpy as np
import librosa
from tabulate import tabulate  # Cần install: pip install tabulate

def analyze_mountain_anatomy(path, bpm, candidates=[334, 454, 467, 482]):
    y, sr = librosa.load(path, mono=True)
    beat_len = 60.0 / bpm
    PRE_MS, POST_MS = 100, 200
    seg_smp = int((PRE_MS + POST_MS) / 1000 * sr)
    beat_idx = int(PRE_MS / (PRE_MS + POST_MS) * seg_smp)
    win_20ms = int(0.02 * sr)

    # Tính toán Baseline năng lượng toàn bài để phục vụ ý tưởng của Claude
    global_envelope = np.abs(y)
    baseline_energy = float(np.mean(global_envelope))

    matrix = []

    for cand in candidates:
        off_s = cand / 1000.0
        segs = []
        # Gom toàn bộ các phân đoạn phách để tính toán mẫu dựng (Template)
        for t in np.arange(off_s, len(y)/sr - beat_len, beat_len):
            s = int((t - PRE_MS/1000) * sr)
            e = s + seg_smp
            if s >= 0 and e <= len(y):
                segs.append(y[s:e])
        
        if len(segs) == 0:
            continue
            
        tmpl = np.mean(segs, axis=0)
        env = np.abs(tmpl)
        
        # 1. Đo đạc năng lượng Cửa sổ Left vs Right (Có trừ nhiễu nền Baseline theo Claude)
        left_window = env[beat_idx - win_20ms : beat_idx]
        right_window = env[beat_idx : beat_idx + win_20ms]
        
        energy_left = float(np.mean(left_window))
        energy_right = float(np.mean(right_window))
        
        # 2. Phân tích hình thái đỉnh của Template (Waveform Peak)
        tmpl_pk_idx = np.argmax(np.abs(tmpl))
        tmpl_pk_time = (tmpl_pk_idx - beat_idx) / sr * 1000  # ms tính từ vạch 0
        tmpl_pk_amp = float(np.abs(tmpl)[tmpl_pk_idx])
        
        # 3. Phân tích đặc trưng tần số (Spectral Features) tại điểm bùng nổ
        # Trích xuất cửa sổ 40ms ngay tại điểm phách
        signal_center = tmpl[beat_idx : beat_idx + int(0.04 * sr)]
        stft_res = np.abs(librosa.stft(signal_center, n_fft=512, hop_length=128))
        
        if stft_res.shape[1] > 0:
            spec_centroid = float(np.mean(librosa.feature.spectral_centroid(S=stft_res, sr=sr)))
            spec_rolloff = float(np.mean(librosa.feature.spectral_rolloff(S=stft_res, sr=sr)))
        else:
            spec_centroid, spec_rolloff = 0.0, 0.0

        # Nạp dữ liệu vào ma trận giải phẫu
        matrix.append([
            f"{cand}ms",
            f"{energy_left:.4f}",
            f"{energy_right:.4f}",
            f"{energy_right / (energy_left + 1e-6):.2f}", # Contrast ratio
            f"{tmpl_pk_time:+.1f}ms",
            f"{tmpl_pk_amp:.3f}",
            f"{spec_centroid:.0f}Hz",
            f"{spec_rolloff:.0f}Hz"
        ])

    headers = [
        "Candidate", "Energy Left", "Energy Right", "Contrast Ratio",
        "Peak Timing", "Peak Amp", "Spec Centroid", "Spec Rolloff"
    ]
    
    print("\n" + "="*85)
    print("                OSU! CRUCIBLE: MOUNTAIN ANATOMY MATRIX REPORT")
    print("="*85)
    print(tabulate(matrix, headers=headers, tablefmt="fancy_grid"))
    print(f"\n[*] Ambient Phonk Baseline Energy (Noise Floor): {baseline_energy:.4f}")

if __name__ == "__main__":
    analyze_mountain_anatomy("top.mp3", 122.998)