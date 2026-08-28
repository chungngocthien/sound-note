"""
diagnose_low_freq.py
====================
Kiểm tra giả thuyết tối giản của Claude: Bandpass 40-100Hz (Sub-bass) trước khi tính Sharpness 
để xem ngọn núi 454ms có tự động quy hàng lên vị trí số #1 mà không cần Voting/Comb Filter.
"""
import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d

def run_low_freq_sharpness(path, bpm, gt_offset=465):
    y, sr = librosa.load(path, mono=True)
    
    # Thực hiện lọc dải thấp (Butterworth hoặc STFT mask chuyên biệt cho Sub-bass)
    print("[Crucible] Thực hiện cô lập dải tần 40Hz - 100Hz (Kick-Bass)...")
    D = librosa.stft(y, hop_length=512)
    freqs = librosa.fft_frequencies(sr=sr)
    # Khóa chặt dải sub-bass, triệt hạ hoàn toàn tiếng Cowbell (thường nằm ở mốc > 800Hz)
    mask = (freqs >= 40) & (freqs <= 100)
    D_low = D.copy()
    D_low[~mask, :] = 0
    y_filtered = librosa.istft(D_low, hop_length=512)

    beat_len = 60.0 / bpm
    beat_ms = beat_len * 1000
    PRE_MS, POST_MS = 100, 200
    seg_smp = int((PRE_MS + POST_MS) / 1000 * sr)

    def stack_template(offset_ms):
        off = offset_ms / 1000
        segs = []
        for t in np.arange(off, len(y_filtered)/sr - beat_len, beat_len):
            s = int((t - PRE_MS/1000) * sr)
            e = s + seg_smp
            if s >= 0 and e <= len(y_filtered):
                chunk = y_filtered[s:e]
                if np.abs(chunk).max() > 0.001: # Hạ thấp ngưỡng chặn vì biên độ lọc dải thấp nhỏ hơn
                    segs.append(chunk)
        return np.mean(segs, axis=0) if segs else None

    def sharpness(tmpl):
        beat_idx = int(PRE_MS / (PRE_MS + POST_MS) * len(tmpl))
        win = int(0.02 * sr)
        left = tmpl[beat_idx - win : beat_idx]
        right = tmpl[beat_idx : beat_idx + win]
        return float(np.var(right) / (np.var(left) + 1e-10))

    offsets = np.arange(0, int(beat_ms), 1)
    print("[Crucible] Đang quét năng lượng phách trên tín hiệu dải thấp...")
    sharpnesses = np.array([sharpness(stack_template(o)) or 0.0 for o in offsets])
    smoothed = uniform_filter1d(sharpnesses, size=5)
    
    # Tìm ngôi vương mới sau khi lọc dải thấp
    new_peak_ms = offsets[np.argmax(smoothed)]
    print(f"\n[KẾT QUẢ THÍ NGHIỆM TỐI THƯỢNG]:")
    print(f"  -> Đỉnh cao nhất mới sau khi lọc Sub-bass: {new_peak_ms}ms")
    print(f"  -> Khoảng cách lỗi so với Ground Truth Osu! ({gt_offset}ms): {new_peak_ms - gt_offset:+.0f}ms")

if __name__ == "__main__":
    run_low_freq_sharpness("top.mp3", 122.998, 465)