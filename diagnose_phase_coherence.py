"""
diagnose_phase_coherence.py
===========================
Hiện thực hóa đề xuất của ChatGPT: Đo độ đồng pha (Phase Coherence) thông qua 
Mean Cross-Correlation giữa các phân đoạn để tìm ra phách trống chân cơ học ổn định nhất.
"""
import sys
import numpy as np
import librosa

def analyze_phase_coherence(path, bpm, candidates=[334, 454, 467, 482]):
    print(f"\n{'='*75}")
    print(f"CHATGPT PHASE COHERENCE CRUCIBLE: {path} (BPM: {bpm})")
    print(f"{'='*75}\n")

    y, sr = librosa.load(path, mono=True)
    beat_len = 60.0 / bpm
    PRE_MS, POST_MS = 30, 50  # Cửa sổ hẹp (-30ms đến +50ms) để khóa chặt Transient đầu trống
    seg_smp = int((PRE_MS + POST_MS) / 1000 * sr)

    for cand in candidates:
        off_s = cand / 1000.0
        segs = []
        
        # 1. Trích xuất toàn bộ các phân đoạn phách đơn lẻ (Không average trước)
        for t in np.arange(off_s, len(y)/sr - beat_len, beat_len):
            s = int((t - PRE_MS / 1000) * sr)
            e = s + seg_smp
            if s >= 0 and e <= len(y):
                chunk = y[s:e]
                # Chuẩn hóa biên độ của từng beat để khử biến động âm lượng (Loudness War)
                norm = np.linalg.norm(chunk)
                if norm > 0.01:
                    segs.append(chunk / norm)
        
        if len(segs) < 10:
            print(f"Candidate {cand}ms: Không đủ dữ liệu phân đoạn.")
            continue
            
        segs = np.array(segs) # Shape: (N_beats, seg_smp)
        
        # 2. Tạo Template đại diện bằng trung vị (Median lọc nhiễu tốt hơn Mean)
        template = np.median(segs, axis=0)
        template_norm = np.linalg.norm(template) + 1e-10
        template /= template_norm

        # 3. Tính Normalized Cross-Correlation của từng Beat so với Template
        correlations = []
        for beat in segs:
            # Dot product của hai vector đã chuẩn hóa chính là Cosine Similarity / Correlation
            corr = float(np.dot(template, beat))
            correlations.append(corr)
            
        mean_corr = float(np.mean(correlations))
        std_corr = float(np.std(correlations))
        
        # Chỉ số Chí mạng của ChatGPT: Trọng số đồng pha (Coherence Score)
        # Đồng pha cao khi tương quan trung bình lớn và độ lệch chuẩn nhỏ
        coherence_score = mean_corr / (std_corr + 1e-5)

        flag = " ← [OSU! GROUND TRUTH LOCK]" if cand == 454 else ""
        print(f"Candidate {cand}ms:{flag}")
        print(f"  -> Số lượng Beats quét được     : {len(segs)}")
        print(f"  -> Độ đồng pha trung bình (Mean): {mean_corr:.4f}")
        print(f"  -> Độ bất ổn định (Std Dev)    : {std_corr:.4f}")
        print(f"  -> CHỈ SỐ COHERENCE SCORE       : {coherence_score:.2f}\n")

if __name__ == "__main__":
    analyze_phase_coherence("top.mp3", 122.998)