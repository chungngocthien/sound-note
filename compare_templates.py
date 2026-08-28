"""
compare_templates.py
====================
Trích xuất và so sánh trực tiếp cấu trúc Waveform/Envelope giữa Đỉnh giả (334ms) và Đỉnh thật (454ms).
"""
import numpy as np
import librosa
import matplotlib.pyplot as plt

def extract_and_compare(path, bpm, cand_fake=334, cand_real=454):
    y, sr = librosa.load(path, mono=True)
    beat_len = 60.0 / bpm
    PRE_MS, POST_MS = 100, 200
    seg_smp = int((PRE_MS + POST_MS) / 1000 * sr)
    beat_idx = int(PRE_MS / (PRE_MS + POST_MS) * seg_smp)

    def get_template(offset_ms):
        off_s = offset_ms / 1000.0
        segs = []
        for t in np.arange(off_s, len(y)/sr - beat_len, beat_len):
            s = int((t - PRE_MS/1000) * sr)
            e = s + seg_smp
            if s >= 0 and e <= len(y):
                chunk = y[s:e]
                if np.abs(chunk).max() > 0.01:
                    segs.append(chunk)
        return np.mean(segs, axis=0) if segs else None

    tmpl_fake = get_template(cand_fake)
    tmpl_real = get_template(cand_real)

    # Tính toán Envelope bằng cách lấy trị tuyệt đối và lấp đầy (Hilbert proxy)
    env_fake = np.abs(tmpl_fake)
    env_real = np.abs(tmpl_real)
    
    # Đạo hàm của Envelope để đo độ sắc của Transient (Độ dốc luồng hơi)
    d_env_fake = np.diff(env_fake, prepend=0)
    d_env_real = np.diff(env_real, prepend=0)

    # Khởi tạo đồ thị trực quan hóa cấu trúc Waveform vật lý
    fig, axes = plt.subplots(3, 2, figsize=(16, 10), sharex='col')
    fig.patch.set_facecolor('#0d0d12')
    fig.suptitle(f"WAVEFORM CRUCIBLE: {cand_fake}ms (Fake) vs {cand_real}ms (Real Ground Truth)", color='white', fontsize=14)

    time_axis = np.linspace(-PRE_MS, POST_MS, seg_smp)

    # Row 1: Stacked Waveform Raw
    axes[0, 0].plot(time_axis, tmpl_fake, color='#ff8888', lw=1)
    axes[0, 0].set_title(f"Raw Waveform Template at {cand_fake}ms (Cowbell/Vocal Spike)", color='#ddd')
    axes[0, 1].plot(time_axis, tmpl_real, color='#00ff88', lw=1)
    axes[0, 1].set_title(f"Raw Waveform Template at {cand_real}ms (Kick / Physical Downbeat)", color='#ddd')

    # Row 2: Envelope
    axes[1, 0].plot(time_axis, env_fake, color='#ffccff', lw=1.5)
    axes[1, 0].set_ylabel("Amplitude Envelope")
    axes[1, 1].plot(time_axis, env_real, color='#ccffcc', lw=1.5)

    # Row 3: Derivative (Transient Sharpness)
    axes[2, 0].plot(time_axis, d_env_fake, color='#ff3333', lw=1)
    axes[2, 0].set_xlabel("Time relative to beat (ms)")
    axes[2, 0].set_ylabel("d(Envelope)/dt")
    axes[2, 1].plot(time_axis, d_env_real, color='#33ff33', lw=1)
    axes[2, 1].set_xlabel("Time relative to beat (ms)")

    # Định dạng chung cho đồ thị đen tối găm cứng
    for ax in axes.flat:
        ax.set_facecolor('#0d0d12')
        ax.grid(alpha=0.1)
        ax.tick_params(colors='#aaa')
        ax.axvline(x=0, color='white', lw=1, ls='--', alpha=0.5) # Vạch kẻ điểm phách chính diện
        for sp in ax.spines.values(): sp.set_color('#333')

    plt.tight_layout()
    plt.savefig("compare_templates.png", dpi=120, facecolor='#0d0d12')
    print("Mẫu Waveform đã được xuất thành công ra file -> compare_templates.png")

if __name__ == "__main__":
    extract_and_compare("top.mp3", 122.998, 334, 454)