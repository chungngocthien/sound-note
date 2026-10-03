# Thiết kế tái tạo bộ phân tích BPM + Offset

## 1. Mục tiêu

Tài liệu này mô tả một pipeline DSP có chức năng tương đương với thuật toán đã được reverse-engineer từ `Timing Analyzer v0.32.4 alpha`:

- đọc audio và đưa về PCM mono;
- tạo một biểu diễn nhấn mạnh các biến thiên có tính nhịp điệu;
- hạ tốc độ lấy mẫu về khoảng 1000 mẫu/giây;
- biến đổi tín hiệu để các onset/rhythm event nổi bật hơn;
- dùng autocorrelation để tìm chu kỳ lặp;
- theo dõi các peak theo chu kỳ;
- hồi quy chu kỳ theo thời gian để suy ra BPM;
- khi đã biết BPM, fold tín hiệu theo chu kỳ để tìm phase mạnh nhất và suy ra offset;
- tùy chọn kiểm tra timeline sau cùng.

Mục tiêu chính là tạo một **bản thiết kế có thể giao cho LLM hoặc lập trình viên để triển khai lại**. Có hai mức triển khai:

1. **Functional clone**: tái tạo hành vi DSP và kết quả BPM/offset gần tương đương, nhưng không cố tái tạo từng bit của math library.
2. **Bit-oriented clone**: tái tạo thêm các thủ thuật floating-point, LUT và các phép toán kiểu assembly để tiến gần binary gốc.

> Tài liệu này ưu tiên Functional clone. Những phần chưa được chứng minh hoàn toàn ở mức bit được đánh dấu rõ ràng thay vì biến một phỏng đoán thành “sự thật” như loài người thường làm với tài liệu kỹ thuật.

---

## 2. Pipeline tổng thể

```text
Audio file
   |
   v
BASS / decoder
   |
   v
PCM float32, Fs = 44100 Hz
   |
   v
Interleaved channels -> sum to mono
   |
   v
+-------------------------------------------------------+
| Preprocessing                                         |
|                                                       |
| repeat 5 times:                                       |
|   IIR cascade (16 sections)                           |
|   square every sample                                 |
|   IIR cascade (16 sections)                           |
|   accumulator *= (1 + filtered_signal)                |
|                                                       |
| log transform                                         |
| 1-section IIR                                         |
| resample 44100 -> ~1000 Hz                            |
| 1-section IIR                                         |
| 1-section IIR                                         |
| backward alternating-difference transform             |
+-------------------------------------------------------+
   |
   v
~1000 Hz analysis signal
   |
   +------------------------------+
   |                              |
   v                              v
Autocorrelation               Phase folding
   |                              |
   v                              v
Peak tracking                 max phase
   |                              |
   v                              v
Period T (ms)                Offset (ms)
   |
   v
BPM = 60000 / T
```

Nguồn reverse engineering cho thấy chương trình khởi tạo BASS ở 44100 Hz, đọc PCM dạng float, gom các channel về mono trước khi chạy DSP. Hàm tiền xử lý `FUN_00401e80` thực hiện 5 vòng IIR -> square -> IIR -> tích lũy, sau đó log, IIR, resample về 1000 Hz, hai IIR cuối và biến đổi ngược. fileciteturn62file5L1-L18 fileciteturn62file1L1-L70

---

## 3. Input audio

### 3.1 Sample rate

Binary gọi:

```c
BASS_Init(0, 0xac44, ...)
```

với `0xac44 = 44100`.

Ở tầng UI/caller, chương trình còn cảnh báo rằng sample rate khác 44100 có thể cho kết quả không đúng.

Vì vậy implementation nên chuẩn hóa input về:

```text
Fs_source = 44100 Hz
```

### 3.2 Channel -> mono

Audio được đọc thành `float32`. Nếu audio có `C` channel và dữ liệu interleaved:

```python
mono[n] = sum(channel[n, c] for c in range(C))
```

Binary dùng tổng channel, không thấy phép chia cho số channel ở bước này.

### 3.3 Mô hình dữ liệu đề nghị

```python
pcm: np.ndarray[np.float32]     # shape (N,)
fs: int = 44100
```

---

## 4. Khối IIR / biquad

`FUN_00401580` là một cascade của các section bậc hai. Mỗi section được biểu diễn bởi 5 float liên tiếp:

```text
[A0, A1, A2, A3, gain]
```

Từ recurrence đã giải được, có thể biểu diễn một section theo:

```text
s[n] = gain*x[n] - A2*s[n-1] - A3*s[n-2]

y[n] = s[n] + A0*s[n-1] + A1*s[n-2]
```

Tương đương:

```text
H(z) = gain * (1 + A0 z^-1 + A1 z^-2)
             / (1 + A2 z^-1 + A3 z^-2)
```

Cài đặt reference:

```python
import numpy as np


def biquad_section(x: np.ndarray, coeff: np.ndarray) -> np.ndarray:
    a0, a1, a2, a3, gain = map(float, coeff)
    y = np.empty_like(x, dtype=np.float32)

    s1 = 0.0
    s2 = 0.0
    for n, xn in enumerate(x.astype(np.float32, copy=False)):
        s0 = gain * float(xn) - a2 * s1 - a3 * s2
        yn = s0 + a0 * s1 + a1 * s2
        y[n] = np.float32(yn)
        s2 = s1
        s1 = s0

    return y


def iir_cascade(x: np.ndarray, sections: np.ndarray) -> np.ndarray:
    y = x.astype(np.float32, copy=True)
    for coeff in sections:
        y = biquad_section(y, coeff)
    return y
```

### 4.1 Cấu trúc 16 section

Caller của `FUN_00401e80` truyền các range section:

```text
[0,2), [2,6), [6,10), [10,14), [14,16)
```

Tức tổng cộng 16 section, được chia thành nhóm 2 + 4 + 4 + 4 + 2. Assembly cũng xác nhận địa chỉ coefficient được tính từ base `0x4193f0` với stride 5 float cho mỗi section. fileciteturn56file9L1-L18

Dump coefficient cho thấy ít nhất:

```text
0x4193f0: [ 2.0,  1.0, -1.966025,  0.96782225, 4.4931905e-4 ]
0x419404: [ 2.0,  1.0, -1.922287,  0.92404425, 4.3932308e-4 ]
0x419418: [ 0.0, -1.0, -1.9869181, 0.9888196, 0.021197148 ]
0x41942c: [ 0.0, -1.0, -1.9718623, 0.97876525, 0.021197148 ]
...
```

Không được tự điền các coefficient còn thiếu nếu mục tiêu là bit-compatible với binary. Hãy lấy toàn bộ 16 tuple từ raw binary dump.

---

## 5. Preprocessing chính

### 5.1 Khởi tạo accumulator

```python
acc = np.ones_like(pcm, dtype=np.float32)
```

Binary dùng hằng số `1.0` cho accumulator. fileciteturn62file1L1-L12

### 5.2 Lặp đúng 5 lần

Mỗi vòng:

```text
x = source PCM
x1 = IIR16(x)
x2 = x1^2
x3 = IIR16(x2)
acc *= (1 + x3)
```

Reference:

```python
def build_energy_envelope(x: np.ndarray, sections16: np.ndarray) -> np.ndarray:
    acc = np.ones_like(x, dtype=np.float32)

    for _ in range(5):
        f1 = iir_cascade(x, sections16)
        f2 = np.float32(f1 * f1)
        f3 = iir_cascade(f2, sections16)
        acc = np.float32(acc * (np.float32(1.0) + f3))

    return acc
```

Điểm rất quan trọng: accumulator không cộng dồn `f3`; nó **nhân dồn** `1 + f3`. Decompile cho thấy phép nhân này trực tiếp trên vector accumulator. fileciteturn71file9L1-L18

---

## 6. Log transform

Sau 5 vòng, binary gọi `FUN_00413958` cho từng sample rồi convert kết quả double trở lại float.

Assembly caller:

```text
CVT/float -> CVTPS2PD
CALL FUN_00413958
CVTSD2SS
store float
```

fileciteturn59file8L1-L18

### 6.1 Functional clone

Đối với phần mềm mới, chỉ cần:

```python
log_env = np.log(acc)
```

Nếu accumulator có thể không dương do sai số hoặc coefficient không đúng, cần xử lý trước:

```python
log_env = np.log(np.maximum(acc, np.finfo(np.float32).tiny))
```

### 6.2 Bit-oriented clone

`FUN_00413958` rõ ràng đang làm floating-point range reduction dựa trên exponent/mantissa. Nó tạo mantissa chuẩn hóa, tạo index vào LUT theo stride 16 byte và tính residual từ một reciprocal-like table entry. Decompile thể hiện mask mantissa, mask exponent và phép index `... & 0x7f0`. fileciteturn51file3L1-L25

Các constant tại vùng gần `0x41a7c0` cũng chứa:

```text
0.6931471805598903
5.497923018708371e-14
```

Hai giá trị này cộng lại xấp xỉ `ln(2)`, phù hợp với cách lưu high/low correction của một implementation log chính xác. fileciteturn56file0L1-L10

Tuy nhiên, layout toàn bộ bảng và polynomial nội bộ chưa được chứng minh hoàn toàn từ decompile hiện có. Do đó:

```text
Functional clone -> dùng np.log
Bit-exact clone   -> reverse-engineer tiếp LUT + math path
```

Đừng dùng giả định `1/sqrt(x)`. Các giá trị reciprocal quan sát được và logic range reduction không phù hợp với inverse square root.

---

## 7. IIR sau log

Sau log transform, binary chạy:

```text
1 section IIR @ 0x419530
```

Sau đó resample. Caller xác nhận thứ tự này. fileciteturn62file1L41-L60

Coefficient ở `0x419530` cần lấy từ raw dump nếu cần exact clone. Không tự suy đoán từ tên biến Ghidra.

---

## 8. Resample về ~1000 Hz

Binary gọi:

```c
FUN_00401b30(&buffer, 1000.0 / Fs)
```

với constant `1000.0` tại `0x41a710`.

Hàm `FUN_00401b30`:

1. tính output length bằng `ceil(N * 1000/Fs)`;
2. dùng phase/position dạng double;
3. map mỗi output sample về một source index nguyên;
4. copy trực tiếp source sample, nếu vượt biên thì ghi zero.

Decompile cho thấy `ceil` thông qua `FUN_00405db0`, và output index được tạo từ floating-point position rồi chuyển về integer. fileciteturn51file1L1-L17

### 8.1 Functional reference

Nếu không cần bit-compatible interpolation:

```python

def nearest_index_resample(x: np.ndarray, fs_in: float, fs_out: float = 1000.0) -> np.ndarray:
    n_out = int(np.ceil(len(x) * fs_out / fs_in))
    pos = np.arange(n_out, dtype=np.float64) * (fs_in / fs_out)
    idx = np.floor(pos + 0.5).astype(np.int64)

    y = np.zeros(n_out, dtype=np.float32)
    valid = idx < len(x)
    y[valid] = x[idx[valid]]
    return y
```

Để bám binary hơn, cần tái tạo chính xác cách `dVar6`, `dVar7`, `dVar8` và các phép trộn sign/rounding tạo `uVar5`. fileciteturn51file1L1-L25

---

## 9. Hai IIR cuối

Sau resample, binary chạy cùng một section tại `0x419544` **hai lần liên tiếp**:

```text
IIR1 @ 0x419544
IIR1 @ 0x419544
```

Assembly có hai `PUSH 0x419544` và hai `CALL FUN_00401580`, mỗi lần với `EAX = 1`, nên đây là hai cascade section đơn liên tiếp. fileciteturn65file6L1-L16

Dump hiện có xác nhận vùng này là một section độc lập, nhưng cần toàn bộ 5 float để hard-code chính xác.

---

## 10. Backward alternating-difference transform

Đây là phần Gemini đã từng tái tạo sai.

Binary **không** làm:

```python
y[n] = max(0, x[n] - x[n-1])
```

Mà sau hai IIR cuối, nó thực hiện một phép biến đổi in-place từ cuối mảng về đầu.

Decompile cho thấy:

```text
last = previous ^ sign_bit_mask

for i from the end toward the beginning:
    output[i+...] = temporary_source[...] - already_written_output[...]

output[0] = another temporary_source value
```

Phần scalar hóa của recurrence là:

```text
q[0] = x[N-2]
q[k] = x[N-2-k] - q[k-1]
```

với phần tử biên cuối được tạo bằng flip sign bit trên một sample trước đó.

Với vector mẫu:

```text
x = [0.10, 0.12, 0.55, 0.80, 0.40, 0.30, 0.75, 0.90, 0.20]
```

reference scalar thu được:

```text
[ 0.90,
 -0.15,
  0.45,
 -0.05,
  0.85,
 -0.30,
  0.42,
 -0.32,
 -0.90 ]
```

Binary sử dụng `XOR` với `DAT_0041a630` cho một boundary value. Dump cho thấy vùng này chứa sign-bit mask `0x80000000`. Vì vậy implementation bit-exact nên giữ XOR, không thay bằng phép unary minus nếu muốn bảo toàn NaN/sign-zero semantics. fileciteturn62file0L1-L45 fileciteturn56file8L1-L18

### 10.1 Functional implementation

```python

def backward_alternating_transform(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    n = len(x)
    if n == 0:
        return x.copy()
    if n == 1:
        return x.copy()

    y = np.empty_like(x)
    y[0] = x[n - 2]
    for k in range(1, n - 1):
        y[k] = np.float32(x[n - 2 - k] - y[k - 1])
    y[n - 1] = np.float32(-x[n - 2])
    return y
```

Đây là scalar model để hiểu thuật toán. Với exact clone, cần giữ đúng boundary indexing của assembly và sign-bit XOR.

---

## 11. Tín hiệu phân tích cuối

Sau bước 10, ta có một tín hiệu ở khoảng 1000 Hz:

```text
analysis_rate ≈ 1000 samples/sec
```

Do đó:

```text
1 sample ≈ 1 ms
```

Đây là lý do toàn bộ BPM/offset detector có thể dùng trực tiếp index mẫu như milliseconds.

---

# 12. Autocorrelation bằng FFT

Binary sử dụng `FUN_004017a0` để tạo autocorrelation.

### 12.1 FFT size

`FUN_00405db0` là helper có hành vi tương đương `ceil()` ở đường code này. Constant `2.0` nằm tại `0x41a720` và được dùng để chọn số mũ power-of-two.

Mục tiêu là chọn FFT size là power-of-two đủ lớn cho dữ liệu.

Conceptually:

```python
fft_n = 1
while fft_n < len(x):
    fft_n *= 2
```

### 12.2 Wiener-Khinchin

Tính:

```text
X = FFT(x)
P = |X|^2
R = IFFT(P)
```

Binary tạo magnitude squared từ real/imag rồi inverse FFT để thu autocorrelation. Hàm khởi tạo FFT cũng tạo sẵn twiddle/bit-reversal table. fileciteturn71file3L1-L18

Functional reference:

```python

def autocorrelation_fft(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    n = len(x)

    fft_n = 1
    while fft_n < n:
        fft_n <<= 1

    X = np.fft.rfft(x, n=fft_n)
    power = np.abs(X) ** 2
    r = np.fft.irfft(power, n=fft_n)

    return r[:n].astype(np.float64)
```

Một implementation exact hơn có thể tái tạo FFT radix-2 custom của binary thay vì `np.fft`.

---

# 13. Peak detector cho BPM

## 13.1 Local maxima

`FUN_00402280`:

1. tính autocorrelation;
2. xét các điểm trong miền phân tích giới hạn;
3. chỉ chấp nhận sample cao hơn threshold;
4. yêu cầu nó lớn hơn các lân cận trước và sau;
5. còn kiểm tra thêm các sample cách một bước nữa.

Điều này có nghĩa peak detector không đơn giản là `x[i] > x[i-1] and x[i] > x[i+1]`.

Decompile cho thấy điều kiện kiểu:

```text
x[i] > threshold
x[i] > x[i-1]
x[i] > x[i+1]
x[i-1] >= x[i-2]
x[i+1] >= x[i+2]
```

và không chấp nhận các trường hợp bằng nhau ở các vị trí kiểm tra thứ hai. fileciteturn72file4L1-L30

### 13.2 Parabolic interpolation

Mỗi peak nguyên được refine thành số thực:

```text
p = i + 0.5 * (y[i+1] - y[i-1])
          / ((y[i] - y[i+1]) + (y[i] - y[i-1]))
```

Constant `0.5` là `DAT_0041a640`. fileciteturn72file4L1-L30

Reference:

```python

def parabolic_peak(y: np.ndarray, i: int) -> float:
    if i <= 0 or i >= len(y) - 1:
        return float(i)

    ym1 = float(y[i - 1])
    y0 = float(y[i])
    yp1 = float(y[i + 1])
    den = (y0 - yp1) + (y0 - ym1)

    if den == 0.0:
        return float(i)

    return float(i) + 0.5 * (yp1 - ym1) / den
```

---

# 14. Theo dõi peak theo chu kỳ

Sau khi thu được các candidate peak, binary không lấy ngay khoảng cách trung bình giữa mọi peak.

Thay vào đó nó giữ một chu kỳ dự đoán rồi tìm peak tiếp theo quanh vị trí đó.

### 14.1 Ý tưởng

```text
candidate p0
    |
    +--> dự đoán p1 ≈ p0 + T
            |
            +--> tìm peak trong khoảng ±10 samples
                    |
                    +--> refine parabola
                           |
                           +--> cập nhật regression T
                                  |
                                  +--> dự đoán p2
```

Nếu không tìm được peak đủ gần, chuỗi bị đánh dấu lỗi/gap và tracker tiếp tục với mức phạt phù hợp.

Decompile cho thấy cửa sổ tìm peak ±10 sample và logic bỏ qua candidate nếu lệch quá xa. fileciteturn72file4L1-L75

### 14.2 Regression chu kỳ

Binary duy trì:

```text
sum_i2 = Σ i²
sum_ip = Σ i * p_i
```

sau đó:

```text
T = sum_ip / sum_i2
```

đây là least-squares fit qua gốc cho:

```text
p_i ≈ i * T
```

Code cũng tính độ phân tán/sai số từ residual để quyết định chất lượng fit. fileciteturn72file4L70-L110

Functional implementation:

```python

def fit_period(peaks: np.ndarray) -> tuple[float, float]:
    peaks = np.asarray(peaks, dtype=np.float64)
    if len(peaks) < 2:
        raise ValueError("need at least two peaks")

    i = np.arange(1, len(peaks) + 1, dtype=np.float64)
    T = np.sum(i * peaks) / np.sum(i * i)
    residual = peaks - i * T
    rms = float(np.sqrt(np.mean(residual * residual)))
    return float(T), rms
```

Đây là model khái niệm. Để match binary, phải giữ nguyên cách binary khởi tạo `i`, cách reset tracker và cách xử lý missing peak.

---

# 15. Chuyển period -> BPM

Binary có constant:

```text
DAT_0041a6e8 = 60000.0
```

Do analysis domain xấp xỉ 1 ms/sample:

```text
BPM = 60000 / T_ms
```

Tương đương code:

```python
bpm = 60000.0 / period_ms
```

Constant này xuất hiện trực tiếp trong BPM calculation. fileciteturn56file1L1-L20

---

# 16. Điều kiện hợp lệ của BPM estimate

Binary yêu cầu tối thiểu 4 điểm cho regression cuối. Nó cũng tính độ lệch của peak quanh fit và từ đó trả các trạng thái khác nhau.

Trạng thái quan sát được:

```text
-1   -> không đủ dữ liệu / không thể tính
 0x10 -> fit quá phân tán
 0   -> fit tốt
 1   -> có fit nhưng quality không hoàn toàn sạch
```

Binary dùng các constant threshold khác nhau, trong đó dump có các giá trị như `0.6`, `2.4`, `3.0e-5`, `20.0`. Không nên gán tên/ngữ nghĩa chính xác cho mọi threshold nếu chưa truy tới toàn bộ call-site của chúng. fileciteturn72file4L70-L120

Functional clone có thể dùng:

```python
if len(peaks) < 4:
    status = -1
elif rms > max_rms_ms:
    status = 0x10
else:
    status = 0
```

Sau này có thể hiệu chỉnh threshold trên dataset thật.

---

# 17. Offset detector

Offset được tính sau khi BPM đã biết.

## 17.1 Từ BPM sang period

```python
period_ms = 60000.0 / bpm
```

Binary dùng đúng quan hệ này. fileciteturn56file2L1-L20

## 17.2 Phase folding

Ý tưởng của `FUN_00402c70` là:

```text
signal
  |
  +-- chia theo các vị trí n*T
  |
  v
phase accumulator
  |
  v
cộng tín hiệu của nhiều chu kỳ lên cùng phase
  |
  v
phase profile
  |
  v
argmax phase
```

Cụ thể, binary duyệt:

```text
position = n*T + 0.5
sample_index = integer(position)
```

rồi cộng một đoạn tín hiệu tương ứng vào phase buffer. Code lặp cho tới hết số chu kỳ có thể chứa trong signal. fileciteturn71file9L1-L55

### 17.3 Chọn phase mạnh nhất

Sau folding, binary tìm index của sample/phase bin lớn nhất.

Sau đó refine bằng cùng công thức parabolic interpolation:

```text
phase = i + 0.5 * (y[i+1] - y[i-1])
              / ((y[i] - y[i+1]) + (y[i] - y[i-1]))
```

và cuối cùng:

```text
offset_ms = phase - 6.2
```

`6.2` là constant tại `0x4193e0`. fileciteturn56file2L1-L20

### 17.4 Functional implementation

```python

def estimate_offset(signal: np.ndarray, bpm: float, correction_ms: float = 6.2) -> float:
    period = 60000.0 / bpm
    n_periods = int(np.floor(len(signal) / period))
    if n_periods <= 0:
        raise ValueError("signal shorter than one period")

    phase = np.zeros(int(np.ceil(period)) + 2, dtype=np.float64)

    for k in range(n_periods):
        start = int(np.floor(k * period + 0.5))
        if start >= len(signal):
            break
        segment = signal[start:start + len(phase)]
        phase[:len(segment)] += segment

    i = int(np.argmax(phase[1:-1])) + 1
    refined = parabolic_peak(phase, i)
    return float(refined - correction_ms)
```

**Chú ý:** đây là functional model của phase folding, không phải reproduction từng byte của `FUN_00402c70`. Binary có cách cấp phát phase buffer và cách copy/cộng các đoạn với boundary handling riêng. Muốn exact clone phải viết lại các boundary loop trong decompile.

---

# 18. Timeline validation

Sau BPM + offset, binary gọi `FUN_00402fc0` để kiểm tra các beat dự đoán trên toàn timeline.

Logic quan sát được gồm hai scale cửa sổ:

```text
±10 samples
±30 samples
```

và đánh dấu các vùng có vấn đề bằng các bit status. UI biểu diễn các trạng thái này dưới dạng `#`, `+`, `-`, `.`.

Phần validation không cần thiết để tính BPM/offset cốt lõi, nhưng hữu ích để đánh giá kết quả trên toàn bài.

---

# 19. Kiến trúc implementation đề nghị

```text
src/
├── audio.py
├── filters.py
├── preprocess.py
├── resample.py
├── correlation.py
├── bpm.py
├── offset.py
├── validation.py
└── main.py
```

### `audio.py`

Chịu trách nhiệm:

```text
audio file -> float32 PCM mono, Fs=44100
```

### `filters.py`

```text
biquad_section()
iir_cascade()
```

### `preprocess.py`

```text
build_energy_envelope()
log_stage()
post_log_filter()
```

### `resample.py`

```text
resample_to_1000hz()
```

### `correlation.py`

```text
fft_autocorrelation()
```

### `bpm.py`

```text
find_local_peaks()
parabolic_peak()
track_periodic_peaks()
fit_period()
period_to_bpm()
```

### `offset.py`

```text
fold_by_period()
find_phase_peak()
phase_to_offset()
```

### `validation.py`

```text
validate_timeline()
```

---

# 20. Một pipeline Python đầy đủ

```python
from dataclasses import dataclass
import numpy as np


@dataclass
class TimingResult:
    bpm: float
    offset_ms: float
    period_ms: float
    period_rms_ms: float


def analyze(audio_mono_44100: np.ndarray, sections16: np.ndarray,
            pre_section: np.ndarray, post_section: np.ndarray) -> TimingResult:
    # 1. Preprocessing
    acc = build_energy_envelope(audio_mono_44100, sections16)

    # 2. Log
    env = np.log(np.maximum(acc, np.finfo(np.float32).tiny)).astype(np.float32)

    # 3. One dedicated pre-resample IIR section @ 0x419530
    env = biquad_section(env, pre_section)

    # 4. Resample to ~1000 Hz
    env = nearest_index_resample(env, 44100.0, 1000.0)

    # 5. Two post-resample sections
    env = biquad_section(env, post_section)
    env = biquad_section(env, post_section)

    # 6. Backward transform
    env = backward_alternating_transform(env)

    # 7. Autocorrelation
    ac = autocorrelation_fft(env)

    # 8. Peaks
    peaks = find_local_peaks(ac)
    tracked = track_periodic_peaks(ac, peaks)

    # 9. Fit period
    period_ms, rms = fit_period(tracked)
    bpm = 60000.0 / period_ms

    # 10. Offset
    offset_ms = estimate_offset(env, bpm)

    return TimingResult(
        bpm=bpm,
        offset_ms=offset_ms,
        period_ms=period_ms,
        period_rms_ms=rms,
    )
```

`find_local_peaks()` và `track_periodic_peaks()` cần được triển khai đúng theo mục 13-14; chúng không thể chỉ là `scipy.signal.find_peaks()` nếu mục tiêu là mô phỏng behavior của binary.

---

# 21. Pseudocode hoàn chỉnh cho LLM triển khai

LLM nên thực hiện theo thứ tự này, không được tự ý đổi domain hoặc đảo thứ tự các phép biến đổi:

```text
INPUT:
    audio file

A. AUDIO
    decode -> float32
    require/convert Fs = 44100
    interleaved channels -> sum -> mono

B. PREPROCESS
    acc = ones(N)
    repeat 5 times:
        a = IIR16(source)
        b = square(a)
        c = IIR16(b)
        acc = acc * (1 + c)

C. LOG
    env[i] = log(acc[i])

D. PRE-RESAMPLE FILTER
    env = IIR1(env)

E. RESAMPLE
    env = resample(env, 44100 -> 1000 Hz)

F. POST FILTER
    env = IIR1(env)
    env = IIR1(env)

G. BACKWARD TRANSFORM
    reverse alternating recurrence

H. AUTOCORRELATION
    Nfft = next_pow2(N)
    X = FFT(env, Nfft)
    P = real(X)^2 + imag(X)^2
    ac = IFFT(P)

I. PEAK DETECTION
    local maxima + threshold
    refine each peak parabolically

J. PERIOD TRACKING
    predict next peak = current + T
    search within +/-10 ms
    update least-squares period estimate
    reject bad gaps

K. BPM
    T_ms = fitted period
    BPM = 60000 / T_ms

L. OFFSET
    T_ms = 60000 / BPM
    fold signal by phase modulo T
    accumulate over cycles
    find maximum phase
    parabolic refinement
    offset = phase - 6.2 ms

M. OPTIONAL VALIDATION
    test predicted beat positions
    classify timeline errors

OUTPUT:
    BPM
    offset_ms
    optional quality/status
```

---

# 22. Những phần cần giữ nguyên tuyệt đối

Nếu mục tiêu là tái tạo logic chứ không phải viết một detector “có vẻ giống”:

1. **5 vòng accumulator** là nhân dồn `(1 + filtered)`.
2. **Square** nằm giữa hai lần IIR trong mỗi vòng.
3. **Log xảy ra sau vòng 5**, không phải trước.
4. **Một IIR xảy ra trước resample**.
5. **Resample về khoảng 1000 Hz**.
6. **Hai IIR cùng section xảy ra sau resample**.
7. **Backward alternating transform xảy ra sau hai IIR đó**.
8. **Autocorrelation dùng FFT power spectrum**, không phải simple convolution trực tiếp.
9. **Peak được parabolic-refine**.
10. **BPM được suy ra từ fitted period**, không lấy trung bình khoảng cách peak thô.
11. **Offset được tìm bằng phase folding với BPM đã xác định**.
12. **Offset có correction cố định 6.2 ms** trong binary hiện tại.

---

# 23. Những phần có thể thay thế khi chỉ cần functional equivalence

```text
BASS               -> bất kỳ decoder PCM nào
custom FFT         -> numpy/scipy FFT
custom log         -> np.log
custom ceil helper -> np.ceil
MSVC float quirks  -> IEEE float32/float64 bình thường
custom allocator   -> numpy arrays
```

Không nên thay thế:

```text
IIR order
number of cascades
5-pass accumulator
10-sample peak tracking window
phase folding concept
BPM formula
offset correction
```

nếu mục tiêu là giữ behavior của detector.

---

# 24. Bit-exact roadmap

Nếu muốn tiến từ functional clone sang clone gần binary:

## Priority 1: coefficient extraction

Lấy toàn bộ:

```text
0x4193f0 ... 16 x 5 float
0x419530 ... 1 x 5 float
0x419544 ... 1 x 5 float
```

## Priority 2: exact resampler

Tái tạo từng phép toán của `FUN_00401b30`, đặc biệt:

```text
floating position
fraction extraction
floor/ceil behavior
out-of-range handling
```

## Priority 3: exact log path

Tái tạo:

```text
mantissa mask
exponent extraction
index construction
LUT byte layout
reciprocal entry
log high/low data
residual polynomial, nếu còn tồn tại trong path chưa được decompile rõ
```

`FUN_00413958` chắc chắn có cấu trúc kiểu range-reduced log, nhưng layout full LUT/polynomial chưa đủ bằng chứng để viết một implementation bit-exact hoàn toàn. Đừng dùng một Python checker nhìn “hợp lý” để kết luận bit-exact nếu chưa chứng minh address mapping từng entry.

## Priority 4: exact FFT

Thay `np.fft` bằng radix-2 FFT theo cấu trúc của binary nếu cần sample-level matching.

## Priority 5: exact tracker

Giữ nguyên:

```text
candidate ordering
±10 search
missing peak counter
least-squares state
quality thresholds
status codes
```

## Priority 6: exact phase-folding boundary behavior

Đặc biệt phải xác định:

```text
phase buffer length
start index rounding
segment copy/add boundaries
last partial cycle
parabolic edge behavior
```

---

# 25. Test strategy

Không kiểm tra chỉ bằng một bài nhạc.

Nên tạo dataset tối thiểu:

```text
A. synthetic click track
   60 BPM
   90 BPM
   120 BPM
   150 BPM
   180 BPM

B. synthetic shifted click tracks
   offset = 0 ms
   10 ms
   25 ms
   50 ms
   100 ms

C. simple periodic waveforms
   sine/pulse/saw

D. real songs
   stable tempo
   swing
   silence intro
   strong drums
   weak drums
   tempo changes
```

Mỗi test nên lưu:

```text
input hash
Fs
N
preprocessed signal checksum
autocorrelation checksum
peak list
period
BPM
phase profile checksum
offset
```

Việc lưu intermediate checkpoints rất quan trọng. Nếu cuối cùng BPM sai, đừng nhìn mỗi output BPM rồi đoán. So từng tầng DSP để biết lỗi đầu tiên xuất hiện ở đâu.

---

# 26. Acceptance criteria

## Functional clone

Được coi là đạt khi:

```text
- pipeline chạy ổn định trên WAV thực tế
- BPM đúng gần binary trên stable-tempo tracks
- offset có tính lặp lại
- intermediate signal hợp lý
- peak sequence có cùng cấu trúc định kỳ
```

## Near-bit clone

Cần thêm:

```text
- cùng coefficient
- cùng float width ở từng stage
- cùng rounding
- cùng FFT padding/normalization
- cùng peak tracking
- cùng LUT mapping
- cùng boundary handling
```

---

# 27. Tóm tắt kỹ thuật

Thuật toán không phải “một công thức tìm BPM”. Nó là một chuỗi biến đổi DSP nhiều tầng:

```text
PCM
 -> band/feature filtering
 -> repeated nonlinear energy accumulation
 -> log compression
 -> low-rate analysis representation
 -> post filtering
 -> alternating backward transform
 -> autocorrelation
 -> periodic peak tracking
 -> least-squares period
 -> BPM
 -> phase folding
 -> phase maximum
 -> offset
```

Phần mạnh nhất của kiến trúc là nó không cố đo BPM trực tiếp trên waveform. Nó trước hết tạo ra một representation trong đó periodicity dễ quan sát hơn, sau đó dùng autocorrelation để biến bài toán “tìm nhịp trong audio” thành bài toán “tìm khoảng trễ lặp lại”.

Offset lại là một bài toán khác: khi period đã biết, không cần tìm lại BPM. Chỉ cần gấp toàn bộ tín hiệu theo period rồi tìm phase có tổng năng lượng/response mạnh nhất.

---

# 28. Source map cho reverse-engineering hiện tại

| Thành phần | Function / address | Trạng thái hiểu |
|---|---|---|
| Audio decode | `FUN_00401c60` | Hiểu rõ ở mức pipeline |
| 2nd-order IIR | `FUN_00401580` | Đã giải recurrence |
| Main preprocessing | `FUN_00401e80` | Hiểu rõ |
| Log helper | `FUN_00413958` | Semantics log/range reduction đã rõ; bit-exact LUT chưa hoàn toàn |
| Resampler | `FUN_00401b30` | Hiểu rõ cấu trúc |
| FFT | `FUN_004010b0`, `FUN_004017a0` | Hiểu rõ |
| Ceil helper | `FUN_00405db0` | Đã xác định là `ceil()` path |
| BPM detector | `FUN_00402280` | Hiểu thuật toán period tracking |
| Offset detector | `FUN_00402c70` | Hiểu phase folding + parabola + `-6.2 ms` |
| Timeline validation | `FUN_00402fc0` | Hiểu vai trò và cửa sổ chính |

---

# 29. Kết luận thiết kế

Một LLM có thể triển khai functional clone mà **không cần tiếp tục reverse-engineer từng instruction** nếu mục tiêu chỉ là một phần mềm có chức năng tìm BPM + offset tương đương về mặt thuật toán.

Thứ tự triển khai nên là:

```text
1. audio + mono
2. exact IIR abstraction
3. 5-pass energy accumulator
4. log
5. resample 1000 Hz
6. post filters
7. backward transform
8. FFT autocorrelation
9. peak detector + parabola
10. periodic peak tracker
11. least-squares period
12. BPM
13. phase folding
14. offset
15. validation
```

Sau khi functional clone hoạt động, mới thay từng module bằng implementation gần binary. Cách này tạo một hệ thống có thể kiểm chứng theo từng checkpoint thay vì viết một đống code rồi phát hiện BPM sai 7 BPM và bắt đầu cầu nguyện cho compiler.

# Gemini
---

## 30. Bổ sung Toán học & Cấu trúc thuật toán ẩn (Mathematical & DSP Foundations)

Tài liệu này bổ sung các lớp nền toán học (Mathematical Foundations) chưa được gọi tên chính thức trong các mục trên, nhằm giúp LLM / Developer hiểu rõ bản chất lý thuyết thay vì chỉ nhìn vào pseudocode imperative.

### 30.1 Nền tảng Toán học của Preprocessing & Envelope Generation

1. **Biquad IIR Cascade Filterbank (Mục 4):**
   - Về mặt DSP, chuỗi 16 section biquad này tạo thành một **Multi-band IIR Filterbank** (Bộ lọc dải thông/dải chặn tầng). Dạng toán học của từng Section là Transfer Function $H_i(z)$ miền Z-transform:
     $$H_i(z) = \text{gain}_i \cdot \frac{1 + A_{0,i} z^{-1} + A_{1,i} z^{-2}}{1 + A_{2,i} z^{-1} + A_{3,i} z^{-2}}$$
   - Việc ghép chuỗi (Cascade) tương đương với nhân các Transfer Function: $H(z) = \prod_{i=1}^{16} H_i(z)$.

2. **Nonlinear Quadratic Novelty / Energy Envelope (Mục 5):**
   - Phép toán $x_2 = x_1^2$ kết hợp với IIR là một phương pháp trích xuất **Energy Envelope / Instantaneous Power**.
   - Việc nhân dồn tích lũy $\text{acc} \leftarrow \text{acc} \times (1 + f_3)$ tạo ra một **Multiscale Non-linear Energy Map** (Bản đồ năng lượng phi tuyến tính đa tỷ lệ) có dạng toán học tương đương:
     $$\text{acc}[n] = \prod_{k=1}^{5} \left( 1 + \mathcal{F}_{\text{IIR}}\left( \left( \mathcal{F}_{\text{IIR}}(x)[n] \right)^2 \right) \right)$$
   - Cơ chế này nén các đỉnh nhiễu không định kỳ (transients) và khuếch đại dòng năng lượng lặp lại (harmonic/rhythmic energy).

3. **Dynamic Range Compression via Logarithmic Mapping (Mục 6):**
   - Phép biến đổi $\text{log}(\text{acc})$ đưa biên độ năng lượng từ thang nhân (multiplicative domain) về thang cộng (additive domain), tương đương với nguyên lý **Weber-Fechner Law** trong tâm lý âm học (psychoacoustics). Nó làm giảm sự chênh lệch biên độ giữa đoạn nhạc lớn (chorus) và đoạn nhạc nhỏ (intro/outro).

### 30.2 Backward Alternating-Difference Transform (Mục 10)

Thuật toán đệ quy ngược từ cuối mảng:
$$y[0] = x[N-2], \quad y[k] = x[N-2-k] - y[k-1] \quad (1 \le k < n-1)$$

- **Bản chất Toán học:** Đây là một **IIR High-Pass First-Order Difference Filter** chạy đảo ngược thời gian (Time-Reversed Alternating Recurrence).
- **Mục đích DSP:** Mạng đệ quy đổi dấu làm triệt tiêu các thành phần DC Drift (dòng xoay chiều tần số cực thấp/suôn nền) và làm sắc nét các **Onset Edges** (điểm bắt đầu của nốt/tiếng trống), chuyển tín hiệu năng lượng dạng dốc mềm thành các xung dao động đổi dấu có biên độ nhọn.

### 30.3 Autocorrelation & Wiener–Khinchin Theorem (Mục 12)

- **Định lý Wiener–Khinchin:** Định lý phát biểu rằng Hàm tự tương quan (Autocorrelation Function - ACF) của một tín hiệu dừng tương đương với Biến đổi Fourier ngược (IFFT) của Mật độ phổ công suất (Power Spectral Density - PSD) của tín hiệu đó:
  $$R_{xx}(\tau) = \mathcal{F}^{-1} \left\{ \left\vert{} \mathcal{F}\{x(t)\} \right\vert{}^2 \right\}$$
- **Ý nghĩa:** Việc tính autocorrelation bằng FFT giảm độ phức tạp tính toán từ $\mathcal{O}(N^2)$ xuống $\mathcal{O}(N \log N)$, biến bài toán tìm nhịp điệu thành bài toán đo độ tự tương quan theo thời gian trễ $\tau$ (lag period).

### 30.4 Sub-sample Peak Refinement (Sub-grid Parabolic Interpolation) (Mục 13.2 & 17.3)

Công thức hiệu chỉnh đỉnh parabola:
$$p = i + \frac{1}{2} \cdot \frac{y[i+1] - y[i-1]}{(y[i] - y[i+1]) + (y[i] - y[i-1])}$$

- **Nền tảng Toán học:** Xấp xỉ Taylor bậc 2 quanh cực trị cục bộ. Giả sử 3 điểm dữ liệu $(i-1, y_{i-1}), (i, y_0), (i+1, y_{i+1})$ thuộc một Parabola $P(x) = a x^2 + b x + c$. Điểm cực đại liên tục $x^*$ thỏa mãn $P'(x^*)=0$:
  $$x^* = i - \frac{b}{2a} = i + \frac{1}{2} \frac{y_{i+1} - y_{i-1}}{2 y_0 - y_{i-1} - y_{i+1}}$$
- **Ý nghĩa DSP:** Tăng độ phân giải thời gian của Peak lên cấp phân số (Sub-sample accuracy, ví dụ $< 1\text{ ms}$), khắc phục hạn chế độ phân giải $1\text{ ms}$ khi resample về $1000\text{ Hz}$.

### 30.5 Linear Regression through Origin cho Period Estimation (Mục 14.2)

- **Mô hình Bài toán:** Giả sử vị trí đỉnh thứ $i$ (với $i = 1, 2, \dots, K$) tuân theo đường thẳng đi qua gốc tọa độ: $p_i = T \cdot i + \epsilon_i$, với $\epsilon_i$ là nhiễu trắng.
- **Tối ưu hóa Least-Squares:** Cần tìm $T$ sao cho tối thiểu hóa tổng bình phương sai số $S(T) = \sum_{i=1}^K (p_i - T \cdot i)^2$. Cho đạo hàm $\frac{dS}{dT} = 0$:$$-2 \sum_{i=1}^K i (p_i - T \cdot i) = 0 \implies T = \frac{\sum_{i=1}^K i \cdot p_i}{\Dưới đây là khối bổ sung thông tin (viết dưới dạng Markdown codeblock) được thiết kế chuyên biệt nhằm lắp vào cuối tệp `.md` của bạn. 

Khối bổ sung này đóng vai trò như một **Theoretical & Mathematical Appendix (Phụ lục Toán học & Lý thuyết)**. Nó lấp đầy các khoảng trống về mặt bản chất toán học, lý giải *tại sao* thuật toán gốc lại thiết kế như vậy, cũng như làm rõ các điểm mù (corner cases, floating-point precision, FFT normalization) để một LLM khi nhận file này có thể lập trình tái dựng mà không bị mơ hồ.

```markdown
---

## 29. Supplement: Theoretical Framework & Mathematical Completeness

> **Mục đích bổ sung**: Phần này mở rộng cơ sở toán học (Mathematical Foundations) cho toàn bộ pipeline DSP, giải thích bản chất vật lý/tín hiệu của từng bước và bổ sung các tham số kỹ thuật còn thiếu để một LLM khi đảm nhận việc code không đưa ra các giả định sai.

### 29.1. Chi tiết cơ sở toán học (Mathematical Formulations)

#### A. Khối IIR Biquad Section (Direct Form I / II Transposed Analysis)
Phương trình sai phân dạng tổng quát được triển khai trong `FUN_00401580`:
$$y[n] = g \cdot x[n] + A_0 \cdot s[n-1] + A_1 \cdot s[n-2]$$
$$s[n] = g \cdot x[n] - A_2 \cdot s[n-1] - A_3 \cdot s[n-2]$$

Hàm truyền $H(z)$ biểu diễn trong miền Z:
$$H(z) = g \cdot \frac{1 + A_0 z^{-1} + A_1 z^{-2}}{1 + A_2 z^{-1} + A_3 z^{-2}}$$

* **Bản chất DSP**: Đây là cấu trúc cascaded biquad. Các nhóm section được ghép tầng để tạo thành bộ lọc dải thông (Band-pass filter) hoặc lọc dải dừng (Band-stop/Notch) nhằm cô lập các dải tần đặc trưng của onset (như dải âm trầm Kick/Snare).

#### B. Phép biến đổi Envelope Phi tuyến 5-pass (Nonlinear Energy Accumulation)
Tín hiệu sau mỗi lần qua IIR cascade được nâng lên bình phương và tích lũy:
$$\mathcal{E}_0[n] = 1.0$$
$$\mathcal{E}_{k}[n] = \mathcal{E}_{k-1}[n] \cdot \left(1.0 + \mathcal{H}_{16}\left( \left[ \mathcal{H}_{16}(x[n]) \right]^2 \right)\right) \quad \text{với } k = 1 \dots 5$$

* **Bản chất Toán học**: Đây là một bộ **Teager-Kaiser / Square Envelope Energy Extractor** đa tầng lặp lại. Việc bình phương ($x^2$) thực hiện dịch tần (frequency modulation) và tạo ra các thành phần tần số thấp (baseband energy) đại diện cho năng lượng nhịp. Việc nhân dồn $(1 + f_3)$ thay vì cộng dồn tạo ra cơ chế *gain modulation* phi tuyến theo tỷ lệ số nhân, làm nổi bật cực đại các vị trí có sự gia tăng năng lượng đột ngột (transient onset) và nén các đoạn tĩnh.

#### C. Backward Alternating-Difference Transform
Biến đổi ngược chuỗi từ cuối mảng $N$:
$$y[0] = x[N-2]$$
$$y[k] = x[N-2-k] - y[k-1] \quad \text{với } k = 1, \dots, N-2$$
$$y[N-1] = -x[N-2]$$

* **Bản chất Toán học**: Biến đổi này tương đương với một **Alternating First-order Infinite Impulse Response (IIR) High-Pass / Differentiator Filter** đảo chiều thời gian. Nó triệt tiêu thành phần DC offset tích lũy từ bước log-transform và nén biên độ nền, chỉ giữ lại các bước chuyển pha đột ngột (phase/energy discontinuities) để chuẩn bị cho FFT Autocorrelation.

#### D. Suy diễn Parabolic Peak Refinement
Từ 3 điểm $y_{i-1}, y_i, y_{i+1}$ quanh đỉnh địa phương $i$, đỉnh parabol thực $p$ được nội suy bằng cách đặt đạo hàm bằng 0 cho đa thức Lagrange bậc 2:
$$p = i + \Delta i, \quad \Delta i = \frac{1}{2} \cdot \frac{y[i+1] - y[i-1]}{2 y[i] - y[i-1] - y[i+1]}$$

#### E. Hồi quy Tuyến tính Chu kỳ (Least-Squares Period Fit)
Để tìm chu kỳ $T$ tối ưu từ danh sách các điểm cực đại thực tế $p_k \approx k \cdot T$ với $k \in \{1, 2, \dots, M\}$:
$$\min_T \sum_{k=1}^M (p_k - k \cdot T)^2 \implies T = \frac{\sum_{k=1}^M k \cdot p_k}{\sum_{k=1}^M k^2}$$
Độ lệch chuẩn residual (RMS Error):
$$\sigma_T = \sqrt{\frac{1}{M} \sum_{k=1}^M (p_k - k \cdot T)^2}$$

#### F. Tích lũy Gấp Pha (Phase Folding Accumulation)
Tín hiệu phân tích $S[n]$ được folded qua chu kỳ $T$ (chuyển sang miền phase bins $m \in [0, \lfloor T \rceil]$):
$$P[m] = \sum_{k=0}^{K-1} S\left( \lfloor k \cdot T + m + 0.5 \rfloor \right)$$
Offset được suy ra từ đỉnh của $P[m]$ qua parabolic refinement trừ đi độ trễ phần cứng/thuật toán cố định $\Delta_{\text{delay}} = 6.2 \text{ ms}$.

---

### 29.2. Bổ sung các điểm mù & Quy tắc triển khai cho LLM (Edge Cases & Missing Specifications)

Khi triển khai code thực tế, LLM **bắt buộc** phải tuân thủ các quy tắc xử lý biên dưới đây để tránh lệch kết quả với binary gốc:

1. **FFT Normalization & Padding**:
   * Chuỗi autocorrelation $R[\tau]$ phải sử dụng `Zero-Padding` tới $N_{\text{fft}} = 2^{\lceil \log_2(2N - 1) \rceil}$ để tránh hiện tượng hiện tượng quấn phổ (circular autocorrelation).
   * Không thực hiện scaling $1/N$ tại IFFT nếu muốn bảo toàn biên độ tuyệt đối cho peak thresholding.

2. **Xử lý số thực / Precision Rules**:
   * Toàn bộ pipeline IIR filter, Log transform, Resampler và Accumulator dùng kiểu **Float32 (Single Precision)** để khớp với thanh ghi SSE/x87 của binary gốc.
   * Tất cả các bước tính toán Hồi quy (Least-squares fit), Parabolic interpolation, và Phase folding accumulator phải chuyển sang **Float64 (Double Precision)** để tránh trôi biến vị trí khi bài hát có độ dài lớn (> 5 phút, $N > 300,000$ mẫu).

3. **Cửa sổ tìm kiếm Peak & Thresholds Default**:
   * `Peak Detection Minimum Threshold`: Chỉ xét các điểm $y[i] > \mu_y + 0.15 \cdot \sigma_y$ (trong đó $\mu_y, \sigma_y$ là trung bình và độ lệch chuẩn của chuỗi autocorrelation $R$).
   * `Search Window`: Cửa sổ tìm kiếm peak dự đoán $p_{k+1}$ trong bước Period Tracking là $p_k + T \pm \delta$ với $\delta = 10 \text{ ms}$ (tương đương $\pm 10$ samples ở Fs = 1000 Hz).

4. **Độ trễ Bộ lọc (Group Delay Compensation)**:
   * Con số hằng số `-6.2 ms` ở bước Offset chính là **Group Delay tổng cộng** của 16-section IIR cascade + Pre-resample filter + Post-resample filters tại tần số nhịp chủ đạo ($1 \sim 3 \text{ Hz}$).
   * Không tự ý điều chỉnh hằng số `-6.2 ms` này trừ khi thay đổi toàn bộ hệ số Filter Coefficient.

---

### 29.3. Sơ đồ ma trận dữ liệu (Data State Matrix)

| Giai đoạn | Tên tín hiệu | Kích thước / Shape | Kiểu dữ liệu | Tần số lấy mẫu ($F_s$) |
|---|---|---|---|---|
| Input | `pcm` | $(N,)$ | `float32` | 44100 Hz |
| Preprocessing | `acc` | $(N,)$ | `float32` | 44100 Hz |
| Log & Filter | `env` | $(N,)$ | `float32` | 44100 Hz |
| Resampled | `env_1k` | $(M,)$ với $M \approx \lceil N \cdot \frac{1000}{44100} \rceil$ | `float32` | ~1000 Hz |
| Backward xform | `analysis_sig` | $(M,)$ | `float32` | ~1000 Hz |
| Autocorrelation | `R` | $(N_{\text{fft}},)$ | `float64` | ~1000 Hz |
| Phase Folding | `phase_buf` | $(\lceil T \rceil + 2,)$ | `float64` | N/A (Phase domain) |

# ChatGPT
## 31. Verified Reverse-Engineering Addendum: Exact Detector Semantics & Corrections

> **Mục tiêu của phần này:** khóa lại những chi tiết mà binary hiện đã cho phép xác định trực tiếp, đồng thời ngăn implementation LLM tự thêm các giả định DSP không có trong binary.

### 31.1. Correction: Không gọi Envelope Transform là Teager-Kaiser

Phần preprocessing hiện **không có bằng chứng là Teager-Kaiser Energy Operator**.

Teager-Kaiser Energy Operator chuẩn có dạng:

$$
\Psi[x[n]] = x[n]^2 - x[n-1]x[n+1]
$$

Trong binary không xuất hiện tích $x[n-1]x[n+1]$ trong khối envelope. Thay vào đó, mỗi pass thực hiện:

1. Copy input.
2. Chạy cascade IIR 16 sections.
3. Bình phương từng sample.
4. Chạy lại chính cascade IIR.
5. Nhân accumulator pointwise với $(1+y[n])$.

Do đó implementation phải mô hình hóa:

$$
A_0[n] = 1
$$

$$
A_k[n]
=
A_{k-1}[n]
\left(
1 +
\mathcal{H}_{16}
\left(
\mathcal{H}_{16}(x)[n]^2
\right)
\right)
$$

với $k=1,\ldots,5$.

Không được thay thế khối này bằng Teager-Kaiser Operator chỉ vì cả hai đều thường được dùng cho energy/onset analysis.

---

### 31.2. Exact Biquad Section Model

`FUN_00401580` triển khai từng section theo tuple 5 số float:

[b0_like, b1_like, a1_like, a2_like, gain]

Mô hình tương đương:

$$
s[n]
=
g\,x[n]
-
A_2\,s[n-1]
-
A_3\,s[n-2]
$$

$$
y[n]
=
s[n]
+
A_0\,s[n-1]
+
A_1\,s[n-2]
$$

và transfer function:

$$
H(z)
=
g
\frac{
1 + A_0z^{-1} + A_1z^{-2}
}{
1 + A_2z^{-1} + A_3z^{-2}
}
$$

Tất cả arithmetic trong section phải giữ ở Float32 nếu mục tiêu là mô phỏng arithmetic của binary.

---

### 31.3. Coefficient Extraction Is a Mandatory Build Artifact

Toàn bộ coefficient table chính chưa được coi là đầy đủ cho implementation cuối cùng cho tới khi dump được toàn bộ raw 32-bit float.

Các vùng cần lấy:

```text
Main 16-section cascade:
    base = 0x004193f0
    count = 16
    stride = 0x14 bytes
    total = 16 * 20 = 0x140 bytes
    range = [0x004193f0, 0x0041952f]

Pre-resample section:
    base = 0x00419530
    count = 1
    stride = 0x14 bytes
    range = [0x00419530, 0x00419543]

Post-resample section:
    base = 0x00419544
    count = 1
    stride = 0x14 bytes
    range = [0x00419544, 0x00419557]
```

Assembly xác nhận các call:

```text
PUSH 0x4193f0
...
PUSH 0x419530
...
PUSH 0x419544
...
PUSH 0x419544
```

và `0x419544` được gọi hai lần liên tiếp sau resampling.

**Không được dùng các giá trị minh họa/truncated dump thay cho bảng đầy đủ.**

Implementation reference nên lưu coefficient dưới dạng:

```python
MAIN_SECTIONS: list[BiquadSection]  # len == 16
PRE_SECTION: BiquadSection
POST_SECTION: BiquadSection
```

và có unit test kiểm tra:

```python
assert len(MAIN_SECTIONS) == 16
```

---

## 31.4. Exact Autocorrelation FFT Size

`FUN_004017a0` tính:

$$
L = N + param_2
$$

sau đó:

$$
q =
\frac{\log_2(L-0.5)}{\log_2(2)}
$$

rồi chuyển $q$ sang integer bằng helper rounding/ceil path và đặt:

$$
N_{FFT} = 2^q
$$

Trong caller BPM hiện tại `param_2 = 0`, nên:

$$
N_{FFT}
=
2^{\left\lceil \log_2(N-0.5)\right\rceil}
$$

Với $N$ là integer dương, điều này về thực tế là power-of-two nhỏ nhất thỏa:

$$
N_{FFT} \ge N
$$

**Không được tự động thay bằng:**

$$
2^{\lceil\log_2(2N-1)\rceil}
$$

theo công thức linear-autocorrelation thông thường, vì binary không làm như vậy.

Pipeline FFT thực tế:

```text
real signal
    ↓
complex array [x[n], 0]
    ↓
forward FFT
    ↓
|X[k]|² = Re² + Im²
    ↓
inverse FFT
    ↓
divide first N samples by NFFT
    ↓
autocorrelation-like output R[τ]
```

Binary rõ ràng có bước:

```text
fVar9 = 1.0 / NFFT
output[i] = transformed[i] * fVar9
```

sau inverse transform.

Do đó implementation phải dùng:

```python
R = np.real(ifft(abs(fft(x, n=Nfft)) ** 2))
R *= 1.0 / Nfft
```

nếu mục tiêu là functional equivalence với scaling của binary.

---

## 31.5. Exact Peak Detection Predicate

Trong `FUN_00402280`, peak candidate được xét từ:

```text
i = 2
```

tới:

```text
i < min(2000, floor(N * 0.5))
```

với threshold:

```text
R[i] > 0.0
```

Một điểm $i$ chỉ được coi là local maximum nếu đồng thời:

$$
R[i] > 0
$$

$$
R[i-1] < R[i]
$$

$$
R[i+1] < R[i]
$$

và hai phía phải có xu hướng tăng vào peak:

$$
R[i-2] \le R[i-1]
\quad\text{với}\quad
R[i-1] \ne R[i-2]
$$

$$
R[i+2] \le R[i+1]
\quad\text{với}\quad
R[i+1] \ne R[i+2]
$$

Viết compact:

```python
is_peak = (
    R[i] > 0.0
    and R[i-1] < R[i]
    and R[i+1] < R[i]
    and R[i-2] <= R[i-1]
    and R[i-1] != R[i-2]
    and R[i+2] <= R[i+1]
    and R[i+1] != R[i+2]
)
```

Không được thay predicate này bằng:

```python
R[i] >= R[i-1] and R[i] >= R[i+1]
```

vì binary dùng strict inequality cho tâm và có thêm điều kiện ở vị trí ±2.

### 31.5.1. Lag Search Range

Do giới hạn:

```text
max_lag = min(2000, floor(N / 2))
```

và Fs của analysis domain xấp xỉ 1000 Hz:

```text
minimum detectable period ≈ 2 ms
maximum detectable period ≈ 2000 ms
```

Tương ứng nominal BPM range:

```text
upper bound ≈ 60000 / 2   = 30000 BPM
lower bound ≈ 60000 / 2000 = 30 BPM
```

Upper bound chỉ là giới hạn toán học của lag detector, không phải tuyên bố rằng phần mềm được thiết kế để nhận diện nhạc 30000 BPM.

---

## 31.6. Sub-sample Peak Refinement

Sau khi tìm được peak integer $i$, binary dùng:

$$
p
=
i
+
0.5
\frac{R[i+1]-R[i-1]}
{
(R[i]-R[i+1])+(R[i]-R[i-1])
}
$$

tương đương:

$$
p
=
i
+
\frac{1}{2}
\frac{R[i+1]-R[i-1]}
{2R[i]-R[i-1]-R[i+1]}
$$

Constant $0.5$ chính là global `DAT_0041a640`.

Giá trị $p$ được giữ ở Double trong phần detector.

---

## 31.7. Period Tracker: Candidate Initialization & Update

Peak detector không chỉ lấy trung bình khoảng cách giữa các peak.

Binary xây dựng một candidate sequence theo dạng:

```text
peak_1
→ predict next = current + T
→ search around prediction ±10 samples
→ accept local peak if present
→ update least-squares statistics
→ predict again
```

Khi tìm được peak thứ $j$:

```text
sum_i2 += j²
sum_ip += j * p_j
sum_p2 += p_j²
count   += 1
```

với:

```text
j = 1, 2, 3, ...
```

Estimate period:

$$
T =
\frac{\sum_j j\,p_j}
{\sum_j j^2}
$$

Estimate BPM:

$$
BPM = \frac{60000}{T}
$$

Binary thực hiện phép tương đương thông qua:

$$
BPM
=
60000
\frac{\sum_j j^2}
{\sum_j j\,p_j}
$$

### 31.7.1. Missing Peak Behavior

Nếu không tìm thấy peak trong cửa sổ prediction:

```text
±10 samples
```

binary tăng missing/gap counter và không thêm peak vào `sum_ip`/`sum_p2`, nhưng beat index kỳ vọng tiếp theo vẫn được tăng.

Candidate bị ngắt nếu trạng thái missing không còn đạt điều kiện tracker tiếp tục.

Do đó **không được** triển khai missing peak bằng cách:

```python
j -= 1
```

hoặc bằng cách nén lại toàn bộ index của các peak tiếp theo.

Correct conceptual model:

```python
expected_index += 1

if peak_found:
    sum_i2 += expected_index ** 2
    sum_ip += expected_index * peak_pos
    sum_p2 += peak_pos ** 2
    count += 1
else:
    gap_count += 1
```

---

## 31.8. Peak-Sequence Amplitude Consistency

Sau khi tracker thu được một candidate, binary tính:

$$
\bar A =
\frac{\sum_j R[p_j]}{M}
$$

và yêu cầu average peak đủ lớn so với seed/current peak.

Threshold quan sát được trong constant table là:

```text
0.7
```

tức logic tương đương:

$$
\bar A > 0.7 A_{seed}
$$

Candidate-strength bookkeeping tiếp tục sử dụng threshold:

```text
1.2
```

cho việc so sánh với candidate mạnh nhất.

Các constant này **không phải threshold RMS**.

---

## 31.9. Exact BPM Quality / Status Logic

Các quality thresholds hiện có thể map như sau:

### Minimum number of usable peaks

```text
count < 4  → status = -1
```

### Absolute period-fit residual threshold

Binary tính:

$$
\sigma
=
\sqrt{
\frac{
\sum p_j^2
-
\frac{(\sum j p_j)^2}{\sum j^2}
}{
M-1
}
}
$$

và kiểm tra:

```text
sigma > 2.4
    → status = 0x10
```

`2.4` là ngưỡng absolute residual trong analysis-sample units, xấp xỉ ms vì Fs ≈ 1000 Hz.

### Relative fit-error threshold

Binary còn tính một second-order normalized fit error, với ngưỡng:

```text
3e-5
```

Nếu vượt ngưỡng:

```text
status = 0x10
```

### High-confidence status

Nếu:

```text
gap_count == 0
and
sigma <= 0.6
```

thì:

```text
status = 0
```

Nếu fit đủ tốt nhưng:

```text
gap_count > 0
```

hoặc residual vượt ngưỡng high-confidence nhưng vẫn dưới hard-fail threshold:

```text
status = 1
```

Do đó trạng thái có ý nghĩa:

```text
-1  → không đủ dữ liệu/peak
 0  → fit rất tốt, không có gap
 1  → fit chấp nhận được nhưng có dấu hiệu không hoàn hảo
16  → fit không đủ tin cậy
```

---

## 31.10. Exact Phase Folding for Offset

Offset detector nhận BPM đã tìm được:

$$
T =
\frac{60000}{BPM}
$$

Đây là period tính bằng analysis samples/ms.

Binary xác định integer period size thông qua rounding helper.

Sau đó tạo phase accumulator:

```text
phase[m]
```

và với mỗi cycle:

$$
s_k = \operatorname{round}(kT+0.5)
$$

Trong source arithmetic, `$+0.5$` được đưa qua integer conversion path để tạo index nearest-integer.

Với mỗi cycle $k$, binary cộng các sample từ:

```text
signal[s_k + m]
```

vào cùng phase bin.

Functional model:

$$
P[m]
=
\sum_k
S[
\operatorname{round}(kT)+m
]
$$

với điều kiện index vẫn nằm trong input range.

### 31.10.1. Partial Final Cycle

Cycle cuối **không bị loại hoàn toàn** chỉ vì thiếu đủ một period.

Loop vẫn tiếp tục và chỉ bỏ qua các sample vượt quá input boundary.

Do đó implementation phải cho phép:

```text
full cycles
+
partial final cycle
```

thay vì chỉ fold:

```python
floor(N / T)
```

full periods.

---

## 31.11. Exact Offset Peak Search

Sau phase accumulation:

1. tìm phase bin lớn nhất;
2. parabolic refinement quanh bin đó;
3. subtract fixed correction.

Công thức:

$$
p =
i
+
0.5
\frac{P[i+1]-P[i-1]}
{(P[i]-P[i+1])+(P[i]-P[i-1])}
$$

Sau đó:

$$
offset = p - 6.2
$$

với:

```text
6.2 ms = 0x4018cccccccccccd
```

### Important Interpretation Rule

Binary chỉ chứng minh rằng `6.2` là một **fixed offset correction constant**.

Hiện chưa có bằng chứng đủ để ghi:

```text
6.2 ms == exact total group delay
```

cho toàn bộ filterbank tại mọi frequency.

Muốn gọi nó là group delay phải tính:

$$
\tau_g(\omega)
=
-\frac{d}{d\omega}\arg H(e^{j\omega})
$$

của toàn bộ pipeline và chứng minh rằng giá trị tại vùng rhythmic frequency phù hợp với 6.2 ms.

Cho tới khi làm phép kiểm chứng này, tài liệu nên gọi:

```text
fixed empirical/algorithmic offset correction
```

---

## 31.12. Resampler: Exact Index Rule

`FUN_00401b30` không thực hiện interpolation tuyến tính.

Với output position $n$:

$$
t_n
=
n\frac{F_{src}}{F_{dst}} + 0.5
$$

sau đó binary dùng floating-point integer-conversion trick để nhận integer source index.

Functional interpretation:

```python
src_index = round(n * Fs_src / Fs_dst)
```

và:

```python
if 0 <= src_index < N:
    out[n] = input[src_index]
else:
    out[n] = 0.0
```

Analysis target:

```text
Fs_dst = 1000 Hz
```

với input thường:

```text
Fs_src = 44100 Hz
```

nên mỗi output sample tương ứng xấp xỉ:

```text
1 ms
```

của analysis time.

Để đạt bit-equivalence, cần reproduce chính xác floating-point rounding path thay vì chỉ gọi Python `round()`.

---

## 31.13. Backward Alternating Transform: Exact Meaning

Sau hai post-resample IIR sections, binary không thực hiện:

```python
max(0, x[n] - x[n-1])
```

Thay vào đó là recurrence ngược:

$$
y_0 = x_{N-2}
$$

$$
y_k = x_{N-2-k} - y_{k-1},
\qquad
1 \le k < N-1
$$

và phần tử cuối sử dụng bitwise sign-bit operation tương đương phép đổi dấu đối với float hữu hạn thông thường.

Functional reference:

```python
y = zeros_like(x)

y[0] = x[N - 2]

for k in range(1, N - 1):
    y[k] = x[N - 2 - k] - y[k - 1]

y[N - 1] = -x[N - 2]
```

Bit-exact reference phải giữ:

```text
final sign flip = XOR 0x80000000
```

thay vì thay bằng arithmetic negation nếu muốn reproduce IEEE-754 bit pattern.

---

## 31.14. Input / Analysis Domain State Table

| Stage                     | Signal                   |            Approx. Fs | Storage                                           |                |         |
| ------------------------- | ------------------------ | --------------------: | ------------------------------------------------- | -------------- | ------- |
| Decode                    | mono PCM                 |              44100 Hz | float32                                           |                |         |
| Main 5-pass preprocessing | accumulator              |              44100 Hz | float32                                           |                |         |
| Log                       | log(accumulator)         |              44100 Hz | double internally → float32 storage               |                |         |
| Pre-resample filter       | filtered envelope        |              44100 Hz | float32                                           |                |         |
| Resample                  | analysis envelope        |               1000 Hz | float32                                           |                |         |
| Post-resample IIR ×2      | filtered analysis signal |               1000 Hz | float32                                           |                |         |
| Backward transform        | onset/rhythm signal      |               1000 Hz | float32                                           |                |         |
| FFT                       | complex spectrum         |        1000 Hz domain | float32 complex                                   |                |         |
| Power spectrum            | $                        |                     X | ^2$                                               | 1000 Hz domain | float32 |
| IFFT                      | autocorrelation          |        1000 Hz domain | float32                                           |                |         |
| Peak positions            | refined peaks            |        1000 Hz domain | float64                                           |                |         |
| Period fit                | $T$                      | ms / analysis samples | float64                                           |                |         |
| BPM                       | $60000/T$                |                   BPM | float64                                           |                |         |
| Phase fold                | phase accumulator        |          phase domain | float32/float64 depending on exact implementation |                |         |
| Offset                    | refined phase - 6.2      |                    ms | float64                                           |                |         |

---


## 31.15 — REVERSE-ENGINEERED GROUND TRUTH: IIR, DETECTOR CONSTANTS, CONVERSION SEMANTICS

This section supersedes the previous "Ghidra extraction requirements" checklist for items that have now been recovered from the binary.

The following values should be treated as authoritative for the v0.32.4-alpha reconstruction unless later instruction-level testing disproves them.

---

### 31.15.1 — Main 16-section IIR coefficient block

Memory layout:

- Base: `0x004193F0`
- Section count: `16`
- Section stride: `0x14` bytes = `5 * sizeof(float)`
- End address: `0x0041952F`
- Each section contains exactly five Float32 values:

[c0, c1, c2, c3, gain]

The reconstructed recurrence for one section is:

```text
s[n] = gain * x[n] - c2 * s[n-1] - c3 * s[n-2]

y[n] = s[n] + c0 * s[n-1] + c1 * s[n-2]
```

Equivalent transfer function:

```text
              gain * (1 + c0 z^-1 + c1 z^-2)
H(z) = --------------------------------------------
              1 + c2 z^-1 + c3 z^-2
```

The coefficients are Float32 constants and should be loaded using their exact raw bit patterns for Level-3 / bit-exact reconstruction.

#### Section 0

```text
addr 0x004193F0:
  c0   =  2.00000000   raw 0x40000000
  c1   =  1.00000000   raw 0x3F800000
  c2   = -1.96602499   raw 0xBFFBA6B5
  c3   =  0.967822254  raw 0x3F77C333
  gain =  0.000449319050 raw 0x39EB9295
```

#### Section 1

```text
addr 0x00419404:
  c0   =  2.00000000   raw 0x40000000
  c1   =  1.00000000   raw 0x3F800000
  c2   = -1.92228699   raw 0xBFF60D80
  c3   =  0.924044251  raw 0x3F6C8E2A
  gain =  0.000439323077 raw 0x39E654F2
```

#### Section 2

```text
addr 0x00419418:
  c0   =  0.00000000   raw 0x00000000
  c1   = -1.00000000   raw 0xBF800000
  c2   = -1.98691809   raw 0xBFFE5355
  c3   =  0.988819599  raw 0x3F7D2348
  gain =  0.0211971477  raw 0x3CADA5A4
```

#### Section 3

```text
addr 0x0041942C:
  c0   =  0.00000000   raw 0x00000000
  c1   = -1.00000000   raw 0xBF800000
  c2   = -1.97186232   raw 0xBFFC65FC
  c3   =  0.978765249  raw 0x3F7A905C
  gain =  0.0211971477  raw 0x3CADA5A4
```

#### Section 4

```text
addr 0x00419440:
  c0   =  0.00000000   raw 0x00000000
  c1   = -1.00000000   raw 0xBF800000
  c2   = -1.96395516   raw 0xBFFB62E2
  c3   =  0.966656804  raw 0x3F7776D2
  gain =  0.0209600348  raw 0x3CABB461
```

#### Section 5

```text
addr 0x00419454:
  c0   =  0.00000000   raw 0x00000000
  c1   = -1.00000000   raw 0xBF800000
  c2   = -1.95116699   raw 0xBFF9BFD7
  c3   =  0.955917597  raw 0x3F74B704
  gain =  0.0209600348  raw 0x3CABB461
```

#### Section 6

```text
addr 0x00419468:
  c0   =  0.00000000   raw 0x00000000
  c1   = -1.00000000   raw 0xBF800000
  c2   = -1.97018325   raw 0xBFFC2EF7
  c3   =  0.977743566  raw 0x3F7A4D67
  gain =  0.0420483202  raw 0x3D2C3ADC
```

#### Section 7

```text
addr 0x0041947C:
  c0   =  0.00000000   raw 0x00000000
  c1   = -1.00000000   raw 0xBF800000
  c2   = -1.93076444   raw 0xBFF7234A
  c3   =  0.958042085  raw 0x3F75423F
  gain =  0.0420483202  raw 0x3D2C3ADC
```

#### Section 8

```text
addr 0x00419490:
  c0   =  0.00000000   raw 0x00000000
  c1   = -1.00000000   raw 0xBF800000
  c2   = -1.92373407   raw 0xBFF63CEB
  c3   =  0.934358478  raw 0x3F6F321E
  gain =  0.0411380120  raw 0x3D288055
```

#### Section 9

```text
addr 0x004194A4:
  c0   =  0.00000000   raw 0x00000000
  c1   = -1.00000000   raw 0xBF800000
  c2   = -1.89517128   raw 0xBFF294F9
  c3   =  0.913750589  raw 0x3F69EB8F
  gain =  0.0411380120  raw 0x3D288055
```

#### Section 10

```text
addr 0x004194B8:
  c0   =  0.00000000   raw 0x00000000
  c1   = -1.00000000   raw 0xBF800000
  c2   = -1.92596865   raw 0xBFF68624
  c3   =  0.955819964  raw 0x3F74B09E
  gain =  0.0827306286  raw 0x3DA96EAD
```

#### Section 11

```text
addr 0x004194CC:
  c0   =  0.00000000   raw 0x00000000
  c1   = -1.00000000   raw 0xBF800000
  c2   = -1.81211865   raw 0xBFE7F381
  c3   =  0.918312252  raw 0x3F6B1683
  gain =  0.0827306286  raw 0x3DA96EAD
```

#### Section 12

```text
addr 0x004194E0:
  c0   =  0.00000000   raw 0x00000000
  c1   = -1.00000000   raw 0xBF800000
  c2   = -1.83145261   raw 0xBFEA6D0A
  c3   =  0.872521520  raw 0x3F5F5D92
  gain =  0.0793709233  raw 0x3DA28D39
```

#### Section 13

```text
addr 0x004194F4:
  c0   =  0.00000000   raw 0x00000000
  c1   = -1.00000000   raw 0xBF800000
  c2   = -1.76369274   raw 0xBFE1C0AF
  c3   =  0.834738493  raw 0x3F55B16C
  gain =  0.0793709233  raw 0x3DA28D39
```

#### Section 14

```text
addr 0x00419508:
  c0   = -2.00000000   raw 0xC0000000
  c1   =  1.00000000   raw 0x3F800000
  c2   = -1.66992509   raw 0xBFD5C01B
  c3   =  0.772546172  raw 0x3F45C596
  gain =  0.860617816  raw 0x3F5C5173
```

#### Section 15

```text
addr 0x0041951C:
  c0   = -2.00000000   raw 0xC0000000
  c1   =  1.00000000   raw 0x3F800000
  c2   = -1.43855608   raw 0xBFB8229B
  c3   =  0.526959062  raw 0x3F06E6CA
  gain =  0.741378784  raw 0x3F3DCB00
```

Therefore the complete main filter is:

```text
16 sections × 5 Float32 = 80 Float32 values = 320 bytes
```

---

### 31.15.2 — Pre-resample IIR section

Base:

```text
0x00419530
```

Five Float32 values:

```text
c0   =  0.00000000   raw 0x00000000
c1   =  0.00000000   raw 0x00000000
c2   = -1.94799995   raw 0xBFF95810
c3   =  0.948099971  raw 0x3F72B6AE
gain =  0.0480000004 raw 0x3D449BA6
```

Functional form:

```text
s[n] = 0.0480000004*x[n]
       - (-1.94799995)*s[n-1]
       - 0.948099971*s[n-2]

y[n] = s[n]
```

This is applied once after the nonlinear/log stage and before resampling.

---

### 31.15.3 — Post-resample IIR section

Base:

```text
0x00419544
```

Five Float32 values:

```text
c0   =  0.00000000   raw 0x00000000
c1   =  0.00000000   raw 0x00000000
c2   = -1.39999998   raw 0xBFB33333
c3   =  0.479999989  raw 0x3EF5C28F
gain =  0.200000003   raw 0x3E4CCCCD
```

Functional form:

```text
s[n] = 0.200000003*x[n]
       - (-1.39999998)*s[n-1]
       - 0.479999989*s[n-2]

y[n] = s[n]
```

This exact one-section filter is applied twice consecutively after the 1 kHz resampling stage.

---

### 31.15.4 — Verified detector constants

The following constants have direct call-site evidence in the BPM detector.

#### Floating-point rounding/index helper constants

```text
0x0041A640 = 0.5
0x0041A648 = 1.0
0x0041A650 = 4503599627370496.0   // 2^52
0x0041A658 = 0x8000000000000000   // sign-bit mask as 64-bit value
0x0041A670 = 0.0
```

These are not arbitrary DSP coefficients. They participate in the binary's explicit floating-point conversion / nearest-integer construction.

The common transformation is semantically equivalent to:

```text
nearest_integer_like_binary(x)
```

implemented using IEEE-754 bit masking and correction, rather than a normal language-level `round()`.

For Level-3 reconstruction, preserve the actual operation sequence.

---

#### BPM scale / validation constants

```text
0x0041A6E8 = 60000.0
0x0041A6F8 = 1.0e-30
0x0041A728 = 0.6
0x0041A730 = 3.0e-5
0x0041A738 = 2.4
0x0041A740 = 20.0
0x0041A748 = 285.7142857142857
```

Known semantics:

```text
60000.0       -> milliseconds per minute
1.0e-30       -> minimum accepted BPM / candidate guard
0.6           -> low regression sigma threshold for status 0
3.0e-5        -> normalized fit consistency threshold
2.4           -> absolute sigma rejection threshold
20.0          -> used in secondary candidate-selection logic
285.714285714 -> used in secondary candidate-selection logic
```

The last two values (`20.0`, `285.7142857142857`) are confirmed constants at the detector call sites, but their higher-level heuristic meaning should not be invented until the surrounding helper logic is completely decoded.

---

#### Candidate-strength constants

At the raw Float32 addresses:

```text
0x0041A750 = 1.2f
0x0041A754 = 0.7f
```

Semantics:

```text
seed_peak_amplitude * 0.7 < average_tracked_peak_amplitude
```

and:

```text
new_average > current_best_average * 1.2
```

are used during candidate bookkeeping.

Important:

`0x0041A750` and `0x0041A754` are Float32 constants located four bytes apart.

Do NOT reinterpret the 8 bytes starting at `0x0041A750` as one meaningful Float64 constant. The overlapping Float64 interpretation produced by a raw memory dump is irrelevant here.

---

### 31.15.5 — Exact BPM seed-peak search domain

For input analysis vector length `N`:

```text
N = number of Float32 samples passed into FUN_00402280
```

The detector computes:

```text
scan_limit = min(N, 2000)
```

The local maximum scan begins at lag/index:

```text
i = 2
```

and continues while:

```text
i < scan_limit - 2
```

Peak predicate is exactly:

```python
R[i] > 0.0

R[i-1] < R[i]
R[i+1] < R[i]

R[i-2] <= R[i-1]
R[i-1] != R[i-2]

R[i+2] <= R[i+1]
R[i+1] != R[i+2]
```

There is NO evidence for a dynamic threshold such as:

```text
mean + 0.15 * std
```

The actual floor is exactly `0.0`.

---

### 31.15.6 — Important seed restriction before tracking

Although the detector scans candidate peaks out to:

```text
min(N, 2000)
```

the candidate-processing loop explicitly stops considering a seed once:

```text
seed_peak_index > 999
```

because the candidate-processing branch contains:

```text
if (999 < seed_peak_index)
    break;
```

Therefore:

```text
initial peak scan range:
    approximately 2 .. min(N, 2000)

usable tracker seed range:
    peak_index <= 999
```

This distinction must be preserved.

It means that the detector's eventual tracked solution is not equivalent to simply accepting every local maximum in the full 0..2000 search range.

---

### 31.15.7 — Parabolic peak interpolation

For a local peak at integer index `i`:

```python
delta = 0.5 * (R[i+1] - R[i-1]) / (
    (R[i] - R[i+1]) +
    (R[i] - R[i-1])
)

peak_position = i + delta
```

The same mathematical interpolation form is reused by the offset detector.

Do not replace this with a generic quadratic-fit library routine if bit-level reproducibility is required.

---

### 31.15.8 — Tracker prediction and ±10 search

For each candidate seed:

```text
beat_index j starts at 1
sum_i2 = 0
sum_ip = 0
sum_p2 = 0
tracked_count = 0
```

After a candidate peak is accepted:

```python
sum_i2 += j * j
sum_ip += j * peak_position
sum_p2 += peak_position * peak_position

tracked_count += 1

fitted_period = sum_ip / sum_i2
```

The next predicted integer peak location is constructed approximately as:

```python
predicted = round(
    current_peak_position
    + fitted_period
)
```

The implementation does this using the binary's floating-point nearest-integer helper.

The actual candidate search window is:

```text
predicted_index - 10
    .. 
predicted_index + 10
```

A matching detected peak in that window is accepted.

When a peak is found:

```python
current_peak = refined_detected_peak
```

When no peak is found:

```text
gap/miss counter increments
expected beat index still advances
tracking may terminate after the binary's allowed missing-peak condition
```

No separate minimum-distance peak suppression stage was identified.

---

### 31.15.9 — Regression and BPM

The fitted period is:

```text
T_samples = sum_ip / sum_i2
```

Then:

```text
BPM = 60000.0 / T_samples
```

Equivalently:

```text
BPM = 60000.0 * sum_i2 / sum_ip
```

The binary therefore fits:

```text
peak_position ≈ T * beat_index
```

through the origin.

There is no fitted intercept in this regression.

---

### 31.15.10 — Amplitude consistency

For every candidate sequence:

```text
average_peak_amplitude =
    sum_of_tracked_peak_amplitudes / tracked_count
```

The candidate survives amplitude consistency when:

```text
seed_peak_amplitude * 0.7 < average_peak_amplitude
```

The candidate score/bookkeeping also tracks the highest average amplitude.

A new candidate becomes the current amplitude leader when approximately:

```text
average_peak_amplitude > current_best_average * 1.2
```

The exact candidate-selection structure around this logic should be retained, rather than replacing it with a generic "pick strongest BPM" routine.

---

### 31.15.11 — BPM quality/status logic

After tracking:

```text
tracked_count < 4
    => return status -1
```

For accepted sequences:

```text
sigma =
sqrt(
    (
        sum_p2
        - sum_ip^2 / sum_i2
    )
    /
    (tracked_count - 1)
)
```

Absolute rejection:

```text
sigma > 2.4
    => status 0x10
```

Secondary normalized-fit rejection:

```text
normalized_fit_error > 3.0e-5
    => status 0x10
```

Good-fit condition:

```text
no gap
and
sigma <= 0.6
    => status 0
```

Otherwise the candidate can remain accepted with imperfect quality:

```text
=> status 1
```

The detector therefore exposes at least the following observed statuses:

```text
-1  = insufficient / unusable tracking
 0  = strong / clean result
 1  = accepted but imperfect
0x10 = poor-quality fit / rejected-quality result
```

Do not collapse these states into a single Boolean success/failure value.

---

### 31.15.12 — FUN_00402C70 period folding for offset detection

Input:

```text
candidate BPM
```

Initial guard:

```text
BPM < 1.0e-30
    => return -1
```

Period in milliseconds:

```text
T_ms = 60000.0 / BPM
```

The implementation invokes the binary's `ceil()` helper before constructing the internal phase buffer:

```text
ceil(T_ms)
```

and then converts that result using `FUN_00413f70`.

The internal working period therefore depends on the exact conversion path, not only on the mathematical real-valued `T_ms`.

---

### 31.15.13 — Exact phase-folding behavior

The detector computes the number of cycles using the binary's `ceil()` operation:

```text
cycle_count = ceil(N_samples / T_ms)
```

For cycle index `k`:

```text
sample_base ≈ round(k * T_ms)
```

The actual implementation creates the integer base index using the same IEEE-754 nearest-integer construction described above.

Samples from each cycle are accumulated into a phase buffer.

Out-of-range samples are naturally excluded by the source-index bounds.

The final partial cycle is therefore INCLUDED when the source still contains valid samples.

---

### 31.15.14 — Phase search domain

The phase buffer uses an internal offset of `0x23` elements:

```text
phase_base = 0x23
```

The search covers one full period-sized phase region:

```text
0x23 .. 0x23 + period_integer - 1
```

The maximum phase location is found by ordinary greater-than comparison.

There is no evidence of an additional dynamic amplitude threshold in the phase search.

The source performs an optimized block-of-four scan for larger buffers and a scalar tail scan.

---

### 31.15.15 — Offset parabolic interpolation

For winning phase bin `i`:

```python
delta = 0.5 * (P[i+1] - P[i-1]) / (
    (P[i] - P[i+1]) +
    (P[i] - P[i-1])
)

phase_peak = i + delta
```

Final returned offset:

```text
offset_ms = phase_peak - 6.2
```

where:

```text
0x004193E0 = 6.2
```

Important:

`6.2 ms` is a fixed algorithmic correction term.

Do NOT currently describe it as "the exact total group delay of the filter chain".

The binary proves that `6.2` is subtracted here. It does not by itself prove that the number was derived analytically from the total group delay.

---

### 31.15.16 — FUN_00405DB0 / ceil semantics

`FUN_00405DB0(double)` is the binary's CRT/library implementation of `ceil()`.

It is used by:

```text
FUN_004017A0
FUN_00401B30
FUN_00402C70
FUN_00402FC0
```

For ordinary finite values, reconstruction should treat it as:

```python
ceil(x)
```

However, the original implementation contains CRT floating-point exception handling for NaN/infinity and therefore should not be replaced by an uncontrolled custom integer conversion in Level-3 work.

---

### 31.15.17 — FUN_00413F70 conversion helper

`FUN_00413F70` is a compiler/runtime floating-point-to-integer conversion helper used by:

```text
FUN_004017A0
FUN_00402C70
FUN_00402FC0
```

The decompile contains two execution paths depending on:

```text
DAT_010212E8
```

Observed semantics include:

```text
DAT_010212E8 == 0:
    use the binary's ROUND/x87-based conversion path
    followed by explicit correction logic

DAT_010212E8 != 0:
    the relevant machine-code branch uses:

        FSTP double [stack]
        CVTTSD2SI EAX, [stack]

    i.e. conversion of the stored Float64 value to integer
    using truncate-toward-zero machine semantics.
```

Therefore an implementation must NOT blindly replace all calls with:

```python
int(x)
```

or:

```python
round(x)
```

without deciding which original runtime path is active.

For Level-2 functional reconstruction, implement the observable nearest-integer behavior used by the detector.

For Level-3 bit-exact reconstruction, reproduce the actual runtime/control-state path.

---

### 31.15.18 — FUN_00401580 floating-point control side effect

`FUN_00401580` explicitly calls:

```text
__controlfp(0x1000000, 0x3000000)
```

before processing its IIR sections, and:

```text
__controlfp(0x9001F, 0x3000000)
```

after processing.

Therefore the function is not merely:

```text
filter(data)
```

It also temporarily changes the floating-point environment.

This matters for Level-3 reconstruction because:

* rounding mode / precision behavior may affect intermediate values;
* the original program uses legacy x87 / CRT floating-point behavior;
* repeated IIR stages amplify tiny differences.

A modern implementation that only reproduces the algebra but ignores floating-point environment differences should be classified as:

```text
Level 2 — algorithmically equivalent
```

rather than:

```text
Level 3 — bit-exact
```

---

### 31.15.19 — Correct distinction between Float32 and Float64 constants

The reverse-engineered binary contains overlapping data regions.

Therefore each constant MUST be classified by the instruction that consumes it.

Examples:

```text
0x0041A750 = 1.2f
0x0041A754 = 0.7f
```

are Float32 constants.

Whereas:

```text
0x0041A640 = 0.5
0x0041A648 = 1.0
0x0041A650 = 2^52
0x0041A6E8 = 60000.0
0x0041A6F8 = 1e-30
```

are consumed as Float64/extended-precision numerical constants in the relevant code paths.

Do not infer type solely from an 8-byte memory dump.

The instruction operand and promotion path determine the effective type.

---

### 31.15.20 — Remaining unknowns after the Gemini ground-truth dump

The following items remain candidates for further reverse engineering, but they are no longer blockers for implementing the core detector:

```text
1. Exact active floating-point environment at program entry and at
   every FUN_00401580 call.

2. Complete source-level semantics of FUN_0041438A /
   __cintrindisp2 in the secondary BPM candidate-selection branch.

3. Exact rationale of constants:
      20.0
      285.7142857142857

4. Exact vector capacity/guard allocation layout around the
   phase-folding buffer in FUN_00402C70.

5. Exact behavior for pathological lengths such as:
      N < 2
      N < 3
      N < 4
      zero-length input
      NaN / Inf
      extremely large BPM
      extremely small BPM

6. Exact compiler/runtime rounding behavior if strict binary
   reproducibility rather than algorithmic equivalence is required.
```

These are second-order reconstruction tasks.

They should NOT cause the implementation of the already-established pipeline to be postponed.

---

### 31.15.21 — Current reconstruction confidence

```text
HIGH CONFIDENCE
---------------
- 16-section IIR memory layout
- all 16 main IIR coefficient sets
- pre-resample IIR coefficients
- post-resample IIR coefficients
- 0.0 peak threshold
- scan_limit = min(N, 2000)
- seed restriction <= 999
- ±10 tracker search window
- parabolic peak interpolation
- regression through origin
- BPM = 60000 / fitted_period
- amplitude threshold 0.7
- candidate-strength factor 1.2
- sigma thresholds 0.6 and 2.4
- normalized-fit threshold 3e-5
- minimum tracked peaks = 4
- phase-folding structure
- 6.2 ms final correction

MEDIUM CONFIDENCE
-----------------
- exact high-level interpretation of secondary candidate-selection
  constants 20.0 and 285.7142857142857
- exact compiler/runtime FP conversion semantics under every
  environment state
- exact vector allocation/guard representation

DO NOT CLAIM WITHOUT FURTHER PROOF
----------------------------------
- 6.2 ms == analytically derived total filter group delay
- filterbank channel labels / exact named filter topology
- Teager-Kaiser energy detector
- dynamic mean/std peak threshold
- zero-intercept regression as a "design choice" rather than
  simply the implemented least-squares formula
- NFFT = 2^(ceil(log2(2N-1)))
- unnormalized inverse FFT
```

---

### 31.15.22 — Implementation consequence

At this point the implementation can proceed without additional Ghidra extraction for the main algorithm.

The recommended architecture is:

```text
decode 44.1 kHz PCM
        |
        v
mono sum
        |
        v
5 × {
    copy
    16-section IIR cascade
    square
    16-section IIR cascade
    accumulator *= (1 + filtered)
}
        |
        v
FUN_00413958 logarithmic transform
        |
        v
1-section IIR @ 0x419530
        |
        v
44.1 kHz -> 1000 Hz nearest-neighbor resample
        |
        v
1-section IIR @ 0x419544
        |
        v
1-section IIR @ 0x419544
        |
        v
backward alternating transform
        |
        v
FFT
        |
        v
|FFT|²
        |
        v
IFFT
        |
        v
explicit 1/NFFT normalization
        |
        v
local maxima
        |
        v
parabolic refinement
        |
        v
peak tracker
        |
        v
least-squares period fit through origin
        |
        v
BPM
        |
        +----> phase folding
                   |
                   v
             phase maximum
                   |
                   v
             parabolic refinement
                   |
                   v
             offset = phase - 6.2 ms
```

The remaining unknowns should be isolated behind small compatibility functions, so they can later be replaced without changing the main pipeline.

## 31.16. Recommended Implementation Order

LLM không nên code toàn bộ pipeline trong một lần.

Thứ tự khuyến nghị:

```text
1. PCM decode + mono
2. Exact Biquad section
3. Exact 16-section cascade
4. Five-pass accumulator
5. Log/range-reduction implementation
6. Pre-resample IIR
7. Resampler
8. Post-resample IIR ×2
9. Backward transform
10. FFT + power spectrum + IFFT + 1/NFFT normalization
11. Peak predicate
12. Peak interpolation
13. Period tracker
14. BPM quality/status
15. Phase folding
16. Offset interpolation
17. Fixed -6.2 ms correction
18. Timeline validation
```

Mỗi bước phải có intermediate dump:

```text
raw PCM
→ after IIR #1
→ after square
→ after IIR #2
→ accumulator after each of 5 passes
→ after log
→ after pre-filter
→ after resample
→ after post-filter #1
→ after post-filter #2
→ after backward transform
→ autocorrelation
→ peak index list
→ refined peak list
→ period T
→ BPM
→ phase buffer
→ offset
```

Chỉ so sánh BPM cuối cùng là không đủ để debug.

---

## 31.17. Definition of Reconstruction Fidelity

Phân biệt ba cấp độ:

### Level 1: Functional equivalent

Có thể dùng:

```text
standard audio decoder
standard FFT
standard float operations
approximate log
approximate resampling
```

miễn là BPM/offset hoạt động tương tự.

### Level 2: Algorithmically equivalent

Phải khớp:

```text
filter topology
all coefficients
5-pass nonlinear accumulation
log structure
resampling rule
FFT size
IFFT normalization
peak predicate
tracker
quality thresholds
phase folding
6.2 ms correction
```

### Level 3: Bit-exact / binary-compatible

Ngoài Level 2 phải khớp:

```text
Float32 vs Float64 boundaries
x87/SSE conversion
rounding mode
integer conversion helpers
IEEE-754 bit tricks
NaN/Inf/subnormal handling
exact LUT entries
exact memory layout
exact loop boundaries
exact operation ordering
```

Một implementation chỉ đạt Level 1/2 không được gọi là "bit-exact".


# 32. GROUND-TRUTH CONSOLIDATION ROUND 3

This section records the additional algorithmic facts established from the latest Ghidra instruction extraction.

It supersedes only the specific earlier interpretations explicitly marked below.

The purpose is to resolve implementation-contract ambiguity, not to add theoretical DSP interpretation.

---

## 32.1 — Correct 5-pass filter architecture

### STATUS: PROVEN

The previous interpretation:

```text
for pass in 0..5:
    run all 16 IIR sections
    square
    run all 16 IIR sections
    accumulate
````

is INCORRECT.

The assembly shows that the first `FUN_00401580` call receives:

```text
coefficient_base =
    0x004193F0 + start_section * 0x14

section_count =
    end_section - start_section
```

where the outer-loop boundary table is:

```text
[0, 2, 6, 10, 14, 16]
```

Therefore the five outer iterations are:

```text
Pass 0:
    sections  0..1
    base = 0x004193F0
    count = 2

Pass 1:
    sections  2..5
    base = 0x00419418
    count = 4

Pass 2:
    sections  6..9
    base = 0x00419468
    count = 4

Pass 3:
    sections 10..13
    base = 0x004194B8
    count = 4

Pass 4:
    sections 14..15
    base = 0x00419508
    count = 2
```

Each pass then performs:

```text
copy original input -> temporary buffer
apply selected IIR group
square every sample
apply ONE IIR section at 0x00419530
accumulator[i] *= 1.0 + filtered[i]
```

The second filter inside each pass is NOT another copy of the selected filter group.

It is always:

```text
base  = 0x00419530
count = 1
```

The outer accumulation therefore represents five distinct spectral/filter groups.

### Correct abstract structure

```python
bounds = [0, 2, 6, 10, 14, 16]
acc[:] = 1.0

for k in range(5):
    tmp = copy(input_signal)

    start = bounds[k]
    end   = bounds[k + 1]

    iir_in_place(
        tmp,
        coeff_base = 0x004193F0 + start * 0x14,
        section_count = end - start
    )

    tmp[:] = tmp * tmp

    iir_in_place(
        tmp,
        coeff_base = 0x00419530,
        section_count = 1
    )

    acc[:] *= 1.0 + tmp
```

After all five passes:

```text
acc -> FUN_00413958 element-wise
```

followed by:

```text
2-section IIR:
    base = 0x004193F0
    count = 2
```

This 2-section filter is a separate post-log stage and is NOT part of the five-pass accumulator loop.

---

## 32.2 — IIR state reset semantics

### STATUS: PROVEN

Every invocation of `FUN_00401580` initializes its recursive local state to zero:

```text
previous state = 0
second previous state = 0
```

The state variables are local to one function invocation.

Therefore:

```text
state does not persist:
    - between IIR sections
    - between separate FUN_00401580 calls
    - between outer passes
    - between the five accumulator passes
```

Every `FUN_00401580` call starts with zero recursive state.

This is important when rewriting the filter as SIMD/vectorized code.

A library call that carries filter state from one call into the next is NOT equivalent.

---

## 32.3 — Exact bootstrap of Period Tracker

### STATUS: PROVEN

For each seed peak:

```text
seed peak index = p0
```

the tracker initializes:

```text
tracked_count = 0
gap_count = 0

beat_index = 1

sum_i2 = 0
sum_ip = 0
sum_p2 = 0
```

The search position initially starts from the seed's own position in the peak vector.

Because the current prediction is the seed itself, the seed is accepted as the first tracked peak.

After accepting the seed:

```python
tracked_count += 1

sum_i2 += 1 * 1
sum_ip += 1 * p0_refined
sum_p2 += p0_refined * p0_refined

T = sum_ip / sum_i2
```

Therefore after the first accepted point:

```text
T = p0_refined
```

There is no separate externally supplied initial BPM or initial period.

### First prediction

The next prediction is generated from:

```text
current_peak + fitted_period
```

followed by the binary's nearest-integer construction.

Since:

```text
current_peak = p0_refined
fitted_period = p0_refined
```

the first predicted location is approximately:

```text
2 * p0_refined
```

This explains why assigning the seed to `j=0` would be incorrect.

The binary effectively begins the regression at:

```text
j = 1
```

---

## 32.4 — Seed vector and candidate enumeration

### STATUS: PROVEN

The initial local-peak detector appends every peak satisfying the exact peak predicate.

The append operation occurs in increasing scan-index order.

Therefore the peak vector is sorted ascending by integer peak index.

There is no Top-K selection before tracker launch.

The candidate loop processes the vector sequentially.

The loop contains:

```c
if (999 < peak_index)
    break;
```

Therefore:

```text
all detected peaks <= 999:
    eligible to become tracker seeds

first peak > 999:
    stops seed processing entirely
```

Because the vector is sorted, the BREAK also prevents every later peak from being considered.

There is no evidence of a second pass over peaks >999 inside `FUN_00402280`.

---

## 32.5 — Tie-breaking inside the ±10 tracker window

### STATUS: PROVEN

The tracker does NOT search all peaks inside the window and select the maximum-amplitude one.

The search begins from the current vector position and advances forward.

For each candidate entry:

```text
if peak_index < predicted - 10:
    advance

else if peak_index <= predicted + 10:
    accept this peak

else:
    no candidate found
```

Therefore, when multiple peak entries are inside the ±10 interval, the binary selects:

```text
the FIRST peak entry in ascending vector order
that satisfies the window constraints
```

It does not compare:

```text
peak amplitude
distance to prediction
refined distance
```

after multiple matches have been found.

Consequences:

```text
Earlier candidate in the vector wins.

A weaker earlier peak can beat a stronger later peak.

A farther earlier peak can beat a closer later peak.
```

This is a direct implementation property, not a heuristic interpretation.

---

## 32.6 — Exact missing-peak behavior

### STATUS: PROVEN

The tracker maintains:

```text
gap_count
```

When no peak is found inside the ±10 window:

```text
gap_count += 1
```

The tracker then exits the search loop because the top of the next tracker iteration contains:

```text
if gap_count > 0:
    break
```

Therefore:

```text
maximum consecutive missing peaks = 1

but the tracker does NOT continue after the miss
```

The missing peak is effectively a terminal gap.

The candidate may still pass later quality logic as a gapped-but-accepted result.

This explains the distinction:

```text
no gap
    -> potentially status 0

one terminal gap
    -> potentially status 1
```

depending on the remaining quality criteria.

There is no evidence that two or three consecutive missing peaks are tolerated.

---

## 32.7 — Tracker prediction after a successful match

### STATUS: PROVEN

For each accepted peak:

```python
beat_index += 1

sum_i2 += beat_index * beat_index
sum_ip += beat_index * peak_position
sum_p2 += peak_position * peak_position

fitted_period = sum_ip / sum_i2
```

The next predicted peak is then constructed approximately as:

```python
predicted = nearest_integer_like(
    current_peak + fitted_period
)
```

The nearest-integer construction is the binary's IEEE-754 conversion sequence, not a language-level `round()` call.

On a miss:

```text
gap_count becomes 1
```

the tracker exits.

The missed point is not incorporated into the regression statistics.

---

## 32.8 — Exact normalized fit expression

### STATUS: PROVEN

The source computes:

```text
BPM = 60000 * sum_i2 / sum_ip
```

and later evaluates a normalized fit quantity whose algebra reduces to:

```text
normalized_fit_error =
sqrt(
    (
        sum_p2 * sum_i2 / (sum_ip * sum_ip)
        - 1
    )
    /
    (tracked_count - 1)
)
```

Equivalent form:

```text
normalized_fit_error =
sqrt(
    (sum_p2 * sum_i2 / sum_ip^2 - 1)
    / (tracked_count - 1)
)
```

The source additionally multiplies/divides by BPM in an unreduced implementation form.

That BPM factor cancels algebraically.

### Quality thresholds

```text
tracked_count < 4
    -> status -1

sigma > 2.4
    -> status 0x10

normalized_fit_error > 3e-5
    -> status 0x10

no gap AND sigma <= 0.6
    -> status 0

otherwise
    -> status 1
```

---

## 32.9 — Interpretation of 999 seed limit

### STATUS: HIGH CONFIDENCE

`seed_index <= 999` is a real implementation limit of the tracker seed mechanism.

At an analysis rate of approximately 1 kHz:

```text
999 samples ~ 999 ms
```

So a fundamental period significantly greater than ~1 second cannot directly initialize the tracker through a seed peak >999.

However:

```text
DO NOT declare a strict BPM floor of 60000/999
```

because a slower musical tempo may still generate a harmonic/subharmonic candidate with a shorter period that enters the allowed seed range.

What is proven is only:

```text
FUN_00402280 does not directly seed the tracker from a peak index >999.
```

The existence of a separate slow-tempo rescue mechanism has not been established.

---

## 32.10 — Exact resampling index semantics

### STATUS: PROVEN FOR THE NORMAL POSITIVE RANGE

The resampler computes:

```text
source_position =
    output_index * (Fs_src / Fs_dst) + 0.5
```

and then performs the binary's positive floating-point integer extraction.

For the normal positive range relevant to audio resampling, this is equivalent to:

```python
source_index = floor(
    output_index * Fs_src / Fs_dst + 0.5
)
```

For:

```text
Fs_src = 44100
Fs_dst = 1000
```

this becomes:

```python
source_index = floor(
    output_index * 44.1 + 0.5
)
```

This is NOT Python's:

```python
round(...)
```

and NOT bankers rounding.

The resampler then performs:

```text
if source_index < input_length:
    output[n] = input[source_index]
else:
    output[n] = 0
```

No interpolation is performed.

### Output length

The output length is based on:

```text
ceil(
    input_length * Fs_dst / Fs_src
)
```

using the binary's `ceil` implementation.

---

## 32.11 — FFT/autocorrelation boundary model

### STATUS: PROVEN

`FUN_004017A0` determines:

```text
NFFT =
    smallest power of two >= N
```

for the BPM detector's `param_2 = 0` path.

The input is copied as:

```text
real[n] = input[n]
imag[n] = 0
```

No explicit mean-subtraction stage exists in this function before the FFT.

Then:

```text
FFT
-> Re^2 + Im^2
-> imag = 0
-> inverse FFT
-> multiply each output by 1/NFFT
```

The explicit normalization is:

```text
scale = 1.0 / NFFT
```

Therefore an implementation that omits the inverse scaling is incorrect.

### Output length

For the BPM caller:

```text
extra length parameter = 0
```

so the returned autocorrelation vector corresponds to the original analysis length rather than an explicit `2N-1` linear-correlation buffer.

There is no evidence in this function for:

```text
NFFT = next_power_of_two(2N-1)
```

or an equivalent mandatory doubling scheme.

---

## 32.12 — Backward alternating transform is NOT simple time reversal

### STATUS: PROVEN

The transform should not be described as:

```text
reverse(signal)
```

and should not be described as a simple first difference.

The observed structure is:

```text
y[0] = x[N-2]

y[k] = x[N-2-k] - y[k-1]
       for 1 <= k < N-1

y[N-1] = -y[0]
```

where the final negation is implemented by XORing the IEEE-754 sign bit.

Therefore this is a reverse-index recurrence.

It is better described operationally as:

```text
reverse-access + alternating recursive transform
```

than as pure time reversal.

### Important consequence

The fact that the source samples are accessed from the end toward the beginning does NOT by itself prove that the signal is merely time-reversed.

The output sequence is a new recursively transformed signal.

Any explanation claiming:

```text
analysis signal is simply the song played backward
```

is incorrect.

---

## 32.13 — Phase buffer size and 0x23 headroom

### STATUS: HIGH CONFIDENCE

The offset detector computes:

```text
period_int = conversion(ceil(60000 / BPM))
buffer_size = period_int + 0x28
            = period_int + 40
```

The vector initializer `FUN_004041D0` allocates exactly the requested number of elements.

Inside `FUN_00402C70`, phase search begins at:

```text
0x23 = 35
```

and covers approximately one period:

```text
35 .. 35 + period_int - 1
```

The additional:

```text
40 - 35 = 5
```

elements provide trailing headroom for the `i+1` interpolation access.

Thus `0x23` is best understood as an INTERNAL SEARCH OFFSET / HEADROOM CONTRACT.

### Critical distinction

The source then directly computes:

```text
offset = phase_peak_raw - 6.2
```

It does NOT subtract `0x23` at this point.

Therefore an implementation MUST NOT silently transform:

```text
phase_raw
```

into:

```text
phase_raw - 35
```

before applying the final `-6.2`.

The remaining question is whether a later caller converts this raw function result into another public coordinate system.

That caller-level normalization is still unresolved.

---

## 32.14 — Partial final phase cycle

### STATUS: PROVEN

The phase fold uses:

```text
cycle_count = ceil(N / T_ms)
```

The last cycle is therefore included even when incomplete.

For each cycle:

```text
base = nearest_integer_like(k * T_ms)
```

Only source samples whose indices remain inside:

```text
0 <= source_index < N
```

are added.

No evidence exists for explicit wraparound of out-of-range source indices.

Therefore the partial final cycle contributes only the samples physically present in the source.

This is NOT equivalent to requiring a complete final beat.

### Remaining boundary detail

The phase vector has:

```text
period_int + 40
```

elements and is searched beginning at:

```text
35
```

which provides enough surrounding storage for the parabolic `i-1` / `i+1` access under ordinary valid-period conditions.

The exact zero-initialization source argument to `FUN_004041D0` has not yet been independently extracted at instruction level.

---

## 32.15 — Floating-point conversion helper

### STATUS: PROVEN

`FUN_00413F70` is a runtime/compiler conversion helper.

The binary contains two different branches depending on:

```text
DAT_010212E8
```

One path uses the x87 rounding machinery and explicit correction.

The alternate path reaches:

```asm
FSTP double [stack]
CVTTSD2SI EAX, [stack]
```

which is truncate-toward-zero at the machine-instruction level.

Therefore:

```text
FUN_00413F70
```

must remain a compatibility boundary.

Do not replace every call indiscriminately with:

```python
int(x)
```

or:

```python
round(x)
```

without first selecting the active runtime path.

For Level-2 reconstruction, use the observed mathematical nearest-integer behavior at the relevant call sites.

For Level-3 reconstruction, emulate the actual runtime state.

---

# 33. REMAINING UNRESOLVED ITEMS

The following issues remain genuinely unresolved after this extraction round.

They should not be guessed.

---

## 33.1 — 20.0 and 285.7142857142857

### STATUS: UNRESOLVED

Known:

```text
0x0041A740 = 20.0
0x0041A748 = 285.7142857142857
```

The latest extraction proves that `20.0` participates in a direct x87 comparison near:

```text
0x00402838 .. 0x00402855
```

Observed sequence:

```asm
FLD  double [EDI]
FADD double [0x0041A698]      ; +10.0
FLD  double [EBP - 0x7c]
FLD  double [0x0041A740]      ; 20.0
FCOMIP ST0, ST1
...
```

This does NOT yet prove that 20.0 is:

```text
BPM ceiling
period floor
octave heuristic
```

Likewise, the role of:

```text
285.7142857142857
```

has not been reduced to an exact expression.

### Required further Ghidra extraction

Inspect approximately:

```text
0x00402820 .. 0x004028A0
```

and print:

* all x87 instructions
* ST(0)/ST(1)/ST(2) stack state at every instruction
* exact stack-variable origins
* exact branch predicates
* all uses of 0x0041A740
* all uses of 0x0041A748
* values subsequently written to:
  `local_78`
  `local_80`
  `local_a0`
  `local_1c`
  `param_2`

Resolve:

```text
FUN_0041438A
 -> __cintrindisp2
 -> __trandisp2
 -> final x87 dispatch operation
```

Do NOT classify the constants by interpreting:

```text
60000 / 285.714...
```

unless the data-flow proves the compared variable is BPM.

---

## 33.2 — Public Offset Coordinate Contract

### STATUS: UNRESOLVED

Inside `FUN_00402C70` the proven final expression is:

```text
offset_raw = phase_peak_raw - 6.2
```

The remaining question is what the caller does with this value.

### Required extraction

Trace the CALL to:

```text
FUN_00402C70
```

at approximately:

```text
0x00403D98
```

and print at least:

```text
±128 instructions
```

around the call.

Then trace:

```text
EAX return status
the output double produced by FUN_00402C70
all stores/loads of that double
all arithmetic applied to it
all printf/fwprintf/fprintf arguments derived from it
```

Required final conclusion:

```text
raw FUN_00402C70 result
    ->
caller variable
    ->
optional normalization
    ->
displayed/stored offset
```

Specifically determine whether the binary performs:

```text
offset
offset + period
offset modulo period
clamp
nothing
```

Do not assume a user-facing [0,T) contract.

---

## 33.3 — Exact Initialization Value of Phase Buffer

### STATUS: UNRESOLVED

`FUN_004041D0` allocates a vector of:

```text
period_int + 40
```

elements and initializes each element from its input value:

```c
*element = *param_1;
```

The exact register/stack argument passed to `FUN_004041D0` from `FUN_00402C70` has not yet been recovered.

Therefore the following must not yet be declared absolute:

```text
phase_buffer initialized to +0.0
```

although zero initialization is strongly suggested by surrounding local state.

### Required extraction

Print ±48 instructions around the call to:

```text
FUN_004041D0
```

inside:

```text
FUN_00402C70
```

and determine:

```text
ESI = destination vector object
EDI = requested element count
argument source pointer = ?
```

Then inspect the value loaded from that argument source.

---

## 33.4 — Exact edge behavior of FUN_00413958

### STATUS: MEDIUM CONFIDENCE / PARTIALLY RESOLVED

The assembly proves this is a custom/vectorized logarithm implementation with:

* exponent extraction
* mantissa normalization
* LUT lookup
* polynomial/range-reduction approximation
* explicit special-value branches

The latest dump shows branches for:

```text
normal finite input
subnormal input
zero
infinity
NaN
```

No explicit amplitude clamp has been observed.

However the exact behavior for every possible:

```text
negative input
negative zero
signed NaN
positive infinity
negative infinity
```

has not yet been completely reconstructed from the exceptional branch.

### Required extraction

Trace:

```text
0x00413958 .. 0x00413B7D
```

and classify every return path by input class:

```text
+0
-0
positive finite
negative finite
positive subnormal
negative subnormal
+Inf
-Inf
NaN
```

For each return path print:

```text
returned value expression
special constant address if used
whether sign is preserved
```

Also identify LUT entries by actual instruction usage.

Do not label the LUT as:

```text
[reciprocal, log_hi, log_lo]
```

unless the memory layout and consuming instructions prove it.

---

## 33.5 — Backward transform caller placement

### STATUS: HIGH CONFIDENCE FOR THE TRANSFORM, UNRESOLVED FOR COORDINATE SEMANTICS

The transform itself is now sufficiently reconstructed.

Remaining question:

```text
Is the transformed array passed directly to both:

FUN_00402280
FUN_00402C70
```

with no intervening reversal/copy/reindex operation?

Required extraction:

Trace the output vector from the return of:

```text
FUN_00401E80
```

through the caller.

Print all array-copy operations between:

```text
FUN_00401E80
FUN_00402280
FUN_00402C70
```

and identify any:

```text
reverse
memcpy
pointer+offset
negative stride
```

operation.

---

## 33.6 — Timeline Validation FUN_00402FC0

### STATUS: UNRESOLVED

The latest character-constant search produced no direct `'#'`, `'+'`, `'-'`, `'.'` immediates inside `FUN_00402FC0`.

This does NOT imply that the function lacks the display-state feature.

The mapping may occur:

```text
in the caller
through a lookup table
through a switch table
through a formatted string
through a bitmask
```

### Required extraction

Do not search only for character immediates.

Instead:

1. Decompile `FUN_00402FC0`.
2. Print every conditional branch.
3. Print every write to:

   * byte
   * word
   * dword
   * double
4. Identify every variable written immediately before return.
5. Trace all return values into `FUN_004037A0`.
6. Search the caller for:

   * `0x23`
   * `0x2B`
   * `0x2D`
   * `0x2E`
   * strings containing those characters
   * lookup tables indexed by the return/status value.
7. Identify whether `FUN_00402FC0` modifies:

   * BPM
   * offset
   * status
   * only diagnostic display state.

The goal is the exact pseudocode, not a theoretical interpretation of what timeline validation "should" do.

---

# 34. IMPLEMENTATION CONTRACT AFTER ROUND 3

The core implementation can now be treated as:

```text
BASS float PCM @ 44.1 kHz
        |
        v
mono channel summation
        |
        v
five spectral passes
  pass 0: IIR sections 0..1
  pass 1: IIR sections 2..5
  pass 2: IIR sections 6..9
  pass 3: IIR sections 10..13
  pass 4: IIR sections 14..15
        |
        | each pass:
        | square
        | one section @ 0x419530
        | accumulator *= 1 + filtered
        v
FUN_00413958 per sample
        |
        v
IIR sections 0..1 @ 0x4193F0
        |
        v
44.1 kHz -> 1000 Hz
source_index = floor(n*44.1 + 0.5)
nearest-neighbor copy
        |
        v
IIR @ 0x419544
        |
        v
IIR @ 0x419544
        |
        v
backward alternating recurrence
        |
        v
FFT
        |
        v
|FFT|^2
        |
        v
IFFT
        |
        v
multiply by 1/NFFT
        |
        v
local peak detector
        |
        v
seed candidates <= 999
        |
        v
seed bootstrap:
    j = 1
    T = first refined peak
        |
        v
tracker
    ±10 search
    first matching vector entry wins
    one miss terminates tracker
        |
        v
least-squares through origin
        |
        v
BPM
        |
        +----------------------+
        |                      |
        v                      v
phase folding             timeline validation
        |                      |
        v                      |
phase peak                  unresolved
        |
        v
raw offset = phase_peak - 6.2
        |
        v
caller-level normalization
UNRESOLVED
```

---

# 35. CURRENT CONFIDENCE LEDGER

## PROVEN

```text
- 5-pass filterbank is divided into 2/4/4/4/2 section groups.
- Each outer pass uses one distinct section group.
- Each pass then uses the one-section filter at 0x419530.
- Post-loop log stage uses sections 0..1 at 0x4193F0.
- IIR state resets on every FUN_00401580 invocation.
- Every detected local peak is inserted in ascending order.
- Seeds >999 terminate seed processing.
- Every eligible peak can become a seed.
- Seed is tracked as beat index j=1.
- Initial fitted period equals the first refined seed position.
- Tracker window is ±10.
- First matching peak in vector order wins.
- One missing peak terminates tracker.
- Exact normalized fit expression.
- Exact quality thresholds.
- Positive-range resampler index is floor(x+0.5).
- Resampler performs nearest-neighbor sample selection.
- FFT size is the smallest power of two >= N in the BPM path.
- IFFT result is explicitly multiplied by 1/NFFT.
- No explicit DC subtraction in FUN_004017A0.
- Backward alternating transform is not simple time reversal.
- Phase buffer size is period_int + 40.
- Phase search begins at +35.
- Raw offset assignment is phase_peak - 6.2.
- Partial final phase cycle is included.
```

## HIGH CONFIDENCE

```text
- Seed >999 creates an effective direct-seed limitation near 1 second
  of period, but this is NOT yet equivalent to a strict BPM floor.
- The 35-element phase offset is internal buffer headroom/search
  coordinate rather than a value that should automatically be
  subtracted from phase_peak.
- The backward transform should be treated as a reverse-index
  recurrence rather than a pure time reversal.
```

## MEDIUM CONFIDENCE

```text
- Exact negative/special-input behavior of FUN_00413958 remains
  partially reconstructed.
- Exact compiler/runtime floating-point behavior depends on active
  FP environment.
```

## UNRESOLVED

```text
- Exact meaning of 20.0.
- Exact meaning of 285.7142857142857.
- Exact x87 operation dispatched through FUN_0041438A.
- Public/caller-level offset normalization.
- Exact initialization argument of phase buffer.
- Exact coordinate semantics of the backward transform in the
  BPM/offset caller chain.
- Complete FUN_00402FC0 timeline-validation contract.
```

# 36. Ground-Truth Consolidation Round 4

## 36.1. Winner Decision Tree trong `FUN_00402280` đã được giải mã phần lớn

### 36.1.1. Candidate không được chọn bởi một global argmax đơn giản

`FUN_00402280` duyệt candidate theo thứ tự tăng dần và duy trì một `best_average`.

Một candidate chỉ tiếp tục được chấp nhận khi:

```text
avg_peak > 0.7 * seed_peak
```

với:

```text
0x0041a754 = 0.7f
```

Sau đó:

```text
if avg_peak > 1.2 * best_average:
    best_average = avg_peak
    best_candidate = current_candidate
```

với:

```text
0x0041a750 = 1.2f
```

Do candidate được xử lý tuần tự, đây là một **online replacement rule**, không phải:

```text
best = argmax(avg_peak)
```

một cách thuần túy.

Hệ quả:

* candidate xuất hiện sớm có thể tiếp tục là `best`;
* candidate sau chỉ thay `best` khi mạnh hơn ít nhất 20%;
* thứ tự xuất hiện của candidate do đó là một phần của thuật toán.

Không được thay bằng:

```python
best = max(candidates, key=lambda c: c.avg_peak)
```

trong functional clone.

---

## 36.1.2. Có một harmonic / octave disambiguation pass sau khi đã có `best`

Sau khi vòng candidate hoàn tất:

```asm
00402805  MOVSD XMM0,qword ptr [0x0041a748]
0040280D  MOV ECX,ESI
00402816  COMISD XMM0,XMM5
00402824  JBE 0x00402875
```

với:

```text
0x0041a748 = 285.7142857142857
```

Điều kiện đi vào harmonic correction là:

```text
best_period < 285.7142857142857 ms
```

và:

```text
60000 / 285.7142857142857 = 210 BPM
```

Do đó có một nhánh hậu xử lý chỉ hoạt động khi:

```text
best_BPM > 210 BPM
```

Nhánh này duyệt **các candidate xuất hiện sau best**.

---

## 36.1.3. Hai hằng `+10` và `20` tạo thành một cửa sổ harmonic ±10 ms

Tại mỗi candidate sau `best`:

```asm
00402838  FLD double ptr [EDI]
0040283A  FADD double ptr [0x0041a698]   ; +10.0
00402840  FLD double ptr [EBP + -0x7c]   ; best period
00402843  CALL 0x0041438a
00402848  FLD double ptr [0x0041a740]     ; 20.0
0040284E  FCOMIP ST0,ST1
00402852  JNC 0x00402863
```

Nếu hàm tại `FUN_0041438A` là phép remainder tương ứng với `fmod`, logic toán học tương đương:

```text
remainder = fmod(candidate_period + 10.0, best_period)

accept harmonic candidate when:
    remainder <= 20.0
```

Vì candidate sau best thường có period lớn hơn best, cấu trúc này có ý nghĩa tương đương với kiểm tra candidate nằm trong khoảng xấp xỉ:

```text
candidate_period ≈ k * best_period ± 10 ms
```

cho một số nguyên `k`.

Cụ thể:

```text
candidate = k*T + δ

(candidate + 10) mod T <= 20
```

tạo ra vùng chấp nhận gần:

```text
-10 ms <= δ <= +10 ms
```

trong miền remainder thông thường.

Do đó hai hằng:

```text
10.0
20.0
```

không phải hai threshold độc lập; chúng tạo thành một tolerance window đối xứng quanh bội số của `best_period`.

### Trạng thái xác minh

**Mạnh về semantics, chưa bit-exact về implementation**.

`FUN_0041438A` chắc chắn là một CRT two-argument floating dispatcher, nhưng đoạn dump hiện tại chưa đi tới target trong jump table thực hiện phép `FPREM`.

Không được ghi cứng vào implementation contract rằng dispatcher có tên `fmod` cho tới khi target `FPREM` được xác nhận.

---

## 36.1.4. Harmonic correction chọn candidate sau đầu tiên thỏa điều kiện

Khi điều kiện harmonic được thỏa:

```asm
00402863
MOV EDX,dword ptr [EBP + -0x2c]
ADD ESI,ESI
MOVSD XMM5,qword ptr [EDX + ESI*0x8]
```

Sau đó code nhảy ra khỏi vòng tìm harmonic:

```asm
0040286D ...
00402875 ...
```

Không có bước:

```text
max amplitude among harmonic-compatible candidates
```

ở đây.

Do đó semantics là:

```text
for candidate after best:
    if harmonic_condition(candidate, best):
        best = first_matching_candidate
        break
```

Điều này rất quan trọng đối với clone exact.

### Không được thay bằng:

```python
best = max(harmonic_matches, key=lambda c: c.avg_peak)
```

---

# 36.2. Candidate selection hiện có thể mô tả thành hai tầng

Pipeline selection hiện đã đủ rõ để viết thành:

```text
for candidate in ascending_candidates:

    reject unless:
        avg_peak > 0.7 * seed_peak

    if avg_peak > 1.2 * best_average:
        best = candidate
        best_average = avg_peak


if best.period < 285.7142857142857 ms:

    for candidate after best:
        if harmonic_relation(candidate.period, best.period):
            best = candidate
            break
```

Trong đó:

```text
harmonic_relation(a, T):
    fmod(a + 10.0, T) <= 20.0
```

là biểu diễn semantics hiện tại, với cảnh báo rằng operand order của dispatcher vẫn cần xác nhận bằng CRT jump target.

Đây là **Winner Decision Tree** mà Section 33.1 trước đó còn thiếu.

---

# 36.3. Backward-copy primitive đã được xác nhận

`FUN_00404fa0` thực hiện phép copy ngược:

```asm
00404fb4  FLD float ptr [ECX + -0x4]
00404fb7  FSTP float ptr [EAX]
00404fb9  SUB ECX,0x4
00404fbc  ADD EAX,0x4
...
00404fc2  CMP ECX,EDX
00404fc4  JNZ 0x00404fb0
```

Semantics:

```python
while src != end:
    dst[0] = src[-1]
    src -= 1 element
    dst += 1 element
```

hay:

```python
dst = reverse(src[begin:end])
```

Do đó binary có một primitive vector insertion / relocation thực hiện **reverse-copy**, không phải một phép đảo mảng giả định.

`FUN_00404e00` gọi primitive này trong quá trình mở rộng / chèn vector:

```asm
00404ec8 ...
00404ed0 ...
00404ed3 CALL FUN_00404fa0
```

và cũng có các nhánh relocation khác khi buffer overlap.

### Kết luận

**Đã chứng minh rằng reverse-copy tồn tại trong data-structure layer.**

Kết luận mạnh hơn:

```text
acc phase-processing buffer có nguồn từ reverse-copy
```

chỉ được coi là ground truth khi callsite cụ thể bên trong `FUN_00402C70` được nối trực tiếp với `FUN_00404cc0/FUN_00404e00`.

Vì vậy:

```text
reverse-copy primitive = PROVEN
specific semantic role in phase pipeline = STRONG, but callsite should remain traceable
```

---

# 36.4. Không nên mô hình hóa backward stage như một phép reverse toàn bộ tín hiệu

Với sự tồn tại của `FUN_00404fa0`, mô hình chính xác hơn không phải:

```python
y = reverse(x)
```

một cách độc lập.

Data flow được hỗ trợ bởi các helper là:

```text
forward vector
        │
        ├── ordinary filtering
        │
        └── reverse-copy into another vector
                    │
                    └── filtering / accumulation
```

Sau đó các vector được đưa vào một phép kết hợp.

Do đó công thức:

```text
y[k] = x[N-2-k] - y[k-1]
```

được dùng trong Section 32.12 trước đây **không còn đủ chính xác ở mức algorithmic data flow**.

Cần giữ distinction:

```text
reverse-copy
≠
reverse-time transform itself
≠
alternating recurrence itself
```

Ba operation này thuộc ba tầng khác nhau.

---

# 36.5. Offset public coordinate đã được đóng về mặt caller

Trong `FUN_00402C70`, phase peak được chuyển thành:

```text
offset_raw = phase_peak_refined - 6.2
```

Ở main:

```asm
00403F77  MOVSD XMM0,qword ptr [ESP + 0x28]
00403F7F  SUBSD XMM0,qword ptr [0x0041a758]
```

với:

```text
0x0041a758 = 22.5
```

Do đó giá trị in ra là:

```text
offset_printed
    = offset_raw - 22.5
    = phase_peak_refined - 6.2 - 22.5
    = phase_peak_refined - 28.7
```

### Đây là contract quan trọng

Không được tự ý làm:

```text
phase_peak - 35
```

vì `35` là **search base**, không phải coordinate origin.

Không có bằng chứng cho phép trừ thêm 35 ở caller.

---

# 36.6. `35` và `6.2` có vai trò khác nhau

Hiện có ba số cố định dễ bị nhầm:

```text
35.0   = phase search base / headroom
6.2    = internal phase coordinate correction
22.5   = caller-side public offset correction
```

Do đó:

```text
35 ≠ offset correction
6.2 + 22.5 = 28.7 = effective public output correction
```

Cấu trúc offset phải được giữ thành:

```text
phase search index
        │
        ▼
parabolic phase refinement
        │
        ▼
- 6.2
        │
        ▼
FUN_00402C70 result
        │
        ▼
- 22.5 at main output
        │
        ▼
printed Offset
```

---

# 36.7. Main cũng xác nhận công thức normalize offset nhập tay

Nhánh xử lý offset thủ công sử dụng:

```text
X + 22.5
```

sau đó tạo:

```text
(X + 22.5 - 30.0) * P / 60000
```

và dùng integer conversion để xác định số chu kỳ.

Semantics được mô tả ở mức thuật toán là:

```text
q = floor((X - 7.5) / P)

X_normalized = (X + 22.5) - q * P
```

với caveat rằng integer conversion phải tái tạo bằng helper CRT/compiler-compatible, không được thay bừa bằng conversion mặc định của ngôn ngữ.

Điều này cho thấy binary không coi Offset là một timestamp tuyệt đối đơn giản; nó đưa offset về một coordinate system có anchor tại khoảng:

```text
7.5 ms
```

và public correction:

```text
22.5 ms
```

Đây là thêm bằng chứng rằng `35 ms` không phải public origin.

---

# 36.8. FFT analysis scope: toàn bộ file đã được xác nhận

Trong main:

```asm
00403B08  PUSH EDX
00403B09  LEA ECX,[ESP + 0x58]
00403B0D  CALL 0x00401E80
```

`FUN_00401E80` xử lý toàn bộ vector PCM được tạo từ audio input.

Sau đó trong `FUN_00402280`:

```asm
004022CD  MOV EAX,dword ptr [ECX + 0x4]
004022D0  SUB EAX,dword ptr [ECX]
004022D5  SAR EAX,0x2
```

đây chính là chiều dài vector float:

```text
N = (end - begin) / sizeof(float)
```

Không có dấu hiệu cắt thành:

```text
10 s
20 s
30 s
```

hay một central analysis window trước khi BPM detector chạy.

### Kết luận

```text
BPM analysis input scope = full decoded / preprocessed audio vector
```

Không nên đưa thêm một windowing stage giả định vào functional clone.

---

# 36.9. Một sửa đổi quan trọng đối với Section 32.11: FFT không nên tiếp tục được ghi là chỉ `Nfft >= N`

Trong `FUN_00402280`, trước khi gọi `FUN_004017A0`, binary tính:

```text
N
→ N * 0.5
→ compiler-specific integer conversion
→ pass result into FFT helper
```

Đoạn:

```asm
004022D0 ... /4
004022D8 CVTSI2SD ...
004022DC MULSD ..., 0.5
...
00402322 CVTTSD2SI EAX,...
0040234E PUSH EAX
00402352 CALL FUN_004017a0
```

cho thấy caller không đơn giản truyền:

```text
N
```

mà truyền một giá trị dựa trên:

```text
N / 2
```

Điều này phủ nhận phiên bản thiết kế quá đơn giản:

```text
Nfft = smallest_power_of_two_geq(N)
```

### Trạng thái hiện tại

`FUN_004017A0` cần được re-audit để chốt chính xác:

```text
nlags
exact rounding of N/2
FFT logical length
zero-padding length
```

Nhận định rằng:

```text
Nfft ≈ next_power_of_two(N + N/2)
```

là **mạnh nhưng chưa nên ghi là bit-exact ground truth nếu chưa lấy thân FUN_004017A0 mới nhất**.

---

# 36.10. Dù chưa chốt công thức `Nfft`, mục tiêu của phần padding đã có thể suy luận

Nếu FFT length thực sự được chọn theo:

```text
N + N/2
```

thì:

```text
Nfft >= 1.5N
```

Do input thực sự chỉ có `N` sample, phần còn lại là zero-padding.

Điều này có một hệ quả DSP quan trọng:

```text
circular autocorrelation
```

của FFT zero-padded đủ dài sẽ trùng với:

```text
linear autocorrelation
```

trong miền lag đủ nhỏ.

Vì detector chỉ scan:

```text
lag <= min(N, 2000)
```

nên đối với audio đủ dài, miền 0..2000 được đặt trong vùng rất nhỏ so với lượng zero-padding.

### Đây là suy luận, không phải source comment

Có khả năng cao padding lớn hơn cần thiết không phải để “tính thêm BPM”, mà để giảm / loại bỏ circular wrap-around trong miền lag được detector sử dụng.

Không được viết đây là động cơ tác giả binary trừ khi có comment/source-level evidence.

---

# 36.11. Nonlinear 5-pass: nguy cơ giá trị âm trước Log được giảm mạnh

Section `0x00419530` có:

```text
gain = 0.048
a1 = 1.948
a2 = -0.9481
```

với characteristic denominator:

```text
1 - 1.948 z^-1 + 0.9481 z^-2
```

phân tích thành:

```text
(1 - 0.998 z^-1)(1 - 0.95 z^-1)
```

Hai pole đều dương và nhỏ hơn 1.

Impulse response zero-state của section này vì thế là không âm.

Do pass trước nó bình phương tín hiệu:

```text
u[n] = x[n]^2 >= 0
```

nên trong arithmetic lý tưởng:

```text
filtered[n] >= 0
```

và:

```text
1 + filtered[n] >= 1
```

Do đó đường dẫn bình thường không tạo:

```text
1 + filtered < 0
```

chỉ vì transient của bộ lọc.

### DC gain

Tại DC:

```text
H(1)
    = 0.048 / (1 - 1.948 + 0.9481)
    = 480
```

Vì vậy DC gain 480 là đúng.

### Quan trọng

Điều này **không** cho phép kết luận:

```text
FUN_00413958 never sees negative / Inf / NaN
```

Các trường hợp vẫn còn:

```text
floating-point overflow
Inf
NaN
subnormal
rounding artifacts
```

đặc biệt khi các giá trị được nhân dồn qua 5 pass.

Kết luận đúng là:

```text
ordinary negative-output concern from the 0x419530 impulse response:
largely eliminated

exact special-value behavior of FUN_00413958:
still unresolved
```

---

# 36.12. `FUN_00402FC0` đã tiết lộ cấu trúc classification 5-stream

`FUN_00402FC0` tạo năm vector accumulator độc lập:

```text
V0
V1
V2
V3
V4
```

sau đó xử lý mỗi bin / sample và ghi flag vào một output vector.

Cuối hàm:

```asm
00403730  OR ESI,dword ptr [EAX]
...
00403738  ...
0040378B  MOV EAX,ESI
```

Do đó return value là:

```text
bitwise OR
of the classification flags over the output vector
```

Main ánh xạ các bit này thành ký hiệu:

```asm
bit 0x1 -> '#'
bit 0x2 -> '+'
bit 0x4 -> '-'
bit 0x8 -> '.'
```

và bitmask bằng 0 không tạo ký hiệu đặc biệt.

Vì vậy `FUN_00402FC0` không đơn giản trả:

```text
good / bad
```

mà trả một **4-bit composite classification mask**.

---

# 36.13. Timeline / phase validation hiện đã không còn là black box hoàn toàn

Từ `FUN_00402FC0`, ta biết chắc:

```text
Input:
    phase-like coordinate
    BPM / period
    processed signal

Internal:
    5 accumulation vectors
    per-position classification
    per-position bitmask

Output:
    aggregate bitwise-OR status mask
```

Do đó Section 33.6 nên được hạ mức từ:

```text
TIMELINE VALIDATION = completely unresolved
```

thành:

```text
TIMELINE / PHASE CLASSIFICATION:
    control flow = largely known
    accumulator architecture = known
    flag propagation = known
    semantic meaning of all 5 streams = unresolved
```

---

# 36.14. Các nút thắt thực sự còn lại sau Round 4

## Blocker A — `FUN_0041438A` exact intrinsic

Cần đi hết:

```text
FUN_0041438A
    → __cintrindisp2
    → __trandisp2
    → jump-table target
    → arithmetic body
```

để xác nhận:

```text
fmod / remainder intrinsic
operand order
special-value semantics
```

Hiện semantics ở callsite đã rất rõ, nhưng bit-exact implementation chưa đóng.

---

## Blocker B — exact FFT size trong `FUN_004017A0`

Cần lấy prologue/body của:

```text
FUN_004017A0
```

và trả lời chính xác:

```text
What does caller-passed N/2-derived integer mean?
How is N + nlags computed?
What exact integer conversion is used?
What exact power-of-two loop is used?
What array length is zero-padded?
```

Không merge lại statement:

```text
Nfft >= N
```

như một ground truth đã đóng.

---

## Blocker C — tie `reverse-copy` vào đúng vector trong `FUN_00402C70`

Cần trace:

```text
FUN_00402C70
    → FUN_00404cc0 / FUN_00404e00
    → FUN_00404fa0
```

và xác định chính xác:

```text
which vector = forward signal
which vector = reverse-copy
which vector = filtered reverse signal
how indices are paired
```

Primitive reverse-copy đã được chứng minh; semantic role cuối cùng vẫn cần callsite.

---

## Blocker D — semantic meaning of the 4 classification flags

Các bit:

```text
0x1
0x2
0x4
0x8
```

đã được chứng minh là classification flags và mapping UI đã biết:

```text
# + - .
```

Nhưng ý nghĩa DSP / quality chính xác của từng bit vẫn cần giải mã.

Không tự đặt tên chúng thành:

```text
GOOD
EARLY
LATE
WEAK
```

nếu binary chưa chứng minh.

---

## Blocker E — exact special-value behavior of `FUN_00413958`

Các nhánh normal / subnormal / zero / inf / NaN đã tồn tại, nhưng cần chốt:

```text
negative input
zero
negative zero
subnormal
+Inf
-Inf
NaN
overflow
```

đặc biệt nếu clone yêu cầu bit-exact output.

---

# 36.15. Updated implementation contract

Pipeline hiện tại có thể khóa ở mức sau:

```text
BASS decode
    ↓
44100 Hz float PCM
    ↓
multichannel sum → mono
    ↓
5-pass nonlinear filter bank
    ├─ sections 0..1
    ├─ sections 2..5
    ├─ sections 6..9
    ├─ sections 10..13
    └─ sections 14..15
         each pass:
            IIR
            square
            fixed section @ 0x419530
            acc *= (1 + filtered)
    ↓
FUN_00413958
    ↓
2-section IIR
    ↓
nearest-sample resample to ~1000 Hz
    ↓
post-resample IIR ×2
    ↓
backward/reverse-copy-related phase preprocessing
    ↓
FFT autocorrelation over full analysis vector
    ↓
local maxima predicate
    ↓
all candidate seeds <= 999
    ↓
tracker per seed
    ↓
candidate acceptance:
    avg_peak > 0.7 * seed_peak
    ↓
online best update:
    avg_peak > 1.2 * best_average
    ↓
harmonic correction when:
    best_period < 285.7142857142857 ms
    ↓
first later candidate satisfying:
    harmonic remainder <= 20 ms
    ↓
BPM = 60000 / period
    ↓
phase folding
    ↓
parabolic phase refinement
    ↓
offset_raw = phase_peak - 6.2
    ↓
main public output:
offset_printed = offset_raw - 22.5
                  = phase_peak - 28.7
    ↓
optional phase/timeline classification
    ↓
4-bit aggregate status mask
```

---

# 36.16. Confidence ledger

| Item                                               | Status                            |
| -------------------------------------------------- | --------------------------------- |
| Candidate acceptance `avg > 0.7*seed`              | PROVEN                            |
| Online replacement `avg > 1.2*best`                | PROVEN                            |
| Best candidate is not global argmax                | PROVEN                            |
| Harmonic correction threshold 285.7142857142857 ms | PROVEN                            |
| `10` and `20` are used in harmonic test            | PROVEN                            |
| Harmonic test semantics = remainder tolerance      | STRONG                            |
| `FUN_0041438A` exact `fmod` implementation         | UNRESOLVED                        |
| Harmonic operand order                             | STRONG, not yet dispatcher-proven |
| First matching harmonic candidate wins             | PROVEN                            |
| Reverse-copy primitive                             | PROVEN                            |
| Reverse-copy's exact phase-buffer role             | STRONG / CALLSITE PENDING         |
| `35` is not subtracted from public offset          | PROVEN                            |
| `6.2` correction inside phase routine              | PROVEN                            |
| `22.5` caller output correction                    | PROVEN                            |
| Printed offset correction = `28.7 ms`              | PROVEN                            |
| Full-file analysis scope                           | PROVEN                            |
| FFT uses a half-length-derived caller parameter    | PROVEN                            |
| Exact `Nfft` formula                               | UNRESOLVED                        |
| 0x419530 DC gain = 480                             | PROVEN                            |
| Ordinary negative-output concern from this filter  | LARGELY ELIMINATED                |
| Log special-value semantics                        | UNRESOLVED                        |
| Classification returns aggregate bitmask           | PROVEN                            |
| Bit mappings `1:#, 2:+, 4:-, 8:.`                  | PROVEN                            |
| Semantic meaning of classification bits            | UNRESOLVED                        |

# 37. Ground-Truth Consolidation Round 5

## 37.1. Backward Transform đã được tái dựng lại ở mức công thức chính xác

Phân tích đuôi `FUN_00401E80` cho phép phân biệt rõ hai buffer:

```text
F[n] = buffer xuôi đang được ghi đè tại *piVar2
B[n] = buffer được tạo qua reverse-copy, sau đó đi qua post-IIR
```

Trong đó reverse-copy primitive là `FUN_00404fa0`, được gọi gián tiếp qua `FUN_00404cc0`.

Điểm quan trọng là phép kết hợp cuối **không phải**:

```text
y[k] = x[N-2-k] - y[k-1]
```

với cùng một buffer `y` đóng vai trò cả hai vế.

### Công thức đúng

Đặt:

```text id="58enav"
N = số mẫu của buffer F
B = reverse-copy buffer sau các bước lọc tương ứng
```

thì phần thân cuối thực hiện:

```text id="0yeyso"
F[0]     = B[N-2]

F[k]     = B[N-2-k] - F[k-1]
           với 1 <= k <= N-2

F[N-1]   = -F[N-2]
```

Điều này được hỗ trợ trực tiếp bởi vòng lặp:

```asm id="u5ck4j"
out[N-2] = B[0] - out[N-3]
out[N-3] = B[1] - out[N-4]
out[N-4] = B[2] - out[N-5]
...
```

và bởi instruction cuối:

```asm id="ut29jb"
*(float *)out[0] = B[N-2]
```

trong khi sample cuối được chuẩn bị bằng XOR sign-bit:

```asm id="m81qtd"
out[N-1] = out[N-2] XOR SIGN_BIT
        = -out[N-2]
```

### Ý nghĩa

Đây là một **reverse-indexed alternating subtraction driven by B**, không phải:

```text
simple reverse
```

và cũng không phải:

```text
filtfilt()
```

hay:

```text
forward_signal - reversed_signal
```

### Contract cho implementation

Không được triển khai bằng:

```python
y = reverse(x)
```

đơn thuần.

Không được triển khai bằng:

```python
y[k] = x[N-2-k] - y[k-1]
```

trừ khi `x` trong biểu thức được hiểu chính xác là buffer `B` đã được tạo bởi reverse-copy + filtering, còn số hạng trừ là buffer kết quả `F[k-1]`.

Biểu diễn reference an toàn:

```python
F[0] = B[N - 2]

for k in range(1, N - 1):
    F[k] = B[N - 2 - k] - F[k - 1]

F[N - 1] = -F[N - 2]
```

Đây là công thức **đã giải blocker Backward Transform ở mức algorithmic data flow**.

---

## 37.2. Reverse-copy và alternating recurrence là hai operation khác nhau

Pipeline đúng phải phân biệt:

```text
input signal
    │
    ├── forward/current buffer F
    │
    └── reverse-copy
            ↓
          B_raw
            ↓
       post-IIR/filter
            ↓
            B
            │
            └──────────────┐
                           ↓
                     alternating
                     subtraction
                           ↓
                         F_out
```

Do đó:

```text
reverse-copy
```

chỉ tạo input cho nhánh `B`.

Nó không tự nó tạo ra kết quả cuối cùng.

Kết quả cuối cùng xuất hiện từ phép ghép:

```text
B[N-2-k] - F[k-1]
```

Điều này giải thích tại sao việc nhìn riêng `FUN_00404fa0` có thể dẫn đến kết luận sai rằng toàn bộ stage là một phép đảo thời gian.

---

## 37.3. Candidate không có per-candidate `status` guard trước Winner Selection

Output assembly của `FUN_00402280` không cho thấy một trường:

```text
candidate.status
```

được kiểm tra trước hai phép so sánh:

```text
avg_peak > 0.7 * seed_peak
avg_peak > 1.2 * best_average
```

Luồng thực tế tại:

```asm id="f1l0aw"
00402580 ...
0040258A ...
00402593 ...
0040259B ...
0040259E JBE 0x0040279A
```

cho thấy candidate trước hết được lọc bằng:

```text
avg_peak > 0.7 * seed_peak
```

Sau đó:

```asm id="j1b4cj"
004025A4
...
MULSS XMM2, 1.2
004025B7
COMISS XMM1, XMM2
004025BA JBE 0x004025C6
```

quyết định việc thay `best`.

Không có nhánh ở đây kiểm tra:

```text
status == 0
status == 1
status != 0x10
status != -1
```

trước khi candidate tham gia Winner Selection.

### Kết luận

Mô hình cũ:

```text
candidate
    -> compute status
    -> reject if bad
    -> compete for best
```

không còn được source hỗ trợ.

Mô hình đúng hơn là:

```text
seed
    ↓
tracker
    ↓
candidate metrics
    ↓
avg_peak acceptance
    ↓
online best update
    ↓
harmonic correction
    ↓
final quality/status evaluation
```

---

## 37.4. `status = 0 / 1 / 0x10 / -1` là chất lượng của kết quả cuối, không phải guard của từng seed

Ở cuối `FUN_00402280`, binary mới thực hiện các kiểm tra:

```text
count < 4
sigma > 2.4
normalized_fit_error > 3e-5
sigma > 0.6
gap_count > 0
```

và trả:

```text
-1
0
1
0x10
```

Những phép kiểm tra này diễn ra **sau** phần candidate selection và harmonic correction.

Do đó không nên mô hình hóa mỗi seed như:

```text
Candidate {
    bpm
    status
    ...
}
```

rồi dùng `status` để lọc Winner Selection, trừ khi một đoạn code khác chứng minh điều đó.

Trong phần code hiện đã truy vết, `status` là kết quả đánh giá của trạng thái cuối được chọn.

---

## 37.5. Harmonic correction cũng không kiểm tra candidate quality

Nhánh harmonic correction:

```asm id="i8fx3m"
00402830 ...
00402838 FLD [EDI]
0040283A FADD 10.0
00402840 FLD [best_period]
00402843 CALL FUN_0041438A
00402848 FLD 20.0
0040284E FCOMIP
```

chỉ dùng giá trị period của candidate và period hiện tại của `best`.

Candidate được chọn khi điều kiện remainder thành công:

```text
harmonic_relation(candidate_period, best_period) = true
```

Không có load/compare nào trong nhánh này với:

```text
sigma
normalized_error
status
tracked_count
```

Do đó binary có thể chọn một harmonic candidate dựa trên quan hệ period trước khi bước final quality classification được đánh giá.

Đây là một đặc tính của thuật toán, không phải lỗi cần “sửa” trong clone.

---

## 37.6. `FUN_0041438A` vẫn chưa thể ghi chắc là C `fmod`

Dispatcher:

```text
FUN_0041438A
    → __cintrindisp2
    → __trandisp2
    → runtime dispatch table
```

đã được xác nhận.

Tuy nhiên tracer hiện tại dừng tại:

```asm id="4vz5zv"
CALL __trandisp2
```

mà chưa truy được jump-table target chứa phép toán FPU thực sự.

Vì vậy vẫn chưa được phép ghi:

```text
FUN_0041438A == fmod()
```

như một ground truth bit-exact.

### Điều đã biết

Tại callsite:

```text
ST(0) = best_period
ST(1) = candidate_period + 10.0
```

do thứ tự hai `FLD`.

Điều vẫn cần xác nhận:

```text
1. dispatcher thực sự dùng FPREM hay FPREM1
2. operand nào là dividend
3. remainder có semantics giống C fmod hay IEEE remainder
```

Đây là blocker còn lại duy nhất trong Winner Decision Tree.

---

# 37.7. Phase Folding: index ghi buffer bắt đầu từ 0, không phải từ 35

Đây là kết quả rất quan trọng từ decompile `FUN_00402C70`.

Trong loop:

```text
dVar11 = 0
```

sau đó:

```text
dVar13 = dVar11 * T + 0.5
```

được chuyển thành index nguyên:

```text
s_k = round(k*T)
```

Sau đó:

```text
iVar3 = 0
```

và loop đọc:

```text
signal[s_k + iVar3]
```

rồi cộng vào:

```text
phase_buffer[iVar3]
```

cho tới khi vượt quá input length.

Do đó mapping cốt lõi là:

```text
phase_buffer[m] += signal[s_k + m]
```

với:

```text
s_k = round(k*T)
m = 0,1,2,...
```

Không có:

```text
phase_buffer[35 + m]
```

trong bước accumulation.

---

## 37.8. `0x23 = 35` là scan origin, không phải write offset

Sau khi accumulation hoàn tất:

```asm id="h0v7x5"
MOV EDX,0x23
```

và:

```text
iVar6 = 35
iVar1 = 35
```

Peak search bắt đầu tại:

```text
m = 35
```

và duyệt đúng một period:

```text
35 <= m < 35 + period
```

Do đó `35` thuộc **peak-search domain**, không phải coordinate được cộng vào phase bins.

Pipeline đúng là:

```text
phase accumulation:
    bins 0 .. period+39

peak search:
    bins 35 .. 34+period
```

Điều này giải thích tại sao có:

```text
buffer_size = period + 40
```

nhưng search lại bắt đầu tại 35.

### Chưa biết

Binary hiện chưa giải thích rõ tại sao prefix 35 bins bị bỏ qua.

Không được tự ghi:

```text
35 = group delay
```

hoặc:

```text
35 = negative-phase compensation
```

cho tới khi có source evidence.

Điều duy nhất chắc chắn là:

```text
35 = hard-coded lower bound of the phase-peak search
```

---

# 37.9. Offset không có bước trừ 35

Vì accumulation ghi từ bin 0 nhưng peak search bắt đầu từ bin 35, nên:

```text
phase_peak_refined
```

vẫn là index trong cùng phase-buffer coordinate system.

Sau parabolic refinement:

```text
offset_raw =
    phase_peak_refined - 6.2
```

và main thực hiện:

```text
offset_printed =
    offset_raw - 22.5
```

Do đó:

```text
offset_printed =
    phase_peak_refined - 28.7
```

Không có instruction nào cho:

```text
phase_peak_refined - 35
```

ở main.

Vì vậy giá trị `35` không được đưa vào công thức public offset.

---

# 37.10. Phase folding bao gồm cả partial final cycle

Decompile cho thấy số cycle được tính từ:

```text
ceil(N / T)
```

và loop chạy cho tới khi `local_14` đạt giá trị này.

Với cycle cuối:

```text
s_k = round(k*T)
```

các sample hợp lệ được cộng cho tới:

```text
N-1
```

Các phần vượt quá input length không được đọc.

Do đó:

```text
partial final cycle = included
```

chứ không chỉ phân tích các cycle đầy đủ.

---

# 37.11. FFT length đã được sửa thành hàm của `N + nlags`

`FUN_004017A0` không còn phù hợp với mô tả:

```text
Nfft = next_power_of_two(N)
```

Prologue cho thấy:

```asm id="mv4bds"
MOV EAX,[ESI + 4]
SUB EAX,[ESI]
SAR EAX,2
```

tạo:

```text
N = input_length
```

Sau đó:

```asm id="9ik7no"
local_1c = param_2 + N
```

và:

```text
fVar8 = log2(local_1c - 1)
```

sau đó:

```text
ceil(...)
```

và:

```asm id="6d5xw1"
iVar6 = 1 << exponent
```

Do đó:

```text
L = N + param_2
E = ceil(log2(L - 1))
Nfft = 2^E
```

---

## 37.12. `param_2` là giá trị được caller tạo từ khoảng `N/2`

Trong `FUN_00402280`, trước call:

```asm id="4rb5x1"
N
→ * 0.5
→ compiler-specific integer conversion
→ EAX
→ PUSH EAX
→ FUN_004017A0
```

Vì thế:

```text
param_2 ≈ round(N/2)
```

với exact rounding phụ thuộc sequence floating-point đã được disassemble.

Do đó functional model nên dùng:

```text
nlags = binary_compatible_round(N / 2)
L = N + nlags
Nfft = smallest_power_of_two_geq(L)
```

với lưu ý rằng binary biểu diễn công thức nội bộ dưới dạng:

```text
ceil(log2(L - 1))
```

trước khi dịch bit.

Hai cách biểu diễn cho cùng power-of-two size đối với `L > 1`.

---

## 37.13. FFT size thực tế

Do:

```text
nlags ≈ round(N/2)
```

ta có:

```text
Nfft ≈ next_power_of_two(1.5N)
```

nhưng implementation contract nên viết chính xác hơn:

```python
nlags = binary_round(N / 2)
L = N + nlags
Nfft = 1 << ceil_log2(L - 1)
```

Không nên hard-code:

```python
Nfft = next_power_of_two(N)
```

nữa.

---

## 37.14. Zero-padding: phân biệt "size-selection length" và "actual signal length"

`L = N + nlags` được sử dụng để **chọn FFT size**.

Điều này không có nghĩa input thực tế chứa `N + nlags` samples.

Input signal vẫn có:

```text
N samples
```

Do đó nếu FFT buffers được zero-initialized và chỉ `N` sample được copy vào real component, phần zero tail sẽ có độ dài:

```text
zero_pad = Nfft - N
```

Cần phân biệt rõ:

```text
L = N + nlags
```

là logical sizing criterion,

trong khi:

```text
N
```

là số sample audio thực tế.

### Trạng thái

Công thức chọn `Nfft` đã được chứng minh.

Việc buffer initialization và zero-fill cụ thể nên tiếp tục được giữ traceable tới thân `FUN_004017A0`, dù pipeline trước đó đã cho thấy real input được đặt vào FFT và imaginary component khởi tạo bằng 0.

---

# 37.15. FFT vẫn là full-file analysis

`FUN_00402280` lấy:

```text
N = (vector_end - vector_begin) / 4
```

trực tiếp từ vector output của preprocessing.

Không có dấu hiệu ở caller cho:

```text
central window
10-second window
30-second window
random chunk
```

Do đó:

```text
BPM analysis scope = full processed audio vector
```

và FFT size tăng theo toàn bộ file.

Điều này giải thích vì sao binary có thể sử dụng một FFT tương đối lớn, dù peak detector chỉ cần một vùng lag nhỏ.

---

# 37.16. Ý nghĩa của `N + nlags` đối với autocorrelation

Với:

```text
L = N + nlags
```

và:

```text
nlags ≈ N/2
```

FFT size được chọn trên khoảng:

```text
1.5N
```

thay vì chỉ `N`.

Đây là một bằng chứng mạnh rằng binary muốn FFT domain đủ lớn để tránh wrap-around trong vùng autocorrelation quan tâm.

Tuy nhiên, đây là **DSP interpretation**, không phải source-level statement về mục đích của tác giả.

Không nên ghi:

```text
"the author padded by 50% specifically to avoid circular aliasing"
```

như fact.

Có thể ghi:

```text
"The resulting FFT size is large enough to accommodate a length criterion of approximately 1.5N; this is consistent with reducing circular wrap-around in the analyzed correlation region."
```

---

# 37.17. Updated BPM selection contract

Toàn bộ BPM winner selection hiện có thể viết:

```python
for seed in ascending_peaks:

    if seed >= 1000:
        break

    run_tracker(seed)

    if tracked_count == 0:
        continue

    avg_peak = average(tracked_peak_amplitudes)

    if avg_peak <= 0.7 * seed_amplitude:
        continue

    if avg_peak > 1.2 * best_average:
        best_average = avg_peak
        best_candidate = current_candidate

    # candidate retained independently of whether
    # final quality status will later be 0, 1, or 0x10


if best_period < 285.7142857142857:
    for candidate_after_best in later_candidates:

        if harmonic_relation(candidate_after_best.period,
                             best_period):

            best_period = candidate_after_best.period
            break

BPM = 60000 / best_period
```

Trong đó:

```text
status classification
```

diễn ra sau selection.

---

# 37.18. Updated unresolved-blocker list

Sau Round 5, các blocker thực sự còn lại là:

### Blocker A — exact `FUN_0041438A`

Cần truy đến jump-table leaf để xác nhận:

```text
FPREM vs FPREM1
operand order
special cases
```

Đây là blocker cuối của Winner Decision Tree.

### Blocker B — exact post-IIR identity of `B`

Reverse-copy primitive và final recurrence đã rõ.

Cần chỉ còn một việc:

```text
xác nhận chính xác buffer B đi qua section nào,
và buffer F tại thời điểm kết hợp đại diện cho tín hiệu nào.
```

Điều này là data-flow labeling, không còn là blocker về công thức recurrence.

### Blocker C — lý do thiết kế của phase search base 35

Đã biết:

```text
write bins start at 0
search starts at 35
buffer = period + 40
```

Nhưng chưa biết semantic reason của con số 35.

Không ảnh hưởng đến việc clone algorithm nếu hard-code đúng behavior:

```text
search_start = 35
```

### Blocker D — exact zero-fill implementation trong `FUN_004017A0`

FFT length đã đóng.

Còn cần xác nhận:

```text
real[0..N-1] = input
real[N..Nfft-1] = 0
imag[0..Nfft-1] = 0
```

ở mức instruction.

---

# 37.19. Revised confidence ledger

| Item                                                          | Status     |
| ------------------------------------------------------------- | ---------- |
| Candidate acceptance `avg > 0.7 * seed`                       | PROVEN     |
| Online replacement `avg > 1.2 * best`                         | PROVEN     |
| No visible per-candidate status guard before winner selection | PROVEN     |
| Harmonic correction activated below 285.714 ms period         | PROVEN     |
| Harmonic correction uses `+10` and `20`                       | PROVEN     |
| Harmonic correction is remainder-based                        | STRONG     |
| `FUN_0041438A` exact intrinsic                                | UNRESOLVED |
| Reverse-copy primitive                                        | PROVEN     |
| Final backward recurrence                                     | PROVEN     |
| `F[0] = B[N-2]`                                               | PROVEN     |
| `F[k] = B[N-2-k] - F[k-1]`                                    | PROVEN     |
| `F[N-1] = -F[N-2]`                                            | PROVEN     |
| Phase accumulation writes to bins starting at 0               | PROVEN     |
| Phase peak search starts at 35                                | PROVEN     |
| `35` is not subtracted from public offset                     | PROVEN     |
| Public offset correction = `6.2 + 22.5 = 28.7 ms`             | PROVEN     |
| Partial final cycle included                                  | PROVEN     |
| Full-file BPM analysis                                        | PROVEN     |
| FFT sizing depends on `N + round(N/2)`                        | PROVEN     |
| Equivalent `Nfft ≈ next_power_of_two(1.5N)`                   | STRONG     |
| Exact FFT zero-fill instructions                              | PENDING    |
| Final semantic identity of forward buffer `F`                 | PENDING    |
| Exact reason for phase search base 35                         | UNRESOLVED |
| Classification bitmask aggregation                            | PROVEN     |
| Classification bit semantic labels                            | UNRESOLVED |
| `FUN_00413958` exact special-value behavior                   | UNRESOLVED |

# 38. Ground-Truth Consolidation Round 6

## 38.1. `FUN_0041438A` được xác định là dispatcher cho `fmod` ở tầng CRT

Dữ liệu tại:

```text
0x0041C1A0:
    04 66 6D 6F 64 ...
```

tương ứng với chuỗi length-prefixed:

```text
"\x04fmod"
```

Do đó descriptor được `FUN_0041438A` nạp qua:

```asm
0041438A  MOV EDX,0x41c1a0
0041438F  JMP 0x00415330
```

đã xác định rõ dispatcher này phục vụ intrinsic:

```text
fmod
```

### Kết luận

Có thể nâng confidence từ:

```text
FUN_0041438A = UNKNOWN floating intrinsic
```

thành:

```text
FUN_0041438A = MSVC CRT fmod dispatcher
```

Tuy nhiên vẫn chưa được phép đóng bit-exact semantics ở callsite cho tới khi xác định leaf implementation:

```text
fmod
    → __cintrindisp2
    → __trandisp2
    → dispatch handler
    → actual arithmetic
```

Các vấn đề còn lại:

```text
FPREM hay FPREM1
operand order
special-value behavior
```

vẫn pending.

---

## 38.2. Operand stack tại harmonic correction

Ngay trước call:

```asm
00402838  FLD  [candidate_period]
0040283A  FADD 10.0
00402840  FLD  [best_period]
00402843  CALL FUN_0041438A
```

x87 stack có:

```text
ST(0) = best_period
ST(1) = candidate_period + 10.0
```

Do dispatcher đã được xác định là `fmod`, semantics mong đợi của nhánh này là một phép remainder dùng hai giá trị trên.

Biểu diễn ở mức algorithmic vẫn nên giữ:

```text
remainder = fmod(candidate_period + 10.0,
                 best_period)
```

nhưng đánh dấu:

```text
operand mapping = not yet leaf-proven
```

Không được thay bằng một phép modulo integer hoặc một implementation language-level bất kỳ.

---

# 38.3. Hai post-IIR call sau reverse-copy dùng cùng coefficient section `0x419544`

Đoạn:

```asm
00402150  PUSH 0x419544
00402162  CALL FUN_00401580

0040216A  PUSH 0x419544
00402177  CALL FUN_00401580
```

chứng minh hai lời gọi liên tiếp đều sử dụng:

```text
coeff_base = 0x00419544
section_count = 1
```

và `0x419544` chính là:

```text
[0, 0, -1.4, 0.48, 0.2]
```

Do đó phase preprocessing có hai lần áp dụng cùng một section post-IIR.

Điểm quan trọng:

```text
IIR #1 coefficient = 0x419544
IIR #2 coefficient = 0x419544
```

Không được mô hình hóa chúng thành hai section khác nhau.

---

## 38.4. Buffer identity của hai post-IIR vẫn cần giữ traceable

Assembly cho thấy giữa hai call có thay đổi rõ ràng về register/local-object state:

```asm
00402155  MOV ESI,EDI
...
0040216F  LEA ESI,[EBP + -0x58]
```

Do đó không nên mặc định rằng:

```text
IIR1(input) → IIR2(same input)
```

trên cùng một object mà không kiểm tra convention của `FUN_00401580`.

Một số local vector objects được tái cấu trúc giữa hai call.

### Contract an toàn

Đã đóng:

```text
2 × one-section IIR
same section @ 0x419544
```

Chưa đóng:

```text
exact identity of the vector object receiving each call
```

Nút này chỉ còn là data-flow labeling, không còn là unknown về coefficient.

---

# 38.5. `FUN_00404CC0` chắc chắn thực hiện một thao tác reverse-copy trên một vector range

`FUN_00404CC0` gọi:

```text
FUN_00404E00(...)
```

và `FUN_00404E00` gọi:

```text
FUN_00404FA0(...)
```

Trong `FUN_00404FA0`:

```asm
00404FB4  FLD  [ECX - 4]
00404FB7  FSTP [EAX]
00404FB9  SUB  ECX,4
00404FBC  ADD  EAX,4
```

nên source được đọc từ cuối về đầu, destination tăng dần.

Semantics:

```python
dst[i] = src[n - 1 - i]
```

đối với range được truyền.

### Giới hạn kết luận

Callsite hiện tại truyền:

```text
piVar2[1]
*piVar2
local_4c
```

nhưng output mới **chưa đủ để khẳng định tuyệt đối rằng source range chính là resampled buffer**.

Do đó thiết kế nên ghi:

```text
FUN_00404CC0 = reverse-copy of the vector range supplied by the callsite
```

thay vì:

```text
FUN_00404CC0 = reverse(resampled_signal)
```

cho đến khi alias relationship giữa `piVar2` và `local_5c` được xác nhận.

Đây là một distinction quan trọng cho bit-exact implementation.

---

# 38.6. FFT sizing formula đã được đóng

Trong `FUN_004017A0`:

```asm
004017CA  MOV EAX,[ESI + 4]
004017CD  SUB EAX,[ESI]
004017CF  ...
004017D2  SAR EAX,2
```

tạo:

```text
N = input vector length
```

Sau đó:

```text
L = N + param_2
```

và exponent được tính từ:

```text
ceil(log2(L - 0.5))
```

sau đó:

```text
NFFT = 1 << exponent
```

Do `L` là số nguyên dương, biểu diễn này tương đương với:

```python
NFFT = smallest_power_of_two_geq(L)
```

hay:

```python
NFFT = next_power_of_two(N + param_2)
```

### Đây là ground truth mới

Không được sử dụng:

```python
NFFT = next_power_of_two(N)
```

nữa.

---

# 38.7. `param_2` của `FUN_004017A0` chính là số lag/autocorrelation outputs được yêu cầu

Ở caller `FUN_00402280`, binary lấy độ dài input:

```text
N
```

sau đó tính một giá trị từ:

```text
N * 0.5
```

và truyền giá trị này làm `param_2` của `FUN_004017A0`.

Với `N >= 0`, sequence floating-point conversion trong caller cho kết quả tương đương:

```python
nlags = floor(N / 2)
```

ở miền audio hợp lệ.

Do đó:

```text
param_2 = nlags
```

và:

```text
nlags = floor(N/2)
```

---

# 38.8. FFT length thực tế

Kết hợp hai phần trên:

```python
N = len(input_signal)
nlags = floor(N / 2)

L = N + nlags

NFFT = next_power_of_two(L)
```

Tức:

```text
NFFT ≈ next_power_of_two(1.5 N)
```

cho `N` lớn.

Ví dụ:

```text
N = 240,000
nlags = 120,000
L = 360,000
NFFT = 524,288
```

chứ không phải:

```text
262,144
```

như mô hình cũ `next_power_of_two(N)`.

---

# 38.9. Output của `FUN_004017A0` không dài `NFFT`

Đây là correction quan trọng.

Sau FFT → magnitude squared → IFFT, hàm không trả toàn bộ:

```text
NFFT
```

samples.

Phần allocation output dùng:

```text
param_2
```

làm target size:

```c
local_1c = param_2 - current_output_size;
```

và nếu còn thiếu:

```text
for remaining:
    *puVar7 = 0
```

Sau đó writer cuối hàm chỉ ghi:

```text
iVar5 = 0 .. param_2-1
```

vào output vector.

Vì:

```text
param_2 = nlags
```

nên:

```text
len(autocorrelation_output) = nlags
                       = floor(N/2)
```

### Không phải:

```text
NFFT
```

và cũng không phải:

```text
N
```

---

# 38.10. Peak detector thực tế chỉ có quyền truy cập tối đa `floor(N/2)` lags

Caller `FUN_00402280`:

```asm
004022CD  input_end - input_begin
004022D5  / 4
...
0040232E  local_40 = 0x7d0
00402335  CMP EAX,0x7d0
0040233C  MOV local_40,EAX
```

và `EAX` ở đây chính là giá trị `nlags` truyền vào `FUN_004017A0`.

Do đó:

```python
scan_limit = min(nlags, 2000)
```

chứ không phải:

```python
scan_limit = min(N, 2000)
```

### Đây là correction bắt buộc đối với thiết kế trước

Peak loop bắt đầu từ:

```text
i = 2
```

và chạy trước:

```text
scan_limit
```

nên miền peak detection là xấp xỉ:

```text
2 <= lag < min(floor(N/2), 2000) - 2
```

với các điều kiện biên exact như assembly.

---

# 38.11. Hệ quả: toàn bộ file vẫn được dùng, nhưng chỉ một nửa số lag được materialize

Pipeline BPM hiện được hiểu chính xác hơn:

```text
full audio input
    │
    ▼
preprocessed vector, length N
    │
    ▼
FFT size:
    NFFT >= N + floor(N/2)
    │
    ▼
autocorrelation via FFT/IFFT
    │
    ▼
retain / return only:
    floor(N/2) lag values
    │
    ▼
peak detector
    │
    └── inspect only first 2000 lags maximum
```

Điều này giải thích được sự khác biệt giữa:

```text
analysis scope
```

và:

```text
peak-search scope
```

Không có contradiction:

* FFT dùng toàn bộ file;
* IFFT tạo miền correlation dài;
* helper chỉ xuất `N/2` lag đầu;
* peak detector chỉ cần tối đa 2000 lag.

---

# 38.12. Phân biệt ba loại "zero" trong `FUN_004017A0`

Không được gom tất cả thành một khái niệm “zero-padding”.

Có ba operation khác nhau:

### A. Imaginary component initialization

Khi copy input vào complex FFT buffer:

```text
real[i] = input[i]
imag[i] = 0.0
```

được ghi trực tiếp.

### B. FFT-domain tail

Input audio chỉ có:

```text
N samples
```

trong khi complex buffer có:

```text
NFFT samples
```

Phần:

```text
N .. NFFT-1
```

phải có trạng thái zero để thực hiện zero-padding.

Excerpt hiện tại không chứa loop trực tiếp ghi:

```text
real[N..NFFT-1] = 0
```

Do đó cơ chế instruction-level chính xác của phần này nằm trong semantics của:

```text
FUN_004044A0
```

hoặc helper/container được gọi trong bước allocation.

Không nên tự ghi rằng binary có một explicit `memset` cho FFT tail nếu chưa thấy instruction đó.

### C. Output-vector extension

Đây lại là operation đã được chứng minh trực tiếp:

```text
output size target = param_2
```

và phần allocation còn thiếu được ghi bằng:

```text
0
```

Operation này **không phải FFT zero-padding**.

---

# 38.13. Scaling của autocorrelation output

Sau IFFT/magnitude stage:

```c
fVar9 = DAT_0041A718 / (float)iVar6;
```

với:

```text
iVar6 = NFFT
```

Sau đó output được tạo bằng:

```text
output[k] = reconstructed_real[k] * fVar9
```

Nếu `DAT_0041A718 = 1.0`, đã được nhận diện từ constant audit, thì:

```text
output[k] = reconstructed_real[k] / NFFT
```

Do đó phép scale `1/NFFT` tiếp tục được giữ trong implementation contract.

---

# 38.14. Exact BPM autocorrelation contract

Phần này hiện có thể viết thành:

```python
def fft_autocorrelation(signal):
    N = len(signal)

    nlags = floor(N / 2)

    L = N + nlags

    NFFT = next_power_of_two(L)

    complex_buf = complex_zeros(NFFT)

    for i in range(N):
        complex_buf[i].real = signal[i]
        complex_buf[i].imag = 0.0

    # exact FFT implementation
    spectrum = FFT(complex_buf)

    for i in range(NFFT):
        spectrum[i] = abs2(spectrum[i])

    corr_complex = IFFT(spectrum)

    corr = [corr_complex[i].real / NFFT
            for i in range(nlags)]

    return corr
```

Trong đó hai điểm vẫn cần giữ traceable:

```text
complex tail zero initialization = allocator/helper semantics pending
FFT/IFFT numerical implementation = binary-specific
```

Không thay bằng thư viện `numpy.correlate()` nếu mục tiêu là functional/bit-exact clone.

---

# 38.15. Revised BPM detector domain

Sau Round 6:

```text
Input:
    N-sample processed signal

FFT:
    NFFT = next_power_of_two(N + floor(N/2))

Correlation output:
    C[0 .. floor(N/2)-1]

Peak scan:
    i < min(floor(N/2), 2000)
```

Do đó candidate seed peak tối đa được sinh từ:

```text
lag < 1000
```

vẫn đúng đối với các file có:

```text
floor(N/2) >= 1000
```

nhưng giới hạn này không phải do một vector correlation có chiều dài 2000.

Nó là:

```text
min(output_lag_count, 2000)
```

---

# 38.16. Updated blocker ledger

Sau Round 6:

| Item                                                    | Status                            |
| ------------------------------------------------------- | --------------------------------- |
| `0x41C1A0` identifies CRT `fmod` dispatcher             | PROVEN                            |
| Harmonic correction uses `fmod` dispatcher              | PROVEN                            |
| Exact `FPREM`/`FPREM1` leaf                             | UNRESOLVED                        |
| Harmonic operand mapping                                | STRONG, leaf pending              |
| Harmonic thresholds `+10`, `<=20`                       | PROVEN                            |
| Two post-IIR calls use `0x419544`                       | PROVEN                            |
| Exact identity of each post-IIR buffer                  | PENDING                           |
| Reverse-copy primitive                                  | PROVEN                            |
| Reverse-copy source range in call                       | PROVEN structurally               |
| Reverse-copy = specifically resampled signal            | PENDING                           |
| `nlags = floor(N/2)` for normal audio input             | PROVEN                            |
| FFT sizing criterion `N + nlags`                        | PROVEN                            |
| `NFFT = next_power_of_two(N + nlags)`                   | PROVEN                            |
| FFT output/correlation materialization length = `nlags` | PROVEN                            |
| Peak scan limit = `min(nlags, 2000)`                    | PROVEN                            |
| Full-file analysis scope                                | PROVEN                            |
| Imaginary input component initialized to 0              | PROVEN                            |
| Exact FFT-tail zero initialization mechanism            | PENDING                           |
| Autocorrelation scale `1/NFFT`                          | PROVEN given `DAT_0041A718 = 1.0` |
| Phase search base 35                                    | PROVEN                            |
| Phase accumulation starts at bin 0                      | PROVEN                            |
| Public offset correction = 28.7 ms                      | PROVEN                            |

# 39. Ground-Truth Consolidation Round 7

## 39.1. `FUN_0041438A` đã được giải mã bit-exact ở mức phép toán remainder

Toàn bộ binary chứa instruction:

```asm
0x00414394: FXCH
0x00414396: FPREM
0x00414398: FSTSW AX
0x0041439B: WAIT
0x0041439C: SAHF
0x0041439D: JP 0x00414396
0x0041439F: FSTP ST1
```

và descriptor tại `0x0041C1A0` xác định dispatcher này là intrinsic:

```text
fmod
```

### Stack order

Ngay trước call trong harmonic correction:

```asm
FLD [candidate_period]
FADD 10.0
FLD [best_period]
CALL FUN_0041438A
```

nên:

```text
ST(0) = best_period
ST(1) = candidate_period + 10.0
```

Leaf `0x00414394` thực hiện:

```asm
FXCH
```

do đó trở thành:

```text
ST(0) = candidate_period + 10.0
ST(1) = best_period
```

Sau đó:

```asm
FPREM
```

thực hiện remainder của `ST(0)` theo `ST(1)`:

```text
remainder =
    fmod(candidate_period + 10.0,
         best_period)
```

Cuối cùng:

```asm
FSTP ST1
```

loại divisor và giữ remainder trên x87 stack.

### Kết luận

Harmonic correction có thể được đóng thành:

```python
remainder = fmod(candidate_period + 10.0,
                 best_period)

match = (remainder < 20.0)
```

Trong assembly:

```asm
FLD 20.0
FCOMIP ST0,ST1
JNC  accept
```

cần giữ đúng semantics của `COMIP` và branch condition. Do đó implementation exact nên bám branch assembly thay vì tự chuyển thành phép so sánh ngôn ngữ cấp cao nếu cần tái hiện mọi trường hợp biên.

Điểm quan trọng nhất:

```text
intrinsic = FPREM
```

không phải:

```text
FPREM1
```

và operand order là:

```text
(candidate_period + 10) mod best_period
```

Đây là ground truth.

---

# 39.2. Harmonic condition có thể viết dưới dạng cửa sổ ±10 ms quanh bội số của period

Với:

```text
r = fmod(P + 10, T)
```

và điều kiện remainder theo branch binary:

```text
r < 20
```

có thể diễn giải trong miền period dương thông thường thành:

```text
P ≈ k*T ± 10 ms
```

với một số nguyên `k`.

Do đó:

```text
+10
```

không phải correction cho period cuối cùng.

Nó là một phần của cách binary biến tolerance ±10 ms thành phép remainder.

Tương tự:

```text
20
```

không phải “20 ms maximum BPM error”.

Đó là giới hạn của remainder sau khi dịch candidate thêm 10 ms.

---

# 39.3. Hai post-IIR `0x419544` thực sự hoạt động trên hai buffer khác nhau

Call sequence:

```asm
0x0040212F: CALL FUN_00404CC0

0x00402150: PUSH 0x419544
0x00402155: MOV  ESI,EDI
0x00402162: CALL FUN_00401580

0x0040216A: PUSH 0x419544
0x0040216F: LEA  ESI,[EBP-0x58]
0x00402177: CALL FUN_00401580
```

Cho thấy hai invocation dùng cùng:

```text
coeff = 0x419544
count = 1
```

nhưng `ESI` của chúng khác nhau.

### IIR #1

Trước call:

```text
ESI = EDI
```

`EDI` là vector forward được dùng ngay sau resample và là vector nguồn của thao tác reverse-copy.

Do đó IIR #1 tác động lên **forward/output buffer**.

### IIR #2

Trước call:

```text
ESI = &local_58
```

và các field của object này vừa được gán từ object được tạo bởi `FUN_00404CC0`:

```text
local_58 = local_44
local_5c = local_48
local_54 = local_40
```

Do đó IIR #2 tác động lên **buffer reverse-copy/accumulator**.

### Kết luận

Đây không phải:

```text
same_buffer = IIR(IIR(x))
```

một cách tuần tự.

Mô hình đúng là:

```text
forward buffer F_raw
        │
        └── IIR(section 0x419544)
                    ↓
                  F

reverse-copy buffer B_raw
        │
        └── IIR(section 0x419544)
                    ↓
                  B
```

Sau đó `F` và `B` mới được kết hợp.

---

# 39.4. `FUN_00404CC0` tạo nhánh reverse-copy trước post-IIR

Trình tự chính xác:

```text
resampled forward vector
        │
        ├──────────────→ forward IIR
        │
        └→ FUN_00404CC0
              │
              └→ reverse-copy buffer
                        │
                        └→ reverse-branch IIR
```

`FUN_00404FA0` đọc source từ:

```text
src[-1], src[-2], ...
```

và ghi destination theo chiều tăng:

```text
dst[0], dst[1], ...
```

Do đó:

```python
B_raw[i] = F_raw[N - 1 - i]
```

trên range được truyền.

---

# 39.5. Công thức alternating recurrence chính xác, với định danh buffer rõ ràng

Đặt:

```text
F = forward buffer sau post-IIR
B = reverse-copy buffer sau post-IIR
N = số mẫu
```

Phần kết hợp cuối binary thực hiện:

```text
F[0] = B[N-2]

F[k] = B[N-2-k] - F[k-1]
        với 1 <= k <= N-2

F[N-1] = -F[N-2]
```

Đây chính là formula mà các round trước đã phát hiện, nhưng giờ `B` được định danh rõ nên không còn ambiguity.

### Một cách tương đương theo hướng duyệt từ cuối

Binary thực tế ghi các phần tử từ cuối buffer về đầu:

```text
F[N-2] = B[0]     - F[N-3]
F[N-3] = B[1]     - F[N-4]
F[N-4] = B[2]     - F[N-5]
...
F[1]   = B[N-3]   - F[0]
F[0]   = B[N-2]
F[N-1] = -F[N-2]
```

Hai biểu diễn trên là cùng một transform.

### Không được mô hình hóa thành

```python
F = reverse(B)
```

hoặc:

```python
F = F_forward - reverse(F_forward)
```

hay:

```python
F = filtfilt(...)
```

Đó đều là các phép biến đổi khác.

---

# 39.6. Đây là một feedback-like recurrence, không phải pure time reversal

Mặc dù chỉ số của `B` giảm từ cuối về đầu:

```text
B[N-2], B[N-3], ...
```

số hạng thứ hai của mỗi phép trừ lại lấy từ:

```text
F[k-1]
```

Do đó biến đổi cuối cùng có cấu trúc:

```text
reverse-index input
        +
one-step recursive subtraction
        +
special endpoint
```

Không nên gọi toàn bộ stage là:

```text
time reversal
```

Chính xác hơn:

```text
reverse-copy + parallel filtering + reverse-index alternating combination
```

---

# 39.7. `FUN_004044A0` xác nhận zero-init của complex FFT tail

`FUN_004017A0` dùng `FUN_004044A0` để resize complex vectors có stride:

```text
8 bytes / complex element
```

tức:

```text
float real
float imag
```

Khi vector cần mở rộng, `FUN_004044A0` gọi:

```text
FUN_00404870()
FUN_00404D30()
```

Trong `FUN_00404D30`:

```c
for (count):
    element[0] = 0;
    element[1] = 0;
```

Do đó mỗi complex element mới được khởi tạo thành:

```text
(0.0f, 0.0f)
```

### Hệ quả

Khi `FUN_004017A0` tạo FFT buffer có:

```text
NFFT > N
```

sau đó copy:

```text
real[0..N-1] = input
imag[0..N-1] = 0
```

thì phần tail mới của complex buffer:

```text
N .. NFFT-1
```

đã được zero-init.

Vì vậy functional clone có thể mô hình hóa:

```python
fft_buf = complex_zeros(NFFT)

for i in range(N):
    fft_buf[i] = complex(input[i], 0.0)
```

mà không cần giả định một giá trị rác ban đầu.

---

# 39.8. Phân biệt zero-init của FFT buffer với zero-init của output correlation vector

Có hai nơi zero-init hoàn toàn khác nhau.

### FFT complex buffer

```text
element size = 8 bytes
(real, imag)

new elements:
    real = 0
    imag = 0
```

### Correlation output vector

```text
element size = 4 bytes
(float)

target size = param_2 = nlags
```

Output vector cũng được zero-fill phần capacity mới trước khi ghi correlation samples.

Do đó không được suy luận rằng:

```text
zero-padding = output correlation zero-fill
```

Hai operation khác nhau ở hai container khác nhau.

---

# 39.9. `FUN_004017A0` hiện có implementation contract hoàn chỉnh ở mức container/data-flow

Có thể viết:

```python
def fft_autocorrelation(signal):
    N = len(signal)

    nlags = floor(N / 2)

    L = N + nlags

    NFFT = next_power_of_two(L)

    fft_buf = complex_zeros(NFFT)

    for i in range(N):
        fft_buf[i].real = signal[i]
        fft_buf[i].imag = 0.0

    X = FFT(fft_buf)

    for i in range(NFFT):
        X[i] = X[i].real * X[i].real + X[i].imag * X[i].imag
        # implementation retains complex storage,
        # with imaginary part set to zero

    Y = IFFT(X)

    scale = 1.0 / float(NFFT)

    corr = [Y[i].real * scale for i in range(nlags)]

    return corr
```

Các chi tiết còn binary-specific:

```text
FFT butterfly implementation
IFFT implementation
floating-point rounding
```

nhưng contract về kích thước, zero-tail và output length đã đóng.

---

# 39.10. Peak detector domain sau Round 7

Vì:

```text
correlation_length = nlags = floor(N/2)
```

nên:

```python
scan_limit = min(nlags, 2000)
```

không phải:

```python
min(N, 2000)
```

Candidate peak scan sử dụng vector correlation đã materialize này.

Với audio đủ dài để:

```text
floor(N/2) >= 1000
```

seed domain vẫn là:

```text
lag < 1000
```

như các round trước.

Nhưng nguồn của giới hạn `1000` phải được hiểu chính xác:

```text
candidate_cutoff = 1000
```

chứ không phải:

```text
correlation_length = 1000
```

---

# 39.11. Main xác nhận BPM và Offset dùng cùng processed vector

Sau:

```asm
CALL FUN_00401E80
```

main tiếp tục giữ vector object trong vùng local tương ứng.

Sau đó:

```asm
CALL FUN_00402280
```

được thực hiện với pointer tới cùng vector object.

Ở nhánh offset:

```asm
CALL FUN_00402C70
```

cũng truyền pointer tới cùng vector object đó.

Vì vậy:

```text
FUN_00401E80 output
        │
        ├── FUN_00402280 → BPM
        │
        └── FUN_00402C70 → phase/offset
```

không có một preprocessing transform riêng giữa hai nhánh.

### Implementation consequence

Functional clone nên đảm bảo:

```text
processed_signal
```

là cùng một vector logic được truyền cho cả:

```text
estimate_bpm(processed_signal)
estimate_offset(processed_signal, bpm)
```

trừ khi một caller-level copy chỉ phục vụ memory ownership.

---

# 39.12. Updated complete data flow

```text
BASS decode
    ↓
44100 Hz float PCM
    ↓
multichannel sum → mono
    ↓
5-pass nonlinear filter bank
    ↓
log transform
    ↓
2-section IIR
    ↓
nearest-neighbor resample → ~1000 Hz
    ↓
┌──────────────────────────────────────────────┐
│ Post-IIR / phase preprocessing               │
│                                              │
│ Forward branch ── IIR @ 0x419544 ── F       │
│                                              │
│ Reverse-copy branch                          │
│      └─ reverse-copy ── IIR @ 0x419544 ── B │
│                                              │
│ F/B combined by reverse-index recurrence     │
└──────────────────────────────────────────────┘
    ↓
processed_signal
    │
    ├──────────────────────┐
    ↓                      ↓
BPM path                 Offset path
    │                      │
    ↓                      ↓
FFT autocorrelation      phase folding
    │                      │
    ↓                      ↓
nlags=floor(N/2)         period-based
    │                    accumulation
    ↓                      │
NFFT >= N+nlags             ↓
    │                    peak + parabolic
    ↓                    refinement
retain first nlags          │
    │                       ↓
    ↓                    -6.2
peak scan                    │
    ↓                       ↓
candidate tracking       FUN_00402C70
    ↓                       │
winner update               ↓
    ↓                    offset_raw
harmonic correction          │
    ↓                       ↓
BPM                     main -22.5
                            │
                            ↓
                         offset_printed
```

---

# 39.13. Remaining blockers after Round 7

### Blocker A — hoàn tất special-value semantics của `fmod`

Phép toán chính đã đóng:

```text
FPREM
ST0 = candidate+10
ST1 = best
```

Còn có thể tiếp tục truy:

```text
NaN
Inf
zero divisor
exception-mask behavior
```

nhưng các trường hợp này không còn cản trở việc implement normal harmonic path.

### Blocker B — semantic identity của 5 classification streams

`FUN_00402FC0` vẫn cần giải mã:

```text
V0
V1
V2
V3
V4
```

và điều kiện tạo:

```text
#
+
-
.
```

Nhưng phần này nằm sau BPM/offset core, không còn là blocker của BPM estimator.

### Blocker C — lý do semantic của search base 35

Đã đóng behavior:

```text
write bins: 0 .. period+39
search bins: 35 .. 35+period-1
```

Chưa biết tại sao tác giả chọn:

```text
35
```

Điều này không ngăn functional clone.

### Blocker D — exact FFT numerical kernel

Kích thước và container behavior đã đóng.

Còn lại:

```text
radix
twiddle evaluation
rounding mode interactions
inverse normalization internals
```

nếu mục tiêu chuyển từ functional equivalence sang bit-exact numerical equivalence.

### Blocker E — exact semantics của `FUN_00413958`

Vẫn cần đặc biệt kiểm tra:

```text
negative input
±0
subnormal
±Inf
NaN
overflow
```

---

# 39.14. Revised confidence ledger

| Item                                                       | Status     |
| ---------------------------------------------------------- | ---------- |
| `FUN_0041438A` = `fmod` dispatcher                         | PROVEN     |
| Leaf uses `FPREM`                                          | PROVEN     |
| Leaf operand order                                         | PROVEN     |
| Harmonic expression `(candidate+10) mod best`              | PROVEN     |
| Harmonic `20` branch threshold                             | PROVEN     |
| Two post-IIR calls use `0x419544`                          | PROVEN     |
| Post-IIR calls target two distinct buffers                 | PROVEN     |
| Reverse-copy primitive                                     | PROVEN     |
| Forward/reverse parallel filtering model                   | PROVEN     |
| Alternating recurrence with explicit F/B identity          | PROVEN     |
| `F[0] = B[N-2]`                                            | PROVEN     |
| `F[k] = B[N-2-k] - F[k-1]`                                 | PROVEN     |
| `F[N-1] = -F[N-2]`                                         | PROVEN     |
| FFT complex elements are zero-initialized on vector growth | PROVEN     |
| FFT tail `N..NFFT-1` is zero                               | PROVEN     |
| `nlags = floor(N/2)`                                       | PROVEN     |
| `NFFT = next_power_of_two(N+nlags)`                        | PROVEN     |
| correlation output length = `nlags`                        | PROVEN     |
| peak scan limit = `min(nlags,2000)`                        | PROVEN     |
| full processed vector shared by BPM and Offset             | PROVEN     |
| public offset correction = 28.7 ms                         | PROVEN     |
| semantic reason for phase search base 35                   | UNRESOLVED |
| classification bit meanings                                | UNRESOLVED |
| FFT numerical kernel                                       | UNRESOLVED |
| `FUN_00413958` special-value semantics                     | UNRESOLVED |

# 40. Ground-Truth Consolidation Round 8

## 40.1. `FUN_00413958`: xác nhận custom logarithm, nhưng special-value return vẫn không nên tự bịa

`FUN_00413958` là một implementation logarithm tùy biến, không gọi trực tiếp CRT `log`.

Nó thực hiện:

```text id="2f5b0x"
1. lấy exponent/mantissa từ IEEE-754 input
2. range reduction về khoảng gần 1
3. dùng LUT tại vùng ~0x41A868
4. cộng polynomial / correction terms
5. xử lý riêng subnormal / zero / Inf / NaN
```

Các constant:

```text id="rfr06y"
0x0041A7C0 = 0.6931471805598903
0x0041A7C8 = 5.497923018708371E-14
```

có tổng gần:

```text id="e1olh8"
ln(2)
```

phù hợp với decomposition:

```text
log(x) = exponent * ln(2) + log(mantissa)
```

### Normal-number path

Nếu exponent thuộc miền normal:

```text id="8vlhsw"
1 <= exponent <= 2046
```

decompile đi qua nhánh:

```c
uVar2 = exponent - 1;
if (uVar2 < 0x7fe) {
    return;
}
```

sau khi đã tính phần range-reduced approximation.

Điều này xác nhận đây là một custom normal-path logarithm.

---

## 40.2. Subnormal path của `FUN_00413958`

Khi exponent field bằng 0:

```text id="lnc4ra"
uVar2 == 0xffffffff
```

binary không thoát ngay, mà có:

```c
auVar4._0_8_ = dVar3 * DAT_0041a810;
```

với:

```text id="vyc86c"
DAT_0041A810 = 2^52
```

Đây là dấu hiệu rõ ràng của chiến lược:

```text id="fvx6kl"
subnormal x
    → scale by 2^52
    → reclassify / renormalize exponent
    → continue logarithm decomposition
```

Do đó không được coi subnormal là một trường hợp “clamp về 0”.

---

## 40.3. Zero, infinity và NaN có branch riêng

Decompile cũng cho thấy kiểm tra:

```text id="n3wse2"
_DAT_0041A800 == dVar3
```

trong nhánh exponent = 0 và kiểm tra exponent/mantissa cho miền:

```text id="fh4o14"
0x7ff
```

ở cuối hàm.

Điều này chứng minh:

```text id="l9s6wy"
zero
infinity
NaN
```

được phân loại riêng thay vì bị đưa thẳng qua normal logarithm approximation.

### Nhưng chưa đóng exact return payload

Ghidra hiện hiển thị nhiều nhánh là:

```c
return;
```

trong một hàm mà caller kỳ vọng kết quả double quay về trong `XMM0`.

Đây có khả năng là artefact của decompiler / register tracking:

```text id="q5qtgj"
return value register state
```

chưa được khôi phục đầy đủ.

Vì vậy thiết kế **không được** tự ghi:

```python id="a8d7jx"
x <= 0  -> -inf
x == 0  -> -inf
x < 0   -> NaN
x == inf -> inf
x == NaN -> NaN
```

chỉ dựa trên kiến thức về `log()` chuẩn.

### Status

```text id="cbd2a1"
custom log implementation = PROVEN
subnormal renormalization = PROVEN
special-case dispatch = PROVEN
exact returned values for special cases = UNRESOLVED
```

Đối với đường dữ liệu bình thường của Timing Analyzer, vấn đề này không phải blocker chính vì envelope sau nonlinear stage được kỳ vọng nằm trong miền dương hữu hạn.

---

# 40.4. `FUN_00402FC0`: ý nghĩa của các timeline flags đã được xác nhận qua UI contract

Main ánh xạ:

```text id="1mb6h7"
bit 0x1 → '#'
bit 0x2 → '+'
bit 0x4 → '-'
bit 0x8 → '.'
```

và source strings mô tả trực tiếp ý nghĩa của chúng.

### Bit `0x1` / `#`

UI:

```text id="w4l7y7"
"Portions marked with '#' in the timeline probably need additional timing section(s)."
```

Do đó:

```text id="bqjse2"
0x1 = timing-section / timing-structure warning
```

Semantic interpretation:

```text
một vùng timeline có thể không được mô hình hóa tốt
bởi một BPM/period duy nhất
```

và binary gợi ý cần thêm timing section.

Không nên biến nó thành:

```text id="i2f6uw"
"hard BPM change detected"
```

vì source chỉ nói:

```text
probably need additional timing section(s)
```

---

# 40.5. Bit `0x2` / `+` và bit `0x4` / `-`

Main kiểm tra cả hai cùng lúc:

```asm id="6n2p1f"
TEST BL,0x6
```

và in:

```text
"Offset of portions marked with '+' or '-' probably have slightly
different offset than the given timing or calculation result."
```

Do đó:

```text id="xm8plk"
0x2 / '+' = offset deviation warning, positive-symbol class
0x4 / '-' = offset deviation warning, negative-symbol class
```

### Nhưng dấu vật lý chính xác của từng bit vẫn chưa được chứng minh hoàn toàn

Source/UI chứng minh cả hai đại diện cho:

```text
slight local offset difference
```

nhưng không đủ để kết luận chắc chắn rằng:

```text
0x2 = local offset > expected
0x4 = local offset < expected
```

hoặc ngược lại.

Do đó implementation contract nên giữ:

```text id="nxvvn4"
bit 0x2 -> '+'
bit 0x4 -> '-'
```

và chưa gắn thêm định nghĩa toán học về dấu nếu chưa trace điều kiện tạo bit.

---

# 40.6. Bit `0x8` / `.`

Main in:

```text id="l8dy55"
"There may be bpm drifts or offset jitter in portions marked with '.'."
```

Do đó:

```text id="49vffk"
0x8 = local instability warning:
    possible BPM drift or offset jitter
```

Đây là một classification khác với `#`.

So sánh:

```text
#  -> timing section may be needed
+/- -> local offset differs slightly
.  -> possible BPM drift / offset jitter
```

Các flag không đại diện cho một score duy nhất. Một timeline có thể mang nhiều bit cùng lúc.

---

# 40.7. `FUN_00402FC0` return value là aggregate OR-mask

Sau khi ghi flag cho từng timeline element:

```c
uVar3 = 0;
...
uVar3 = uVar3 | *puVar5;
```

và cuối hàm:

```text id="gmcps1"
return uVar3;
```

Do đó:

```python id="xcm7xe"
aggregate_mask = OR(all per-position flags)
```

Ví dụ:

```text
0x00 -> không có cảnh báo nào
0x01 -> có ít nhất một '#'
0x06 -> có '+' hoặc '-'
0x08 -> có '.'
0x0F -> cả bốn class đều xuất hiện
```

### Quan trọng

`0x06` không phải một class riêng.

Nó chỉ có nghĩa:

```text
bit 0x2 hoặc bit 0x4 đã xuất hiện
```

---

# 40.8. Classification thực tế nằm sau BPM/offset, không điều khiển BPM winner

Call order của main:

```text id="m3ck0t"
FUN_00401E80
    ↓
FUN_00402280
    ↓
FUN_00402C70
    ↓
FUN_00402FC0
```

`FUN_00402FC0` chỉ được gọi sau khi BPM và offset đã có.

Do đó timeline validation:

```text id="a7d0ef"
does not feed back into FUN_00402280
```

trong call graph hiện tại.

Không được implement:

```python id="3vxz2a"
BPM = candidate accepted only if timeline_validation == 0
```

trừ khi một caller khác chứng minh feedback path.

---

# 40.9. `FUN_004010B0`: xác nhận đây là FFT butterfly kernel, nhưng flag Forward/Inverse chưa được đóng

Thân `FUN_004010B0` rõ ràng thực hiện:

```text
bit-reversal / permutation
→ radix-2 butterfly stages
→ complex twiddle multiplication
```

Ví dụ:

```c
a = pfVar2
b = pfVar4

pfVar4 = a - b
pfVar2 = a + b
```

và các stage sau sử dụng bảng:

```text id="cbj9ne"
DAT_004211C0
DAT_004211C4
```

đã được `FUN_00401000` khởi tạo như bảng sin/cos.

Do đó có thể đóng:

```text
FUN_004010B0 = custom radix-2 complex FFT kernel
```

---

# 40.10. Không nên suy luận `param_1=0` từ decompile như một FFT-direction flag

Ghidra hiện phát hiện:

```c
FUN_004010b0(0,0);
```

nhưng signature:

```c
void __thiscall FUN_004010b0(byte param_1,
                             int param_2,
                             float *param_3)
```

có dấu hiệu calling-convention / register mapping chưa hoàn toàn đáng tin cậy.

Bản thân function dùng:

```text id="z0j7dm"
iVar6 = 1 << param_1
```

để xác định số phần tử / stage domain.

Điều này không phù hợp với giả thuyết:

```text id="mp4j7v"
param_1 = forward/inverse boolean
```

vì khi đó `1 << 0` hoặc `1 << 1` chỉ tạo FFT domain kích thước 1 hoặc 2.

### Kết luận

Không được ghi:

```text
0 = forward
1 = inverse
```

cho đến khi reconstruct lại callsite register-level.

---

# 40.11. Hai lần `FUN_004010B0` trong `FUN_004017A0`

`FUN_004017A0` gọi kernel hai lần:

```text id="j9bw3j"
Call #1
    ↓
magnitude squared
    ↓
Call #2
```

Giữa hai lần gọi có:

```c
real = real^2 + imag^2
imag = 0
```

Vì vậy pipeline chắc chắn là:

```text id="kq3b6v"
input
  ↓
FFT
  ↓
|X[k]|²
  ↓
inverse-transform
  ↓
autocorrelation
```

Đây chính là cấu trúc Wiener-Khinchin / power-spectrum autocorrelation.

### Tuy nhiên

Direction của từng call:

```text
Call #1 = forward?
Call #2 = inverse?
```

chưa nên đánh dấu là bit-exact ground truth từ signature Ghidra hiện tại.

---

# 40.12. Vùng phase `[0..34]` không có bước transform đặc biệt

Trong `FUN_00402C70`:

```text id="4vkq3m"
phase accumulation:
    phase[m] += signal[round(k*T) + m]
```

với:

```text id="5f7x8m"
m bắt đầu từ 0
```

Sau đó peak search bắt đầu:

```text id="zq7e3c"
m = 35
```

Không có code nào trong đoạn đã trace cho thấy:

```text
phase[0..34] = 0
```

sau accumulation,

hoặc:

```text
phase[m] = special_transform(phase[m])
```

trước peak scan.

Do đó vùng:

```text id="w3d0nf"
0 .. 34
```

chỉ là phần buffer được tạo ra nhưng không thuộc miền peak-search.

### Contract

```text
phase_buffer_size = integer_period + 40

accumulation:
    write starting from bin 0

peak search:
    start = 35
    search over approximately one period
```

Không được chèn thêm:

```python id="c1a40x"
phase = phase[35:]
```

vì điều đó làm thay đổi coordinate system của `phase_peak`.

---

# 40.13. Vì sao 35 vẫn chưa được semantic hóa

Có một quan hệ số học đáng chú ý:

```text
35 - 28.7 = 6.3 ms
```

gần `6.2`, nhưng đây chỉ là coincidence chưa đủ để biến:

```text id="xj58kt"
35
```

thành:

```text
35 = 6.2 + public_correction
```

Binary không thực hiện phép toán đó.

Các số có vai trò độc lập:

```text
35   = hard-coded peak-search start
6.2  = internal phase correction
22.5 = caller output correction
28.7 = 6.2 + 22.5
```

---

# 40.14. Updated near-final pipeline

```text
BASS decode @ 44.1 kHz
    ↓
channel sum → mono
    ↓
5 nonlinear filter passes
    ↓
custom log transform
    ↓
2-section IIR
    ↓
nearest-neighbor resample → ~1000 Hz
    ↓
parallel phase preprocessing:
    ├─ forward buffer → 0x419544 IIR
    └─ reverse-copy buffer → 0x419544 IIR
    ↓
reverse-index alternating combination
    ↓
processed analysis signal
    │
    ├──────────────────────────────┐
    ↓                              ↓
BPM path                         Offset path
    │                              │
    ↓                              ↓
FFT autocorrelation            phase folding
    │                              │
    ├─ FFT                        ├─ period = 60000/BPM
    ├─ |X|²                       ├─ round(kT)
    └─ inverse FFT                ├─ accumulate bins
    │                              ├─ search start 35
    ↓                              └─ parabola
return floor(N/2) lags                  ↓
    │                              phase - 6.2
    ↓                                   ↓
peak scan                         main - 22.5
    ↓                                   ↓
periodic tracker                  public offset
    ↓
online candidate winner
    ↓
harmonic correction:
    fmod(candidate + 10, best) < 20
    ↓
BPM
    │
    └──────────────┐
                   ↓
             Timeline validation
                   ↓
             per-position flags
                   ├─ # = 0x1
                   ├─ + = 0x2
                   ├─ - = 0x4
                   └─ . = 0x8
                   ↓
             aggregate OR-mask
```

---

# 40.15. Final unresolved ledger

| Item                                            | Status                                    |
| ----------------------------------------------- | ----------------------------------------- |
| `FUN_00413958` custom logarithm                 | PROVEN                                    |
| `FUN_00413958` subnormal renormalization        | PROVEN                                    |
| `FUN_00413958` zero/Inf/NaN branch existence    | PROVEN                                    |
| Exact special-value return payload              | UNRESOLVED                                |
| Timeline `#` mapping                            | PROVEN                                    |
| Timeline `+` mapping                            | PROVEN                                    |
| Timeline `-` mapping                            | PROVEN                                    |
| Timeline `.` mapping                            | PROVEN                                    |
| Exact mathematical sign interpretation of `+/-` | UNRESOLVED                                |
| Timeline aggregate OR-mask                      | PROVEN                                    |
| Timeline validation feeds BPM winner            | NOT OBSERVED                              |
| `FUN_004010B0` custom FFT kernel                | PROVEN                                    |
| First FFT call = forward                        | STRONG, not yet calling-convention-proven |
| Second FFT call = inverse                       | STRONG, not yet calling-convention-proven |
| Exact Forward/Inverse control parameter         | UNRESOLVED                                |
| Complex FFT tail zero-init                      | PROVEN                                    |
| `NFFT = next_power_of_two(N + floor(N/2))`      | PROVEN                                    |
| Correlation output length = floor(N/2)          | PROVEN                                    |
| Peak scan `min(floor(N/2),2000)`                | PROVEN                                    |
| Phase accumulation starts at bin 0              | PROVEN                                    |
| Peak search starts at bin 35                    | PROVEN                                    |
| No special transform on bins 0..34              | PROVEN for traced code                    |
| `35` semantic reason                            | UNRESOLVED                                |
| Public offset correction = 28.7 ms              | PROVEN                                    |