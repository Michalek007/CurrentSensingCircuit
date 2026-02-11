import numpy as np
from scipy.signal import windows
from scipy.fft import fft
import struct
import matplotlib.pyplot as plt

# OUT_FILE = "uart_capture.bin"
OUT_FILE = "uart_capture_G100_0u_2.bin"
# OUT_FILE = "uart_capture_G100_0u.bin"
# OUT_FILE = "uart_capture_0u.bin"
# OUT_FILE = "uart_capture_0u8.bin"
# OUT_FILE = "uart_capture_0u8_2.bin"
# OUT_FILE = "uart_capture_J2.bin"


MAX_ADC_VALUE = 2 ** 14 - 1  # after oversampling
VCC = 3.3  # V
V_REF = VCC/2  # reference level may need to be updated
S = 100_000  # V -> nA # 0.01mV/nA
# S = 1000 # 1mV/1nA


def FIR(fp, fr, fs, window):
    f0 = (fr + fp)/2
    alpha = 0
    if window == "boxcar":
        alpha = 0.9
    elif window == "bartlett":
        alpha = 3.3
    elif window == "hann":
        alpha = 3.1
    elif window == "hamming":
        alpha = 3.3
    elif window == "blackman":
        alpha = 5.6
    elif window == "parzen":
        alpha = 7.0
    elif window == "flattop":
        alpha = 9.7
    if alpha == 0:
        return 1, 0
    R = np.ceil(alpha*fs/(fr - fp))
    K = int(R) + 1
    n = np.linspace(0, K - 1, K)
    hx = 2*f0/fs*np.sinc(2*f0/fs*(n - R/2))
    w = windows.get_window(window, K, fftbins=False)
    h = np.multiply(hx, w)
    t = np.linspace(0, (len(h) - 1)/fs, len(h))
    return h, t


def circleConv(x1, x2):
    Nx1 = len(x1)
    Nx2 = len(x2)
    Ny = Nx1 + Nx2 - 1

    y = np.zeros(Ny)
    x1buff = np.zeros(Ny)
    x1buff[0:Nx1] = x1
    x2buff = np.zeros(Ny)
    x2buff[0:Nx2] = x2

    x2inv = np.zeros(Ny)
    for i in range(Ny):
        x2inv[i] = x2buff[-i-1]

    x2invbuff = np.zeros(Ny)
    for i in range(Ny):
        x2invbuff[0] = x2inv[-1]
        x2invbuff[1:] = x2inv[0:-1]
        x2inv = np.copy(x2invbuff)
        y[i] = np.sum(np.multiply(x1buff, x2inv))
    
    return y


def DFT(t, a):
    y = np.zeros(len(t), dtype=np.complex128)
    for k in range(len(t)):
        for n in range(len(t)):
            y[k] = y[k] + (a[n]*np.exp(-1j*2*np.pi*n*k/len(t)))
        # y[k] = y[k]*2/len(t)
    y = y[0:len(y)//2]
    f = np.linspace(0, len(y), len(y))/len(t)/(t[1] - t[0])
    return y, f


def example():
    fp = 100
    fr = 150
    fs = 500
    window = "bartlett"

    N = 500
    ts = np.linspace(0, (N-1)/fs, N)
    f1 = 60
    f2 = 200
    x = -2*np.sin(2*np.pi*f1*ts) + 10*np.sin(2*np.pi*f2*ts)
    h, t = FIR(fp, fr, fs, window)
    y = circleConv(x, h)
    t2 = np.linspace(0, (len(y) - 1)/fs, len(y))

    [H, f] = DFT(t, h)

    plt.subplot(2, 1, 1)
    plt.plot(ts, x)
    plt.title("Signal")
    plt.grid()

    plt.subplot(2, 1, 2)
    plt.plot(t2, y)
    plt.title("Filtred signal")
    plt.grid()
    plt.show()

    plt.plot(t, h)
    plt.title("Impulse response")
    plt.grid()
    plt.show()

    plt.subplot(2, 1, 1)
    plt.plot(f, np.abs(H))
    plt.title("Magnitude")
    plt.grid()

    plt.subplot(2, 1, 2)
    plt.plot(f, np.angle(H)*180/np.pi)
    plt.title("Phase")
    plt.grid()

    plt.show()


def DFT_zietek(t, y):
    N = len(t)
    dt = t[1] - t[0]
    yf = 2.0 / N * np.abs(fft(y)[0:N // 2])
    xf = np.fft.fftfreq(N, d=dt)[0:N // 2]
    return yf, xf

def DFT_onesided(t, values):
    N = len(values)
    dt = t[1] - t[0]

    H = np.fft.rfft(values)
    f = np.fft.rfftfreq(N, d=dt)

    return H, f


def main():
    with open(OUT_FILE, "rb") as f:
        raw = f.read()

    count = len(raw) // 2
    values = struct.unpack(f">{count}H", raw)

    print(f"Loaded {count} samples")

    # fs = 4687.5  # Hz
    fs = 18750  # Hz
    fp = 100  # 100
    fr = 200  # 200

    plt.plot(values)
    plt.title("Signal")
    plt.grid()
    plt.show()

    # plot(values)

    h, t = FIR(fp, fr, fs, "hamming")
    y = circleConv(values, h)

    y_idx_min = len(h)
    y_idx_max = len(y)-len(h)
    y = y[y_idx_min:y_idx_max]

    mean = np.mean(y)

    # y = y[10_000:80_000]
    plt.plot(y)
    plt.title("Filtred signal")
    # plt.ylim([mean-100, mean+100])
    # plt.ylim([mean+1000, mean + 1300])
    # plt.ylim([mean+145, mean + 152])
    plt.grid()
    plt.show()

    # plot(y)

    t2 = np.linspace(0, (len(values) - 1)/fs, len(values))
    [H, f] = DFT_zietek(t2, values-mean)

    plt.plot(f, np.abs(H))
    plt.title("Magnitude")
    # plt.xlim([0, 2000])
    plt.grid()
    plt.show()


def plot(values):
    vadc_values = []
    for i in range(len(values)):
        vadc_values.append(values[i]*VCC/MAX_ADC_VALUE)

    plt.figure()
    plt.plot(vadc_values)
    plt.title("Measured voltage")
    plt.xlabel("Sample index")
    plt.ylabel("Vadc [V]")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    i_values = []
    for i in range(len(vadc_values)):
        i_values.append((vadc_values[i]-V_REF)*S)

    plt.figure()
    plt.plot(i_values)
    plt.title("Measured current")
    plt.xlabel("Sample index")
    plt.ylabel("I [nA]")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()