import struct
import time
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


def read_and_plot():
    with open(OUT_FILE, "rb") as f:
        raw = f.read()

    count = len(raw) // 2
    values = struct.unpack(f">{count}H", raw)

    print(f"Loaded {count} samples")

    plt.figure()
    plt.plot(values)
    plt.title("UART Capture")
    plt.xlabel("Sample index")
    plt.ylabel("Value (uint16)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

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
    read_and_plot()
