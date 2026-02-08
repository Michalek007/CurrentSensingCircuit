import serial
import struct
import time
import matplotlib.pyplot as plt

SERIAL_PORT = "COM17"
BAUDRATE = 460800
N_BYTES = 128
TIMEOUT = 0.1
RECORD_SECONDS = 2
OUT_FILE = "uart_capture.bin"

MAX_ADC_VALUE = 2 ** 14 - 1  # after oversampling
VCC = 3.3  # V
V_REF = VCC/2  # reference level may need to be updated

if N_BYTES % 2 != 0:
    raise ValueError("N_BYTES must be even (uint16 conversion)")


def capture_uart():
    ser = serial.Serial(
        port=SERIAL_PORT,
        baudrate=BAUDRATE,
        timeout=TIMEOUT,
        bytesize=serial.EIGHTBITS,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
    )

    print(f"Opened {SERIAL_PORT} @ {BAUDRATE} baud")
    print(f"Recording for {RECORD_SECONDS} seconds...")

    buffer = bytearray()
    start_time = time.time()

    with open(OUT_FILE, "wb") as f:
        while time.time() - start_time < RECORD_SECONDS:
            data = ser.read(N_BYTES - len(buffer))
            if not data:
                continue

            buffer.extend(data)

            if len(buffer) == N_BYTES:
                f.write(buffer)
                buffer.clear()

    ser.close()
    print("Capture complete.")


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
        i_values.append((vadc_values[i]-V_REF)*1000)

    plt.figure()
    plt.plot(i_values)
    plt.title("Measured current")
    plt.xlabel("Sample index")
    plt.ylabel("I [nA]")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    capture_uart()
    read_and_plot()
