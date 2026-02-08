import serial
import threading
import struct
import sys
import time

# ---------------- CONFIG ----------------
SERIAL_PORT = "COM17"   # e.g. "COM3" on Windows
BAUDRATE = 460800
N_BYTES = 128                # must be EVEN (uint16)
TIMEOUT = 0.1                 # seconds
# ----------------------------------------

if N_BYTES % 2 != 0:
    raise ValueError("N_BYTES must be even (uint16 conversion)")

def uart_reader(ser: serial.Serial, n_bytes: int):
    """
    Continuously read exactly n_bytes from UART,
    convert to uint16 (big-endian), and print.
    """
    buffer = bytearray()

    while True:
        try:
            # Read whatever is available
            data = ser.read(n_bytes - len(buffer))
            if not data:
                continue

            buffer.extend(data)

            # Once we have a full batch
            if len(buffer) == n_bytes:
                # Convert bytes -> uint16 big-endian
                # '>' = big-endian, 'H' = uint16
                count = n_bytes // 2
                values = struct.unpack(f'>{count}H', buffer)

                print(values)

                buffer.clear()

        except serial.SerialException as e:
            print("Serial error:", e)
            break
        except Exception as e:
            print("Unexpected error:", e)
            break


def main():
    ser = serial.Serial(
        port=SERIAL_PORT,
        baudrate=BAUDRATE,
        timeout=TIMEOUT,
        bytesize=serial.EIGHTBITS,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
    )

    print(f"Opened {SERIAL_PORT} @ {BAUDRATE} baud")

    reader_thread = threading.Thread(
        target=uart_reader,
        args=(ser, N_BYTES),
        daemon=True
    )
    reader_thread.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        ser.close()


if __name__ == "__main__":
    main()
