import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import chirp

def main() -> None:
    sample_rate = 48000
    frequency = 2000
    duration = 1.0

    time = np.arange(
        0,
        duration,
        1 / sample_rate,
    )

    signal = chirp(
        time,
        f0=1000,
        f1=2000,
        t1=duration,
        method="linear",
    )

    spectrum = np.fft.rfft(signal)
    frequencies = np.fft.rfftfreq(
        len(signal),
        d=1 / sample_rate,
    )

    magnitude = np.abs(spectrum)

    plt.figure(figsize=(12, 4))
    plt.plot(frequencies, magnitude)

    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Magnitude")
    plt.title("FFT of a 1 kHz sine wave")

    plt.xlim(0, 5000)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
