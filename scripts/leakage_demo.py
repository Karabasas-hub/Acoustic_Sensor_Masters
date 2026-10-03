import numpy as np
import matplotlib.pyplot as plt

def main() -> None:
    sample_rate = 48000
    frequency = 1005.5
    duration = 1.0

    time = np.arange(
        0,
        duration,
        1 / sample_rate,
    )

    signal = np.sin(
        2 * np.pi * frequency * time
    )

    window = np.hanning(len(signal))

    windowed_signal = signal * window

    rectangular_spectrum = np.fft.rfft(signal)

    hann_window = np.hanning(len(signal))
    hann_spectrum = np.fft.rfft(
        signal * hann_window
        )

    frequencies = np.fft.rfftfreq(
        len(signal),
        d=1 / sample_rate,
    )

    rectangular_magnitude = np.abs(
        rectangular_spectrum
    )

    hann_magnitude = np.abs(
        hann_spectrum
    )

    plt.figure(figsize=(12, 4))

    plt.plot(
        frequencies,
        rectangular_magnitude,
        label="Rectangular Window",
    )

    plt.plot(
        frequencies,
        hann_magnitude,
        label="Hann Window",
    )


    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Magnitude")
    plt.title("Spectral Leakage: Rectangular vs Hann Window")

    plt.xlim(900, 1100)
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()