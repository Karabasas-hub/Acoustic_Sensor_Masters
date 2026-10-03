import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft

def main() -> None:
    sample_rate = 48000
    duration = 2.0

    time = np.arange(
        0,
        duration,
        1 / sample_rate,
    )

    #frequency increase from 500Hz to 4000Hz
    #this a linear chirp signal

    f_start = 500
    f_end = 4000

    k = (f_end - f_start) / duration

    phase = 2 * np.pi * (
        f_start * time
        + 0.5 * k * time**2
    )

    signal = np.sin(phase)

    frequencies, times, stft_matrix = stft(
        signal,
        fs=sample_rate,
        window="hann",
        nperseg=8192,
        noverlap=256,
    )

    magnitude = np.abs(stft_matrix)

    magnitude_db = 20 * np.log10(
        magnitude / np.max(magnitude)
    )

    plt.figure(figsize=(12, 6))

    plt.pcolormesh(
        times,
        frequencies,
        magnitude_db,
        shading="gouraud",
    )

    plt.colorbar(
        label = "Magnitude [dB]"
    )

    plt.xlabel("Time [s]")
    plt.ylabel("Frequency [Hz]")
    plt.title("STFT of a Chirp")

    plt.ylim(0, 5000)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()