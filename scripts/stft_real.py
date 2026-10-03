import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft

from acoustic_mesh.audio import load_audio

def main() -> None:

    signal, sample_rate = load_audio(
        "data/raw/example.wav"
    )

    frequencies, times, stft_matrix = stft(
        signal,
        fs=sample_rate,
        window="hann",
        nperseg=8192,
        noverlap=6144,
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
        label="Mangitude [dB]"
    )

    plt.xlabel("Time [s]")
    plt.ylabel("Frequency [Hz]")
    plt.title("STFT of Real Speech Recording")

    plt.ylim(0, 4000)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()