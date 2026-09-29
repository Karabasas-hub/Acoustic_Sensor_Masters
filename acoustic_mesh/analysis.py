import numpy as np
import matplotlib.pyplot as plt

def plot_waveform(
        signal: np.ndarray,
        sample_rate: int,
) -> None:
    time = np.arange(len(signal)) / sample_rate

    plt.figure(figsize=(12, 4))
    plt.plot(time, signal)

    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.title("Audio waveform")

    plt.tight_layout()
    plt.show()