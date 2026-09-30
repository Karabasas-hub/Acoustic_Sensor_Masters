import numpy as np
import matplotlib.pyplot as plt

def main() -> None:
    sample_rate = 48000
    frequency = 1000
    duration = 0.01

    time = np.arange(
        0,
        duration,
        1 / sample_rate,
    )

    signal = np.sin(2 * np.pi * frequency * time)

    plt.figure(figsize=(12, 4))
    #plt.plot(time, signal)
    plt.scatter(time, signal, s=8)

    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.title("1kHz sine wave sampled at 48kHz")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()