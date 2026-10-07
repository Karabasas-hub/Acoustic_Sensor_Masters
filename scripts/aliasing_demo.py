"""
Tones sampled at 8kHz and where each of them shows up

Figure 1 - a 7kHz tone its 8kHz samples and the 1kHz tone through the same dots
Figure 2 - spectra of 3, 5 and 7 kHz tones each sampled at 8kHz
"""

import numpy as np
import matplotlib.pyplot as plt
from acoustic_mesh.analysis import plot_spectrum
from acoustic_mesh.sampling import alias_frequency

SAMPLE_RATE = 8000  # Hz: 8000 snapshots per second
TONES = [3000, 5000, 7000]  # Hz: one below SAMPLE_RATE / 2 two above it

def main() -> None:

    print(f"Nyquist frequency: {SAMPLE_RATE / 2:.0f} Hz")

    # Figure 1 - the samples of a 7 kHz tone also fit a 1 kHz tone

    # A point every microsecond stands in for the real, smooth sound
    # The samples are taken every 1/8000 s = 0.125ms

    smooth_time = np.arange(0, 0.002, 1 / 1000000)  # 0 to 2 ms, 2000 points
    sample_time = np.arange(0, 0.002, 1 / SAMPLE_RATE)  # 0 to 2 ms, 16 snapshots

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(
        smooth_time * 1000,
        np.cos(2 * np.pi * 7000 * smooth_time),
        linewidth=0.8,
        label="real tone: 7kHz",
    )
    ax.plot(
        smooth_time * 1000,
        np.cos(2 * np.pi * 1000 * smooth_time),
        linestyle="--",
        label="tone the samples suggest: 1 kHz",
    )
    ax.plot(
        sample_time * 1000,
        np.cos(2 * np.pi * 7000 * sample_time),
        "o",
        label="8 kHz samples",
    )

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Amplitude")
    ax.set_ylim(-1.2, 1.6)
    ax.set_title("At 8 kHz, a 7 kHz tone gives the same dots and a 1 kHz tone")
    ax.legend(loc="upper center", ncol=3)
    fig.tight_layout()

    # Figure 2 - where 3, 5 and 7 kHz land in the spectrum

    n = np.arange(SAMPLE_RATE)

    # The frequency of each FFT bin: 0, 1, 2, .. , 4000 Hz (1 s of data -> Hz apart)

    frequencies = np.fft.rfftfreq(len(n), d=1 / SAMPLE_RATE)

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(12, 8))
    for ax, tone in zip(axes, TONES):
        samples = np.cos(2 * np.pi * tone * n / SAMPLE_RATE)    # sample n is at (n / fs)
        spectrum = np.abs(np.fft.rfft(samples)) # size of each frequency bin
        measured = frequencies[np.argmax(spectrum)] # frequency of the tallest bin
        predicted = alias_frequency(tone, SAMPLE_RATE)

        print(f"{tone} Hz -> predicted {predicted:.0f} Hz, FFT peak at {measured:.0f}")
        plot_spectrum(
            frequencies,
            spectrum,
            ax=ax,
            title=f"{tone} Hz tone samples at 8kHz",
        )

    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
