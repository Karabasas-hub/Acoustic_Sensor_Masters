"""
Downsampling 48 kHz with and without an anti-aliasing filter

The signal holds a 1 kHz tone we want to keep and a 6.5 kHz tone above the new Nyquist frequency (4 kHz).
Without a filter the 6.5 kHz tones comes back as a fake 1.5 kHz tone.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import decimate

from acoustic_mesh.analysis import plot_spectrum, to_db
from acoustic_mesh.sampling import alias_frequency

FAST_RATE = 48000   # the original sample rate
SLOW_RATE = 8000    # the sample rate we want
FACTOR = FAST_RATE // SLOW_RATE     # keep one sample in every six
WANTED = 1000   # below the new Nyquist (4 kHz) should survive
UNWANTED = 6500     # above it - should be removed

def main() -> None:

    # One second at 48 kHz - the wanted tone plus a half-as-loud unwanted tone
    t = np.arange(FAST_RATE) / FAST_RATE    # 48000 sample times in seconds
    signal = np.cos(2 * np.pi * WANTED * t) + 0.5 * np.cos(2 * np.pi * UNWANTED * t)

    # Method 1 - keep every 6th sample, throw the rest away. No filter
    naive = signal[::FACTOR]

    # Method 2 - low-pass filter first, then keep every 6th sample
    filtered = decimate(signal, FACTOR)

    ghost = alias_frequency(UNWANTED, SLOW_RATE)    # where the unwanted tone will land
    print(f"New Nyquist frequency: {SLOW_RATE / 2:.0f}")
    print(f"The {UNWANTED} Hz tone aliases to {ghost:.0F}")

    # Spectra of both 8 kHz versions. Each is 8000 samples long -> 1 Hz between bins.
    frequencies = np.fft.rfftfreq(len(naive), d=1 / SLOW_RATE)
    naive_spectrum = np.abs(np.fft.rfft(naive))
    filtered_spectrum = np.abs(np.fft.rfft(filtered))

    # How loud is it at the ghost frequency compared with the wanted one?

    ghost_bin = np.argmin(np.abs(frequencies - ghost))  # the bin closest to the ghost
    level_naive = to_db(naive_spectrum)[ghost_bin]
    level_filtered = to_db(filtered_spectrum)[ghost_bin]
    print(f"Level at {ghost:.0f} Hz, no filter: {level_naive:6.1f} dB")
    print(f"Level at {ghost:.0f} Hz, with filter: {level_filtered:6.1f} dB")

    fig, (top, bottom) = plt.subplots(2, 1, sharex=True, figsize=(12, 7))
    