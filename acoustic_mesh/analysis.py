import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

DB_FLOOR = -120.0

def to_db(
        x: np.ndarray | float,
        reference: float | None = None,
        power: bool = False,
        floor_db: float = DB_FLOOR,
) -> np.ndarray:
    """
    Convert amplitudes (or powers) to decibels relative to 'reference'

    Amplitude-like values (samples, pressure, |FFT|) use 20*log10.
    Power-like values (|FFT|**2, PSD) need pwoer=True, which uses 10*log10
    reference=None makes the largest value 0 dB
    """
    x = np.abs(np.asarray(x, dtype=np.float64))
    if reference is None:
        reference = float(x.max())
    factor = 10.0 if power else 20.0

    if reference <= 0:
        return np.full(x.shape, floor_db)

    smallest_ratio = 10.0 ** (floor_db / factor)
    return factor * np.log10(np.maximum(x / reference, smallest_ratio))


def plot_waveform(
        signal: np.ndarray,
        sample_rate: float,
        ax: Axes | None = None,
        title: str = "Waveform",
) -> Axes:
    #Plot samples against time in seconds
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 4))
    time = np.arange(len(signal)) / sample_rate

    ax.plot(time, signal, linewidth=0.8)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Amplitude [FS]")
    ax.set_title(title)
    return ax

def plot_spectrum(
        frequencies: np.ndarray,
        magnitude: np.ndarray,
        db: bool = True,
        reference: float | None = None,
        ax: Axes | None = None,
        title: str = "Spectrum",
        label: str | None = None,
) -> Axes:
    #Plot a magnitude spectrum, in dB by default
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 4))
    values = to_db(magnitude, reference) if db else magnitude
    ax.plot(frequencies, values, linewidth=0.8, label=label)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Magnitude [dB]" if db else "Magnitude")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    return ax

def plot_spectrogram(
        times: np.ndarray,
        frequencies: np.ndarray,
        magnitude: np.ndarray,
        reference: float | None = None,
        dynamic_range_db: float = 80.0,
        ax: Axes | None = None,
        title: str = "Spectrogram",
) -> Axes:
    #Plot |STFT| i dB, drawing each time-frequency cell as it is
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 6))
    levels = to_db(magnitude, reference)
    top = levels.max()
    mesh = ax.pcolormesh(
        times,
        frequencies,
        levels,
        shading="nearest",
        vmin=top - dynamic_range_db,
        vmax=top,
    )

    ax.figure.colorbar(mesh, ax=ax, label="Magnitude [dB]")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Frequency [Hz]")
    ax.set_title(title)
    return ax