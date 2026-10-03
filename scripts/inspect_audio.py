from acoustic_mesh.audio import load_audio
from acoustic_mesh.analysis import plot_waveform, to_db

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = REPO_ROOT / "data" / "raw" / "example.wav"


def main() -> None:
    signal, sample_rate = load_audio(
        EXAMPLE
    )

    duration = len(signal) / sample_rate
    peak = np.max(np.abs(signal))
    rms = np.sqrt(np.mean(signal**2))

    print(f"Sample rate: {sample_rate} Hz")
    print(f"Nyquist frequency: {sample_rate / 2:.0f} Hz")
    print(f"Samples: {len(signal)}, shape {signal.shape}, {signal.dtype}")
    print(f"Duration: {duration:.3f} s")
    print(f"Min / max sample: {signal.min():.4f} / {signal.max():.4f}")
    print(f"Peak level: {to_db(peak, reference=1.0):.1f} dBFS")
    print(f"RMS level: {to_db(rms, reference=1.0):.1f}")
    print(f"Crest factor: {to_db(peak / rms, reference=1.0):.1f} dB")

    plot_waveform(signal, sample_rate, title=EXAMPLE.name)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()