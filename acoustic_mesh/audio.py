from pathlib import Path

import numpy as np
import soundfile as sf

def load_audio(path: str | Path, mono: bool = True) -> tuple[np.ndarray, int]:
    """
    Read an audio file as float64 samples in [-1, 1] plus its sample rate

    Multichannel files come back as shape (frames, channels)
    with mono=True the channels are averaged into one signal of shape (frames, )
    """

    signal, sample_rate = sf.read(Path(path), dtype="float64", always_2d=False)

    if mono and signal.ndim == 2:
        signal = signal.mean(axis=1)

    return signal, sample_rate

