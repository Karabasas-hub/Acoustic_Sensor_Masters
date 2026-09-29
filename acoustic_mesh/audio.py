from pathlib import Path

import numpy as np
import soundfile as sf

def load_audio(path: str | Path) -> tuple[np.ndarray, int]:
    path = Path(path)

    signal, sample_rate = sf.read(
        path,
        always_2d=False,
        dtype="float32",
    )

    return signal, sample_rate

