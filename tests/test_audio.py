from pathlib import Path
import numpy as np
import pytest
from acoustic_mesh.audio import load_audio

EXAMPLE = Path(__file__).resolve().parents[1] / "data" / "raw" / "example.wav"

def test_load_example_wav():
    signal, sample_rate = load_audio(EXAMPLE)

    assert sample_rate == 8000
    #telephone quality file
    assert signal.shape == (23808, )
    #mono: one row of 23808 samples
    assert signal.dtype == np.float64
    #load_audio asks for float64

    #same min and max that inspect_audio.py prints

    assert signal.max() == pytest.approx(0.3828, abs=0.0001)
    assert signal.min() == pytest.approx(-0.2891, abs=0.0001)