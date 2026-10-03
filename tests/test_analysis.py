import matplotlib

matplotlib.use("Agg")
# draw in memory only: no plot windows pop up during tests
import matplotlib.pyplot as plt
import numpy as np
import pytest

from acoustic_mesh.analysis import plot_spectrogram, plot_spectrum, plot_waveform, to_db
# Amplitudes use 20*log10: half -> -6.02 dB, a tenth -> -20 dB, a thousandth -> -60 dB.
def test_to_db_amplitudes():

    result = to_db(np.array([1.0, 0.5, 0.1, 0.001]))

    assert result == pytest.approx([0.0, -6.0206, -20.0, -60.0], abs=0.001)
# Powers use 10*log10, so the same ratios give half the dB values.
def test_to_db_powers():

    result = to_db(np.array([1.0, 0.5, 0.1, 0.001]), power=True)

    assert result == pytest.approx([0.0, -3.0103, -10.0, -30.0], abs=0.001)

def test_to_db_dbfs():
# reference=1.0 means "compared with full scale": half of full scale is -6.02 dBFS.
    assert to_db(0.5, reference=1.0) == pytest.approx(-6.0206, abs=0.001)

def test_to_db_zero_hits_the_floor():
# log10(0) would be minus infinity; the floor turns it into -120 dB.
    result = to_db(np.array([1.0, 0.0]))
    assert result[1] == -120.0

def test_to_db_silence():
# All zeros: nothing to compare against, so every value is the floor.
    result = to_db(np.zeros(5))
    assert np.all(result == -120.0)

# plots: only check that they draw without crashing

def test_plots_draw_without_errors():
    time = np.linspace(0, 1, 100)
    #100 moments from 0 to 1s

    plot_waveform(np.sin(2 * np.pi * 5 * time), sample_rate=100)
    plot_spectrum(np.arange(10), np.ones(10)) #flast spectrum 10 bins
    plot_spectrogram(np.arange(5), np.arange(4), np.ones((4, 5))) # 4x5 tiles
    plt.close("all") # throw the test figures away