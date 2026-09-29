from acoustic_mesh.audio import load_audio
from acoustic_mesh.analysis import plot_waveform


def main() -> None:
    signal, sample_rate = load_audio(
        "data/raw/example.wav"
    )

    print("Sample rate:", sample_rate)
    print("Number of samples:", len(signal))
    print("Array shape:", signal.shape)
    print("Data type:", signal.dtype)

    duration = len(signal) / sample_rate
    print("Duration [s]:", duration)

    print("Minimum sample:", signal.min())
    print("Maximum sample:", signal.max)

    plot_waveform(signal, sample_rate)

if __name__ == "__main__":
    main()