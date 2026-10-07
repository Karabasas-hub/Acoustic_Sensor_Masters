# Sampling helpers

#alias_frequency predicts where a tone shows up after sampling

import numpy as np

def alias_frequency(
        frequency: float | np.ndarray,  #the real tone in Hz, one numer of many
        sample_rate: float, #samples per second
) -> np.ndarray:
    
    """
    returns the frequency a tone appears at after sampling.

    samples cannot tell f apart from f + k * sample_rate (k as a whole number)
    samples also cannot tell f apart from its mirror image 
    every tone lands between 0 and sample_rate/2
    """

    # Make it an array so one number and many numbers work the same way
    f = np.asarray(frequency, dtype=np.float64)

    # How many whole sample-rate steps is the tone away from 0 Hz?
    # Round to the nearest whole number: 7000 / 8000 = 0.875 -> 1 step
    steps = np.round(f / sample_rate)

    # Slide the tone down by that many steps (7000 - 8000 = -1000),
    # then drop the minus sign: a mirror image sound the same ( -> 1000)
    return np.abs(f - steps * sample_rate)