import numpy as np
import matplotlib.pyplot as plt

def make_eeg_dataset(
    n_channels=32,
    n_samples=1000,
    n_noisy=4,
    seed=42,
):
    rng = np.random.default_rng(seed)

    # Shared time axis
    t = np.arange(n_samples) / 250

    # Give each channel a slightly different EEG-like signal
    frequencies = rng.uniform(8, 13, size=n_channels)
    phases = rng.uniform(0, 2 * np.pi, size=n_channels)

    signals = (
        np.sin(
            2 * np.pi * frequencies[:, None] * t
            + phases[:, None]
        )
        + 0.25 * rng.standard_normal((n_channels, n_samples))
    )

    # Add large artifacts to a few randomly selected channels
    noisy_channels = rng.choice(
        n_channels, size=n_noisy, replace=False
    )

    signals[noisy_channels] += (
        8 * rng.standard_normal(
            (n_noisy, n_samples)
        )
    )

    return signals

def plot_channel_comparison(eeg=None, time=None):
    if eeg is None:
        eeg = make_eeg_dataset()
    #Remove noisy channels
    mean_signal_amplitude = np.mean(np.abs(eeg), axis=1)
    threshold = np.percentile(mean_signal_amplitude, 90)
    condition = mean_signal_amplitude > threshold
    
    eeg_bad = eeg[condition]
    eeg = eeg[~condition]

    
    if time is None:
        step = 1/1000
        time = np.arange(0, 1, step)
    
    fig, ax = plt.subplots(1,2, figsize=(12,6))
    fig.suptitle('Comparing Good and Bad Channels')
    
    ax[0].plot(time, np.mean(eeg, axis=0))
    ax[0].set_title('Mean signal in good channels')
    ax[0].set_ylabel('Signal ($\\mu V$)')
    ax[0].set_xlabel('Time (s)')
    
    ax[1].plot(time, np.mean(eeg_bad, axis=0))
    ax[1].set_title('Mean signal in bad channels')
    ax[1].set_ylabel('Signal ($\\mu V$)')
    ax[1].set_xlabel('Time (s)')
    fig.tight_layout()
    plt.show()
