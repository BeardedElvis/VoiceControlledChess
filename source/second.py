import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import Audio

""" Testing """

SAMPLING_RATE = 16000

audio_fpath = "./data/New Folder/cleaned/"
audio_clips = os.listdir(audio_fpath)
print("No. of .wav files in audio folder: ", len(audio_clips))

wave, _ = librosa.load(audio_fpath+audio_clips[0], sr=SAMPLING_RATE)

librosa.display.waveplot(wave, sr=SAMPLING_RATE)

def gen_sin(freq, amplitude, sr=1000):
    return np.sin(
        (freq * 2 * np.pi * np.linspace(0, sr, sr)) / sr
    ) * amplitude

def plot_wave_composition(defs, hspace = 1.0):
    fig_size = plt.rcParams["figure.figsize"]

    plt.rcParams["figure.figsize"] = [14.0, 10.0]

    waves = [
        gen_sin(freq, amp)
        for freq, amp in defs
    ]

    fig, axs = plt.subplots(nrows = len(defs) + 1)

    for ix, wave in enumerate(waves):
        sns.lineplot(data = wave, ax = axs[ix])
        axs[ix].set_ylabel('{}'.format(defs[ix]))

        if ix != 0:
            axs[ix].set_title('+')

    plt.subplots_adjust(hspace = hspace)

    sns.lineplot(data = sum(waves), ax = axs[len(defs)])
    axs[len(defs)].set_ylabel('sum')
    axs[len(defs)].set_xlabel('time')
    axs[len(defs)].set_title('=')

    plt.rcParams["figure.figsize"] = fig_size

    return waves, sum(waves)

wave_defs = [
    (2, 1),
    (3, 0.8),
    (5, 0.2),
    (7, 0.1),
    (9, 0.25)
]

waves, the_sum = plot_wave_composition(wave_defs)

ffts = np.fft.fft(the_sum)
freqs = np.fft.fftfreq(len(the_sum))

frequencies, coeffs = zip(
    *list(
        filter(
            lambda row: row[1] > 10,
            [ (int(abs(freq * 1000)), coef) for freq, coef in zip(freqs[0:(len(ffts) // 2)], np.abs(ffts)[0:(len(ffts) // 2)]) ]
        )
    )
)

sns.barplot(x = list(frequencies), y = coeffs)

Audio(data = gen_sin(800, 1, 16000), rate = SAMPLING_RATE)
