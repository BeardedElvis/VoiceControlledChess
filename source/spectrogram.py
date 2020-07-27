# sound file reading + spectrogram
import os
import librosa
import librosa.display
import matplotlib.pyplot as plt

# sound recording
import sounddevice as sd

# machine learning
import tensorflow as tf
from tensorflow import keras
import numpy as np

audio_fpath = "./input/audio/audio/"
audio_clips = os.listdir(audio_fpath)
print("No. of .wav files in audio folder: ", len(audio_clips))

specs = np.empty((len(audio_clips), 1025, 64))

print(specs.shape)
number = 0
for fileName in audio_clips:
    x, sr = librosa.load(audio_fpath+fileName, sr=44100, mono=True)

    X = librosa.stft(x)
    Xdb = librosa.amplitude_to_db(abs(X))

    specs[number] = Xdb
    number += 1
    # print(Xdb.shape)
# print(fileName)
# print(type(x), type(sr))
# print(x.shape, sr)

# plt.figure(figsize=(14, 5))
# librosa.display.waveplot(x, sr=sr)

# plt.figure(figsize=(14, 5))
# librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')
# plt.colorbar()
# number = 0
# for fileName in audio_clips:
#     plt.figure(figsize=(7, 5))
#     plt.title(fileName)
#     librosa.display.specshow(specs[number], sr=sr, x_axis='time', y_axis='log')
#     plt.colorbar()
#     plt.show()
#     number += 1

# To record:
# seconds = 3
# print('Start talking')
# x = sd.rec(int(seconds * sr), samplerate=sr, channels=1)
# sd.wait()

# x = np.reshape(x, x.size)
# print(type(x), type(sr))
# print(x.shape, sr)

# X = librosa.stft(x)
# Xdb = librosa.amplitude_to_db(abs(X))

# plt.figure(figsize=(7, 5))
# librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='log')
# plt.colorbar()

# plt.show()