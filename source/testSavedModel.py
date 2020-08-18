# sound file reading + spectrogram
print("Importing os")
import os
print("Importing librosa")
import librosa
print("Importing librosa.display")
import librosa.display
print("Importing pyplot")
import matplotlib.pyplot as plt

# sound recording
print("Importing sounddevice")
import sounddevice as sd

# machine learning
print("Importing tensorflow")
import tensorflow as tf
print("Importing keras")
from tensorflow import keras
print("Importing numpy")
import numpy as np

print("Importing random")
import random

# Load testing set
audio_fpath = "./input/audio/testNumbers/"
audio_clips = os.listdir(audio_fpath)
print("No. of .wav files in audio folder: ", len(audio_clips))

# test_specs = np.empty((len(audio_clips), 1025, 64))
test_specs_list = []
test_labels = np.empty(len(audio_clips), dtype=int)
spec_length = 0

# print("Test specs: ", test_specs.shape)
number = 0
# for fileName in audio_clips:
#     test_labels[number] = ord(fileName[0]) - 65
#     x, sr = librosa.load(audio_fpath+fileName, sr=44100, mono=True)

#     X = librosa.stft(x)
#     Xdb = librosa.amplitude_to_db(abs(X))

#     test_specs[number] = Xdb
#     number += 1

for fileName in audio_clips:
    test_labels[number] = ord(fileName[0]) - 49
    x, sr = librosa.load(audio_fpath+fileName, sr=44100, mono=True)

    X = librosa.stft(x)
    Xdb = librosa.amplitude_to_db(abs(X))

    spec_length = max(spec_length, Xdb.shape[1])

    # test_specs[number] = Xdb
    test_specs_list.append(Xdb)
    number += 1

spec_length = 190
for i in range(len(test_specs_list)):
    test_specs_list[i] = np.pad(test_specs_list[i], ((0,0),(0, spec_length - test_specs_list[i].shape[1])), 'constant')

test_specs = np.array(test_specs_list)

# Load model
print("Loading model...")
model = tf.keras.models.load_model('./models/my_model')
print("Model loaded!\n")

model.evaluate(test_specs, test_labels, verbose=2)

# probability_model = tf.keras.Sequential([model,
                                        # tf.keras.layers.Softmax()])

# Use trained model

chosen_spec = random.randint(0, len(test_labels) - 1)
print("Testing spec nr ", chosen_spec)

spec = test_specs[chosen_spec]

spec = (np.expand_dims(spec,0))

predictions_single = model.predict(spec)

print("\nPredicted: ",np.argmax(predictions_single[0]))

print("Actual: ", test_labels[chosen_spec])
