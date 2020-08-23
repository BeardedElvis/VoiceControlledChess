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

def load_specs(audio_fpath, spec_length, ascii_offset):
    audio_clips = os.listdir(audio_fpath)
    print("No. of .wav files in folder: ", len(audio_clips))

    specs = np.empty((len(audio_clips), 1025, spec_length))
    labels = np.empty(len(audio_clips), dtype=int)

    for i in range(len(audio_clips)):
        fileName = audio_clips[i]
        # Read label
        labels[i] = ord(fileName[0]) - ascii_offset
        # Load wav
        x, sr = librosa.load(audio_fpath+fileName, sr=44100, mono=True)
        # Transform to spectrogram
        X = librosa.stft(x)
        Xdb = librosa.amplitude_to_db(abs(X))
        # Normalise values
        Xdb = Xdb - np.amin(Xdb)
        Xdb = Xdb / np.amax(Xdb)
        # Add to array
        specs[i] = Xdb

    return specs, labels

# Load testing set
print("\nLoading testing data")
test_specs, test_labels = load_specs("./input/audio/testRecordedNumbers/", 173, 49)

# Load model
print("Loading model...")
model = tf.keras.models.load_model('./models/numbers_model')
print("Model loaded!\n")

model.evaluate(test_specs, test_labels, verbose=2)

# Use trained model
chosen_spec = random.randint(175, len(test_labels) - 1)
print("Testing spec nr ", chosen_spec,"\nLabel ", test_labels[chosen_spec] + 1)

spec = test_specs[chosen_spec]

spec = (np.expand_dims(spec,0))

predictions_single = model.predict(spec)

print("\nPredicted: ",np.argmax(predictions_single[0]))

print("Actual: ", test_labels[chosen_spec])
