# sound file reading + spectrogram
print("Importing os")
import os
print("Importing librosa")
import librosa
print("Importing librosa.display")
import librosa.display
print("Importing audioread")
import audioread
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
print("Importing ImageDataGenerator")
from keras.preprocessing.image import ImageDataGenerator
print("Importing numpy")
import numpy as np

print("Importing random")
import random

spec_length = 173

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

def create_model(model_path, spec_length, train_specs, train_labels, test_specs, test_labels, epochs):
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(1025, spec_length)),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dense(8)
    ])

    model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

    # Fit model
    model.fit(train_specs, train_labels, epochs=epochs)

    # Save model
    model.save(model_path)

    print("\n")

    # Test model
    test_loss, test_acc = model.evaluate(test_specs, test_labels, verbose=2)

    return model

""" CREATE MODEL FOR NUMBERS """

# Load training set
print("\nLoading training data")
train_specs, train_labels = load_specs("./input/audio/trainRecordedNumbers/", spec_length, 49)

# Load testing set
print("\nLoading testing data")
test_specs, test_labels = load_specs("./input/audio/testRecordedNumbers/", spec_length, 49)

# Create model
model = create_model('./models/numbers_model', spec_length, train_specs, train_labels, test_specs, test_labels, 10)

# Use trained model
chosen_spec = random.randint(0, len(test_labels) - 1)
spec = test_specs[chosen_spec]

spec = (np.expand_dims(spec,0))

predictions_single = model.predict(spec)

print("\nPredicted: ",np.argmax(predictions_single[0]) + 1)

print("Actual: ", test_labels[chosen_spec] + 1)

""" CREATE MODEL FOR LETTERS """

# Load training set
print("\nLoading training data")
train_specs, train_labels = load_specs("./input/audio/trainRecordedLetters/", spec_length, 65)

# Load testing set
print("\nLoading testing data")
test_specs, test_labels = load_specs("./input/audio/testRecordedLetters/", spec_length, 65)

# Create model
model = create_model('./models/letters_model', spec_length, train_specs, train_labels, test_specs, test_labels, 10)

# Use trained model
chosen_spec = random.randint(0, len(test_labels) - 1)
spec = test_specs[chosen_spec]

spec = (np.expand_dims(spec,0))

predictions_single = model.predict(spec)

print("\nPredicted: ", chr(np.argmax(predictions_single[0]) + 65))

print("Actual: ", chr(test_labels[chosen_spec] + 65))