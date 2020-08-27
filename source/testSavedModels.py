# sound file reading + spectrogram
print("Importing os")
import os
print("Importing librosa")
import librosa  # Version 0.8.0

# machine learning
print("Importing tensorflow")
import tensorflow as tf # Version 2.0.0
print("Importing keras")
from tensorflow import keras    # Version 2.2.4-tf  
print("Importing numpy")
import numpy as np  # Version 1.16.0

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

        print("Loaded " + str(i + 1) + "/" + str(len(audio_clips)) + " files", end='\r')

    print('\n')

    return specs, labels

print("\nTesting numbers model")

# Load testing set
print("\nLoading testing data")
test_specs, test_labels = load_specs("./input/audio/testRecordedNumbers/", 173, 49)

# Load model
print("Loading numbers model...")
model = tf.keras.models.load_model('./models/numbers_model')
print("Model loaded!")

model.evaluate(test_specs, test_labels, verbose=2)

# Use trained model
chosen_spec = random.randint(0, len(test_labels) - 1)
print("\nTesting spec nr", chosen_spec)

spec = test_specs[chosen_spec]

spec = (np.expand_dims(spec,0))

predictions_single = model.predict(spec)

print("\nPredicted:",np.argmax(predictions_single[0]) + 1)

print("Actual:", test_labels[chosen_spec] + 1)

wait = input("\nPress any key to continue...")

print("\nTesting letters model")

# Load testing set
print("\nLoading testing data")
test_specs, test_labels = load_specs("./input/audio/testRecordedLetters/", 173, 65)

# Load model
print("Loading letters model...")
model = tf.keras.models.load_model('./models/letters_model')
print("Model loaded!")

model.evaluate(test_specs, test_labels, verbose=2)

# Use trained model
chosen_spec = random.randint(0, len(test_labels) - 1)
print("\nTesting spec nr", chosen_spec)

spec = test_specs[chosen_spec]

spec = (np.expand_dims(spec,0))

predictions_single = model.predict(spec)

print("\nPredicted:",chr(np.argmax(predictions_single[0]) + 65))

print("Actual:", chr(test_labels[chosen_spec] + 65))