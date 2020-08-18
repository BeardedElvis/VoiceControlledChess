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
print("Importing numpy")
import numpy as np

print("Importing random")
import random

# Load training set
audio_fpath = "./input/audio/trainNumbers/"
audio_clips = os.listdir(audio_fpath)
print("No. of .wav files in training folder: ", len(audio_clips))

# train_specs = np.empty((len(audio_clips), 1025, 64))
train_specs_list = []
labels = np.empty(len(audio_clips), dtype=int)
spec_length = 0

# print("Train specs: ",train_specs.shape)
number = 0
for fileName in audio_clips:
    labels[number] = ord(fileName[0]) - 49
    x, sr = librosa.load(audio_fpath+fileName, sr=44100, mono=True)

    X = librosa.stft(x)
    Xdb = librosa.amplitude_to_db(abs(X))

    spec_length = max(spec_length, Xdb.shape[1])

    # train_specs[number] = Xdb
    train_specs_list.append(Xdb)
    number += 1

#______________________________________________________________________________

# Load testing set
audio_fpath = "./input/audio/testNumbers/"
audio_clips = os.listdir(audio_fpath)
print("No. of .wav files in testing folder: ", len(audio_clips))

# test_specs = np.empty((len(audio_clips), 1025, 64))
test_specs_list = []
test_labels = np.empty(len(audio_clips), dtype=int)

# print("Test specs: ", test_specs.shape)
number = 0
for fileName in audio_clips:
    test_labels[number] = ord(fileName[0]) - 49
    x, sr = librosa.load(audio_fpath+fileName, sr=44100, mono=True)

    X = librosa.stft(x)
    Xdb = librosa.amplitude_to_db(abs(X))

    spec_length = max(spec_length, Xdb.shape[1])

    # test_specs[number] = Xdb
    test_specs_list.append(Xdb)
    number += 1

for i in range(len(train_specs_list)):
    train_specs_list[i] = np.pad(train_specs_list[i], ((0,0),(0, spec_length - train_specs_list[i].shape[1])), 'constant')

train_specs = np.array(train_specs_list)

for i in range(len(test_specs_list)):
    test_specs_list[i] = np.pad(test_specs_list[i], ((0,0),(0, spec_length - test_specs_list[i].shape[1])), 'constant')

test_specs = np.array(test_specs_list)

## To display
##____________________________________________________________________
# print(fileName)
# print(type(x), type(sr))
# print(x.shape, sr)

# plt.figure(figsize=(14, 5))
# librosa.display.waveplot(x, sr=sr)

# plt.figure(figsize=(14, 5))
# librosa.display.specshow(train_specs[0], sr=sr, x_axis='time', y_axis='log')
# plt.colorbar()
# plt.show()

# plt.figure()
# plt.imshow(train_specs[0])
# plt.colorbar()
# plt.grid(False)
# plt.show()

# number = 0
# for fileName in audio_clips:
#     plt.figure(figsize=(7, 5))
#     plt.title(fileName)
#     # librosa.display.specshow(train_specs[number], sr=sr, x_axis='time', y_axis='log')
#     plt.imshow(train_specs[number])
#     plt.colorbar()
#     plt.show()
#     number += 1

#__________________________________________________________________________

# Normalise
for spec in train_specs:
    spec = spec - np.amin(spec)
    spec = spec / (np.amax(spec) - np.amin(spec))

for spec in test_specs:
    spec = spec - np.amin(spec)
    spec = spec / (np.amax(spec) - np.amin(spec))


# #______________________________________________________________________

# # Test
# plt.figure(figsize=(30,30))
# for i in range(15):
#     plt.subplot(6,5,2*(i+1))
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     librosa.display.specshow(train_specs[i], sr=sr, x_axis='time', y_axis='log')
#     # plt.imshow(train_specs[i], cmap=plt.cm.binary)
#     plt.xlabel(labels[i])
# plt.show()

#___________________________________________________________________________

# Create model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(1025, spec_length)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(8)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# model.summary()

# Fit model
model.fit(train_specs, labels, epochs=10)

# Save model
model.save('./models/my_model')

print("\n")

# Test model
test_loss, test_acc = model.evaluate(test_specs, test_labels, verbose=2)

probability_model = tf.keras.Sequential([model,
                                        tf.keras.layers.Softmax()])

# predictions = probability_model.predict(test_specs)

# for i in range(len(test_specs)):
#     print("\nPredicted: ",np.argmax(predictions[i]))

#     print("Actual: ",test_labels[i])

# Graph tests ---- Not working
# def plot_image(i, predictions_array, true_label, img):
#   predictions_array, true_label, img = predictions_array, true_label[i], img[i]
#   plt.grid(False)
#   plt.xticks([])
#   plt.yticks([])

#   plt.imshow(img, cmap=plt.cm.binary)

#   predicted_label = np.argmax(predictions_array)
#   if predicted_label == true_label:
#     color = 'blue'
#   else:
#     color = 'red'

#   plt.xlabel("{} {:2.0f}% ({})".format(audio_clips[predicted_label][0],
#                                 100*np.max(predictions_array),
#                                 audio_clips[true_label][0]),
#                                 color=color)

# def plot_value_array(i, predictions_array, true_label):
#   predictions_array, true_label = predictions_array, true_label[i]
#   plt.grid(False)
#   plt.xticks(range(10))
#   plt.yticks([])
#   thisplot = plt.bar(range(10), predictions_array, color="#777777")
#   plt.ylim([0, 1])
#   predicted_label = np.argmax(predictions_array)

#   thisplot[predicted_label].set_color('red')
#   thisplot[true_label].set_color('blue')

# i = 0
# plt.figure(figsize=(6,3))
# plt.subplot(1,2,1)
# plot_image(i, predictions[i], test_labels, test_specs)
# plt.subplot(1,2,2)
# plot_value_array(i, predictions[i],  test_labels)
# plt.show()

#____________________________________________________________________

# Use trained model
chosen_spec = random.randint(0, len(test_labels) - 1)
spec = test_specs[chosen_spec]

spec = (np.expand_dims(spec,0))

predictions_single = probability_model.predict(spec)

print("\nPredicted: ",np.argmax(predictions_single[0]))

print("Actual: ", test_labels[chosen_spec])

##____________________________________________________________________

## To record:
##____________________________________________________________________
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
##____________________________________________________________________

