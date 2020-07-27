import os
import os.path
import numpy as np
import pandas as pd
import tensorflow as tf
import librosa
import librosa.display
import hickle as hkl
import random
import glob
from hypothesis import assume, given, settings
import hypothesis.strategies as st
import hypothesis.extra.numpy as npst
from multiprocessing import Pool
import unittest
import copy

SAMPLING_RATE = 16000

RUN_TESTS = True

noise_files = glob.glob('./data/*.wav')
noises = {}

def dummy_load_wave(example):
    row, params = example
    path = row.filename

    return np.ones((SAMPLING_RATE)) * float(path.split('/')[0]), row

def dataset_params( batch_size=32,
                    epochs = 50000,
                    parallelize = True,
                    max_text_length = None,
                    min_text_length = None,
                    max_wave_length = 80000,
                    shuffle = True,
                    random_shift_min = -4000,
                    random_shift_max = 4000,
                    random_stretch_min = 0.7,
                    random_stretch_max = 1.3,
                    random_noise = 0.75,
                    random_noise_factor_min = 0.2,
                    random_noise_factor_max = 0.5,
                    augment = False):
    return {
        'parallelize': parallelize,
        'shuffle': shuffle,
        'max_text_length': max_text_length,
        'min_text_length': min_text_length,
        'max_wave_length': max_wave_length,
        'random_shift_min': random_shift_min,
        'random_shift_max': random_shift_max,
        'random_stretch_min': random_stretch_min,
        'random_stretch_max': random_stretch_max,
        'random_noise': random_noise,
        'random_noise_factor_min': random_noise_factor_min,
        'random_noise_factor_max': random_noise_factor_max,
        'epochs': epochs,
        'batch_size': batch_size,
        'augment': augment
    }

def experiment_params(  data,
                        optimizer = 'Adam',
                        lr = 1e-4,
                        alphabet = " 'abcdefghijklmnopqrstuvwxyz",
                        causal_convolutions = True,
                        stack_dilation_rates = [1, 3, 9, 27, 81],
                        stacks = 2,
                        stack_kernel_size = 3,
                        stack_filters = 32,
                        sampling_rate = SAMPLING_RATE,
                        n_fft = 160*4,
                        frame_step = 160,
                        lower_edge_hertz = 0,
                        upper_edge_hertz = 8000,
                        num_mel_bins = 160,
                        clip_gradients = None,
                        codename = 'regular',
                        **kwargs):
    params = {
        'optimizer': optimizer,
        'lr': lr,
        'data': data,
        'alphabet': alphabet,
        'causal_convolutions': causal_convolutions,
        'stack_dilation_rates': stack_dilation_rates,
        'stacks': stacks,
        'stack_kernel_size': stack_kernel_size,
        'stack_filters': stack_filters,
        'sampling_rate': sampling_rate,
        'n_fft': n_fft,
        'frame_step': frame_step,
        'lower_edge_hertz': lower_edge_hertz,
        'upper_edge_hertz': upper_edge_hertz,
        'num_mel_bins': num_mel_bins,
        'clip_gradients': clip_gradients,
        'codename': codename
    }

    #import pdb; pdb.set_trace()

    if kwargs is not None and 'data' in kwargs:
        params['data'] = { **params['data'], **kwargs['data']}
        del kwargs['data']

    if kwargs is not None:
        params = { **params, **kwargs }

    return params

def experiment_name(params, excluded_keys=['alphabet', 'data', 'lr', 'clip_gradients']):

    def represent(key, value):
        if key in excluded_keys:
            return None
        else:
            if isinstance(value, list):
                return '{}_{}'.format(key, '_'.join([str(v) for v in value]))
            else:
                return '{}_{}'.format(key, value)

    parts = filter(
        lambda p: p is not None,
        [
            represent(k, params[k])
            for k in sorted(params.keys())
        ]
    )

    return '/'.join(parts)

def random_stretch(audio, params):
    rate = random.uniform(
        params['random_stretch_min'],
        params['random_stretch_max']
    )
    return librosa.effects.time_stretch(audio, rate)

def random_shift(audio, params):
    _shift = random.randrange(
        params['random_shift_min'],
        params['random_shift_max']
    )

    if _shift < 0:
        pad = (_shift * -1, 0)
    else:
        pad = (0, _shift)

    return np.pad(audio, pad, mode='constant')

def random_noise(audio, params):
    _factor = random.uniform(
        params['random_noise_factor_min'],
        params['random_noise_factor_max']
    )

    if params['random_noise'] > random.uniform(0, 1):
        _path = random.choice(noise_files)

        if _path in noises:
            wave = noises[_path]
        else:
            if os.path.isfile(_path + '.wave.hkl'):
                wave = hkl.load(_path + '.wave.hkl')
                noises[_path] = wave
            else:
                wave, _ = librosa.load(_path, sr = SAMPLING_RATE)
                hkl.dump(wave, _path + '.wave.hkl')
                noises[_path] = wave

        noise = random_shift(
            wave,
            {
                'random_shift_min': -16000,
                'random_shift_max': 16000
            }
        )

        max_noise = np.max(noise[0:len(audio)])
        max_wave = np.max(audio)

        noise = noise * (max_wave / max_noise)

        return _factor * noise[0:len(audio)] + (1.0 - _factor) * audio
    else:
        return audio

def to_path(filename):
    return './data/' + filename

def load_wave(example, absolute=False):
    row, params = example

    _path = row.filename if absolute else to_path(row.filename)

    if os.path.isfile(_path + 'wave.hkl'):
        wave = hkl.load(_path + '.wave.hkl').astype(np.float32)
    else:
        wave, _ = librosa.load(_path, sr=SAMPLING_RATE)
        hkl.dump(wave, _path + '.wave.hkl')

    if len(wave) <= params['max_wave_length']:
        if params['augment']:
            wave = random_noise(
                random_stretch(
                    random_shift(
                        wave,
                        params
                    ),
                    params
                ),
                params
            )
    else:
        wave = None
    
    return wave, row

def input_fn(input_dataset, params, load_wave_fn=load_wave):
    def _input_fn():
        """
            Returns raw audio wave along with the label
        """

        dataset = input_dataset
        print(params)

        if 'max_text_length' in params and params['max_text_length'] is not None:
            print('Constraining dataset to the max_text_length')
            dataset = input_dataset[input_dataset.text.str.len() < params['max_text_length']]
        if 'min_text_length' in params and params['min_text_length'] is not None:
            print('Constraining dataset to the min_text_length')
            dataset = input_dataset[input_dataset.text.str.len() >= params['min_text_length']]
        if 'max_wave_length' in params and params['max_wave_length'] is not None:
            print('Constraining datset to the max_wave_length')

        print('Resulting dataset length: {}'.format(len(dataset)))

        def generator_fn():
            pool = Pool()
            buffer = []

            for epoch in range(params['epochs']):
                for _, row in dataset.sample(frac=1).iterrows():
                    buffer.append((row, params))

                    if len(buffer) >= params['batch_size']:
                        if params['parallelize']:
                            audios = pool.map(
                                load_wave_fn,
                                buffer
                            )
                        else:
                            audios = map(
                                load_wave_fn,
                                buffer
                            )
                        for audio, row in audios:
                            if audio is not None:
                                if np.isnan(audio).any():
                                    print('SKIPPING! NaN coming from pipeline!')
                                else:
                                    yield (audio, len(audio)), row.text.encode()

                        buffer = []
        return tf.data.Dataset.from_generator(
            generator_fn,
            output_types = ((tf.float32, tf.int32), (tf.string)),
            output_shapes = ((None,()), (()))
        ) \
        .padded_batch(
            batch_size = params['batch_size'],
            padded_shapes = (
                (tf.TensorShape([None]), tf.TensorShape(())),
                tf.TensorShape(())
            )
        )

    return _input_fn

def encode_labels(labels, params):
    characters = list(params['alphabet'])

    table = tf.contrib.lookup.HashTable(
        tf.contrib.lookup.KeyValueTensorInitializer(
            characters,
            list(np.arange(len(characters), dtype='int64')),
            key_dtype=tf.string,
            value_dtype=tf.int64
        ),
        -1,
        name = 'char2id'
    )

    return table.lookup(
        tf.string_split(labels, delimiter = '')
    )

def decode_codes(codes, params):
    characters = list(params['alphabet'])

    table = tf.contrib.lookup.HashTable(
        tf.contrib.lookup.KeyValueTensorInitializer(
            list(np.arange(len(characters), dtype='int64')),
            characters,
            key_dtype=tf.int64,
            value_dtype=tf.string
        ),
        '',
        name = 'id2char'
    )

    return table.lookup(codes)

def compute_lengths(original_lengths, params):
    """
    Computes the length of data for CTC
    """

    return tf.cast(
        tf.floor(
            (tf.cast(original_lengths, dtype = tf.float32) - params['n_fft']) /
                params['frame_step']
        ) + 1,
        tf.int32
    )

def decode_logits(logits, lengths, params):
    if len(tf.shape(lengths).shape) == 1:
        lengths = tf.reshape(lengths, [1])
    else:
        lengths = tf.squeeze(lengths)
        
    predicted_codes, _ = tf.nn.ctc_beam_search_decoder(
        tf.transpose(logits, (1, 0, 2)),
        lengths,
        merge_repeated=True
    )
    
    codes = tf.cast(predicted_codes[0], tf.int32)
    
    sentence = decode_codes(codes, params)
    
    return sentence, codes

class LogMelSpectrogram(tf.layers.Layer):
    def __init__(self,
                 sampling_rate,
                 n_fft,
                 frame_step,
                 lower_edge_hertz,
                 upper_edge_hertz,
                 num_mel_bins,
                 **kwargs):
        super(LogMelSpectrogram, self).__init__(**kwargs)

        self.sampling_rate = sampling_rate
        self.n_fft = n_fft
        self.frame_step = frame_step
        self.lower_edge_hertz = lower_edge_hertz
        self.upper_edge_hertz = upper_edge_hertz
        self.num_mel_bins = num_mel_bins

    def call(self, inputs, training = True):
        stfts = tf.contrib.signal.stft(
            inputs,
            frame_length = self.n_fft,
            frame_step = self.frame_step,
            fft_length = self.n_fft,
            pad_end = False
        )

        power_spectrograms = tf.real(stfts * tf.conj(stfts))

        num_spectrogram_bins = power_spectrograms.shape[-1].value

        linear_to_mel_weight_matrix = tf.constant(
            np.transpose(
                librosa.filters.mel(
                    sr = self.sampling_rate,
                    n_fft = self.n_fft + 1,
                    n_mels = self.num_mel_bins,
                    fmin = self.lower_edge_hertz,
                    fmax = self.upper_edge_hertz
                )
            ),
            dtype = tf.float32
        )

        mel_spectrograms = tf.tensordot(
            power_spectrograms,
            linear_to_mel_weight_matrix,
            1
        )

        mel_spectrograms.set_shape(
            power_spectrograms.shape[:-1].concatenate(
                linear_to_mel_weight_matrix.shape[-1:]
            )
        )

        return tf.log(mel_spectrograms + 1e-6)

class AtrousConv1D(tf.layers.Layer):
    def __init__(self,
                 filters,
                 kernel_size,
                 dilation_rate,
                 use_bias = True,
                 kernel_initializer = tf.glorot_normal_initializer(),
                 causal = True
                ):
        super(AtrousConv1D, self).__init__()

        self.filters = filters
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.causal = causal

        self.conv1d = tf.layers.Conv1D(
            filters = filters,
            kernel_size = kernel_size,
            dilation_rate = dilation_rate,
            padding = 'valid' if causal else 'same',
            use_bias = use_bias,
            kernel_initializer = kernel_initializer
        )

    def call(self, inputs):
        if self.causal:
            padding = (self.kernel_size - 1) * self.dilation_rate
            inputs = tf.pad(inputs, tf.constant([(0, 0), (1, 0), (0, 0)]) * padding)

        return self.conv1d(inputs)

class ResidualBlock(tf.layers.Layer):
    def __init__(self, filters, kernel_size, dilation_rate, causal, **kwargs):
        super(ResidualBlock, self).__init__(**kwargs)

        self.dilated_conv1 = AtrousConv1D(
            filters = filters,
            kernel_size = kernel_size,
            dilation_rate = dilation_rate,
            causal = causal
        )

        self.dilated_conv2 = AtrousConv1D(
            filters = filters,
            kernel_size = kernel_size,
            dilation_rate = dilation_rate,
            causal = causal
        )

        self.out = tf.layers.Conv1D(
            filters = filters,
            kernel_size = 1
        )

    def call(self, inputs, training = True):
        data = tf.layers.batch_normalization(
            inputs,
            training = training
        )

        filters = self.dilated_conv1(data)
        gates = self.dilated_conv2(data)

        filters = tf.nn.tanh(filters)
        gates = tf.nn.tanh(gates)

        out = tf.nn.tanh(
            self.out(
                filters * gates
            )
        )

        return out + inputs, out

class ResidualStack(tf.layers.Layer):
    def __init__(self, filters, kernel_size, dilation_rates, causal, **kwargs):
        super(ResidualStack, self).__init__(**kwargs)

        self.blocks = [
            ResidualBlock(
                filters = filters,
                kernel_size = kernel_size,
                dilation_rate = dilation_rate,
                causal = causal
            )
            for dilation_rate in dilation_rates
        ]

    def call(self, inputs, training = True):
        data = inputs
        skip = 0

        for block in self.blocks:
            data, current_skip = block(data, training = training)
            skip += current_skip

        return skip

class SpeechNet(tf.layers.Layer):
    def __init__(self, params, **kwargs):
        super(SpeechNet, self).__init__(**kwargs)

        self.to_log_mel = LogMelSpectrogram(
            sampling_rate = params['sampling_rate'],
            n_fft = params['n_fft'],
            frame_step = params['frame_step'],
            lower_edge_hertz=params['lower_edge_hertz'],
            upper_edge_hertz=params['upper_edge_hertz'],
            num_mel_bins=params['num_mel_bins']
        )

        self.expand = tf.layers.Conv1D(
            filters = params['stack_filters'],
            kernel_size = 1,
            padding = 'same'
        )

        self.stacks = [
            ResidualStack(
                filters = params['stack_filters'],
                kernel_size = params['stack_kernel_size'],
                dilation_rates = params['stack_dilation_rates'],
                causal = params['causal_convolutions']
            )
            for _ in range(params['stacks'])
        ]

        self.out = tf.layers.Conv1D(
            filters = len(params['alphabet']) + 1,
            kernel_size = 1,
            padding = 'same'
        )

    def call(self, inputs, training = True):
        data = self.to_log_mel(inputs)

        data = tf.layers.batch_normalization(
            data,
            training = training
        )

        if len(data.shape) == 2:
            data = tf.expand_dims(data, 0)

        data = self.expand(data)

        for stack in self.stacks:
            data = stack(data, training = training)

        data = tf.layers.batch_normalization(
            data,
            training = training
        )

        return self.out(data) + 1e-8

class TestNotebook(unittest.TestCase):
    #WORKS
    def test_it_works(self):
        self.assertEqual(2 + 2, 4)

    #WORKS
    def test_dataset_returns_data_in_order(self):
        params = experiment_params(
            dataset_params(
                batch_size = 2,
                epochs = 1,
                augment = False
            )
        )
        data = pd.DataFrame(
            data = {
                'text': [ str(i) for i in range(10) ],
                'filename':  [ '{}/wav'.format(i) for i in range(10) ]
            }
        )
        dataset = input_fn(data, params['data'], dummy_load_wave)()
        iterator = dataset.make_one_shot_iterator()
        next_element = iterator.get_next()

        with tf.Session() as session:
            try:
                while True:
                    audio, label = session.run(next_element)
                    audio, length = audio
                    for _audio, _label in zip(list(audio), list(label)):
                        self.assertEqual(_audio[0], float(_label))

                    for _length in length:
                        self.assertEqual(_length, SAMPLING_RATE)

            except tf.errors.OutOfRangeError:
                pass

    #WORKS
    @given(st.text(alphabet="abcdefghijk234!@#$%^&*", max_size=10))
    @settings(max_examples=10, deadline=None)
    def test_encode_and_decode_work(self, text):
        assume(text != '')

        params = { 'alphabet': 'abcdefghijk234!@#$%^&*' }

        label_ph = tf.placeholder(tf.string, shape=(1), name = 'text')
        codes_op = encode_labels(label_ph, params)
        decode_op = decode_codes(codes_op, params)

        with tf.Session() as session:
            session.run(tf.global_variables_initializer())
            session.run(tf.tables_initializer(name = 'init_all_tables'))

            codes, decoded = session.run(
                [codes_op, decode_op],
                {
                    label_ph: np.array([text])
                }
            )

            # note(codes)
            # note(decoded)

            self.assertEqual(text, ''.join(map(lambda s: s.decode('UTF-8'), decoded.values)))
            self.assertEqual(codes.values.dtype, np.int64)
            self.assertEqual(len(codes.values), len(text))

    #FAILED
    @given(
        npst.arrays(
            np.float32,
            (4, 30, len('abcdefghijk1234!@#$%^&*')),
            elements=st.floats(0, 1, width=32)
        )
    )
    @settings(max_examples=10, deadline=None)
    def test_decode_logits_doesnt_crash(self, logits):
        params = { 'alphabet': 'abcdefghijk1234!@#$%^&*' }
        
        lengths = np.array([15, 15, 15, 15], dtype=np.int32)
        
        logits_ph = tf.placeholder(
            tf.float32,
            shape=(4, 30, len(params['alphabet']))
        )
        
        lengths_ph = tf.placeholder(
            tf.int32,
            shape=(4)
        )
        
        decode_op, codes_op = decode_logits(
            logits_ph,
            lengths_ph,
            params
        )
        
        with tf.Session() as session:        
            session.run(tf.global_variables_initializer())
            session.run(tf.tables_initializer(name='init_all_tables'))
            
            codes, decoded = session.run(
                [codes_op, decode_op],
                {
                    logits_ph: logits,
                    lengths_ph: lengths
                }
            )
            
            results = np.ones(codes.dense_shape) * -1
            
            for ix, value in zip(codes.indices, codes.values):
                results[ix[0], ix[1]] = value

            for row in results:
                self.assertLessEqual(len(row[row != -1]), 15)
    
    #FAILED
    @given(
        st.sampled_from([22000, 16000, 8000]),
        st.sampled_from([1024, 512]),
        st.sampled_from([1024, 512]),
        npst.arrays(
            np.float32,
            (4, 16000),
            elements = st.floats(-1, 1, width=32)
        )
    )
    @settings(max_examples = 10, deadline=None)
    def test_log_mel_conversion_works(self, sampling_rate, n_fft, frame_step, audio):
        lower_edge_hertz = 0.0
        upper_edge_hertz = sampling_rate / 2.0
        num_mel_bins = 64

        def librosa_melspectrogram(audio_item):
            spectrogram = np.abs(
                librosa.core.stft(
                    audio_item,
                    n_fft = n_fft,
                    hop_length = frame_step,
                    center = False
                )
            )**2

            return np.log(
                librosa.feature.melspectrogram(
                    S = spectrogram,
                    sr = sampling_rate,
                    n_mels = num_mel_bins,
                    fmin = lower_edge_hertz,
                    fmax = upper_edge_hertz
                ) + 1e-6
            )
        
        audio_ph = tf.placeholder(tf.float32, (4, 16000))

        librosa_log_mels = np.transpose(
            np.stack([
                librosa_melspectrogram(audio_item)
                for audio_item in audio
            ]),
            (0, 2, 1)
        )

        log_mel_op = tf.check_numerics(
            LogMelSpectrogram(
                sampling_rate = sampling_rate,
                n_fft = n_fft,
                frame_step = frame_step,
                lower_edge_hertz = lower_edge_hertz,
                upper_edge_hertz = upper_edge_hertz,
                num_mel_bins = num_mel_bins
            )(audio_ph),
            message = "log mels"
        )

        with tf.Session() as session:
            session.run(tf.global_variables_initializer())

            log_mels = session.run(
                log_mel_op,
                {
                    audio_ph: audio
                }
            )

            np.testing.assert_allclose(
                log_mels,
                librosa_log_mels,
                rtol = 1e-1,
                atol = 0
            )

    #FAILED
    @given(
        st.sampled_from([22000, 16000, 8000]),
        st.sampled_from([1024, 512, 640]),
        st.sampled_from([1024, 512, 160]),
        npst.arrays(
            np.float32,
            (st.integers(min_value = 16000, max_value = 16000*5)),
            elements = st.floats(-1, 1, width=32)
        )
    )
    @settings(max_examples = 10, deadline=None)
    def test_compute_lengths_works( self,
                                    audio_wave,
                                    sampling_rate,
                                    n_fft,
                                    frame_step
                                    ):
        assume(n_fft >= frame_step)

        original_wave_length = audio_wave.shape[0]

        audio_waves_ph = tf.placeholder(tf.float32, (None, None), name = "audio_waves")
        original_lengths_ph = tf.placeholder(tf.int32, (None), name = "original_lengths")

        lengths_op = compute_lengths(
            original_lengths_ph,
            {
                'frame_step': frame_step,
                'n_fft': n_fft
            }
        )

        self.assertEqual(lengths_op.dtype, tf.int32)

        log_mel_op = LogMelSpectrogram(
            sampling_rate = sampling_rate,
            n_fft = n_fft,
            frame_step = frame_step,
            lower_edge_hertz = 0.0,
            upper_edge_hertz = 8000.0,
            num_mel_bins = 13
        )(audio_waves_ph)

        with tf.Session() as session:
            session.run(tf.global_variables_initializer())

            lengths, log_mels = session.run(
                [lengths_op, log_mel_op],
                {
                    audio_waves_ph: np.array([audio_wave]),
                    original_lengths_ph: np.array([original_wave_length])
                }
            )

            self.assertEqual(lengths[0], log_mels.shape[1])

    #FAILED
    def test_causal_conv1d_works(self):
        conv_size2_dilation_1 = AtrousConv1D(
            filters = 1,
            kernel_size = 2,
            dilation_rate = 1,
            kernel_initializer = tf.ones_initializer(),
            use_bias = False
        )
        conv_size3_dilation_1 = AtrousConv1D(
            filters = 1,
            kernel_size = 3,
            dilation_rate = 1,
            kernel_initializer = tf.ones_initializer(),
            use_bias = False
        )
        conv_size2_dilation_2 = AtrousConv1D(
            filters = 1,
            kernel_size = 2,
            dilation_rate = 2,
            kernel_initializer = tf.ones_initializer(),
            use_bias = False
        )
        conv_size2_dilation_3 = AtrousConv1D(
            filters = 1,
            kernel_size = 2,
            dilation_rate = 3,
            kernel_initializer = tf.ones_initializer(),
            use_bias = False
        )

        data = np.array(list(range(1, 31)))
        data_ph = tf.placeholder(tf.float32, (1, 30, 1))

        size2_dilation_1_1 = conv_size2_dilation_1(data_ph)
        size2_dilation_1_2 = conv_size2_dilation_1(size2_dilation_1_1)

        size3_dilation_1_1 = conv_size3_dilation_1(data_ph)
        size3_dilation_1_2 = conv_size3_dilation_1(size3_dilation_1_1)

        size2_dilation_2_1 = conv_size2_dilation_2(data_ph)
        size2_dilation_2_2 = conv_size2_dilation_2(size2_dilation_2_1)

        size2_dilation_3_1 = conv_size2_dilation_3(data_ph)
        size2_dilation_3_2 = conv_size2_dilation_3(size2_dilation_3_1)

        with tf.Session() as session:
            session.run(tf.global_variables_initializer())

            outputs = session.run(
                [
                    size2_dilation_1_1,
                    size2_dilation_1_2,
                    size3_dilation_1_1,
                    size3_dilation_1_2,
                    size2_dilation_2_1,
                    size2_dilation_2_2,
                    size2_dilation_3_1,
                    size2_dilation_3_2
                ],
                {
                    data_ph: np.reshape(data, (1, 30, 1))
                }
            )

            for ix, out in enumerate(outputs):
                out = np.squeeze(out)
                outputs[ix] = out

                self.assertEqual(out.shape[0], len(data))

            np.testing.assert_equal(
                outputs[0],
                np.array([1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59], dtype=np.float32)
            )

            np.testing.assert_equal(
                outputs[1],
                np.array([1, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116], dtype=np.float32)
            )

            np.testing.assert_equal(
                outputs[2],
                np.array([1, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48, 51, 54, 57, 60, 63, 66, 69, 72, 75, 78, 81, 84, 87], dtype=np.float32)
            )

            np.testing.assert_equal(
                outputs[3],
                np.array([1, 4, 10, 18, 27, 36, 45, 54, 63, 72, 81, 90, 99, 108, 117, 126, 135, 144, 153, 162, 171, 180, 189, 198, 207, 216, 225, 234, 243, 252], dtype=np.float32)
            )

            np.testing.assert_equal(
                outputs[4],
                np.array([1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58], dtype=np.float32)
            )

            np.testing.assert_equal(
                outputs[5],
                np.array([1, 2, 5, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112], dtype=np.float32)
            )

            np.testing.assert_equal(
                outputs[6],
                np.array([1, 2, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57], dtype=np.float32)
            )

            np.testing.assert_equal(
                outputs[7],
                np.array([1, 2, 3, 6, 9, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108], dtype=np.float32)
            )

    #FAILED
    @given(
        st.sampled_from([64, 32]),
        st.sampled_from([7, 3]),
        st.sampled_from([1, 4]),
        npst.arrays(
            np.float32,
            (4, 16000),
            elements = st.floats(-1, 1, width=32)
        )
    )
    @settings(max_examples = 10, deadline=None)
    def test_residual_block_works(self, audio_waves, filters, size, dilation_rate):
        with tf.Graph().as_default() as g:
            audio_ph = tf.placeholder(tf.float32, (4, None))

            log_mel_op = LogMelSpectrogram(
                sampling_rate = SAMPLING_RATE,
                n_fft = 512,
                frame_step = 256,
                lower_edge_hertz = 0,
                upper_edge_hertz = 8000,
                num_mel_bins = 10
            )(audio_ph)

            expanded_op = tf.layers.Dense(filters)(log_mel_op)

            _, block_op = ResidualBlock(
                filters = filters,
                kernel_size = size,
                causal = True,
                dilation_rate = dilation_rate
            )(expanded_op, training = True)

            loss_op = tf.reduce_sum(block_op)

            variables = tf.trainable_variables
            self.assertTrue(any(["batch_normalization" in var.name for var in variables]))

            grads_op = tf.gradients(
                loss_op,
                variables
            )

            for grad, var in zip(grads_op, variables):
            #     if grad is None:
            #         note(var)

                self.assertTrue(grad is not None)

            with tf.Session(graph = g) as session:
                session.run(tf.global_variables_initializer())

                result, expanded, grads, _ = session.run(
                    [block_op, expanded_op, grads_op, loss_op],
                    {
                        audio_ph: audio_waves
                    }
                )

                self.assertFalse(np.array_equal(result, expanded))
                self.assertEqual(result.shape, expanded.shape)
                self.assertEqual(len(grads), len(variables))
                self.assertFalse(any([np.isnan(grad).any() for grad in grads]))

    #FAILED
    @given(
        st.sampled_from([64, 32]),
        st.sampled_from([7, 3]),
        npst.arrays(
            np.float32,
            (4, 16000),
            elements = st.floats(-1, 1, width=32)
        )
    )
    @settings(max_examples = 10, deadline=None)
    def test_residual_stacks_work(self, audio_waves, filters, size):
        dilation_rates = [1, 2, 4]

        with tf.Graph().as_default() as g:
            audio_ph = tf.placeholder(tf.float32, (4, None))

            log_mel_op = LogMelSpectrogram(
                sampling_rate = SAMPLING_RATE,
                n_fft = 512,
                frame_step = 256,
                lower_edge_hertz = 0,
                upper_edgeHErtz = 8000,
                num_mel_bins = 10
            )(audio_ph)

            expanded_op = tf.layers.Dense(filters)(log_mel_op)

            stack_op = ResidualStack(
                filters = filters,
                kernel_size = size,
                causal = True,
                dilation_rates = dilation_rates
            )(expanded_op, training = True)

            loss_op = tf.reduce_sum(stack_op)

            variables = tf.trainable_variables
            self.assertTrue(any(["batch_normalization" in var.name for var in variables]))

            grads_op = tf.gradients(
                loss_op,
                variables
            )

            for grad, var in zip(grads_op, variables):
                self.assertTrue(grad is not None)

            with tf.Session() as session:
                session.run(tf.global_variables_initializer())

                result, expanded, grads, _ = session.run(
                    [stack_op, expanded_op, grads_op, loss_op],
                    {
                        audio_ph: audio_waves
                    }
                )

                self.assertFalse(np.array_equal(result, expanded))
                self.assertEqual(result.shape, expanded.shape)
                self.assertEqual(len(grads), len(variables))
                self.assertFalse(any([np.isnan(grad).any() for grad in grads]))

    #FAILED
    @given(
        npst.arrays(
            np.float32,
            (4, 16000),
            elements = st.floats(-1, 1, width=32)
        ),
    )
    @settings(max_examples = 10, deadline=None)
    def test_speech_net_works(self, audio_waves):
        with tf.Graph().as_default() as g:
            audio_ph = tf.placeholder(tf.float32, (4, None))

            logits_op = SpeechNet(
                experiment_params(
                    {},
                    stack_dilation_rates = [1, 2, 4],
                    stack_kernel_size = 3,
                    stack_filters = 32,
                    alphabet = 'abcd'
                )
            )(audio_ph)

            loss_op = tf.reduce_sum(logits_op)

            variables = tf.trainable_variables()
            self.assertTrue(any(["batch_normalization" in var.name for var in variables]))

            grads_op = tf.gradients(
                loss_op,
                variables
            )

            for grad, var in zip(grads_op, variables):
                self.assertTrue(grad is not None)

            with tf.Session() as session:
                session.run(tf.global_variables_initializer())

                result, grads, _ = session.run(
                    [logits_op, grads_op, loss_op],
                    {
                        audio_ph: audio_waves
                    }
                )

                self.assertEqual(result.shape[2], 5)
                self.assertEqual(len(grads), len(variables))
                self.assertFalse(any([np.isnan(grad).any() for grad in grads]))
    
    #FAILED
    @given(
        npst.arrays(
            np.float32,
            (4, 16000),
            elements=st.floats(-1, 1, width=32)
        )
    )
    @settings(max_examples=10, deadline=None)
    def test_batch_normalization_when_not_training_doesnt_have_gradients(self, audio_waves):
        with tf.Graph().as_default() as g:
            audio_ph = tf.placeholder(tf.float32, (4, None))

            logits_op = SpeechNet(
                experiment_params(
                    {},
                    stack_dilation_rates= [1, 2, 4],
                    stack_kernel_size= 3,
                    stack_filters= 32,
                    alphabet= 'abcd'
                )
            )(audio_ph, training=False)

            # really dumb loss function just for the sake
            # of testing:
            loss_op = tf.reduce_sum(logits_op)

            variables = tf.trainable_variables()

            grads_op = tf.gradients(
                loss_op,
                variables
            )
        
            for grad, var in zip(grads_op, variables):
                # if grad is None:
                #     note(var)

                self.assertTrue(grad is not None)

            with tf.Session(graph=g) as session:
                session.run(tf.global_variables_initializer())

                result, grads, _ = session.run(
                    [logits_op, grads_op, loss_op],
                    {
                        audio_ph: audio_waves
                    }
                )
                
                no_batch_norms = list(
                    filter(
                        lambda var: 'batch_normaslization' not in var.name,
                        variables
                    )
                )

                self.assertEqual(len(grads), len(no_batch_norms))
                self.assertFalse(any([np.isnan(grad).any() for grad in grads]))
    


if __name__ == '__main__' and RUN_TESTS:
    import doctest

    doctest.testmod()
    unittest.main(
        argv=['first-arg-is-ignored', 'TestNotebook.test_decode_logits_doesnt_crash'],
        failfast=True,
        exit=False
    )

def model_fn(features, labels, mode, params):
    if isinstance(features, dict):
        audio = features['audio']
        original_lengths = features['length']
    else:
        audio, original_length = features

    lengths = compute_lengths(original_length, params)

    if labels is not None:
        codes = encode_labels(labels, params)

    network = SpeechNet(params)

    is_training = mode==tf.estimator.ModeKeys.TRAIN

    logits = network(audio, training = is_training)
    text, predicted_codes = decode_codes(logits, lengths, params)

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'logits': logits,
            'text': tf.sparse_tensor_to_dense(
                text,
                ''
            )
        }

        export_outputs = {
            'predictions': tf.estimator.export.PredictOutput(predictions)
        }

        return tf.estimator.EstimatorSpec(
            mode,
            predictions = predictions,
            export_outputs = export_outputs
        )
    else:
        loss = tf.reduce_mean(
            tf.nn.ctc_loss(
                labels = codes,
                inputs = logits,
                sequence_length = lengths,
                time_major = False,
                ignore_longer_outputs_than_inputs = True
            )
        )

        mean_edit_distance = tf.reduce_mean(
            tf.edit_distance(
                tf.cast(predicted_codes, tf.int32),
                codes
            )
        )

        distance_metric = tf.metrics.mean(mean_edit_distance)

        if mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(
                mode,
                loss = loss,
                eval_metric_ops = {'edit_distance': distance_metric}
            )

        elif mode == tf.estimator.ModeKeys.TRAIN:
            global_step = tf.train.get_or_create_global_step()

            tf.summary.text(
                'train_predicted_text',
                tf.sparse_tensor_to_dense(text, '')
            )
            tf.summary.scalar('train_edit_distance', mean_edit_distance)

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_op = tf.contrib.layers.optimize_loss(
                    loss = loss,
                    global_step = global_step,
                    learning_rate = params['lr'],
                    optimizer = (params['optimizer']),
                    update_ops = update_ops,
                    clip_gradients = params['clip_gradients'],
                    summaries = [
                        "learning_rate",
                        "loss",
                        "global_gradient_norm"
                    ]
                )

            return tf.estimator.EstimatorSpec(
                mode,
                loss = loss,
                train_op = train_op
            )

def experiment(data_params = dataset_params(), **kwargs):
    params = experiment_params(
        data_params,
        **kwargs
    )

    print(params)

    estimator = tf.estimator.Estimator(
        model_fn = model_fn,
        model_dir = 'stats/{}'.format(experiment_name(params)),
        params = params
    )

    #import pdb; pdb.set_trace()

    train_spec = tf.estimator.TrainSpec(
        input_fn = input_fn(
            train_data,
            params['data']
        )
    )

    features = {
        "audio": tf.placeholder(dtype = tf.float32, shape = [None]),
        "length": tf.placeholder(dtype = tf.int32, shape = [])
    }

    serving_input_receiver_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(
        features
    )

    best_exporter = tf.estimator.BestExporter(
        name = "best_exporter",
        serving_input_receiver_fn = serving_input_receiver_fn,
        exports_to_keep = 5
    )

    eval_params = copy.deepcopy(params['data'])
    eval_params['augment'] = False

    eval_spec = tf.estimator.EvalSpec(
        input_fn = input_fn(
            eval_data,
            eval_params
        ),
        throttle_secs = 60*30,
        exporters = best_exporter
    )

    tf.estimator.train_and_evaluate(
        estimator,
        train_spec,
        eval_spec
    )

def test(data_params = dataset_params(), **kwargs):
    params = experiment_params(
        data_params,
        **kwargs
    )

    print(params)

    estimator = tf.estimator.Estimator(
        model_fn = model_fn,
        model_dir = 'stats/{}'.format(experiment_name(params)),
        params = params
    )

    eval_params = copy.deepcopy(params['data'])
    eval_params['augment'] = False
    eval_params['epochs'] = 1
    eval_params['shuffle'] = False

    estimator.evaluate(
        input_fn = input_fn(
            test_data,
            eval_params
        )
    )

def predict_test(**kwargs):
    params = experiment_params(
        dataset_params(
            augment = False,
            shuffle = False,
            batch_size = 1,
            epochs = 1,
            parallelize = False
        ),
        **kwargs
    )

    print(len(test_data))

    estimator = tf.estimator.Estimator(
        model_fn = model_fn,
        model_dir = 'stats/{}'.format(experiment_name(params)),
        params = params
    )

    return list(
        estimator.predict(
            input_fn = input_fn(
                test_data,
                params['data']
            )
        )
    )