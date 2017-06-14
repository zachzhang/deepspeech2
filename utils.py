import librosa
import soundfile as sf
import numpy as np

window_size = .02
window_stride = .01


def load_audio(path):
    sound, samplerate = sf.read(path)

    if len(sound.shape) > 1:
        if sound.shape[1] == 1:
            sound = sound.squeeze()
        else:
            sound = sound.mean(axis=1)  # multiple channels, average
    return sound, samplerate


# (freq, time)
def parse_audio(path):
    x, fs = load_audio(path)

    n_fft = int(fs * window_size)
    hop_length = int(fs * window_stride)
    win_length = n_fft

    D = librosa.stft(x, n_fft=n_fft, hop_length=hop_length,
                     win_length=win_length)

    spect, phase = librosa.magphase(D)
    spect = np.log1p(spect)

    mean = spect.mean()
    std = spect.std()

    return (spect - mean) / std


def pad_batch(X):
    max_len = max([x.shape[1] for x in X])

    inputs = np.zeros((len(X), X[0].shape[0], max_len))

    for i in range(len(X)):
        x = X[i]
        inputs[i, :, :x.shape[1]] = x

    return inputs


def get_batch(df, batch_size):
    batch = df.sample(batch_size)

    X = [parse_audio(f) for f in list(batch.iloc[:, 1])]
    X = pad_batch(X)

    y = list(batch.iloc[:, 2])
    y_sizes = [len(_y) for _y in y]

    return (X, y, y_sizes)
