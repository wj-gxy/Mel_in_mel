import math
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
import random
import torch
import torch.utils.data
import numpy as np
from librosa.util import normalize
from scipy.io.wavfile import read
from scipy.signal import resample
from librosa.filters import mel as librosa_mel_fn
# from glob import glob
# import resampy
MAX_WAV_VALUE = 32768.0


def load_wav(full_path, sr):
    sampling_rate, data = read(full_path)
    if sampling_rate != sr:
        data = resample(data, int(len(data) * sr / sampling_rate))
    return data, sr


def dynamic_range_compression(x, C=1, clip_val=1e-5):
    return np.log(np.clip(x, a_min=clip_val, a_max=None) * C)


def dynamic_range_decompression(x, C=1):
    return np.exp(x) / C


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


def spectral_de_normalize_torch(magnitudes):
    output = dynamic_range_decompression_torch(magnitudes)
    return output


mel_basis = {}
hann_window = {}

''''''
def mel_spectrogram(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
    # if torch.min(y) < -1.:
    #     print('min value is ', torch.min(y))
    # if torch.max(y) > 1.:
    #     print('max value is ', torch.max(y))

    global mel_basis, hann_window
    if fmax not in mel_basis:
        mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
        mel_basis[str(fmax) + '_' + str(y.device)] = torch.from_numpy(mel).float().to(y.device)
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

    y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
                                mode='reflect')
    y = y.squeeze(1)

    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[str(y.device)],
                      center=center, pad_mode='reflect', normalized=False, onesided=True, return_complex=False)

    spec = torch.sqrt(spec.pow(2).sum(-1) + (1e-9))
    # spec = torch.log10(torch.maximum(torch.tensor(1e-10), spec))
    spec = torch.matmul(mel_basis[str(fmax) + '_' + str(y.device)], spec)
    spec = spectral_normalize_torch(spec)

    return spec
'''
#aishell-3

def get_dataset_filelist(a):
    wave_list = []
    file_list = os.listdir(a.input_wavs_dir)
    for i in file_list:
        path_1 = os.path.join(a.input_wavs_dir, i, 'wav')  # data_aishell3
        filenames = glob(f'{path_1}/*.wav', recursive=True)
        wave_list.extend(filenames)
    training_file = wave_list[0:-3000]
    training_files = []
    for i in range(len(training_file)):
        sampled_files = random.sample(training_file, 2)
        carrier_file, hidden_message_files = sampled_files[0], sampled_files[1]
        training_files.append((carrier_file, hidden_message_files))

    valid_data = wave_list[-3000:]
    validation_files = []
    for i in range(len(valid_data)):
        sampled_files = random.sample(valid_data, 2)
        carrier_file, hidden_message_files = sampled_files[0], sampled_files[1]
        validation_files.append((carrier_file, hidden_message_files))
    return training_files, validation_files
'''
'''
##VCTK
def get_dataset_filelist(a):
    wave_list = []
    file_list = os.listdir(a.input_wavs_dir)
    for i in file_list:
        path_1 = os.path.join(a.input_wavs_dir, i)  # data_aishell3
        filenames = glob(f'{path_1}/*.wav', recursive=True)
        wave_list.extend(filenames)
    training_file = wave_list[0:-500]
    training_files = []
    for i in range(len(training_file)):
        sampled_files = random.sample(training_file, 2)
        carrier_file, hidden_message_files = sampled_files[0], sampled_files[1]
        training_files.append((carrier_file, hidden_message_files))

    valid_data = wave_list[-150:]
    validation_files = []
    for i in range(len(valid_data)):
        sampled_files = random.sample(valid_data, 2)
        carrier_file, hidden_message_files = sampled_files[0], sampled_files[1]
        validation_files.append((carrier_file, hidden_message_files))
    return training_files, validation_files
'''

#LJ Speech
def get_dataset_filelist(a):
    with open(a.input_training_file, 'r', encoding='utf-8') as fi:
        training_file = [os.path.join(a.input_wavs_dir, x.split('|')[0] + '.wav')
                          for x in fi.read().split('\n') if len(x) > 0]
        training_files = []
        for i in range(len(training_file)):
            sampled_files = random.sample(training_file, 2)
            carrier_file, hidden_message_files = sampled_files[0], sampled_files[1]
            training_files.append((carrier_file, hidden_message_files))
    with open(a.input_validation_file, 'r', encoding='utf-8') as fi:
        validation_file = [os.path.join(a.input_wavs_dir, x.split('|')[0] + '.wav')
                            for x in fi.read().split('\n') if len(x) > 0]
        validation_files = []
        for i in range(len(validation_file)):
            sampled_files = random.sample(validation_file, 2)
            carrier_file, hidden_message_files = sampled_files[0], sampled_files[1]
            validation_files.append((carrier_file, hidden_message_files))
    return training_files, validation_files


class MelDataset(torch.utils.data.Dataset):
    def __init__(self, training_files, segment_size, n_fft, num_mels,
                 hop_size, win_size, sampling_rate, fmin, fmax, split=True, shuffle=True, n_cache_reuse=1,
                 device=None, fmax_loss=None, fine_tuning=False, base_mels_path=None):
        self.audio_files = training_files
        random.seed(1234)
        if shuffle:
            random.shuffle(self.audio_files)
        self.segment_size = segment_size
        self.sampling_rate = sampling_rate
        self.split = split
        self.n_fft = n_fft
        self.num_mels = num_mels
        self.hop_size = hop_size
        self.win_size = win_size
        self.fmin = fmin
        self.fmax = fmax
        self.fmax_loss = fmax_loss
        self.cached_wav = None
        self.n_cache_reuse = n_cache_reuse
        self._cache_ref_count = 0
        self.device = device
        self.fine_tuning = fine_tuning
        self.base_mels_path = base_mels_path

    def __getitem__(self, index):
        cover, serect = self.audio_files[index]
        # filename = self.audio_files[index]
        if self._cache_ref_count == 0:
            cover_audio, sampling_rate = load_wav(cover,sr=self.sampling_rate)
            audio = cover_audio / MAX_WAV_VALUE
            audio = normalize(audio) * 0.95

            serect_audio, sampling_rate = load_wav(serect,sr=self.sampling_rate)
            serect_audio = serect_audio / MAX_WAV_VALUE
            serect_audio = normalize(serect_audio) * 0.95

            self.cached_wav = audio

            self._cache_ref_count = self.n_cache_reuse
        else:
            audio = self.cached_wav
            self._cache_ref_count -= 1

        audio = torch.FloatTensor(audio)
        audio = audio.unsqueeze(0)
        serect_audio = torch.FloatTensor(serect_audio)
        serect_audio = serect_audio.unsqueeze(0)

        if not self.fine_tuning:
            if self.split:
                if audio.size(1) >= self.segment_size:
                    max_audio_start = audio.size(1) - self.segment_size
                    audio_start = random.randint(0, max_audio_start)
                    audio = audio[:, audio_start:audio_start + self.segment_size]
                else:
                    audio = torch.nn.functional.pad(audio, (0, self.segment_size - audio.size(1)), 'constant')

                if serect_audio.size(1) >= self.segment_size:
                    max_audio_start = serect_audio.size(1) - self.segment_size
                    audio_start = random.randint(0, max_audio_start)
                    serect_audio = serect_audio[:, audio_start:audio_start + self.segment_size]
                else:
                    serect_audio = torch.nn.functional.pad(serect_audio, (0, self.segment_size - serect_audio.size(1)),
                                                           'constant')

            mel = mel_spectrogram(audio, self.n_fft, self.num_mels,
                                  self.sampling_rate, self.hop_size, self.win_size, self.fmin, self.fmax,
                                  center=False)

            serect_mel = mel_spectrogram(serect_audio, self.n_fft, self.num_mels,
                                         self.sampling_rate, self.hop_size, self.win_size, self.fmin, self.fmax,
                                         center=False)

        mel_loss = mel_spectrogram(audio, self.n_fft, self.num_mels,
                                   self.sampling_rate, self.hop_size, self.win_size, self.fmin, self.fmax_loss,
                                   center=False)
        serect_mel_loss = mel_spectrogram(serect_audio, self.n_fft, self.num_mels,
                                          self.sampling_rate, self.hop_size, self.win_size, self.fmin, self.fmax_loss,
                                          center=False)

        return (mel.squeeze(), audio.squeeze(0), mel_loss.squeeze(),
                serect_mel.squeeze(), serect_audio.squeeze(0), serect_mel_loss.squeeze())

    def __len__(self):
        return len(self.audio_files)


if __name__ == '__main__':
    input_wavs_dir = '/home/lgd/miniconda3/envs/one/hifi-gan-master/LJSpeech-1.1/re_wave/test/LJ001-0001.wav'
    cover_audio, sampling_rate = load_wav(input_wavs_dir)
    audio = torch.FloatTensor(cover_audio)
    audio = audio.unsqueeze(0)
    audio = audio[:, :8192]

    d = mel_spectrogram(audio, 1204, 80, 22050, 256, 1024, 0, 8000, center=False)
    print(d.size())
