from __future__ import absolute_import, division, print_function, unicode_literals
import glob
import os
import json
import torch
import numpy as np
from scipy.io.wavfile import write
from env import AttrDict
from meldataset import MAX_WAV_VALUE
from istftnet_model import Generator
from model_4 import stego_model
from meldataset import MelDataset, mel_spectrogram
from torch.utils.data import DataLoader
import soundfile as sf
from env import AttrDict, build_env
from hifigan.istftnet_models import Generator as Gen
import torch.nn.functional as F
from stft import TorchSTFT

h = None
device = None
import math
from boltons import fileutils
from TEST.MOS_test import mos_dn, utmos_


def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict


def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + '*')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return ''
    return sorted(cp_list)[-1]


def generate_speckle_noise(tensor, noise_norm, mean=0.0, stddev=1.0):
    return tensor * (1 + noise_norm * tensor.data.new(tensor.size()).normal_(mean, stddev))


def generate_gaussian_noise(tensor, noise_norm, mean=0.0, stddev=1.0):
    return tensor + noise_norm * tensor.data.new(tensor.size()).normal_(mean, stddev)


def istftnet(mel, hifi_model_path):
    config_file = '/home/lgd/miniconda3/envs/all/Mel_in/cp_hifigan/istftnet/lj/config_v1.json'
    with open(config_file) as f:
        data = f.read()
    json_config = json.loads(data)
    h = AttrDict(json_config)
    gen = Gen(h).to(device)
    state_dict_g = torch.load(hifi_model_path, map_location=device)
    gen.load_state_dict(state_dict_g['generator'])
    gen.eval()
    y = gen(mel)
    return y


config_file = '/home/lgd/miniconda3/envs/all/Mel_in/cp_hifigan/istftnet/lj/config_v1.json'
with open(config_file) as f:
    data = f.read()
json_config = json.loads(data)
h = AttrDict(json_config)
stft = TorchSTFT(filter_length=h.gen_istft_n_fft, hop_length=h.gen_istft_hop_size, win_length=h.gen_istft_n_fft,
                 device=device).to(device)
generator = Generator(h).to(device)
checkpoint_file = '/home/lgd/miniconda3/envs/all/Mel_in/cp_hifigan/istftnet/lj/g_00325000'
state_dict_g = load_checkpoint(checkpoint_file, device)
generator.load_state_dict(state_dict_g['generator'])
model = stego_model().to(device)
model.load_state_dict(state_dict_g['model'])
generator.eval()
model.eval()
generator.remove_weight_norm()

path1 = '/home/lgd/miniconda3/envs/all/iSTFTNet/cp_hifigan/g_00675000'
path2 = '/home/lgd/miniconda3/envs/all/hifi-gan-E/ZH/22050/g_00285000'

import random

input_validation_file = '/home/lgd/miniconda3/envs/dataset/LJSpeech-1.1/validation.txt'  # '/home/lgd/miniconda3/envs/all/Mel_in/sample/test'
input_wav = '/home/lgd/miniconda3/envs/dataset/LJSpeech-1.1/wavs'  # '/home/lgd/miniconda3/envs/dataset/aishell3'  #

wave_list = []
'''
file_list = os.listdir(input_wav)
for i in file_list:
    path_1 = os.path.join(input_wav, i, 'wav')  # data_aishell3
    filenames = glob.glob(f'{path_1}/*.wav', recursive=True)
    wave_list.extend(filenames)
    valid_data = wave_list[-150:]
    validation_files = []
    for i in range(len(valid_data)):
        sampled_files = random.sample(valid_data, 2)
        carrier_file, hidden_message_files = sampled_files[0], sampled_files[1]
        validation_files.append((carrier_file, hidden_message_files))
'''
with open(input_validation_file, 'r', encoding='utf-8') as fi:
    validation_file = [os.path.join(input_wav, x.split('|')[0] + '.wav')
                       for x in fi.read().split('\n') if len(x) > 0]
    validation_files = []
    for i in range(len(validation_file)):
        sampled_files = random.sample(validation_file, 2)
        carrier_file, hidden_message_files = sampled_files[0], sampled_files[1]
        validation_files.append((carrier_file, hidden_message_files))

tESTset = MelDataset(validation_files, 16384*3, h.n_fft, h.num_mels,
                     h.hop_size, h.win_size, h.sampling_rate, h.fmin, h.fmax, n_cache_reuse=0,
                     shuffle=False if h.num_gpus > 1 else True, fmax_loss=h.fmax, device=device)

test_loader = DataLoader(tESTset, batch_size=1)

MOS_gen = []
MOS_ground = []
MOS_cover = []
STOI = []
PESQ = []
from pystoi import stoi
from torchaudio.transforms import Resample
import torchaudio
def resample(tensor, x):
    reample1 = Resample(22050, x).to(device)
    reample2 = Resample(x, 22050).to(device)
    y = reample1(tensor)
    z = reample2(y)
    return z
def highpass_filter(wav, cutoff_freq= 500):
    filtered_audio = torchaudio.functional.highpass_biquad(waveform=wav, sample_rate=16000, cutoff_freq=cutoff_freq)
    return filtered_audio
with torch.no_grad():
    for i, batch in enumerate(test_loader):
        x_mel, x_aud, x_mel_loss, y_mel, y_aud, y_mel_s = batch
        x_mel.to(device)
        y_mel.to(device)
        spec, phase = generator(x_mel, y_mel)  # 生成语音
        y_g_hat = stft.inverse(spec, phase)
        # y_g_hat = generate_gaussian_noise(y_g_hat, 0.001)  # gaussian
        # y_g_hat = generate_speckle_noise(y_g_hat, 0.01) #speckle
        # y_g_hat = resample(y_g_hat, 44100)
        y_g_hat = highpass_filter(y_g_hat, cutoff_freq=500)
        # y_g_hat_mel_loss = mel_spectrogram(y_g_hat.squeeze(1), h.n_fft, h.num_mels, h.sampling_rate, h.hop_size,
        #                                    h.win_size, h.fmin, h.fmax_for_loss)
        #
        # re_serect_mel = model(y_g_hat_mel_loss)

        cover_1, cover_2 = istftnet(x_mel, path1)
        cover = stft.inverse(cover_1, cover_2)
        y_g_hat_mel = mel_spectrogram(y_g_hat.squeeze(1), h.n_fft, h.num_mels, h.sampling_rate,
                                      h.hop_size, h.win_size, h.fmin, h.fmax_for_loss)

        re_serect_mel = model(y_g_hat_mel)

        secret_1, secret_2 = istftnet(y_mel, path1)
        secret = stft.inverse(secret_1, secret_2)

        re_sec_1, re_sec_2 = istftnet(re_serect_mel, path1)
        re_sec = stft.inverse(re_sec_1, re_sec_2)

        re_sec = (re_sec * MAX_WAV_VALUE).squeeze()
        secret = (secret * MAX_WAV_VALUE).squeeze()

        audio = y_g_hat.squeeze()
        audio = audio * MAX_WAV_VALUE
        cover = cover.squeeze()
        cover = cover * MAX_WAV_VALUE

        audio = audio.cpu().numpy().astype('int16')
        cover = cover.cpu().numpy().astype('int16')
        secret = secret.cpu().numpy().astype('int16')
        re_sec = re_sec.cpu().numpy().astype('int16')

        sf.write(f'/home/lgd/miniconda3/envs/all/Mel_in/robustness/hf/secret{i}.wav', secret, 22050)
        sf.write(f'/home/lgd/miniconda3/envs/all/Mel_in/robustness/hf/re_sec{i}.wav', re_sec, 22050)
        # sf.write(f'/home/lgd/miniconda3/envs/all/Mel_in/robustness/gau_0.01/setgo_{i}.wav', audio, 22050)
        # sf.write(f'/home/lgd/miniconda3/envs/all/Mel_in/robustness/gau_0.01/cover_{i}.wav', cover, 22050)

