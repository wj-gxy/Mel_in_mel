from __future__ import absolute_import, division, print_function, unicode_literals
import glob
import os
import json
import torch
import numpy as np
from istftnet_model import Generator
from model_4 import stego_model
from meldataset import MelDataset, mel_spectrogram
from torch.utils.data import DataLoader
import soundfile as sf
from env import AttrDict, build_env
from hifigan.istftnet_models import Generator as Gen
import torch.nn.functional as F
from stft import TorchSTFT
import torchaudio
import time
h = None
device = torch.device("cuda:0")
import math
from boltons import fileutils
from TEST.MOS_test import mos_dn, utmos_
from pystoi import stoi
from pypesq import pesq
from torchaudio.transforms import Resample
from scipy.signal import medfilt

def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict

def resample(tensor, x):
    reample1 = Resample(22050, x).to(device)
    reample2 = Resample(x, 22050).to(device)
    y = reample1(tensor)
    z = reample2(y)
    return z


def m_filter(wav, window_size):  # 中值滤波
    wav = wav.squeeze(0).detach().cpu().numpy()
    # 应用低通滤波器waveform: Tensor, sample_rate: int, cutoff_freq:
    filtered_audio = medfilt(wav, 3)
    filtered_audio_torch = torch.from_numpy(filtered_audio)
    return filtered_audio_torch.to(device)

def highpass_filter(wav, cutoff_freq= 500):
    filtered_audio = torchaudio.functional.highpass_biquad(waveform=wav, sample_rate=16000, cutoff_freq=cutoff_freq)
    return filtered_audio

def lpf(wav, C):  # 低通滤波
    # 应用低通滤波器waveform: Tensor, sample_rate: int, cutoff_freq:
    filtered_audio = torchaudio.functional.lowpass_biquad(waveform=wav, sample_rate=22050, cutoff_freq=C, Q=0.707)
    return filtered_audio


def generate_speckle_noise(tensor, noise_norm, mean=0.0, stddev=1.0):
    return tensor * (1 + noise_norm * tensor.data.new(tensor.size()).normal_(mean, stddev))


def generate_gaussian_noise(tensor, noise_norm, mean=0.0, stddev=1.0):
    return tensor + noise_norm * tensor.data.new(tensor.size()).normal_(mean, stddev)


def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + '*')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return ''
    return sorted(cp_list)[-1]


def istftnet(mel, hifi_model_path):
    config_file = '/home/lgd/miniconda3/envs/all/Mel_in/cp_hifigan/istftnet/lj/config_v1.json'
    with open(config_file) as f:
        data = f.read()
    json_config = json.loads(data)
    h = AttrDict(json_config)
    gen = Gen(h).to(device)
    state_dict_g = torch.load(hifi_model_path, map_location=device)
    gen.load_state_dict(state_dict_g['generator'])
    # gen_p_total = sum([param.numel() for param in gen.parameters()])
    # print('gen:', gen_p_total / (1024 * 1024))
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

generator_p_total = sum([param.numel() for param in generator.parameters()])
print('generator_p_total:', generator_p_total/(1024*1024))

path1 = '/home/lgd/miniconda3/envs/all/iSTFTNet/cp_hifigan/g_00675000'
path2 = '/home/lgd/miniconda3/envs/all/iSTFTNet/cp_hifigan/aihell/g_00755000'


import random
input_validation_file = '/home/lgd/miniconda3/envs/dataset/LJSpeech-1.1/validation.txt'  # '/home/lgd/miniconda3/envs/all/Mel_in/sample/test'
input_wav = '/home/lgd/miniconda3/envs/dataset/aishell3' #'/home/lgd/miniconda3/envs/dataset/LJSpeech-1.1/wavs'  #

wave_list = []

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
        validation_files.append((carrier_file, hidden_message_files))'''

tESTset = MelDataset(validation_files, 16384, h.n_fft, h.num_mels,
                     h.hop_size, h.win_size, h.sampling_rate, h.fmin, h.fmax, n_cache_reuse=0,
                     shuffle=False if h.num_gpus > 1 else True, fmax_loss=h.fmax, device=device)

test_loader = DataLoader(tESTset, batch_size=1)

MOS_GR = []
MOS_G = []
MOS_C = []

MOS_GR_T = []
MOS_G_T = []
MOS_C_T = []

MOS_RES = []
MOS_S = []

MOS_RES_T = []
MOS_S_T = []

STOI_C = []
STOI_S = []

PESQ_C = []
PESQ_S = []

all_time = []
with torch.no_grad():
    for i, batch in enumerate(test_loader):
        x_mel, x_aud, x_mel_loss, y_mel, y_aud, y_mel_s = batch
        x_mel = x_mel.to(device)
        print(x_mel.device)
        y_mel = y_mel.to(device)

        # t1 = time.perf_counter()
        spec, phase = generator(x_mel, y_mel)  # 生成语音
        y_g_hat = stft.inverse(spec, phase)
        # t2 = time.perf_counter()
        # print(t2 - t1)
        # all_time.append(t2 - t1)

        # y_g_hat = generate_gaussian_noise(y_g_hat, 0.001) #gaussian
        # y_g_hat = generate_speckle_noise(y_g_hat, 0.001) #speckle
        # y_g_hat = resample(y_g_hat, 16000)  #reasmple22050,8000,44100
        # y_g_hat = lpf(y_g_hat, 3000)  # Low-pass Filtering5000,4000
        # y_g_hat = m_filter(y_g_hat, 3)  # m_filterm,window=3
        # y_g_hat = y_g_hat.to(device)
        # y_g_hat = highpass_filter(y_g_hat, cutoff_freq=500)
        cover_1, cover_2 = istftnet(x_mel, path2)
        cover = stft.inverse(cover_1, cover_2)

        if ((y_g_hat.squeeze(0).cpu() >= -1).all() and (y_g_hat.squeeze(0).cpu() <= 1).all()
                and (cover.squeeze(0).cpu() >= -1).all() and (cover.squeeze(0).cpu() <= 1).all()):
            mos_g = mos_dn(y_g_hat.detach().cpu().numpy().squeeze(), 16000)
            print(mos_g)
            MOS_G.append(mos_g)

            mos_c = mos_dn(cover.detach().cpu().numpy().squeeze(), 16000)
            MOS_C.append(mos_c)

            mos_gr = mos_dn(x_aud.detach().cpu().numpy().squeeze(), 16000)
            MOS_GR.append(mos_gr)
            print('mos:', mos_gr)
            mos_g_T = utmos_(y_g_hat.detach().squeeze(1).cpu(), 22050)
            MOS_G_T.append(mos_g_T)

            mos_c_t = utmos_(cover.detach().squeeze(1).cpu(), 22050)
            MOS_C_T.append(mos_c_t)

            mos_gr_t = utmos_(x_aud.detach().cpu(), 22050)
            MOS_GR_T.append(mos_gr_t)

        st_c = stoi(cover.detach().cpu().numpy().squeeze(), y_g_hat.detach().cpu().numpy().squeeze(), 22050)
        print(st_c)
        if not math.isnan(st_c):
            STOI_C.append(st_c)

        pe_c = pesq(cover.detach().cpu().numpy().squeeze(), y_g_hat.detach().cpu().numpy().squeeze(), 16000)
        if not math.isnan(pe_c):
            PESQ_C.append(pe_c)

        y_g_hat_mel = mel_spectrogram(y_g_hat.squeeze(1), h.n_fft, h.num_mels, h.sampling_rate,
                                      h.hop_size, h.win_size, h.fmin, h.fmax_for_loss)

        secret_1, secret_2 = istftnet(y_mel, path2)
        secret = stft.inverse(secret_1, secret_2)

        re_serect_mel = model(y_g_hat_mel)
        re_sec_1, re_sec_2 = istftnet(re_serect_mel, path2)
        re_sec = stft.inverse(re_sec_1, re_sec_2)

        if ((re_sec.squeeze(0).cpu() >= -1).all() and (re_sec.squeeze(0).cpu() <= 1).all()
                and (secret.squeeze(0).cpu() >= -1).all() and (secret.squeeze(0).cpu() <= 1).all()):
            mos_re = mos_dn(re_sec.detach().cpu().numpy().squeeze(), 16000)
            MOS_RES.append(mos_re)

            mos_s = mos_dn(secret.detach().cpu().numpy().squeeze(), 16000)
            MOS_S.append(mos_s)

            mos_re_T = utmos_(re_sec.detach().squeeze(1).cpu(), 22050)
            MOS_RES_T.append(mos_re_T)

            mos_s_t = utmos_(secret.detach().squeeze(1).cpu(), 22050)
            MOS_S_T.append(mos_s_t)

        st_s = stoi(re_sec.detach().cpu().numpy().squeeze(), secret.detach().cpu().numpy().squeeze(), 22050)

        if not math.isnan(st_s):
            STOI_S.append(st_s)
            print(st_s)
        pe_s = pesq(re_sec.detach().cpu().numpy().squeeze(), secret.detach().cpu().numpy().squeeze(), 16000)
        if not math.isnan(pe_s):
            PESQ_S.append(pe_s)
            print(pe_s)

    print('MOS_GR= %f' % np.mean(MOS_GR))
    print('MOS_G= %f' % np.mean(MOS_G))
    print('MOS_C= %f' % np.mean(MOS_C))
    print('MOS_GR_T= %f' % np.mean(MOS_GR_T))
    print('MOS_G_T= %f' % np.mean(MOS_G_T))
    print('MOS_C_T= %f' % np.mean(MOS_C_T))
    print('STOI_C= %f' % np.mean(STOI_C))
    print('PESQ_C= %f' % np.mean(PESQ_C))

    print(' MOS_RES= %f' % np.mean(MOS_RES))
    print(' MOS_S= %f' % np.mean(MOS_S))
    print('MOS_RES_T= %f' % np.mean(MOS_RES_T))
    print('MOS_S_T= %f' % np.mean(MOS_S_T))
    print('STOI_S = %f' % np.mean(STOI_S))
    print('PESQ_S= %f' % np.mean(PESQ_S))
    print(all_time)
    print(np.mean(all_time))
