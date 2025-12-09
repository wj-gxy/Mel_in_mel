from __future__ import absolute_import, division, print_function, unicode_literals
import glob
import os
import json
import torch
import numpy as np
from scipy.io.wavfile import write
from env import AttrDict
from meldataset import MAX_WAV_VALUE
from model import Generator
from model_4 import stego_model
from meldataset import MelDataset, mel_spectrogram
from torch.utils.data import DataLoader
import soundfile as sf
from env import AttrDict, build_env
from hifigan.models import Generator as Gen
import torch.nn.functional as F

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




def hifigan(mel, hifi_model_path):
    config_file = '/home/lgd/miniconda3/envs/all/Mel_in/hifigan/config.json'
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


config_file = '/home/lgd/miniconda3/envs/all/Mel_in/cp_hifigan/hifigan/hifigan_lj/config.json'
with open(config_file) as f:
    data = f.read()
json_config = json.loads(data)
h = AttrDict(json_config)
generator = Generator(h).to(device)
checkpoint_file = '/home/lgd/miniconda3/envs/all/Mel_in/cp_hifigan/hifigan/hifigan_lj/g_00445000'
state_dict_g = load_checkpoint(checkpoint_file, device)
generator.load_state_dict(state_dict_g['generator'])
model = stego_model().to(device)
model.load_state_dict(state_dict_g['model'])
generator.eval()
model.eval()
generator.remove_weight_norm()

path1 = '/home/lgd/miniconda3/envs/all/hifi-gan-master/hifigan/cp_hifigan/LJ_V1/generator_v1'
path2 = '/home/lgd/miniconda3/envs/all/hifi-gan-E/ZH/22050/g_00285000'

import random

input_validation_file = '/home/lgd/miniconda3/envs/dataset/LJSpeech-1.1/validation.txt'  # '/home/lgd/miniconda3/envs/all/Mel_in/sample/test'
input_wav = '/home/lgd/miniconda3/envs/dataset/LJSpeech-1.1/wavs'#'/home/lgd/miniconda3/envs/dataset/aishell3'  #

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
    for i in range(300):
        sampled_files = random.sample(validation_file, 2)
        carrier_file, hidden_message_files = sampled_files[0], sampled_files[1]
        validation_files.append((carrier_file, hidden_message_files))


tESTset = MelDataset(validation_files, 16384*3, h.n_fft, h.num_mels,
                     h.hop_size, h.win_size, h.sampling_rate, h.fmin, h.fmax, n_cache_reuse=0,
                     shuffle=False if h.num_gpus > 1 else True, fmax_loss=h.fmax, device=device)
test_loader = DataLoader(tESTset, batch_size=1)

MOS_gen = []
MOS_ground = []
MOS_cover =[]
STOI = []
PESQ = []
with torch.no_grad():
    for i, batch in enumerate(test_loader):
        x_mel, x_aud, x_mel_loss, y_mel, y_aud, y_mel_s = batch
        x_mel.to(device)
        y_mel.to(device)
        y_g_hat = generator(x_mel, y_mel)

        cover = hifigan(x_mel, path1)
        y_g_hat_mel = mel_spectrogram(y_g_hat.squeeze(1), h.n_fft, h.num_mels, h.sampling_rate,
                                                                    h.hop_size, h.win_size, h.fmin, h.fmax_for_loss)

        re_serect_mel = model(y_g_hat_mel)
        secret = hifigan(y_mel, path1)
        secret = (secret * MAX_WAV_VALUE).squeeze()
        re_sec = hifigan(re_serect_mel, path1)
        re_sec = (re_sec * MAX_WAV_VALUE).squeeze()


        audio = y_g_hat.squeeze()
        audio = audio * MAX_WAV_VALUE
        cover = cover.squeeze()

        cover = cover * MAX_WAV_VALUE
        audio = audio.cpu().numpy().astype('int16')
        cover = cover.cpu().numpy().astype('int16')
        secret = secret.cpu().numpy().astype('int16')
        re_sec = re_sec.cpu().numpy().astype('int16')


        sf.write(f'asr_model/secret{i}.wav', secret, 22050)
        sf.write(f'asr_model/re_sec{i}.wav', re_sec, 22050)
        sf.write(f'asr_model/stego_{i}.wav', audio, 22050)
        sf.write(f'asr_model/cover_{i}.wav', cover, 22050)

        '''
        st_ = test_stoi(cover.squeeze(), y_g_hat.squeeze())
        if not math.isnan(st_):
            STOI.append(st_)

        pe_ = test_pesq(cover.squeeze(), y_g_hat.squeeze())
        if not math.isnan(pe_):
            PESQ.append(pe_)

        if ((y_g_hat.squeeze(0).cpu() >= -1).all() and (y_g_hat.squeeze(0).cpu() <= 1).all()):
            # mos_groud = mos_dn((x_aud.squeeze(0)).cpu(), 16000)
            # mos_cover= mos_dn((cover.squeeze(0).squeeze(0)).cpu(), 16000)
            # mos_glow = mos_dn((y_g_hat.squeeze(0).squeeze(0)).cpu(), 16000)

            mos_cover = utmos_((cover.squeeze(0)), 22050)
            mos_glow = utmos_((y_g_hat.squeeze(0)), 22050)
            mos_groud = utmos_((x_aud), 22050)

            MOS_cover.append(mos_cover)
            MOS_ground.append(mos_groud)
            MOS_gen.append(mos_glow)
            print(mos_glow)

        
        # y_g_hat_mel = mel_spectrogram(y_g_hat.squeeze(1), h.n_fft, h.num_mels, h.sampling_rate,
        #                               h.hop_size, h.win_size, h.fmin, h.fmax_for_loss)
        #
        # re_serect_mel = model(y_g_hat_mel)
        # audio = y_g_hat.squeeze()
        # audio = audio * MAX_WAV_VALUE
        # cover = hifigan(x_mel, path1)
        # cover = (cover * MAX_WAV_VALUE).squeeze()
        # secret = hifigan(y_mel, path1)
        # secret = (secret * MAX_WAV_VALUE).squeeze()
        # re_sec = hifigan(re_serect_mel, path1)
        # re_sec = (re_sec * MAX_WAV_VALUE).squeeze()
        # audio = audio.cpu().numpy().astype('int16')
        # sf.write('sample/containers.wav', audio, 22050)
        # cover = cover.cpu().numpy().astype('int16')
        # sinr = signal_noise_ratio(cover, audio)
        # print(sinr)
        # sf.write('sample/cover.wav', cover, 22050)
        # secret = secret.cpu().numpy().astype('int16')
        # sf.write('sample/secret.wav', secret, 22050)
        # re_sec = re_sec.cpu().numpy().astype('int16')
        # Snr = signal_noise_ratio(secret, re_sec)
        # print(Snr)
        # sf.write('sample/re_sec.wav', re_sec, 22050)

        
        cover = hifi_gan.decode_batch(x_mel)
        print(cover.size())
        secret = hifi_gan.decode_batch(y_mel)
        re_sec = hifi_gan.decode_batch(re_serect_mel)
        audio = audio.cpu().numpy().astype('int16')
        # cover = cover.squeeze(1).cpu().numpy().astype('int16')
        # secret = secret.squeeze(1).cpu().numpy().astype('int16')
        # re_sec = re_sec.squeeze(1).cpu().numpy().astype('int16')
        sf.write('sample/containers.wav', audio, 22050)
        sf.write('sample/cover.wav', np.ravel(cover.squeeze(1).cpu().numpy()), 22050)
        sf.write('sample/secret.wav', np.ravel(secret.squeeze(1).cpu().numpy()), 22050)
        sf.write('sample/re_sec.wav', np.ravel(re_sec.squeeze(1).cpu().numpy()), 22050)'''
    print('MOS_gen= %f' % np.mean(MOS_gen))
    print('MOS_ground= %f' % np.mean(MOS_ground))
    print('MOS_cover= %f' % np.mean(MOS_cover))
    print('stoi= %f' % np.mean(STOI))
    print('pesq= %f' % np.mean(PESQ))
