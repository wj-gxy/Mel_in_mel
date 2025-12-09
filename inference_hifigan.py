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
import time
h = None
device = torch.device("cuda")
import math
from boltons import fileutils
from TEST.MOS_test import mos_dn, utmos_
from pystoi import stoi
from pypesq import pesq


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
    # gen_p_total = sum([param.numel() for param in gen.parameters()])
    # print('gen:', gen_p_total / (1024 * 1024))
    gen.eval()
    y = gen(mel)
    return y


config_file = '/home/lgd/miniconda3/envs/all/Mel_in/cp_hifigan/hifigan/hifigan_lj/config.json'
with open(config_file) as f:
    data = f.read()
json_config = json.loads(data)
h = AttrDict(json_config)
generator = Generator(h).to(device)
checkpoint_file = '/home/lgd/miniconda3/envs/all/Mel_in/cp_hifigan/hifigan/hifigan_lj/new_45/g_00315000'
#'/home/lgd/miniconda3/envs/all/Mel_in/cp_hifigan/hifigan/hifigan_lj/g_00445000'
state_dict_g = load_checkpoint(checkpoint_file, device)
generator.load_state_dict(state_dict_g['generator'])
model = stego_model().to(device)
model.load_state_dict(state_dict_g['model'])
generator.eval()
model.eval()
generator.remove_weight_norm()

# generator_p_total = sum([param.numel() for param in generator.parameters()])
# print('generator_p_total:', generator_p_total/(1024*1024))


path1 = '/home/lgd/miniconda3/envs/all/hifi-gan-master/hifigan/cp_hifigan/LJ_V1/generator_v1'
path2 = '/home/lgd/miniconda3/envs/all/hifi-gan-E/ZH/22050/g_00325000'

import random

input_validation_file = '/home/lgd/miniconda3/envs/dataset/LJSpeech-1.1/validation.txt'  # '/home/lgd/miniconda3/envs/all/Mel_in/sample/test'
input_wav = '/home/lgd/miniconda3/envs/dataset/LJSpeech-1.1/wavs'  #'/home/lgd/miniconda3/envs/dataset/aishell3'  #

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

# torch.manual_seed(42)
random.seed(1234)  # 控制 random.sample()
np.random.seed(1234)  # 控制 np.random 相关操作

with open(input_validation_file, 'r', encoding='utf-8') as fi:
    validation_file = [os.path.join(input_wav, x.split('|')[0] + '.wav')
                       for x in fi.read().split('\n') if len(x) > 0]
    validation_files = []
    for i in range(100):
        sampled_files = random.sample(validation_file, 2)
        carrier_file, hidden_message_files = sampled_files[0], sampled_files[1]
        validation_files.append((carrier_file, hidden_message_files))
print(validation_files)
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
        y_mel = y_mel.to(device)
        # print(x_mel.device)
        t1 = time.perf_counter()
        y_g_hat = generator(x_mel, y_mel)
        t2 = time.perf_counter()
        # print(t2-t1)
        all_time.append(t2-t1)
        cover = hifigan(x_mel, path1)
        y_g_hat_mel = mel_spectrogram(y_g_hat.squeeze(1), h.n_fft, h.num_mels, h.sampling_rate,
                                      h.hop_size, h.win_size, h.fmin, h.fmax_for_loss)

        re_serect_mel = model(y_g_hat_mel)
        secret = hifigan(y_mel, path1)

        re_sec = hifigan(re_serect_mel, path1)
        if ((y_g_hat.squeeze(0).cpu() >= -1).all() and (y_g_hat.squeeze(0).cpu() <= 1).all()
                and (cover.squeeze(0).cpu() >= -1).all() and (cover.squeeze(0).cpu() <= 1).all()):
            mos_g = mos_dn(y_g_hat.detach().cpu().numpy().squeeze(), 16000)
            print('stego:',mos_g)
            MOS_G.append(mos_g)
            mos_c = mos_dn(cover.detach().cpu().numpy().squeeze(), 16000)
            print('cover:',mos_c)
            MOS_C.append(mos_c)
            mos_gr = mos_dn(x_aud.detach().cpu().numpy().squeeze(), 16000)
            MOS_GR.append(mos_gr)
            print('mos_gr:', mos_gr)

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

        re_serect_mel = model(y_g_hat_mel)

        secret = hifigan(y_mel, path1)

        re_sec = hifigan(re_serect_mel, path1)
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
    print('MOS_stego= %f' % np.mean(MOS_G))
    print('MOS_C= %f' % np.mean(MOS_C))
    print('MOS_GR_T= %f' % np.mean(MOS_GR_T))
    print('MOS_stego_T= %f' % np.mean(MOS_G_T))
    print('MOS_C_T= %f' % np.mean(MOS_C_T))
    print('STOI_C= %f' % np.mean(STOI_C))
    print('PESQ_C= %f' % np.mean(PESQ_C))

    print(' MOS_RES= %f' % np.mean(MOS_RES))
    print(' MOS_S= %f' % np.mean(MOS_S))
    print('MOS_RES_T= %f' % np.mean(MOS_RES_T))
    print('MOS_S_T= %f' % np.mean(MOS_S_T))
    print('STOI_S = %f' % np.mean(STOI_S))
    print('PESQ_S= %f' % np.mean(PESQ_S))
    print(np.mean(all_time))


