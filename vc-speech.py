from __future__ import absolute_import, division, print_function, unicode_literals
import torch
import matplotlib.pyplot as plt
import glob
import os
import json
import torch
from env import AttrDict, build_env
from hifigan.models import Generator as Gen
from model import Generator
from model_4 import stego_model
import torchaudio

h = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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


def hifigan(mel, hifi_model_path= '/home/lgd/miniconda3/envs/all/hifi-gan-E/ZH/16k_vqmic/g_00430000'):
    config_file = '/home/lgd/miniconda3/envs/all/hifi-gan-E/ZH/16k_vqmic/16_vqconfig.json'
    with open(config_file) as f:
        data = f.read()
    json_config = json.loads(data)
    h = AttrDict(json_config)
    gen = Gen(h).to(device)
    state_dict_g = torch.load(hifi_model_path, map_location=device)
    gen.load_state_dict(state_dict_g['generator'])
    gen.eval().to(device)
    y = gen(mel)
    return y



config_file = '/home/lgd/miniconda3/envs/all/Mel_in/cp_hifigan/hifigan/vctk/config.json'
with open(config_file) as f:
    data = f.read()
json_config = json.loads(data)
h = AttrDict(json_config)

generator = Generator(h).to(device)
checkpoint_file = '/home/lgd/miniconda3/envs/all/Mel_in/cp_hifigan/hifigan/vctk/g_00745000'
state_dict_g = load_checkpoint(checkpoint_file, device)
generator.load_state_dict(state_dict_g['generator'])
model = stego_model().to(device)
model.load_state_dict(state_dict_g['model'])
generator.eval()
model.eval()
generator.remove_weight_norm()




from meldataset import MelDataset, mel_spectrogram, MAX_WAV_VALUE
from torch.utils.data import DataLoader
import soundfile as sf
import random
validation_files= []
path_1 = '/home/lgd/miniconda3/envs/all/Trace_2/ConsistencyVC/output/60_exp_crosslingual_whispers-three-emo-loss'
filenames = glob.glob(f'{path_1}/*.wav', recursive=True)
# sampled_files = random.sample(filenames, 2)
filenames[1] ='/home/lgd/miniconda3/envs/all/Mel_in/fig/vc/scat/p239_342_to_p257_116.wav'








a = ['/home/lgd/miniconda3/envs/all/Mel_in/fig/vc/scat/p258_016_to_p313_385.wav',
     '/home/lgd/miniconda3/envs/all/Mel_in/fig/vc/scat/p258_072_to_p313_404.wav',
     '/home/lgd/miniconda3/envs/all/Mel_in/fig/vc/scat/p258_243_to_p313_368.wav']



b = ['/home/lgd/miniconda3/envs/all/Mel_in/fig/vc/scat/p313_368.wav',
     '/home/lgd/miniconda3/envs/all/Mel_in/fig/vc/scat/p313_385.wav',
     '/home/lgd/miniconda3/envs/all/Mel_in/fig/vc/scat/p313_404.wav']

carrier_file, hidden_message_files = b[2], a[2]
print(carrier_file)
print(hidden_message_files)
validation_files.append((carrier_file, hidden_message_files))

tESTset = MelDataset(validation_files, 16384, h.n_fft, h.num_mels,
                     h.hop_size, h.win_size, h.sampling_rate, h.fmin, h.fmax, n_cache_reuse=0,
                     shuffle=False if h.num_gpus > 1 else True, fmax_loss=h.fmax, device=device)
test_loader = DataLoader(tESTset, batch_size=1)

with torch.no_grad():
    for i, batch in enumerate(test_loader):
        x_mel, x_aud, x_mel_loss, y_mel, y_aud, y_mel_s = batch
        x_mel = x_mel.to(device)
        print(x_mel.size())
        y_mel = y_mel.to(device)


        # cover = hifigan(y_mel)

        y_g_hat_1 = generator(y_mel, x_mel)

        # cover =  cover* MAX_WAV_VALUE

        cover = y_aud.squeeze()* MAX_WAV_VALUE
        cover = cover.cpu().numpy().astype('int16')


        y_g_hat_1 = y_g_hat_1.squeeze()* MAX_WAV_VALUE
        y_g_hat_1 = y_g_hat_1.cpu().numpy().astype('int16')

        print(cover)
        print(y_g_hat_1)
    sf.write(f'/home/lgd/miniconda3/envs/all/Mel_in/fig/vc/scat/C_3.wav', cover, 16000)
    sf.write(f'/home/lgd/miniconda3/envs/all/Mel_in/fig/vc/scat/S_3.wav',y_g_hat_1, 16000)

# Save the waverform
