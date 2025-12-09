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
from speechbrain.inference.TTS import Tacotron2
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


def hifigan(mel, hifi_model_path= '/home/lgd/miniconda3/envs/all/hifi-gan-master/hifigan/cp_hifigan/LJ_V1/generator_v1'):
    config_file = '/home/lgd/miniconda3/envs/all/Mel_in/hifigan/config.json'
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


######SPN#############




# Intialize TTS (tacotron2) and Vocoder (HiFIGAN)
tacotron2 = Tacotron2.from_hparams(source="speechbrain/tts-tacotron2-ljspeech", savedir="tmpdir_tts")
text = "in being comparatively modern."

# Running the TTS
mel_output, mel_length, alignment = tacotron2.encode_text(text)
print(mel_length)

waveforms = hifigan(mel_output.to(device))
print(waveforms.size())
torchaudio.save('example_TTS.wav',waveforms.squeeze(1).cpu(), 22050)

from meldataset import MelDataset, mel_spectrogram, MAX_WAV_VALUE
from torch.utils.data import DataLoader
validation_files= []
path_1 = '/home/lgd/miniconda3/envs/dataset/LJSpeech-1.1/re_wave/test'
filenames = glob.glob(f'{path_1}/*.wav', recursive=True)
carrier_file, hidden_message_files = filenames[1], filenames[0]
validation_files.append((carrier_file, hidden_message_files))

tESTset = MelDataset(validation_files, waveforms.size(2), h.n_fft, h.num_mels,
                     h.hop_size, h.win_size, h.sampling_rate, h.fmin, h.fmax, n_cache_reuse=0,
                     shuffle=False if h.num_gpus > 1 else True, fmax_loss=h.fmax, device=device)

test_loader = DataLoader(tESTset, batch_size=1)

with torch.no_grad():
    for i, batch in enumerate(test_loader):
        x_mel, x_aud, x_mel_loss, y_mel, y_aud, y_mel_s = batch
        x_mel = x_mel.to(device)
        print(x_mel.size())
        y_mel = y_mel.to(device)
        mel_output = mel_output.to('cuda')
        print(mel_output.size())

        cover = hifigan(y_mel)
        waveforms = hifigan(mel_output)
        y_g_hat = generator(y_mel, mel_output)
        y_g_hat_1 = generator(y_mel, x_mel)

        # cover =  cover* MAX_WAV_VALUE

torchaudio.save('C.wav',cover.squeeze(1).cpu(), 22050)
torchaudio.save('S_text_1.wav',y_g_hat.squeeze(1).cpu(), 22050)
torchaudio.save('S_audio_1.wav',y_g_hat_1.squeeze(1).cpu(), 22050)

# Save the waverform
