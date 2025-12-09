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
from speechbrain.inference.TTS import Tacotron2,MSTacotron2
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


######SPN#############

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# tacotron2 = Tacotron2.from_hparams(source="speechbrain/tts-tacotron2-ljspeech", savedir=tmpdir_tts)
# Intialize TTS (tacotron2) and Vocoder (HiFIGAN)
mstacotron2 = MSTacotron2.from_hparams(source="speechbrain/tts-mstacotron2-libritts",
                                       savedir='/home/lgd/miniconda3/envs/all/Mel_in/tmpdir_tts_1')

input_text = "Where are you going to play tomorrow? Do you want to come with me?"
reference_audio_path = '/home/lgd/miniconda3/envs/dataset/aishell3/SSB0005/wav/SSB00050001.wav'
# Running the TTS
mel_output, mel_length, alignment = mstacotron2.clone_voice(input_text, reference_audio_path)
print(mel_length)

waveforms = hifigan(mel_output.to(device))

torchaudio.save('example_TTS.wav',waveforms.squeeze(1).cpu(), 22050)


