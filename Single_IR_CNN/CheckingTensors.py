import soundfile as sf
import numpy as np
import torch
import torchaudio
import os

for filename in os.listdir('input/'):
    fileAndPath = 'input/' + filename
    sound, sr = torchaudio.load(fileAndPath)
    print(f'({type(sound[0])}), {filename}:\t {len(sound[0])}')
    # print(sound)
for filename in os.listdir('target/'):
    fileAndPath = 'target/' + filename
    sound, sr = torchaudio.load(fileAndPath)
    print(f'({type(sound[0])}), {filename}:\t {len(sound[0])}')
    # print(sound)