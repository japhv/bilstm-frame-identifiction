import os
import glob
import torch
import torchaudio as ta
from sys import argv


def get_audio(path):
    audio_data, srs = [], []
    files = os.listdir(path)
    for file in glob.glob(os.path.join(path, '*.wav')):
        data, sr = ta.load(file)
        audio_data.append(data)

    return audio_data, srs


def main():
    data_path = './nsynth-test/audio/'
    data, sample_rates = get_audio(data_path)
    print(len(data))


if __name__ == "__main__":
    main()
