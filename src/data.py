import torch.utils.data as data
import torchvision.transforms as transforms
from pytorch_nsynth.nsynth import NSynth
from scipy import signal
import numpy as np


def get_data_loader(type, **kwargs):
    assert (type in ("train", "test", "valid", )), "Invalid data loader type: {}".format(type)

    data_path = {
        "train": "/local/sandbox/nsynth/nsynth-train",
        "test" : "/local/sandbox/nsynth/nsynth-test",
        "valid": "/local/sandbox/nsynth/nsynth-valid"
    }

    # audio samples are loaded as an int16 numpy array
    # rescale intensity range as float [-1, 1]
    toFloat = transforms.Lambda(lambda x: x / np.iinfo(np.int16).max)

    transformations = transforms.Compose([
        transforms.Lambda(lambda x: signal.resample(x, 32,000)),
        toFloat,
        transforms.Lambda(lambda x: np.expand_dims(x, axis=0))
    ])

    dataset = NSynth(
        data_path[type],
        transform=transformations,
        blacklist_pattern=["synth_lead"],  # blacklist synth_lead instrument
        categorical_field_list=["instrument_family", "instrument_source"])
    loader = data.DataLoader(dataset, **kwargs)
    return loader

