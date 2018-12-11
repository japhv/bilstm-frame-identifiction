import matplotlib as mpl
mpl.use('Agg')
mpl.rcParams['agg.path.chunksize'] = 10000
import torch.utils.data as data
import torchvision.transforms as transforms
from pytorch_nsynth.nsynth import NSynth
from scipy.signal import resample
import numpy as np


def get_data_loader(type, fast=True, **kwargs):
    assert (type in ("train", "test", "valid", )), "Invalid data loader type: {}".format(type)

    data_path = {
        "train": "/local/sandbox/nsynth/nsynth-train",
        "test" : "/local/sandbox/nsynth/nsynth-test",
        "valid": "/local/sandbox/nsynth/nsynth-valid"
    }

    categorical_field_list = ["instrument_family"]

    if fast:
        transformations = transforms.Compose([
            transforms.Lambda(lambda x: x / np.iinfo(np.int16).max),
            transforms.Lambda(lambda x: resample(x, 32000)),
            transforms.Lambda(lambda x: x[:12000]),
            transforms.Lambda(lambda x: np.expand_dims(x, axis=0))
        ])

    else:
        transformations = transforms.Compose([
            transforms.Lambda(lambda x: x / np.iinfo(np.int16).max),
            transforms.Lambda(lambda x: x[:16000]),
            transforms.Lambda(lambda x: np.expand_dims(x, axis=0))
        ])


    dataset = NSynth(
        data_path[type],
        transform=transformations,
        blacklist_pattern=["synth_lead"],  # blacklist synth_lead instrument
        categorical_field_list=categorical_field_list)
    loader = data.DataLoader(dataset, **kwargs)
    return loader

