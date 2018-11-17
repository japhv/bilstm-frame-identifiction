import torch.utils.data as data
import torchvision.transforms as transforms
from pytorch_nsynth.nsynth import NSynth

import numpy as np


def get_data_loader(type, batch_size=1, shuffle=False):
    assert (type in ("train", "test", "valid", )), "Invalid data loader type: {}".format(type)

    data_path = {
        "train": "/local/sandbox/nsynth/nsynth-train",
        "test" : "/local/sandbox/nsynth/nsynth-test",
        "valid": "/local/sandbox/nsynth/nsynth-valid"
    }

    # audio samples are loaded as an int16 numpy array
    # rescale intensity range as float [-1, 1]
    toFloat = transforms.Lambda(lambda x: x / np.iinfo(np.int16).max)
    # use instrument_family and instrument_source as classification targets

    dataset = NSynth(
        data_path[type],
        transform=toFloat,
        blacklist_pattern=["synth_lead"],  # blacklist string instrument
        categorical_field_list=["instrument_family", "instrument_source"])
    loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader

