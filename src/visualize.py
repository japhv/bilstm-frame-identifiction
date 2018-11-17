import os
import matplotlib as mpl
if os.environ.get('DISPLAY', '') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import numpy as np


def plot_loss(model_loss, model_name):
    fig, ax = plt.subplots()

    train_loss = model_loss["train"]
    valid_loss = model_loss["val"]

    epochs = len(train_loss)

    x = np.linspace(1, epochs, epochs)

    ax.set_title("Average Model Loss over Epochs on {}".format(model_name))

    ax.set_xlabel("Epochs")
    ax.set_ylabel("Average Loss")

    # Adjust x-axis ticks
    tick_spacing = 5
    ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))

    ax.plot(x, train_loss, color='purple', label="train", marker=".")
    ax.plot(x, valid_loss, color='red', label="validation", marker="x")

    os.makedirs("./graphs", exist_ok=True)
    fig.savefig("./graphs/{}_epoch_loss".format(model_name))