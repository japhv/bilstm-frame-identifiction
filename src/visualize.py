import os
import matplotlib as mpl
if os.environ.get('DISPLAY', '') == '':
    print('no display found. Using non-interactive Agg backend')
mpl.use('Agg')
mpl.rcParams['agg.path.chunksize'] = 10000

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import waveform
import itertools
from sklearn.metrics import confusion_matrix


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


def plot_histograms(class_acc, class_names, spreads, type='Simple'):
    arr = []
    for i in range(10):
        arr.append(class_acc[i][0] / class_acc[i][1])

    x = np.arange(10)

    # Histogram plot of all 10 classes
    hfig, az = plt.subplots()
    plt.title('Relative accuracy of all classes')
    plt.bar(x, height=arr)
    plt.xticks(x, class_names)
    plt.xlabel('Class')
    plt.ylabel('Accuracy')
    hfig.savefig('NSynth_accuracy_hist_' + type + '.png', bbox_inches='tight')

    # Plots of each class predictions
    for i in range(10):
        hfig, az = plt.subplots()
        plt.title('Prediction spread of class: ' + class_names[i])
        plt.bar(x, height=spreads[i])
        plt.xticks(x, class_names)
        plt.xlabel('Classes')
        plt.ylabel('# Predicted')
        hfig.savefig('NSynth_class_' + class_names[i] + '_accuracy_hist_' + type + '.png',
                     bbox_inches='tight')


def save_samples(tensor, predicted, label, done, correct, class_names):
    if done[label]:
        return 0
    done[label] = True

    waveform.plot_wave([class_names[label]], [tensor.cpu()], predicted, correct, sr=16000)
    waveform.plot_specgram([class_names[label]], [tensor.cpu()], predicted, correct, sr=16000)

    return 1


def plot_confusion_matrix(y_test, y_pred, classes, type='Simple',
                          title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    cm = confusion_matrix(y_test, y_pred)
    np.set_printoptions(precision=2)

    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print("Normalized confusion matrix")
    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig('./confusion_matrix_' + type + '.png')