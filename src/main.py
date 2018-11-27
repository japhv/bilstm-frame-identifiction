"""
NSynth classification using PyTorch

Authors: Japheth Adhavan, Jason St. George
Reference: Sasank Chilamkurthy <https://chsasank.github.io>
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import utils
import models
import data
import visualize
import time
import copy
import argparse

import logging
logging.basicConfig(level=logging.INFO)

gpu_idx = utils.get_free_gpu()
device = torch.device("cuda:{}".format(gpu_idx))
logging.info("Using device cuda:{}".format(gpu_idx))


def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=5, test=False, bonus=False):
    """

    :param model:
    :param criterion:
    :param optimizer:
    :param scheduler:
    :param num_epochs:
    :return:
    """
    since = time.time()

    # best_model_wts = copy.deepcopy(model.state_dict())
    best_model_wts = model.state_dict()
    best_acc = 0.0
    dataset_sizes = {x: len(dataloaders[x].dataset) for x in ["train", "val"]}
    model_loss = {x: [0 for _ in range(num_epochs)] for x in ["train", "val"]}
    spreads = [[0 for y in range(10)] for x in range(10)]
    test_c = [0 for x in range(10)]
    test_t = [0 for x in range(10)]
    c_done = [False for x in range(10)]
    i_done = [False for x in range(10)]
    c_samples, i_samples = 0, 0
    y_test, y_pred = [], []

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)

        phases = ['val'] if test else ['train', 'val'] # Skip training if we're testing
        for phase in phases:
            if phase == 'train':
                scheduler.step()
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            # iterate over data
            for batch_idx, (samples, if_label, is_label, targets) in enumerate(dataloaders[phase]):
                inputs = samples.to(device)
                if bonus:
                    labels = is_label.to(device)
                else:
                    labels = if_label.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    # Add spread/confusion matrix statistics
                    if phase == 'val':
                        correct = (preds == labels).squeeze()
                        np_predicted = preds.cpu().numpy()  # Get vector of int class output labels
                        y_pred.extend(np_predicted)
                        y_test.extend(if_label.cpu().numpy())

                        if i_samples < 10 and c_samples < 10:
                            for i in range(len(outputs)):
                                label = str(labels[i])  # e.g. 'tensor(0)'
                                label = int(label[7])  # 0
                                test_c[label] += correct[i].item()
                                test_t[label] += 1

                                if np_predicted[i] != label:
                                    spreads[label][np_predicted[i]] += 1
                                    if i_samples < 10:
                                        i_samples += visualize.save_samples(inputs[i],
                                                                            np_predicted[i], label,
                                                                            i_done, False, CLASS_NAMES)
                                else:
                                    if c_samples < 10:
                                        c_samples += visualize.save_samples(inputs[i], None, label,
                                                                            c_done, True, CLASS_NAMES)

                # aggregate statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                if phase == 'train' and batch_idx % 50 == 0:
                    logging.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch,
                        (batch_idx + 1) * len(samples),
                        dataset_sizes[phase],
                        100. * (batch_idx + 1) / len(dataloaders[phase]),
                        loss.item()))


            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            model_loss[phase][epoch] = epoch_loss

            logging.info('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase.capitalize(), epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                # best_model_wts = copy.deepcopy(model.state_dict())
                best_model_wts = model.state_dict()

        print()

    time_elapsed = time.time() - since
    logging.info('Training completed in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    logging.info('Best overall val Acc: {:4f}'.format(best_acc))

    for i in range(10):
        spreads[i][i] = test_c[i]

    accuracies = []
    for i in range(10):
        accuracies.append((test_c[i], test_t[i]))
        print('Accuracy of %5s : %2d %%' % (
            i, 100 * test_c[i] / test_t[i]))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, model_loss, accuracies, spreads, y_test, y_pred


def main(args):
    np.warnings.filterwarnings('ignore')
    train_loader = data.get_data_loader("train", batch_size=args.batch_size,
                                        shuffle=True, num_workers=4, bonus=args.bonus)
    valid_loader = data.get_data_loader("valid", bonus=args.bonus)

    dataloaders = {
        "train": train_loader,
        "val": valid_loader
    }

    if args.bonus:
        model = models.BonusNetwork().to(device)
        type = "Bonus"
        CLASS_NAMES = ['acoustic', 'electronic', 'synthetic']

    else:
        CLASS_NAMES = ['bass', 'brass', 'flute', 'guitar', 'keyboard',
                       'mallet', 'organ', 'reed', 'string', 'vocal']

        if args.epicmodel:
            model = models.EpicNetwork().to(device)
            type = "Epic"
        else:
            model = models.SimpleNetwork().to(device)
            type = "Simple"

    model.double()
    criterion = nn.CrossEntropyLoss()
    optimizer_conv = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=args.step, gamma=args.gamma)

    if not args.test:
        print('Training...')
        model, model_loss, accuracies, spread, y_test, y_pred = train_model(model, dataloaders,
                                                                            criterion, optimizer_conv,
                                                                            exp_lr_scheduler,
                                                                            num_epochs=args.epochs,
                                                                            bonus=args.bonus)
        torch.save(model.state_dict(), "./" + type + "Network.pt")
    else:
        print('Testing...')
        model = model.load_state_dict(torch.load("./" + type + "Network.pt"))
        model, model_loss, accuracies, spread, y_test, y_pred = train_model(model, dataloaders, criterion,
                                                                            optimizer_conv,
                                                                            exp_lr_scheduler,
                                                                            test=True, num_epochs=1)

    visualize.plot_loss(model_loss, type + "Network")
    visualize.plot_histograms(accuracies, CLASS_NAMES, spread, type=type)
    visualize.plot_confusion_matrix(y_test, y_pred, CLASS_NAMES, type=type)

    print('Completed Successfully!')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='NSynth classifier')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test', action='store_true', default=False,
                        help='disables training, loads model')
    parser.add_argument('--bonus', action='store_true', default=True,
                        help='sets bonus flag for transforms to mfcc')
    parser.add_argument('--epicmodel', action='store_true', default=False,
                        help='If True, use epic network, otherwise use simple network')
    parser.add_argument('--epochs', type=int, default=5, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--step', type=int, default=3, metavar='N',
                        help='number of epochs to decrease learn-rate (default: 3)')
    parser.add_argument('--gamma', type=float, default=0.1, metavar='N',
                        help='factor to decrease learn-rate (default: 0.1)')
    main(parser.parse_args())