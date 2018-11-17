"""
NSynth classification using PyTorch

Authors: Japheth Adhavan, Jason St. George
Reference: Sasank Chilamkurthy <https://chsasank.github.io>
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

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



def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=25):
    """

    :param model:
    :param criterion:
    :param optimizer:
    :param scheduler:
    :param num_epochs:
    :return:
    """
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    dataset_sizes = { x: len(dataloaders[x].dataset) for x in ["train", "val"] }

    model_loss = {x : [0 for _ in range(num_epochs)] for x in ["train", "val"]}

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for batch_idx, (samples, if_label, is_label, targets) in enumerate(dataloaders[phase]):
                inputs = samples.to(device)
                labels = if_label.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
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
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    logging.info('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    logging.info('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, model_loss


def main(args):
    train_loader = data.get_data_loader("train", batch_size=32, shuffle=True, num_workers=4)
    valid_loader = data.get_data_loader("valid")

    dataloaders = {
        "train": train_loader,
        "val": valid_loader
    }

    model = models.SimpleNetwork().to(device)
    model.double()

    criterion = nn.CrossEntropyLoss()

    if not args.test:
        # Observe that only parameters of final layer are being optimized as
        # opoosed to before.
        optimizer_conv = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

        # Decay LR by a factor of 0.1 every 7 epochs
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)

        model, model_loss = train_model(model, dataloaders, criterion, optimizer_conv, exp_lr_scheduler)

        visualize.plot_loss(model_loss, "SimpleNetwork")

        torch.save(model.state_dict(), "./models/SimpleNetwork.pt")
    else:
        model.load_state_dict(torch.load("./models/SimpleNetwork.pt"))



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='NSynth classifier')
    parser.add_argument('--test', action='store_true', default=False,
                        help='disables training, loads model')
    main(parser.parse_args())