import os

import torch
import torchvision
from tqdm import tqdm
import numpy as np

from src.tracking_triplet.utils import matplotlib_imshow

def save_checkpoint(state, is_best, output_path):
    """Save checkpoint if a new best is achieved"""
    if is_best:
        save_path = os.path.join(output_path, f"checkpoint_epoch_{state['epoch']}_loss_{state['loss']}.pth")
        torch.save(state, save_path)
    else:
        print("Validation Accuracy did not improve")

def write_images_tensorboard(images, writer):
    img_grid = torchvision.utils.make_grid(images)
    matplotlib_imshow(img_grid)
    writer.add_image('batch', img_grid)


def train_epoch(model, train_loader, optimizer, criterion, log_interval, cuda, writer, epoch):
    model.train()
    losses = []
    i = 0
    for data, target in tqdm(train_loader, desc="Training epoch"):
        if cuda:
            data = data.cuda()
            target = target.cuda()

        optimizer.zero_grad()

        outputs = model(data)
        loss, number_triplets = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if i % log_interval == 0:
            print(f"\n***{i} Avgloss: {np.mean(losses)} | triplets: {number_triplets}")
            writer.add_scalar('training loss', np.mean(losses), epoch * len(train_loader) + i)
            if i == 0:
                write_images_tensorboard(data.cpu(), writer)

        i += 1
    return np.mean(losses)

@torch.no_grad()
def val_epoch(model, val_loader, criterion, cuda, writer, epoch):
    model.eval()
    losses = []
    i = 0
    for data, target in tqdm(val_loader, desc="Validation epoch"):

        if cuda:
            data = data.cuda()
            target = target.cuda()

        outputs = model(data)
        loss, number_triplets = criterion(outputs, target)
        losses.append(loss.item())

        writer.add_scalar('validation loss', np.mean(losses), epoch * len(val_loader) + i)
        write_images_tensorboard(data.cpu(), writer)
        i += 1

    return np.mean(losses)


def fit(model, epochs, train_loader, val_loader, scheduler, optimizer, criterion, log_interval, cuda, writer, output_path):
    best_loss = 100
    for epoch in range(1, epochs+1):
        print(f"\nEpoch {epoch}")
        train_loss = train_epoch(model, train_loader, optimizer, criterion, log_interval, cuda, writer, epoch)
        val_loss = val_epoch(model, val_loader, criterion, cuda, writer, epoch)

        print(f'Summary-{epoch} | train_loss: {train_loss:.2f} | val_loss {val_loss:.2f}')

        scheduler.step()

        is_best = bool(val_loss < best_loss)
        save_checkpoint({
            'epoch': 1 + epoch + 1,
            'state_dict': model.state_dict(),
            'loss': val_loss
        }, is_best, output_path)


