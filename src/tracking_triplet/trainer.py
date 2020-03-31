import torch
from tqdm import tqdm
import numpy as np


def train_epoch(model, train_loader, optimizer, criterion, log_interval, cuda):
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

        i += 1
    return np.mean(losses)

@torch.no_grad()
def val_epoch(model, val_loader, criterion, cuda):
    model.eval()
    losses = []
    for data, target in tqdm(val_loader, desc="Validation epoch"):
        if cuda:
            data = data.cuda()
            target = target.cuda()

        outputs = model(data)
        loss, number_triplets = criterion(outputs, target)
        losses.append(loss.item())

    return np.mean(losses)


def fit(model, epochs, train_loader, val_loader, scheduler, optimizer, criterion, log_interval, cuda):

    for epoch in range(1, epochs+1):
        print(f"\nEpoch {epoch}")
        train_loss = train_epoch(model, train_loader, optimizer, criterion, log_interval, cuda)
        val_loss = val_epoch(model, val_loader, criterion, cuda)

        print(f'Summary-{epoch} | train_loss: {train_loss:.2f} | val_loss {val_loss:.2f}')

        scheduler.step()