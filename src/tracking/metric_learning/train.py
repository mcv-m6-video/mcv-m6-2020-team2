import os
import argparse
import datetime

import torch
from torch.utils.data import DataLoader
from torch import optim
from torch.optim.lr_scheduler import StepLR
from torchvision.datasets import ImageFolder
from torchvision import transforms as T
from torch.utils.tensorboard import SummaryWriter

from tracking.metric_learning.sampler import PKSampler
from tracking.metric_learning.loss import TripletMarginLoss
from tracking.metric_learning.network import EmbeddingNet
from tracking.metric_learning.embeddings import plot_embeddings, plot_to_image


def train_epoch(model, optimizer, criterion, data_loader, epoch, print_freq=20):
    model.train()
    running_loss = 0
    running_frac_pos_triplets = 0
    for i, data in enumerate(data_loader):
        optimizer.zero_grad()
        samples, targets = data[0].cuda(), data[1].cuda()

        embeddings = model(samples)

        loss, frac_pos_triplets = criterion(embeddings, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_frac_pos_triplets += float(frac_pos_triplets)

        if i % print_freq == print_freq - 1:
            i += 1
            avg_loss = running_loss / print_freq
            avg_trip = 100.0 * running_frac_pos_triplets / print_freq
            print('[{:d}, {:d}] | loss: {:.4f} | % avg hard triplets: {:.2f}%'.format(epoch, i, avg_loss, avg_trip))
            running_loss = 0
            running_frac_pos_triplets = 0

    return avg_loss


def find_best_threshold(dists, targets):
    best_thresh = 0.01
    best_correct = 0
    for thresh in torch.arange(0.0, 1.51, 0.01):
        predictions = dists <= thresh.cuda()
        correct = torch.sum(predictions == targets.cuda()).item()
        if correct > best_correct:
            best_thresh = thresh
            best_correct = correct

    accuracy = 100.0 * best_correct / dists.size(0)

    return best_thresh, accuracy


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    embeds, labels = [], []

    for data in loader:
        samples, _labels = data[0].cuda(), data[1]
        out = model.get_embedding(samples)
        embeds.append(out)
        labels.append(_labels)

    embeds = torch.cat(embeds, dim=0)
    labels = torch.cat(labels, dim=0)

    dists = torch.cdist(embeds, embeds)

    labels = labels.unsqueeze(0)
    targets = labels == labels.t()

    mask = torch.ones(dists.size()).triu() - torch.eye(dists.size(0))
    dists = dists[mask == 1]
    targets = targets[mask == 1]

    threshold, accuracy = find_best_threshold(dists, targets)

    print('accuracy: {:.3f}%, threshold: {:.2f}'.format(accuracy, threshold))

    return accuracy


def save(model, epoch, save_dir, file_name):
    os.makedirs(save_dir, exist_ok=True)
    file_name = 'epoch_' + str(epoch) + '__' + file_name
    save_path = os.path.join(save_dir, file_name)
    torch.save(model, save_path)


def get_transform(train):
    transforms = []
    # transforms.append(T.Resize((128, 128)))
    if train:
        transforms.append(T.ColorJitter(0.5, 0.5, 0.5, 0)),
        transforms.append(T.RandomHorizontalFlip())
    transforms.append(T.ToTensor())
    transforms.append(T.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225]))
    return T.Compose(transforms)


def train(args):
    train_dataset = ImageFolder(root=os.path.join(args.data_path, 'train'), transform=get_transform(train=True))
    train_sampler = PKSampler(train_dataset.targets, p=16, k=16)
    train_loader = DataLoader(train_dataset, batch_size=256, sampler=train_sampler, num_workers=4)

    val_dataset = ImageFolder(root=os.path.join(args.data_path, 'val'), transform=get_transform(train=False))
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=4)

    model = EmbeddingNet(num_dims=128)
    model.cuda()

    criterion = TripletMarginLoss(margin=1.0, mining='batch_hard')
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    scheduler = StepLR(optimizer, 8, gamma=0.1)

    writer = SummaryWriter(os.path.join(args.log_path, datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))

    for epoch in range(20):
        print('Training...')
        loss = train_epoch(model, optimizer, criterion, train_loader, epoch)
        writer.add_scalar('train_loss', loss, epoch)
        scheduler.step()

        print('Evaluating...')
        acc = evaluate(model, val_loader)
        writer.add_scalar('val_acc', acc, epoch)

        if epoch % 5 == 4:
            print('Plotting embeddings...')
            figure = plot_embeddings(model, val_loader)
            writer.add_image('embeddings', plot_to_image(figure), epoch)

        print('Saving...')
        save(model, epoch, args.save_path, 'ckpt.pth')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, default='../../../data/metric_learning')
    parser.add_argument('--save-path', type=str, default='./checkpoints')
    parser.add_argument('--log-path', type=str, default='./runs')
    train(parser.parse_args())
