import io

import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from torchvision.transforms.functional import to_tensor


@torch.no_grad()
def extract_embeddings(model, loader):
    model.eval()
    embeds, labels = [], []

    for images, _labels in loader:
        if torch.cuda.is_available():
            images = images.cuda()
        out = model.get_embedding(images).cpu().numpy()
        embeds.append(out)
        labels.append(_labels.numpy())

    embeds = np.vstack(embeds)
    labels = np.concatenate(labels)

    return embeds, labels


def plot_to_image(figure):
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(figure)
    buf.seek(0)
    image = Image.open(buf)
    image = to_tensor(image)
    return image


def plot_embeddings(model, loader, max_classes=None):
    embeds, labels = extract_embeddings(model, loader)

    embeds = TSNE(n_components=2, verbose=1).fit_transform(embeds)

    figure = plt.figure(figsize=(10, 10))
    if max_classes is not None:
        selected_classes = np.random.choice(loader.dataset.classes, max_classes)
    else:
        selected_classes = loader.dataset.classes
    for cls in selected_classes:
        idx = loader.dataset.class_to_idx[cls]
        inds = labels == idx
        plt.scatter(embeds[inds, 0], embeds[inds, 1], alpha=0.5)

    plt.legend(loader.dataset.classes)

    return figure


if __name__ == '__main__':
    from torchvision.datasets import ImageFolder
    from torch.utils.data import DataLoader
    from tracking.metric_learning.train import get_transform

    for setx in ["train", "val"]:
        dataset = ImageFolder(root='../../../data/week5_dataset_metriclearning/train', transform=get_transform(train=False))
        dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)

        model = torch.load('../metric_learning/models/model.pth', map_location=torch.device('cpu'))

        plot_embeddings(model, dataloader, max_classes=10)
        plt.savefig(f"embedding_{setx}.png")
