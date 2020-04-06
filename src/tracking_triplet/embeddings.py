import torch
from matplotlib import pyplot as plt
import numpy as np
import random
from sklearn.manifold import TSNE

def _generate_colors(number_colors):
    get_colors = lambda n: list(map(lambda i: f"#{random.randint(0, 0xFFFFFF):06x}",range(n)))
    return (get_colors(number_colors))

def plot_embeddings(embeddings, targets, n_classes, filename, max_classes=20, xlim=None, ylim=None):
    colors = _generate_colors(n_classes)
    classes = list(np.arange(n_classes))
    embeddings = TSNE(n_components=2).fit_transform(embeddings)
    plt.figure(figsize=(10,10))
    for i in range(min(max_classes, n_classes)):
        inds = np.where(targets==i)[0]
        plt.scatter(embeddings[inds,0], embeddings[inds,1], alpha=0.5, color=colors[i])
    if xlim:
        plt.xlim(xlim[0], xlim[1])
    if ylim:
        plt.ylim(ylim[0], ylim[1])
    plt.legend(classes)
    plt.savefig(filename)


def extract_embeddings(dataloader, model, n_dims):
    with torch.no_grad():
        model.eval()
        embeddings = np.zeros((len(dataloader.dataset), n_dims))
        labels = np.zeros(len(dataloader.dataset))
        k = 0
        for images, target in dataloader:
            if torch.cuda.is_available():
                images = images.cuda()
            embeddings[k:k+len(images)] = model.get_embedding(images).data.cpu().numpy()
            labels[k:k+len(images)] = target.numpy()
            k += len(images)
    return embeddings, labels