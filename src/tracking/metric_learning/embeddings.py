import torch
import numpy as np
import random
from sklearn.manifold import TSNE
from tqdm import tqdm
from matplotlib import pyplot as plt


def _generate_colors(number_colors):
    get_colors = lambda n: list(map(lambda i: f"#{random.randint(0, 0xFFFFFF):06x}",range(n)))
    return (get_colors(number_colors))

def plot_embeddings(dataset, embeddings, targets, filename, max_classes=10, title=''):
    print("Computing TSNE")
    embeddings = TSNE(n_components=2).fit_transform(embeddings)

    plt.figure(figsize=(10,10))
    selected_classes = np.random.choice(dataset.classes, max_classes)
    for cls in tqdm(selected_classes, desc=f'Preparing plot. Saved to {filename}'):
        i = dataset.class_to_idx[cls]
        inds = np.where(targets==i)[0]
        plt.scatter(embeddings[inds,0], embeddings[inds,1], alpha=0.5)

    plt.legend(dataset.classes)
    plt.title(title)
    plt.savefig(filename)


@torch.no_grad()
def extract_embeddings(dataloader, model, cuda):
    model.eval()
    embeddings = []
    labels = []
    for images, target in tqdm(dataloader, desc='Generating embedding'):
        images = images.cuda() if cuda else images
        out =  model.get_embedding(images).data.cpu().numpy()
        embeddings.append(out)
        labels.append(target.numpy())

    embeddings = np.vstack(embeddings)
    labels = np.concatenate(labels)
    return embeddings, labels