import os

from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from  matplotlib.gridspec import GridSpec
import numpy as np
import torch

def show_batch(dataloader, n_view=10, n_cars=10):
    fig = plt.figure(figsize=(20, 20))
    grid = ImageGrid(fig, 111, nrows_ncols=(n_cars, n_view), axes_pad=0.1)

    n_images = n_cars*n_view
    for sample_batched in dataloader:
        images, labels = sample_batched
        for count, (ax, im) in enumerate(zip(grid, images)):
            im = np.transpose(im.numpy(), (1, 2, 0))
            ax.imshow(im)
            ax.axis('off')
            if count > n_images:
                break
        plt.show()
        plt.axis('off')
        break


def show_triplets(triplets, data, cuda, output_path, epoch, n_cars=15):
    fig = plt.figure(figsize=(20, 20))
    grid = ImageGrid(fig, 111, nrows_ncols=(n_cars, 3), axes_pad=0.1)

    print(len(triplets))
    triplets = triplets[:n_cars]
    flat_list = [item for triplet in triplets for item in triplet]

    for i, triplet_id in enumerate(flat_list):
        ax = grid[i]
        im = data[triplet_id].cpu().numpy() if cuda else data[triplet_id].numpy()
        ax.imshow(np.transpose(im, (1, 2, 0)))
        ax.title.set_text(f"{triplet_id}")
        ax.axis('off')

    plt.savefig(f"{output_path}/triplets_{epoch}.png")
    # plt.show()
    plt.axis('off')


def matplotlib_imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))


def write_triplets_tensorboard(triplets, data, epoch):
    print(len(triplets))
    triplets = triplets[:15]

    gs = GridSpec(len(triplets), 3)
    gs.update(wspace=0.1, hspace=0.1, left=0.1, right=0.4, bottom=0.1, top=0.9)
    flat_list = [item for triplet in triplets for item in triplet]

    for i in range(len(triplets)*3):
        plt.subplot(gs[i])
        if torch.cuda.is_available():
            image = np.transpose(data[flat_list[i]].cpu().numpy(), (1, 2, 0))
        else:
            image = np.transpose(data[flat_list[i]].numpy(), (1, 2, 0))

        plt.imshow(image)
        plt.axis('off')

    if not os.path.exists("../triplets/"):
        os.makedirs("../triplets/")
    plt.show()
    # plt.savefig(f"../triplets/triplets_{epoch}.png")
    plt.axis('off')