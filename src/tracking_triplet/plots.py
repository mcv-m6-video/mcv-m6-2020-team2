from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np

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

