from torch.utils.data.sampler import BatchSampler
import numpy as np
from torchvision.datasets import ImageFolder


class ChallengeDataset(ImageFolder):

    def __init__(self, rootdir, transforms):
        super().__init__(rootdir, transforms)
        self.dataset_path = rootdir
        self.transforms = transforms
        print(f'Found {len(self)} images belonging to {len(self.classes)} classes')


class BalancedBatchSampler(BatchSampler):
    """
    BatchSampler samples n_classes and within these classes samples n_samples.
    Returns batches of size n_classes * n_samples
    """

    def __init__(self, labels, n_classes, n_samples):
        self.labels = np.array(labels)
        self.classes = list(set(self.labels))
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.n_dataset = len(self.labels)
        self.batch_size = self.n_samples * self.n_classes

        self.label_to_indices = {label: np.where(self.labels == label)[0] for label in self.classes}
        np.random.seed(42)


    def __iter__(self):
        count = 0
        while count + self.batch_size < self.n_dataset:
            indices = []
            classes = np.random.choice(self.classes, self.n_classes, replace=False)

            for class_name in classes:
                sample = np.random.choice(self.label_to_indices[class_name], self.n_samples, replace=False)
                indices.extend(sample)

            yield indices
            count += self.batch_size

    def __len__(self):
        return self.n_dataset // self.batch_size




