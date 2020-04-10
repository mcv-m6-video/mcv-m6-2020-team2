import os
import numpy as np
import copy

from torchvision.datasets import ImageFolder
from torch.utils.data.sampler import BatchSampler


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

    def __init__(self, labels, dirs, n_classes, n_samples):
        self.labels = np.array(labels)
        self.classes = list(set(self.labels))
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.n_dataset = len(self.labels)
        self.batch_size = self.n_samples * self.n_classes

        self.cameras = np.asarray([int(os.path.split(dir[0])[-1].split("_")[0].split("c")[-1]) for dir in dirs])
        self.n_cameras = len(set(self.cameras))
        self.label_to_indices = {label: np.where(self.labels == label)[0] for label in self.classes}
        self.cams_to_indices = {camera: np.where(self.cameras == camera)[0] for camera in set(self.cameras)}
        self.cams_in_label = {label: set([self.cameras[i] for i in range(len(self.cameras)) if self.labels[i] == label]) for label in self.classes}
        np.random.seed(42)


    def __iter__(self):
        count = 0
        while count + self.batch_size < self.n_dataset:
            indices = []
            classes = np.random.choice(self.classes, self.n_classes, replace=False)
            for class_name in classes:
                cameras = np.random.choice(list(set(self.cams_in_label[class_name])), min(len(self.cams_in_label[class_name]), self.n_samples),replace=False)
                imgs_per_cam = max(1, int(np.floor(self.n_samples / len(cameras))))
                for cam_name in cameras:
                    available_indices = copy.deepcopy(list(set(self.label_to_indices[class_name]).intersection(set(self.cams_to_indices[cam_name]))))
                    sample = np.random.choice(available_indices, min(imgs_per_cam, len(available_indices)), replace=False)
                    indices.extend(sample)

            yield indices
            count += self.batch_size

    def __len__(self):
        return self.n_dataset // self.batch_size




