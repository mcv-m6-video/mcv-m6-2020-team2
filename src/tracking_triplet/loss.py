import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from src.tracking_triplet.utils import write_triplets_tensorboard


class OnlineTripletLoss(nn.Module):
    """
    Online Triplets loss
    Takes a batch of embeddings and corresponding labels.
    Triplets are generated using triplet_selector object that take embeddings and targets and return indices of
    triplets
    """
    def __init__(self, margin):
        super().__init__()
        self.margin = margin

    def forward(self, embeddings, targets):

        triplets = self._get_triplets(embeddings, targets)

        if embeddings.is_cuda:
            embeddings = embeddings.cuda()
            triplets = triplets.cuda()

        ap_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 1]]).pow(2).sum(1) # anchor - positive
        an_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 2]]).pow(2).sum(1) # achor - negative
        losses = F.relu(ap_distances - an_distances + self.margin)

        return losses.mean(), triplets


    def _pair_wise_distance(self, vectors):
        return -2 * vectors.mm(torch.t(vectors)) + \
               vectors.pow(2).sum(dim=1).view(1, -1) + \
               vectors.pow(2).sum(dim=1).view(-1, 1)


    def _get_triplets(self, embeddings, targets):
        """
        Select hardest positive and negative from a given anchor
        """
        if embeddings.is_cuda:
            embeddings = embeddings.cpu()
            targets = targets.cpu()

        distance_matrix = self._pair_wise_distance(embeddings)
        triplets = []

        for label in set(targets):
            label_mask = (targets == label)

            positive_indices = np.where(label_mask)[0]
            negative_indices = np.where(np.logical_not(label_mask))[0]

            for anchor_positive in positive_indices:
                # Get distances to all possible positives and select the one with maximum distance
                hardest_positive = positive_indices[np.argmax([distance_matrix[anchor_positive, pair_positive] for pair_positive in positive_indices])]

                # Get distances to all possible negatives and select the one with minimum distance
                hardest_negative= negative_indices[np.argmin([distance_matrix[anchor_positive, pair_negative] for pair_negative in negative_indices])]

                triplets.append([anchor_positive, hardest_positive, hardest_negative])

        return torch.LongTensor(np.array(triplets))
