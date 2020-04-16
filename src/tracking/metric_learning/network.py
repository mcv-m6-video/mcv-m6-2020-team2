from torch import nn
from torchvision import models


class EmbeddingNet(nn.Module):

    def __init__(self, num_dims=128):
        super().__init__()

        base_encoder = models.resnet18(pretrained=True)
        self.backbone = nn.Sequential(*list(base_encoder.children())[:-1], nn.Flatten())
        self.projection = nn.Sequential(nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, num_dims))

    def forward(self, x):
        x = self.backbone(x)
        return self.projection(x)

    def get_embedding(self, x):
        return self.backbone(x)
