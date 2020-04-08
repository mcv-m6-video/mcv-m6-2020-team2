from torch import nn
from torchvision import models


class EmbeddingNet(nn.Module):

    def __init__(self, num_dims, architecture="resnet"):
        super().__init__()

        if architecture == 'resnet':
            self.model = models.resnet18(pretrained=True)
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, num_dims)

        elif architecture == 'mobile':
            self.model = models.mobilenet_v2(pretrained=True)

            new_classifier = nn.Sequential(*list(self.model.classifier.children())[:-1],
                                           # nn.AdaptiveAvgPool2d((1, 1)),
                                           nn.Linear(self.model.last_channel, num_dims))
            self.model.classifier = new_classifier

        elif architecture == "vgg":
            self.model = models.vgg16(pretrained=True)
            new_classifier = nn.Sequential(*list(self.model.classifier.children())[:-1], nn.Linear(4096, num_dims))
            self.model.classifier = new_classifier


    def forward(self, x):
        return self.model(x)

    def get_embedding(self, x):
        return self.forward(x)