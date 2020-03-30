import torch
from torch import nn
from torchvision import models
import torchvision.transforms.functional as F
import cv2


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(*list(models.resnet18(pretrained=True).children())[:-1])

    def forward(self, x):
        return self.model(x)

    def get_embedding(self, img):
        with torch.no_grad():
            img = F.to_tensor(img).unsqueeze(0).cuda()
            return self.forward(img).squeeze().cpu().numpy()

    def get_embeddings(self, batch):
        with torch.no_grad():
            batch = torch.stack([F.to_tensor(img) for img in batch]).cuda()
            return self.forward(batch).squeeze().cpu().numpy()


if __name__ == '__main__':
    cap = cv2.VideoCapture('../../../data/AIC20_track3/train/S03/c010/vdo.avi')
    ret, img = cap.read()
    img = cv2.resize(img, (128, 128))

    encoder = Encoder()
    encoder.eval()
    embedding = encoder.get_embedding(img)
    print(embedding)
