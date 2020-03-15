from collections import defaultdict, OrderedDict

import torch
import torch.utils.data
import torchvision

import numpy as np
import cv2

from src.detection.engine import train_one_epoch, evaluate
import src.detection.utils as utils


def parse_annotations(ann_file):
    import xml.etree.ElementTree as ET
    tree = ET.parse(ann_file)
    root = tree.getroot()

    annotations = defaultdict(list)
    for track in root.findall('track'):
        if track.attrib['label'] == 'car':
            for box in track.findall('box'):
                frame = int(box.attrib['frame'])
                xtl = float(box.attrib['xtl'])
                ytl = float(box.attrib['ytl'])
                xbr = float(box.attrib['xbr'])
                ybr = float(box.attrib['ybr'])
                annotations[frame].append([xtl, ytl, xbr, ybr])

    return OrderedDict(annotations)


class AICityDataset(torch.utils.data.Dataset):
    def __init__(self, video_file, ann_file, transform=None):
        self.video_file = video_file
        self.ann_file = ann_file
        self.transform = transform

        self.boxes = parse_annotations(self.ann_file)
        self.video_cap = cv2.VideoCapture(self.video_file)

    def __getitem__(self, idx):
        self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, img = self.video_cap.read()

        if self.transform is not None:
            img = self.transform(img)

        boxes = torch.as_tensor(self.boxes.get(idx, []), dtype=torch.float32)
        labels = torch.ones((len(boxes),), dtype=torch.int64)
        image_id = torch.tensor([idx])

        target = {'boxes': boxes, 'labels': labels, 'image_id': image_id}

        return img, target

    def __len__(self):
        return int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT))


def get_model(num_classes, architecture='maskrcnn'):
    # load an instance segmentation model pre-trained pre-trained on COCO
    if architecture == 'fasterrcnn':
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    elif architecture == 'maskrcnn':
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    else:
        raise ValueError('Unknown detection architecture.')

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

    # remove mask heads
    model.roi_heads.mask_roi_pool = None
    model.roi_heads.mask_head = None
    model.roi_heads.mask_predictor = None

    return model


def main():
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # our dataset has two classes only - background and car
    num_classes = 2
    # use our dataset and defined transformations
    transform = torchvision.transforms.ToTensor()
    dataset = AICityDataset(video_file='../../data/AICity_data/train/S03/c010/vdo.avi',
                            ann_file='../../data/ai_challenge_s03_c010-full_annotation.xml',
                            transform=transform)

    # split the dataset in train and test set
    indices = np.random.permutation(len(dataset))
    split = int(len(dataset)*0.75)
    train_sampler = torch.utils.data.SubsetRandomSampler(indices[:split])
    test_sampler = torch.utils.data.SubsetRandomSampler(indices[split:])

    # define training and validation data loaders
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=8, sampler=train_sampler, num_workers=1, collate_fn=utils.collate_fn)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=1, sampler=test_sampler, num_workers=1, collate_fn=utils.collate_fn)

    # get the model using our helper function
    model = get_model(num_classes)

    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # let's train it for some epochs
    num_epochs = 1

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, test_loader, device=device)


if __name__ == '__main__':
    main()
