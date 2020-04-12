import os
import time

import torch
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from tracking.metric_learning.dataloader import BalancedBatchSampler, ChallengeDataset
from tracking.metric_learning.network import EmbeddingNet
from tracking.metric_learning.loss import OnlineTripletLoss
from tracking.metric_learning.trainer import fit
from tracking.metric_learning.embeddings import extract_embeddings, plot_embeddings
from tracking.metric_learning.plotutils import show_batch

visualize=False
images_x_class = 10
margin = 1.
n_dimensions = 128
n_epochs = 50
log_interval = 50
lr = 1e-3
arch = 'resnet'
n_workers = 4

name = f'model_batchbig_{arch}_{n_dimensions}_images{images_x_class}-{str(time.time()).split(".")[0]}'
output_path = f"../results_metric_learning/{name}"
if not os.path.exists(output_path):
    os.makedirs(output_path)

writer = SummaryWriter(f'../runs/{name}')
dataset='../data/week5_dataset_metriclearning/'

tr = transforms.Compose([transforms.Resize((80,100)),
                         transforms.ToTensor()
                         ])

train_dataset = ChallengeDataset(rootdir=os.path.join(dataset,'train'), transforms=tr)
val_dataset = ChallengeDataset(rootdir=os.path.join(dataset, 'test'), transforms=tr)

train_n_classes = 100 # len(train_dataset.classes)
val_n_classes = len(val_dataset.classes)
print(f"Training batch size: {train_n_classes*images_x_class}")
print(f"Validation batch size: {val_n_classes*images_x_class}")
train_batch_sampler = BalancedBatchSampler(train_dataset.targets, train_dataset.imgs, n_classes=train_n_classes, n_samples=images_x_class)
val_batch_sampler = BalancedBatchSampler(val_dataset.targets, val_dataset.imgs, n_classes=val_n_classes, n_samples=images_x_class)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler=train_batch_sampler, num_workers=n_workers)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_sampler=val_batch_sampler, num_workers=n_workers)


# Preparing network
model = EmbeddingNet(num_dims=n_dimensions, architecture=arch)
cuda_flag = torch.cuda.is_available()
if cuda_flag:
    model.cuda()

loss_fn = OnlineTripletLoss(margin)
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)

if visualize:
    # show_batch(train_loader, n_view=images_x_class, n_cars=10)
    train_embeddings, train_labels = extract_embeddings(train_loader, model, cuda_flag)
    plot_embeddings(train_loader.dataset, train_embeddings, train_labels, filename=f"train_{name}_before.png", title='Train before')


fit(model, n_epochs, train_loader, val_loader, scheduler, optimizer, loss_fn, log_interval,
    cuda_flag, writer, output_path)

torch.save(model, output_path+"/model.pth")

train_embeddings, train_labels = extract_embeddings(train_loader, model, n_dimensions)
plot_embeddings(train_loader.dataset, train_embeddings, train_labels, filename=f"train_{name}_final.png", title='Train after')

val_embeddings, val_labels = extract_embeddings(val_loader, model, n_dimensions)
plot_embeddings(val_loader.dataset, val_embeddings, val_labels, filename=f"val_{name}_final.png", title='Validation after')
