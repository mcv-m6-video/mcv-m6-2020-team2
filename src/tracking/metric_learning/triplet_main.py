import torch
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tracking.metric_learning.dataloader import BalancedBatchSampler, ChallengeDataset
from tracking.metric_learning import EmbeddingNet
from tracking.metric_learning import OnlineTripletLoss
from tracking.metric_learning import fit
import os
from tracking.metric_learning import extract_embeddings, plot_embeddings
from tracking.metric_learning import show_batch
import time

visualize=False
images_x_class = 4
margin = 1.
n_dimensions = 16
n_epochs = 20
log_interval = 10
lr = 5e-4
arch = 'resnet'
n_workers = 4

name = f'model_{arch}_{n_dimensions}_images{images_x_class}-{str(time.time()).split(".")[0]}'
output_path = f"results/{name}"
if not os.path.exists(output_path):
    os.makedirs(output_path)

writer = SummaryWriter(f'runs/{name}')
dataset='../../data/week5_dataset_metriclearning/'

tr = transforms.Compose([transforms.Resize((80,100)),
                         transforms.ColorJitter(),
                         transforms.RandomHorizontalFlip(),
                         transforms.RandomPerspective(),
                         transforms.RandomRotation(15),
                         transforms.ToTensor(),
                         transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
                         ])

train_dataset = ChallengeDataset(rootdir=os.path.join(dataset,'train'), transforms=tr)
val_dataset = ChallengeDataset(rootdir=os.path.join(dataset, 'test'), transforms=tr)

# # Something has to tell me the number of classes
train_n_classes = len(train_dataset.classes)
val_n_classes = len(val_dataset.classes)
train_batch_sampler = BalancedBatchSampler(train_dataset.targets, n_classes=train_n_classes, n_samples=images_x_class)
val_batch_sampler = BalancedBatchSampler(val_dataset.targets, n_classes=val_n_classes, n_samples=images_x_class)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler=train_batch_sampler, num_workers=n_workers)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_sampler=val_batch_sampler, num_workers=n_workers)


# # Preparing network
model = EmbeddingNet(num_dims=n_dimensions, architecture=arch)
if torch.cuda.is_available():
    model.cuda()

loss_fn = OnlineTripletLoss(margin)
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)

if visualize:
    show_batch(train_loader, n_view=images_x_class, n_cars=10)
    train_embeddings, train_labels = extract_embeddings(train_loader, model, torch.cuda.is_available())
    plot_embeddings(train_loader.dataset, train_embeddings, train_labels, filename=f"train_{name}_before.png", title='Train before')


fit(model, n_epochs, train_loader, val_loader, scheduler, optimizer, loss_fn, log_interval,
    torch.cuda.is_available(), writer, output_path)

torch.save(model, output_path+"/model.pth")

train_embeddings, train_labels = extract_embeddings(train_loader, model, n_dimensions)
plot_embeddings(train_loader.dataset, train_embeddings, train_labels, filename=f"train_{name}_final.png", title='Train after')

val_embeddings, val_labels = extract_embeddings(val_loader, model, n_dimensions)
plot_embeddings(val_loader.dataset, val_embeddings, val_labels, filename=f"val_{name}_final.png", title='Validation after')
