import torch
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from src.tracking_triplet.dataloader import BalancedBatchSampler, ChallengeDataset
from src.tracking_triplet.network import EmbeddingNet
from src.tracking_triplet.loss import OnlineTripletLoss
from src.tracking_triplet.trainer import fit
import os
from src.tracking_triplet.embeddings import extract_embeddings, plot_embeddings
from src.tracking_triplet.utils import show_batch

visualize=False
images_x_class = 5


dataset='data/week5_dataset_metriclearning/'
tr = transforms.Compose([transforms.Resize((80,100)), transforms.ToTensor()])
train_dataset = ChallengeDataset(rootdir=os.path.join(dataset,'train'), transforms=tr)
val_dataset = ChallengeDataset(rootdir=os.path.join(dataset, 'test'), transforms=tr)

# # Something has to tell me the number of classes
train_n_classes = len(train_dataset.classes)
val_n_classes = len(val_dataset.classes)
train_batch_sampler = BalancedBatchSampler(train_dataset.targets, n_classes=train_n_classes, n_samples=images_x_class)
val_batch_sampler = BalancedBatchSampler(val_dataset.targets, n_classes=val_n_classes, n_samples=images_x_class)

kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
train_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler=train_batch_sampler, **kwargs)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_sampler=val_batch_sampler, **kwargs)

if visualize:
    show_batch(train_loader, n_view=images_x_class, n_cars=10)

# Preparing network
margin = 1.
n_dimensions = 128
model = EmbeddingNet(num_dims=n_dimensions) # Feature Vector dimension

if torch.cuda.is_available():
    model.cuda()

loss_fn = OnlineTripletLoss(margin)

lr = 1e-3
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
n_epochs = 20
log_interval = 10

name = f'model_{n_dimensions}_images-{images_x_class}'
writer = SummaryWriter(f'runs/{name}')
output_path = f"results/{name}"
if not os.path.exists(output_path):
    os.makedirs(output_path)


train_embeddings_otl, train_labels_otl = extract_embeddings(train_loader, model, n_dimensions)
plot_embeddings(train_embeddings_otl, train_labels_otl, train_n_classes, filename=f"train_{name}_before.png")


fit(model, n_epochs, train_loader, val_loader, scheduler, optimizer, loss_fn, log_interval,
    torch.cuda.is_available(), writer, output_path)


train_embeddings_otl, train_labels_otl = extract_embeddings(train_loader, model, n_dimensions)
plot_embeddings(train_embeddings_otl, train_labels_otl, train_n_classes, filename=f"train_{name}_final.png")
torch.save(model, output_path+"/model.pth")

# val_embeddings_otl, val_labels_otl = extract_embeddings(val_loader, model, n_dimensions)
# plot_embeddings(val_embeddings_otl, val_labels_otl, val_n_classes, filename=f"val_{name}_final.png")
