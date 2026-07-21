import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets,transforms,models
from torch.utils.data import DataLoader,Dataset
import random

simclr_transform=transforms.Compose([
    transforms.RandomResizedCrop(32),
    transforms.RandomHorizontalFlip(),
    transforms.RandomApply([transforms.ColorJitter(0.4,0.4,0.4,0.1)],p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

class SimCLRDataset(Dataset):
   def __init__(self,dataset1):
        self.dataset=dataset1
   def __getitem__(self, index):
        img,_=self.dataset[index]
        return simclr_transform(img),simclr_transform(img)

   def __len__(self):
        return len(self.dataset)

base_dataset=datasets.CIFAR10(root="./data",train=True,download=True)
train_data=SimCLRDataset(base_dataset)
train_loader=DataLoader(train_data,batch_size=128,shuffle=True)

class SimClRModel(nn.Module):
    def __init__(self):
        super().__init__()
        base=models.resnet18(pretrained=False)
        self.encoder=nn.Sequential(*list(base.children())[:-1])
        self.projector=nn.Sequential(
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Linear(256,128)
        )
    def forward(self,x):
        x=self.encoder(x)
        x=torch.flatten(x,1)
        x=self.projector(x)
        return F.normalize(x,dim=1)

def nt_xent_loss(z1,z2,temperature=0.5):
    N=z1.size(0)
    z=torch.cat([z1,z2],dim=0)
    sim=F.cosine_similarity(z.unsqueeze(1),z.unsqueeze(0),dim=2)
    sim=sim/temperature
    labels=torch.arange(N).repeat(2)
    labels=(labels.unsqueeze(0)==labels.unsqueeze(1)).float()
    labels=labels.to(z.device)

    mask=torch.eye(2*N,dtype=torch.bool).to(z.device)
    sim.masked_fill_(mask,-9e15)
    sim_exp=torch.exp(sim)
    sim_sum=sim_exp.sum(dim=1,keepdim=True)

    log_prob=sim-torch.log(sim_sum)
    loss=-(labels*log_prob).sum(dim=1)/labels.sum(dim=1)
    return loss.mean()

model=SimClRModel()
optimizer=torch.optim.Adam(model.parameters(),lr=0.01)

for epoch in range(5):
    for x1,x2 in train_loader:
        z1=model(x1)
        z2=model(x2)
        loss=nt_xent_loss(z1,z2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"loss is {loss.item()}")




