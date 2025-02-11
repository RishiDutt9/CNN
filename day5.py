import torch
from torchvision import datasets,transforms
from torch.utils.data import DataLoader


#Define transformation

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))

])
#Load Cifer 10 Dataset

train_dataset = datasets.CIFAR10(root = "./data",train=True, download = True,trainsform = transforms)
test_dataset = datasets.CIFAR10(root = "./data",train=False, download = True,trainsform = transforms)

#Create the DataLoader

train_loader = DataLoader(train_dataset,batch_size = 64,shuffle = True)
train_loader = DataLoader(train_dataset,batch_size = 64,shuffle = False)

print(f"Training Datasize {len(train_dataset)}")
print(f"Training Datasize {len(test_dataset)}")