import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

from typing import Dict, Tuple


class MNISTSumDataset(Dataset):
    def __init__(self, train: bool) -> None:
        mnist_transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.mnist_ds = datasets.MNIST(
            './dataset', train=train, download=True,
            transform=mnist_transform
        )
    
    def __len__(self):
        return len(self.mnist_ds)
    
    def __getitem__(self, idx) -> Dict:
        random_num = ((torch.rand(1)[0]*9).int()).float()  # random no. b/w 0-9
        img, lbl = self.mnist_ds[idx]
        sum_lbl = random_num + lbl
        return {"img": img, "rand_num": random_num, "lbl": lbl, "sum_lbl": sum_lbl}


def get_data() -> Tuple[DataLoader, DataLoader]:
    
    train_ds = MNISTSumDataset(train=True)
    val_ds = MNISTSumDataset(train=False)

    train_kwargs = {'batch_size': 64}
    test_kwargs = {'batch_size': 1000}

    if torch.cuda.is_available():
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)


    train_loader = DataLoader(train_ds,**train_kwargs)
    val_loader = DataLoader(val_ds,**test_kwargs)

    return train_loader, val_loader