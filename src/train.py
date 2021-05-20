import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .data import get_data
from .model import Net


def train_mnist(args):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    train_loader, val_loader = get_data()

    model = Net().to(device)
    print(model)
    opt = optim.Adam(model.parameters(), lr=3e-4)

    cross_el_loss = nn.CrossEntropyLoss()
    l1_loss = nn.L1Loss()

    for epoch in range(args.epochs):
        # Train
        model.train()
        for batch_idx, data in enumerate(train_loader):
            img = data["img"].to(device)
            rand_num = data["rand_num"].to(device)
            cls_lbl = data["lbl"].to(device)
            sum_lbl = data["sum_lbl"].unsqueeze(dim=1).to(device)

            opt.zero_grad()
            cls_pred, sum_pred = model(img, rand_num)

            loss_cls = cross_el_loss(cls_pred, cls_lbl)
            loss_sum = l1_loss(sum_pred, sum_lbl)

            loss = loss_cls + loss_sum
            
            loss.backward()
            opt.step()

        print(f"Epoch: {epoch}, Train Cls loss: {loss_cls.item()}, Train Sum loss: {loss_sum.item()}, Train Total loss: {loss.item()}")

        # Eval
        model.eval()
        total_cls = 0
        correct_cls = 0
        total_sum = 0
        correct_sum = 0
        with torch.no_grad():
            for batch_idx, data in enumerate(val_loader):
                img = data["img"].to(device)
                rand_num = data["rand_num"].to(device)
                cls_lbl = data["lbl"].to(device)
                sum_lbl = data["sum_lbl"].unsqueeze(dim=1).to(device)

                cls_pred, sum_pred = model(img, rand_num)

                # Can be made better
                for idx, i in enumerate(cls_pred):
                    if torch.argmax(i) == cls_lbl[idx]:
                        correct_cls += 1
                    total_cls += 1
                
                # Can be made better
                for idx, i in enumerate(sum_pred):
                    p = sum_pred[idx][0].int()
                    l = sum_lbl[idx][0].int()
                    if (p == l):
                        correct_sum += 1
                    total_sum += 1
        
        print(f"Epoch: {epoch}, Test Cls accuracy: {correct_cls / total_cls}, Test Sum accuracy: {correct_sum / total_sum}")
        


