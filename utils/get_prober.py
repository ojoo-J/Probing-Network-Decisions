import torch
import torch.nn as nn


class Prober(nn.Module):
    def __init__(self):
        super(Prober, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(4096, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        x = x.float()
        output = self.layers(x.view(x.size(0), -1))
        return output


class Prober2(nn.Module):
    def __init__(self):
        super(Prober2, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(4096, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        x = x.float()
        output = self.layers(x.view(x.size(0), -1))
        return output


class ProberMNIST(nn.Module):
    def __init__(self, latent_dim=[256, 128, 64]):
        super(ProberMNIST, self).__init__()

        ls = []
        for i in range(len(latent_dim)-1):
            ls.append(nn.Linear(latent_dim[i], latent_dim[i+1]))
            ls.append(nn.ReLU())
        ls.append(nn.Linear(latent_dim[-1], 2))
        self.layers = nn.Sequential(*ls)

    def forward(self, x):
        x = x.float()
        output = self.layers(x.view(x.size(0), -1))
        return output
    
class ProberFashionMNIST(nn.Module):
    def __init__(self, latent_dim=[256, 128, 64]):
        super(ProberFashionMNIST, self).__init__()

        ls = []
        for i in range(len(latent_dim)-1):
            ls.append(nn.Linear(latent_dim[i], latent_dim[i+1]))
            ls.append(nn.ReLU())
        ls.append(nn.Linear(latent_dim[-1], 2))
        self.layers = nn.Sequential(*ls)

    def forward(self, x):
        x = x.float()
        output = self.layers(x.view(x.size(0), -1))
        return output


class ProberCIFAR10(nn.Module):
    def __init__(self, num_hidden=8192):
        super(ProberCIFAR10, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(num_hidden, 2048),
            nn.ReLU(),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 2),
        )

    def forward(self, x):
        x = x.float()
        output = self.layers(x.view(x.size(0), -1))
        return output


class ProberImageNette(nn.Module):
    def __init__(self, num_hidden=2048, out_hidden=2048):
        super(ProberImageNette, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(num_hidden, out_hidden),
            nn.ReLU(),
            nn.Linear(out_hidden, 512),
            nn.ReLU(),
            nn.Linear(512, 2),
        )

    def forward(self, x):
        x = x.float()
        output = self.layers(x.view(x.size(0), -1))
        return output


class ProberCIFAR100(nn.Module):
    def __init__(self, num_hidden=512):
        super(ProberCIFAR100, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(num_hidden, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        x = x.float()
        output = self.layers(x.view(x.size(0), -1))
        return output


@torch.no_grad()
def evaluate_img2correct(model, dataloader, device):
    model.eval()

    train_x = []
    train_y = []
    pred_y = []
    for x, y in dataloader:
        train_x.append(x)
        train_y.append(y)

        pred = model(x.to(device))
        _, pred_class = pred.max(dim=1)

        pred_y.append(pred_class.detach().cpu())

    train_x = torch.cat(train_x)
    train_y = torch.cat(train_y)
    pred_y = torch.cat(pred_y)

    acc = (train_y == pred_y).sum() / len(pred_y)
    print(f"Acc : {acc:.4f}")
    return train_x, train_y, pred_y, acc
