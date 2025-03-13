import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
import timm
from torch.optim import Adam
from tqdm import tqdm

# Core model architectures
class CNN(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=False),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.05),
        )
        
        self.fc_layer = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(3136, 512),  
            nn.ReLU(inplace=False),
            nn.Linear(512, 256),
            nn.ReLU(inplace=False),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes),
        )
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.conv_layer(x)
        x = x.view(x.size(0), -1)
        return self.fc_layer(x)

    def load_pretrained(self, path):
        if path is None:
            return
        checkpoint = torch.load(path)
        state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
        self.load_state_dict(state_dict)

    def train_step(self, batch, device):
        x, target = batch[0], batch[1]
        x, target = x.to(device), target.to(device)
        
        output = self(x)
        loss = self.criterion(output, target)
        
        pred = output.argmax(dim=1)
        acc = (pred == target).float().mean() * 100
        
        return loss, acc

    def evaluate(self, data_loader, device):
        self.eval()
        total_loss = 0
        total_acc = 0
        
        with torch.no_grad():
            for batch in data_loader:
                loss, acc = self.train_step(batch, device)
                total_loss += loss.item()
                total_acc += acc.item()
        
        return {
            "loss": total_loss / len(data_loader),
            "accuracy": total_acc / len(data_loader)
        }

    def train_model(self, train_loader, val_loader, epochs=10, device='cuda', lr=1e-3):
        self.train()
        optimizer = Adam(self.parameters(), lr=lr)
        
        for epoch in range(epochs):
            # Training
            train_loss = 0
            train_acc = 0
            for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}'):
                optimizer.zero_grad()
                loss, acc = self.train_step(batch, device)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                train_acc += acc.item()
            
            # Print metrics
            train_loss /= len(train_loader)
            train_acc /= len(train_loader)
            val_metrics = self.evaluate(val_loader, device)
            
            print(f'Epoch {epoch+1}/{epochs}:')
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'Val Loss: {val_metrics["loss"]:.4f}, Val Acc: {val_metrics["accuracy"]:.2f}%')

class ResNet(nn.Module):
    def __init__(self, arch='resnet18', num_classes=10):
        super().__init__()
        self.model = timm.create_model(arch, num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

    def load_pretrained(self, path):
        if path is None:
            return
        checkpoint = torch.load(path)
        state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
        self.load_state_dict(state_dict)

class Prober(nn.Module):
    def __init__(self, input_dim=256, hidden_dims=[128, 64]):
        super().__init__()
        layers = []
        dims = [input_dim] + hidden_dims + [2]
        for i in range(len(dims)-1):
            layers.extend([
                nn.Linear(dims[i], dims[i+1]),
                nn.ReLU() if i < len(dims)-2 else nn.Identity()
            ])
        self.layers = nn.Sequential(*layers)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.layers(x.float())

    def load_pretrained(self, path):
        if path is None:
            return
        checkpoint = torch.load(path)
        state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
        self.load_state_dict(state_dict)

    def train_step(self, batch, device):
        _, hidden, target = batch[:3]
        hidden, target = hidden.to(device), target.to(device)
        
        output = self(hidden)
        loss = self.criterion(output, target)
        
        pred = output.argmax(dim=1)
        acc = (pred == target).float().mean() * 100
        
        return loss, acc

    def evaluate(self, data_loader, device):
        self.eval()
        total_loss = 0
        total_acc = 0
        
        with torch.no_grad():
            for batch in data_loader:
                loss, acc = self.train_step(batch, device)
                total_loss += loss.item()
                total_acc += acc.item()
        
        return {
            "loss": total_loss / len(data_loader),
            "accuracy": total_acc / len(data_loader)
        }

    def train_model(self, train_loader, val_loader, epochs=10, device='cuda', lr=1e-3):
        self.train()
        optimizer = Adam(self.parameters(), lr=lr)
        
        for epoch in range(epochs):
            # Training
            train_loss = 0
            train_acc = 0
            for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}'):
                optimizer.zero_grad()
                loss, acc = self.train_step(batch, device)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                train_acc += acc.item()
            
            # Print metrics
            train_loss /= len(train_loader)
            train_acc /= len(train_loader)
            val_metrics = self.evaluate(val_loader, device)
            
            print(f'Epoch {epoch+1}/{epochs}:')
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'Val Loss: {val_metrics["loss"]:.4f}, Val Acc: {val_metrics["accuracy"]:.2f}%')

class CombinedNN(nn.Module):
    def __init__(self, classifier, prober, hidden_mean, hidden_std):
        super().__init__()
        self.classifier = classifier
        self.prober = prober
        self.hidden_mean = hidden_mean
        self.hidden_std = hidden_std
        
        # Freeze both models
        for model in [self.classifier, self.prober]:
            for param in model.parameters():
                param.requires_grad = False

    def forward(self, x):
        hidden = self.classifier(x)
        hidden = (hidden - self.hidden_mean) / self.hidden_std
        return self.prober(hidden)

    @torch.no_grad()
    def evaluate_all(self, data_loader, device):
        self.eval()
        vals = defaultdict(list)
        for img, label in data_loader:
            img, label = img.to(device), label.to(device)
            
            cls_out = self.classifier(img)
            cls_prob = F.softmax(cls_out, dim=1)
            cls_pred = cls_prob.argmax(dim=1)
            
            prb_out = self(img)
            prb_prob = F.softmax(prb_out, dim=1)
            prb_pred = prb_prob.argmax(dim=1)
            
            vals["image"].append(img)
            vals["label"].append(label)
            vals["clf_out"].append(cls_out)
            vals["clf_prob"].append(cls_prob)
            vals["clf_pred"].append(cls_pred)
            vals["prb_out"].append(prb_out)
            vals["prb_prob"].append(prb_prob)
            vals["prb_pred"].append(prb_pred)
            
        return {k: torch.cat(v).cpu() for k, v in vals.items()}

class MNISTFlow(nn.Module):
    """Flow-based generator for MNIST"""
    def __init__(self, device="cuda"):
        super().__init__()
        self.device = device
        self.data_info = {
            "data_set": "MNIST",
            "data_shape": [1, 28, 28],
            "n_bits": 8,
            "temp": 1,
            "num_classes": 10,
            "class_names": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
            "data_mean": [0.1307],
            "data_std": [0.3081],
        }
        
        # Import here to avoid circular imports
        from counterfactuals.generative_models.factory import get_generative_model
        self.model, self.model_type = get_generative_model(
            generative_model_type="Flow", 
            data_info=self.data_info, 
            device=device
        )
        
    def encode(self, x):
        return self.model.encode(x)
        
    def decode(self, z):
        return self.model.decode(z)
        
    def forward(self, x):
        return self.model(x)

    def load_pretrained(self, path):
        if path is None:
            return
        checkpoint = torch.load(path)
        state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
        self.load_state_dict(state_dict)

# Factory functions with dataset-specific configurations
def get_classifier(dataset, ckpt_path=None):
    CONFIGS = {
        'mnist': {'model': CNN, 'kwargs': {'in_channels': 1, 'num_classes': 10}},
        'cifar10': {'model': ResNet, 'kwargs': {'num_classes': 10}},
        'imagenet': {'model': ResNet, 'kwargs': {'num_classes': 1000}},
    }
    config = CONFIGS[dataset.lower()]
    model = config['model'](**config['kwargs'])
    if ckpt_path:
        model.load_pretrained(ckpt_path)
    return model

def get_prober(dataset, ckpt_path=None, **kwargs):
    model = Prober(**kwargs)
    if ckpt_path:
        model.load_pretrained(ckpt_path)
    return model

def get_generator(dataset, ckpt_path=None, device='cuda'):
    """Get generator model"""
    if dataset.lower() == 'mnist':
        model = MNISTFlow(device=device)
        if ckpt_path:
            model.load_pretrained(ckpt_path)
        return model
    raise ValueError(f"Generator not implemented for dataset: {dataset}")

def get_combined(classifier, prober, hidden_mean, hidden_std):
    return CombinedNN(classifier, prober, hidden_mean, hidden_std) 