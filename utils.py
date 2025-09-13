import os
import torch
import torch.nn as nn
import random
from torchvision import datasets
from torchvision.transforms import v2, InterpolationMode
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split

class Dataloader:
    def __init__(self, data_dir, batch_size=16):
        self.data_dir = data_dir
        self.batch_size = batch_size

        self.transform = v2.Compose([
            v2.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485]*3, std=[0.229]*3)
        ])

        self.num_classes = None
        self.train = None
        self.valid = None
        self.test = None

    def setup(self):
        full_train = datasets.ImageFolder(root=f"{self.data_dir}/Train_gray", transform=self.transform)
        self.test = datasets.ImageFolder(root=f"{self.data_dir}/Test_gray", transform=self.transform)

        self.num_classes = len(full_train.classes)

        # dividir train en train/valid
        n_train = int(len(full_train) * 0.7)
        n_valid = len(full_train) - n_train
        self.train, self.valid = random_split(full_train, lengths=[n_train, n_valid])

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size,
                          drop_last=True, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.valid, batch_size=self.batch_size,
                          drop_last=False, shuffle=False, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size,
                          drop_last=False, shuffle=False, num_workers=4)

# Semillas

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Guardar el modelo

def _safe_dir() -> str:
    try:
        base = os.path.dirname(os.path.abspath(__file__))  # type: ignore[name-defined]
    except NameError:
        base = os.getcwd()
    return base

def save_model(model: nn.Module, name: str) -> None:
    path = os.path.join(_safe_dir(), name)
    torch.save(model.state_dict(), path)


def load_model(model: nn.Module, name: str) -> nn.Module:
    path = os.path.join(_safe_dir(), name)
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state)
    return model
