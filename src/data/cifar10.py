import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.247, 0.243, 0.261)


def _worker_init_fn(worker_id: int) -> None:
    import numpy as np
    import random
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


class _HFCifar10Dataset(Dataset):
    """Thin wrapper around a HuggingFace CIFAR-10 split.

    Each HF sample is {'img': PIL.Image, 'label': int}.
    """

    def __init__(self, hf_split, transform):
        self._data = hf_split
        self._transform = transform

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int):
        sample = self._data[idx]
        img = sample["img"].convert("RGB")
        label = sample["label"]
        return self._transform(img), label


def get_cifar10_loaders(
    data_dir: str = "./data",
    batch_size: int = 128,
    num_workers: int = 4,
    resize: int | None = None,
) -> tuple[DataLoader, DataLoader]:
    """Return (train_loader, test_loader) for CIFAR-10.

    Loads from HuggingFace (uoft-cs/cifar10) and caches under data_dir.

    Args:
        data_dir: Directory for HuggingFace dataset cache.
        batch_size: Mini-batch size for both loaders.
        num_workers: Number of DataLoader worker processes.
        resize: If given, resize images to this square size (e.g. 224 for ViT).
    """
    base_transforms = []
    if resize is not None:
        base_transforms.append(transforms.Resize((resize, resize)))

    train_transform = transforms.Compose(
        base_transforms
        + [
            transforms.RandomCrop(resize if resize else 32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )

    test_transform = transforms.Compose(
        base_transforms
        + [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )

    hf_ds = load_dataset(
        "uoft-cs/cifar10",
        cache_dir=data_dir,
    )

    train_dataset = _HFCifar10Dataset(hf_ds["train"], train_transform)
    test_dataset = _HFCifar10Dataset(hf_ds["test"], test_transform)

    import torch
    pin_memory = torch.cuda.is_available()  # pin_memory is unsupported on MPS

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=_worker_init_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=_worker_init_fn,
    )

    return train_loader, test_loader
