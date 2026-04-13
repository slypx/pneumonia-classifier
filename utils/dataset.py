# utils/dataset.py
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from .config import TRAIN_DIR, VAL_DIR, TEST_DIR, BATCH_SIZE, NUM_WORKERS

def get_transforms(train: bool = True):
    if train:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

def create_dataloaders():
    train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=get_transforms(True))
    val_dataset   = datasets.ImageFolder(VAL_DIR,   transform=get_transforms(False))
    test_dataset  = datasets.ImageFolder(TEST_DIR,  transform=get_transforms(False))

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=NUM_WORKERS)
    val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=NUM_WORKERS)
    test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=NUM_WORKERS)

    return train_loader, val_loader, test_loader, train_dataset.classes
