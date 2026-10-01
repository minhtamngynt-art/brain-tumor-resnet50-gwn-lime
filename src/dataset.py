import random
from pathlib import Path
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split

CLASS_NAMES = ["notumor", "glioma", "pituitary"]
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

class MRIDataset(Dataset):
    def __init__(self, frame, transform):
        self.frame = frame.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        row = self.frame.iloc[idx]
        image = Image.open(row["path"]).convert("RGB")
        return self.transform(image), int(row["label"])


def get_transforms(img_size=224):
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    eval_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    return train_tf, eval_tf


def prepare_dataloaders(data_dir="./Training", n_per_class=1100, img_size=224, batch_size=32, seed=42):
    data_dir = Path(data_dir)
    extensions = {".jpg", ".jpeg", ".png", ".bmp"}

    rows = []
    for label, class_name in enumerate(CLASS_NAMES):
        class_dir = data_dir / class_name
        paths = sorted(p for p in class_dir.iterdir() if p.suffix.lower() in extensions)

        if len(paths) < n_per_class:
            raise ValueError(f"{class_name}: found {len(paths)} images, need {n_per_class}")

        rng = random.Random(seed)
        selected = rng.sample(paths, n_per_class)

        rows.extend(
            {"path": str(path), "label": label, "class_name": class_name}
            for path in selected
        )

    df = pd.DataFrame(rows)

    train_df, temp_df = train_test_split(
        df, test_size=0.20, stratify=df["label"], random_state=seed
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, stratify=temp_df["label"], random_state=seed
    )

    train_tf, eval_tf = get_transforms(img_size)

    train_ds = MRIDataset(train_df, train_tf)
    val_ds = MRIDataset(val_df, eval_tf)
    test_ds = MRIDataset(test_df, eval_tf)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=torch.cuda.is_available()
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=torch.cuda.is_available()
    )

    return train_loader, val_loader, test_loader, train_df, val_df, test_df
