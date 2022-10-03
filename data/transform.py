import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transform():
    return A.Compose([
        A.Rotate(limit=180, p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.Normalize(), ToTensorV2()
    ])

def get_val_transform():
    return A.Compose([A.Normalize(), ToTensorV2()])