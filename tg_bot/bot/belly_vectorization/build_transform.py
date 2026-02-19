import torch
from torchvision.transforms import v2 as T

def build_transforms(image_size: int = 224):
    imagenet_mean = (0.485, 0.456, 0.406)
    imagenet_std = (0.229, 0.224, 0.225)

    train_tf = T.Compose([
        T.Resize((image_size, image_size)),

        # брюшко может быть вверх ногами -> 50% переворачиваем на 180
        T.RandomApply([T.RandomRotation(degrees=(180, 180))], p=0.5),

        # лёгкие фотометрические аугментации
        T.RandomApply(
            [T.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.15, hue=0.02)],
            p=0.7
        ),
        T.RandomApply([T.GaussianBlur(kernel_size=5)], p=0.2),

        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])

    val_tf = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])

    return train_tf, val_tf
