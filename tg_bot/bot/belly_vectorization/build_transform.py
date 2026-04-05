import torch
from PIL import Image
from torchvision.transforms import v2 as T


class PadToSquare:
    def __init__(self, fill=0):
        self.fill = fill

    def __call__(self, img):
        if not isinstance(img, Image.Image):
            img = T.ToPILImage()(img)

        width, height = img.size
        max_side = max(width, height)

        pad_left = (max_side - width) // 2
        pad_right = max_side - width - pad_left
        pad_top = (max_side - height) // 2
        pad_bottom = max_side - height - pad_top

        return T.functional.pad(
            img,
            padding=[pad_left, pad_top, pad_right, pad_bottom],
            fill=self.fill,
        )


def build_transforms(image_size: int):
    imagenet_mean = (0.485, 0.456, 0.406)
    imagenet_std = (0.229, 0.224, 0.225)

    common_pre = [
        PadToSquare(fill=0),
        T.Resize((image_size, image_size), antialias=True),
    ]

    train_tf = T.Compose([
        *common_pre,

        # брюшко может быть вверх ногами -> 50% переворачиваем на 180
        #
        # пока убрал поворот на 180, т.к. добавил повернутое на 180 изображение как отдельный сэмпл
        #
        # T.RandomApply([T.RandomRotation(degrees=(180, 180))], p=0.5),

        # лёгкие геометрические аугментации
        T.RandomApply([
            T.RandomAffine(
                degrees=0,
                translate=(0.03, 0.03),
                scale=(0.95, 1.05),
                shear=(-3, 3),
            )
        ], p=0.3),

        # фотометрия
        T.RandomApply([
            T.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.1,
                hue=0.02,
            )
        ], p=0.6),

        T.RandomApply([T.GaussianBlur(kernel_size=5)], p=0.2),

        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])

    val_tf = T.Compose([
        *common_pre,
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])

    return train_tf, val_tf
