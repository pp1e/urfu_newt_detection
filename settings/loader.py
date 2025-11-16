import os
import torch
import segmentation_models_pytorch as smp
from dotenv import load_dotenv

load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN")

MODEL_KARELINA_PATH = os.getenv("MODEL_KARELINA_PATH")
MODEL_RIBBED_PATH = os.getenv("MODEL_RIBBED_PATH")

DEVICE = os.getenv("DEVICE", "cpu")

# конфигурация модели
ENCODER = os.getenv("ENCODER", "resnet34")
ENCODER_WEIGHTS = os.getenv("ENCODER_WEIGHTS", "imagenet")
CLASSES = int(os.getenv("CLASSES", 1))


def build_model():
    return smp.Unet(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        in_channels=3,
        classes=CLASSES,
    )


def load_model(path: str):
    checkpoint = torch.load(path, map_location=DEVICE)

    # твоя структура чекпоинта:
    # {"model_state": ..., "epoch": ..., "val_dice": ..., ...}
    state_dict = checkpoint.get("model_state", checkpoint)

    model = build_model()
    model.load_state_dict(state_dict)

    model.to(DEVICE)
    model.eval()
    return model


MODEL_KARELINA = load_model(MODEL_KARELINA_PATH)
MODEL_RIBBED = load_model(MODEL_RIBBED_PATH)
