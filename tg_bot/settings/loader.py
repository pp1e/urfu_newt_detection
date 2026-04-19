import os
import torch
from dotenv import load_dotenv
from transformers import SegformerForSemanticSegmentation

load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN")

SEGMENTATION_MODEL_PATH = os.getenv("SEGMENTATION_MODEL_PATH")

DEVICE = os.getenv("DEVICE", "cpu")

CLASSES = int(os.getenv("CLASSES", 1))
IMG_SIZE = int(os.getenv("IMG_SIZE", 768))
THRESHOLD = float(os.getenv("THRESHOLD", 0.5))


def build_model():
    return SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b2-finetuned-ade-512-512",
        num_labels=CLASSES,
        ignore_mismatched_sizes=True,
    )


def load_model(path: str):
    checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)

    state_dict = checkpoint.get("model_state", checkpoint)

    model = build_model()
    model.load_state_dict(state_dict)

    model.to(DEVICE)
    model.eval()
    return model


SEGMENTATION_MODEL = load_model(SEGMENTATION_MODEL_PATH)
