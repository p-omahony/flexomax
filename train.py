import torch
from torchvision.transforms import ToTensor
from models.vgg16 import VGG16
from torch.utils.data import DataLoader
from encoders import Encoder
import lightning as L
from lightning.pytorch.loggers import CSVLogger
from pathlib import Path
from datasets import ClothingDataset
from tqdm import tqdm

DATA_PATH = Path('data')
TRAIN_PATH = DATA_PATH / Path('train')
VAL_PATH = DATA_PATH / Path('val')

LABELS = ['Blazer', 'Dress', 'Hat', 'Hoodie', 'Longsleeve', 'Outwear', 'Pants',
          'Polo', 'Shirt', 'Shoes', 'Shorts', 'Skirt', 'T-Shirt', 'Undershirt']

IMAGES_PATH = DATA_PATH / Path('images_compressed')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
IMG_EXT = '.jpg'
RESIZE_SHAPE = (224, 224)

train_images = list((TRAIN_PATH / 'images').iterdir())
val_images = list((VAL_PATH / 'images').iterdir())
label_encoder = Encoder(LABELS)

train_dataset = ClothingDataset(
    train_images,
    label_encoder=label_encoder,
    transform=ToTensor()
)
val_dataset = ClothingDataset(
    val_images,
    label_encoder=label_encoder,
    transform=ToTensor()
)

train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=16)

model = VGG16(n_classes=len(LABELS))
print(model)

logger = CSVLogger("logs", name="first_exp")

trainer = L.Trainer(
    max_epochs=5,
    logger=logger,
    accelerator=DEVICE
)


# trainer.fit(model, train_dataloader, val_dataloader)