import torch
from torchvision.transforms import ToTensor, Compose, Normalize, Resize, ToPILImage
from models.vgg16 import VGG16
from torch.utils.data import DataLoader
from encoders import Encoder
import lightning as L
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from pathlib import Path
from datasets import ClothingDataset
from tqdm import tqdm

DATA_PATH = Path('top_data')
TRAIN_PATH = DATA_PATH / Path('train')
VAL_PATH = DATA_PATH / Path('val')

# LABELS = ['Blazer', 'Dress', 'Hat', 'Hoodie', 'Longsleeve', 'Outwear', 'Pants',
#           'Polo', 'Shirt', 'Shoes', 'Shorts', 'Skirt', 'T-Shirt', 'Undershirt']

LABELS = ['T-Shirt', 'Longsleeve', 'Pants', 'Shoes']

LABELS = ['Tshirts', 'Shirts', 'Casual Shoes', 'Watches', 'Sports Shoes', 'Kurtas', 'Tops', 'Handbags', 'Heels', 'Sunglasses', 'Wallets', 'Flip Flops', 'Sandals', 'Briefs', 'Belts', 'Backpacks', 'Socks', 'Formal Shoes', 'Perfume and Body Mist', 'Jeans', 'Shorts', 'Trousers', 'Flats', 'Bra', 'Dresses', 'Sarees', 'Earrings', 'Deodorant', 'Nail Polish', 'Lipstick']

IMAGES_PATH = DATA_PATH / Path('images_compressed')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
IMG_EXT = '.jpg'
RESIZE_SHAPE = (64, 64)

train_images = list((TRAIN_PATH / 'images').iterdir())
val_images = list((VAL_PATH / 'images').iterdir())
label_encoder = Encoder(LABELS)

# transforms = Compose([
#     ToTensor(),
#     Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])

transforms = Compose([
    ToPILImage(),
    Resize(size=RESIZE_SHAPE),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


train_dataset = ClothingDataset(
    train_images,
    label_encoder=label_encoder,
    transform=transforms
)

val_dataset = ClothingDataset(
    val_images,
    label_encoder=label_encoder,
    transform=transforms
)

train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=16)

model = VGG16(n_classes=len(LABELS))
print(model)

logger = CSVLogger("logs", name="first_exp")

val_loss_ckpt_callback = ModelCheckpoint(
    dirpath="checkpoints",
    filename='{epoch}-{val_loss:.2f}',
    monitor='val_loss',
    mode='min',
    save_top_k=2
)

val_acc_ckpt_callback = ModelCheckpoint(
    dirpath="checkpoints",
    filename='{epoch}-{val_balanced_accuracy:.2f}',
    monitor='val_balanced_accuracy',
    mode='max',
    save_top_k=2
)

trainer = L.Trainer(
    max_epochs=100,
    logger=logger,
    callbacks=[val_acc_ckpt_callback, val_loss_ckpt_callback],
    accelerator=DEVICE
)


trainer.fit(model, train_dataloader, val_dataloader)