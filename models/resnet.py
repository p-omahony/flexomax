import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score
from torch.optim import AdamW
import lightning as L

class ResNet(L.LightningModule):
    def __init__(self, n_classes):
        super(ResNet, self).__init__()
        self.n_classes = n_classes

        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
        self.final_classifier = nn.Linear(in_features=512, out_features=self.n_classes)
        self.model.fc = self.final_classifier

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        loss = self.loss_func(y_hat, y)
        bal_acc = balanced_accuracy_score(torch.argmax(y, axis=1).cpu().numpy(), torch.argmax(y_hat, axis=1).cpu().numpy())

        self.log("train_loss", loss, on_step=False, on_epoch=True)
        self.log("train_balanced_accuracy", bal_acc, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        loss = self.loss_func(y_hat, y)
        bal_acc = balanced_accuracy_score(torch.argmax(y, axis=1).cpu().numpy(), torch.argmax(y_hat, axis=1).cpu().numpy())

        self.log("val_loss", loss, on_step=False, on_epoch=True)
        self.log("val_balanced_accuracy", bal_acc, on_step=False, on_epoch=True)

        return loss
    
    def loss_func(self, y_hat, y):
        ce_loss = nn.CrossEntropyLoss()
        return ce_loss(y_hat, y)
    
    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=0.001)
    
