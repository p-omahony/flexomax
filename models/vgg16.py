from torchvision.models import vgg16
from sklearn.metrics import balanced_accuracy_score
from torch.optim import Adam
import torch
import lightning as L
import torch.nn as nn

class Classifier(nn.Module):
    def __init__(self, n_classes):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(in_features=25088, out_features=4096, bias=True)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(in_features=4096, out_features=4096, bias=True)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(in_features=4096, out_features=n_classes, bias=True)

    def forward(self, inpt):
        inpt = self.fc1(inpt)
        inpt = self.relu1(inpt)
        inpt = self.dropout1(inpt)
        inpt = self.fc2(inpt)
        inpt = self.relu2(inpt)
        inpt = self.dropout2(inpt)
        otpt = self.fc3(inpt)
        
        return otpt

class VGG16(L.LightningModule):
    def __init__(self, n_classes):
        super().__init__()
        self.model = vgg16()
        self.model.classifier = Classifier(n_classes)

    def forward(self, inputs):
        return self.model(inputs)
    
    def loss_func(self, y_hat, y_true):
        ce_loss = nn.CrossEntropyLoss()
        return ce_loss(y_hat, y_true)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)

        loss = self.loss_func(y_hat, y)
        bal_acc = balanced_accuracy_score(
            torch.argmax(y.detach(), axis=1).cpu().numpy(),
            torch.argmax(y_hat.detach(), axis=1).cpu().numpy()
        )

        self.log("train_loss", loss, on_epoch=True)
        self.log("train_balanced_accuracy", bal_acc, on_epoch=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)

        loss = self.loss_func(y_hat, y)
        bal_acc = balanced_accuracy_score(
            torch.argmax(y.detach(), axis=1).cpu().numpy(),
            torch.argmax(y_hat.detach(), axis=1).cpu().numpy()
        )

        self.log("val_loss", loss, on_epoch=True)
        self.log("val_balanced_accuracy", bal_acc, on_epoch=True)

    
    def configure_optimizers(self):
        return Adam(self.parameters(), lr=0.001)
    
