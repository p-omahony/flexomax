import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score
from torch.optim import AdamW
import torch


class CNNModel(nn.Module):
    def __init__(self, n_classes):
        super(CNNModel, self).__init__()

        self.n_classes = n_classes
        
        # Convolution 1
        self.cnn1  = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=1, padding=0)
        #self.bt1   = nn.BatchNorm2d(16)  # Batch normalization layer
        self.relu1 = nn.LeakyReLU()
        
        # Max pool 1
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
     
        # Convolution 2
        self.cnn2  = nn.Conv2d(in_channels=16, out_channels=64, kernel_size=5, stride=1, padding=0)
        #self.bt2   = nn.BatchNorm2d(64)
        self.relu2 = nn.LeakyReLU()
        
        # Max pool 2
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        
        # Fully connected 1
        self.fc1      = nn.Linear(13 * 13 * 64, self.n_classes)
        #self.drop     = nn.Dropout(p=0.5)
        #self.fc2      = nn.Linear(10,2)
        self.sigmoid1 = nn.Sigmoid()
    
    def forward(self, x):
        # Convolution 1
        out = self.cnn1(x)
        #out = self.bt1(out)
        out = self.relu1(out)
        
        # Max pool 1
        out = self.maxpool1(out)
        
        # Convolution 2 
        out = self.cnn2(out)
        #out = self.bt2(out)
        out = self.relu2(out)
        
        # Max pool 2 
        out = self.maxpool2(out)
        
        # flatten
        out = out.view(out.size(0), -1)

        # Linear function (readout)
        out = self.fc1(out)
        #out = self.drop(out)
        #out = self.fc2(out)
        out = self.sigmoid1(out)
        
        return out
    
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