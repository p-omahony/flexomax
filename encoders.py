import numpy as np

class Encoder:
    def __init__(self, labels):
        self.labels = labels
        self.n_classes = len(labels)
        self.eyes = np.eye(self.n_classes)

    def encode(self, inpt):
        return self.eyes[self.labels.index(inpt)]
    
    def decode(self, inpt):
        idx = np.argmax(inpt)
        return self.labels[idx]

