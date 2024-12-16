import torchood
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


classifier = SimpleModel()
classifier = torchood.EvidenceNeuralNetwork(classifier)
