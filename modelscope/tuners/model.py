import torch


class TunerModel(torch.nn.Module):

    def __init__(self, module, tuner_type):
        super().__init__()
        self.module = module
        self.tuner_type = tuner_type

    def save_pretrained(self):
        pass

    def load_pretrained(self):
        pass



