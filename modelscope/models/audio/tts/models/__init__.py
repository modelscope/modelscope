from .robutrans import RobuTrans
from .vocoder_models import Generator


def create_am_model(name, hparams):
    if name == 'robutrans':
        return RobuTrans(hparams)
    else:
        raise Exception('Unknown model: ' + name)
