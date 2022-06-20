from .robutrans import RobuTrans


def create_model(name, hparams):
    if name == 'robutrans':
        return RobuTrans(hparams)
    else:
        raise Exception('Unknown model: ' + name)
