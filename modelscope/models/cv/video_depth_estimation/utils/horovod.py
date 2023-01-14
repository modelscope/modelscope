# Part of the implementation is borrowed and modified from PackNet-SfM,
# made publicly available under the MIT License at https://github.com/TRI-ML/packnet-sfm
try:
    import horovod.torch as hvd
    HAS_HOROVOD = True
except ImportError:
    HAS_HOROVOD = False


def hvd_disable():
    global HAS_HOROVOD
    HAS_HOROVOD = False


def hvd_init():
    if HAS_HOROVOD:
        hvd.init()
    return HAS_HOROVOD


def on_rank_0(func):

    def wrapper(*args, **kwargs):
        if rank() == 0:
            func(*args, **kwargs)

    return wrapper


def rank():
    return hvd.rank() if HAS_HOROVOD else 0


def world_size():
    return hvd.size() if HAS_HOROVOD else 1


@on_rank_0
def print0(string='\n'):
    print(string)


def reduce_value(value, average, name):
    """
    Reduce the mean value of a tensor from all GPUs

    Parameters
    ----------
    value : torch.Tensor
        Value to be reduced
    average : bool
        Whether values will be averaged or not
    name : str
        Value name

    Returns
    -------
    value : torch.Tensor
        reduced value
    """
    return hvd.allreduce(value, average=average, name=name)
