# The implementation is adopted from stable-diffusion-webui, made public available under the Apache 2.0 License
# at https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/master/modules/devices.py

import contextlib
import sys

import torch

if sys.platform == 'darwin':
    from modules import mac_specific


def has_mps() -> bool:
    if sys.platform != 'darwin':
        return False
    else:
        return mac_specific.has_mps


def get_cuda_device_string():
    return 'cuda'


def get_optimal_device_name():
    if torch.cuda.is_available():
        return get_cuda_device_string()

    if has_mps():
        return 'mps'

    return 'cpu'


def get_optimal_device():
    return torch.device(get_optimal_device_name())


def get_device_for(task):
    return get_optimal_device()


def torch_gc():

    if torch.cuda.is_available():
        with torch.cuda.device(get_cuda_device_string()):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    if has_mps():
        mac_specific.torch_mps_gc()


def enable_tf32():
    if torch.cuda.is_available():

        # enabling benchmark option seems to enable a range of cards to do fp16 when they otherwise can't
        # see https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/4407
        if any(
                torch.cuda.get_device_capability(devid) == (7, 5)
                for devid in range(0, torch.cuda.device_count())):
            torch.backends.cudnn.benchmark = True

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


enable_tf32()

cpu = torch.device('cpu')
device = device_interrogate = device_gfpgan = device_esrgan = device_codeformer = torch.device(
    'cuda')
dtype = torch.float16
dtype_vae = torch.float16
dtype_unet = torch.float16
unet_needs_upcast = False


def cond_cast_unet(input):
    return input.to(dtype_unet) if unet_needs_upcast else input


def cond_cast_float(input):
    return input.float() if unet_needs_upcast else input


def randn(seed, shape):
    torch.manual_seed(seed)
    return torch.randn(shape, device=device)


def randn_without_seed(shape):
    return torch.randn(shape, device=device)


def autocast(disable=False):
    if disable:
        return contextlib.nullcontext()

    return torch.autocast('cuda')


def without_autocast(disable=False):
    return torch.autocast('cuda', enabled=False) if torch.is_autocast_enabled() and \
        not disable else contextlib.nullcontext()


class NansException(Exception):
    pass


def test_for_nans(x, where):
    if not torch.all(torch.isnan(x)).item():
        return

    if where == 'unet':
        message = 'A tensor with all NaNs was produced in Unet.'

    elif where == 'vae':
        message = 'A tensor with all NaNs was produced in VAE.'

    else:
        message = 'A tensor with all NaNs was produced.'

    message += ' Use --disable-nan-check commandline argument to disable this check.'

    raise NansException(message)
