from .base_tuner import Tuner
import torch
from lora_diffusion import inject_trainable_lora


def lora_state_dict(self: torch.nn.Module, *args, **kwargs):
    state_dict = self._state_dict_origin(*args, **kwargs)
    return {k: state_dict[k] for k in state_dict if "lora_" in k}


class LoRATuner(Tuner):

    def tune(self, model, **kwargs):
        model.requires_grad_(False)
        require_grad_params = inject_trainable_lora(model, **kwargs)
        model._state_dict_origin = model.state_dict
        model.state_dict = lora_state_dict
        return require_grad_params

    def add_hook(self, trainer, **kwargs):
        from modelscope.trainers.hooks import Hook

        class LoRAHook(Hook):

            def __init__(self):
                self._wrapped = False

            def before_run(self, trainer):
                self.wrap_module(trainer)

            def before_eval(self, trainer):
                self.wrap_module(trainer)

            def wrap_module(_, trainer):
                if not self._wrapped:
                    self.tune(trainer.model, **kwargs)
                    self._wrapped = True

        trainer.register_hook(LoRAHook())
