from dataclasses import dataclass


@dataclass
class SwiftConfig:
    pass


class Swift:

    @staticmethod
    def prepare_model(model, config: SwiftConfig):
        """Prepare the module and returns the new module.

        Args:
            model: The model to tune.
            config: The config of the tuner.

        Returns:
            The tuned model.
        """
        from .lora import LoRA, LoRAConfig
        from .adapter import Adapter, AdapterConfig
        from .prompt import Prompt, PromptConfig
        if isinstance(config, LoRAConfig):
            return LoRA.prepare_model(model, config)
        elif isinstance(config, AdapterConfig):
            return Adapter.prepare_model(model, config)
        elif isinstance(config, PromptConfig):
            return Prompt.prepare_model(model, config)
        return None
