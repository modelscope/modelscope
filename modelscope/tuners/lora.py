

def inject_lora(module):
    pass


def add_lora_hook(trainer):
    from modelscope.trainers.hooks import Hook

    class LORAHOOK(Hook):

        def before_run(self, trainer):
            pass

        def before_eval(self, trainer):
            pass

        def wrap_module(self, trainer):
            trainer.model = inject_lora(trainer.model)

        def strategy(self):
            Hook.overload(self.save_checkpoints, name='CheckpointHook.save_checkpoints')
            Hook.overload(self.remove_checkpoints, name='CheckpointHook.remove_checkpoints')
            Hook.overload(self.load_checkpoints, name='CheckpointHook.load_checkpoints')

        def save_checkpoints(self):
            pass

        def load_checkpoints(self):
            pass

        def remove_checkpoints(self):
            pass

    trainer.register_hook(LORAHOOK())
