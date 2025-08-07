from tqdm.auto import tqdm


class ProgressCallback:

    def __init__(self, filename: str, file_size: int):
        self.filename = filename
        self.file_size = file_size

    def update(self, size: int):
        pass

    def end(self):
        pass


class TqdmCallback(ProgressCallback):

    def __init__(self, filename: str, file_size: int):
        super().__init__(filename, file_size)
        self.progress = tqdm(
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
            total=file_size if file_size > 0 else 1,
            initial=0,
            desc='Downloading [' + self.filename + ']',
            leave=True)

    def update(self, size: int):
        self.progress.update(size)

    def end(self):
        self.progress.close()
