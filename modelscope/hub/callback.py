from tqdm.auto import tqdm


class ProgressCallback:

    def __init__(self, filename: str, file_size: int):
        self.filename = filename
        self.file_size = file_size

    def set_current(self, size: int):
        pass

    def update(self, size: int):
        pass

    def end(self):
        pass


class TqdmCallback(ProgressCallback):

    def __init__(self,
                 filename: str,
                 file_size: int,
                 resume_size: int = 0):
        super().__init__(filename, file_size)
        total = file_size if file_size > 0 else 1
        self.progress = tqdm(
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
            total=total,
            initial=min(max(resume_size, 0), total),
            desc='Downloading [' + self.filename + ']',
            leave=True)

    def set_current(self, size: int):
        self.progress.n = min(max(size, 0), self.progress.total or 0)
        self.progress.last_print_n = self.progress.n
        self.progress.refresh()

    def update(self, size: int):
        self.progress.update(size)

    def end(self):
        self.progress.close()
