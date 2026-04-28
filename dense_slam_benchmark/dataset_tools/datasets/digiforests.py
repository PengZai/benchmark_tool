from .base import Dataset


class DigiForests(Dataset):
    def __init__(self, configs):
        super().__init__(configs)
        self.readDatasample()
        for sample_idx, sample in enumerate(self.samples):
            if not self.is_sample_idx_selected(sample_idx):
                continue
            self.loadAsyncrhonizedData(sample)
