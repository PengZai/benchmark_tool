from dense_slam_benchmark.dataset_tools import utils

from .base import Dataset


class TartanAir(Dataset):
    def __init__(self, configs):
        super().__init__(configs)

        with open(self.data_source["trajectory_path"], "r", encoding="utf-8") as f:
            idx = 0
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                x, y, z, qx, qy, qz, qw = map(float, line.split())
                T_w_p = utils.pose_to_T(x, y, z, qx, qy, qz, qw)
                datasample = {"id": idx, "ts": idx, "T_w_p": T_w_p}
                self.samples.append(datasample)
                idx += 1

        for sample_idx, sample in enumerate(self.samples):
            if not self.is_sample_idx_selected(sample_idx):
                continue
            self.loadSyncrhonizedData(sample)
