import enum
from typing import Literal
from pathlib import Path
import numpy as np

voltages = np.array(
    [-0.5, -0.4, -0.3, -0.2, -0.1, -0, 0, 0.1, 0.2, 0.3, 0.4, 0.5],
    # [-0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5],
    dtype=np.float64,
)


class Num(enum.IntEnum):
    CH = 8  # num of channels
    CELLS = 1024  # num of cells
    V = len(voltages)  # num of voltages
    R = 100  # num of records


def load_data(directory: Path, side: Literal["a", "b"]):
    ld = np.empty((Num.CH, Num.CELLS, Num.V, Num.R), dtype=np.float64)

    for idx, voltage in enumerate(voltages):
        fname = directory / f"{voltage:.1f}lvl_side_{side}_fast_data.npy"
        ld[:, :, idx, :] = np.load(fname)

    return ld


if __name__ == "__main__":
    data_dir = Path("sensor_data/04_10_2024_070_068")
    data = load_data(data_dir, side="a")
