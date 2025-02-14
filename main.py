from collections.abc import Collection
import enum
from typing import Callable, Literal
from pathlib import Path
import numpy as np
from intvalpy import Interval, Tol, precision
from intvalpy_fix import IntLinIncR2
from numpy.typing import NDArray
from tqdm import tqdm
from itertools import product

from plot import (
    POS,
    plot_calib_method_1,
    plot_calib_diff_method_1,
    plot_calib_method_2,
    plot_uni_tol_vertices,
)

precision.extendedPrecisionQ = True

voltages = np.array(
    [-0.5, -0.4, -0.3, -0.2, -0.1, -0, 0, 0.1, 0.2, 0.3, 0.4, 0.5],
    # [-0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5],
    dtype=np.float64,
)
eps = 2**-14


class Num(enum.IntEnum):
    CH = 8  # num of channels
    CELLS = 1024  # num of cells
    V = len(voltages)  # num of voltages
    R = 100  # num of records


def load_data(directory: Path, side: Literal["a", "b"]) -> NDArray[np.float64]:
    ld = np.empty((Num.CH, Num.CELLS, Num.V, Num.R), dtype=np.float64)

    for idx, voltage in enumerate(voltages):
        fname = directory / f"{voltage:.1f}lvl_side_{side}_fast_data.npy"
        ld[:, :, idx, :] = np.load(fname)

    return ld


def regression_type_1(
    values: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], int]:
    x = np.repeat(voltages, Num.R)  # [x, y] -> [x, ..., x, y, ..., y]
    y = values.ravel()
    weights = np.full_like(y, eps)

    # same as X_mat below, but x_vec is created faster
    # though Interval creation takes all the gain away
    # x_vec = np.ones((x.size, 2, 2), dtype=x.dtype)
    # x_vec[:, 0, :] = x[:, None]
    # X_mat = Interval(x_vec)

    # we know that y_i = b_0 + b_1 * x_i
    # or, in other words
    # Y = X * b, where X is a matrix with row (x_i, 1), and b is a vector (b_1, b_0)
    X_mat = Interval([[[_x, _x], [1, 1]] for _x in x])
    Y_vec = Interval(np.column_stack((y, weights)), midRadQ=True)

    # find argmax for Tol
    b_vec, tol_val, num_iter, calcfg_num, exit_code = Tol.maximize(X_mat, Y_vec)
    updated = 0
    if tol_val < 0:
        # if Tol value is less than 0, we must iterate over all rows and add some changes to Y_vec, so Tol became 0
        for i in range(len(Y_vec)):
            X_mat_small = Interval([[x[i], x[i]], [1, 1]])
            Y_vec_small = Interval([[y[i], weights[i]]], midRadQ=True)
            value = Tol.value(X_mat_small, Y_vec_small, b_vec)
            if value < 0:
                weights[i] = abs(y[i] - (x[i] * b_vec[0] + b_vec[1])) + 1e-8
                updated += 1

    Y_vec = Interval(np.column_stack((y, weights)), midRadQ=True)
    # find argmax for Tol
    b_vec, tol_val, num_iter, calcfg_num, exit_code = Tol.maximize(X_mat, Y_vec)

    return b_vec, weights, updated


# using twin arithmetics
def regression_type_2(
    points: NDArray[np.float64],
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    int,
    NDArray[np.float64],
    NDArray[np.float64],
]:
    x_new = voltages.copy()
    # already in shape (Num.V, Num.R) -> (12, 100)
    y_new = np.sort(points.copy(), axis=1)

    y25 = y_new[:, 25]
    y75 = y_new[:, 75]
    whisker_len = 1.5 * (y75 - y25)

    y_in_down = y25 - eps
    y_in_up = y75 + eps
    y_ex_down = np.maximum(y25 - whisker_len, y_new[:, 0])
    y_ex_up = np.minimum(y75 + whisker_len, y_new[:, -1])

    # create 2x2 matrix per element 4 times (2x2x4 = 16)
    n_repeat = 4
    X_mat = np.ones(2 * 2 * n_repeat * Num.V).reshape(-1, 2, 2)
    # assign x_new element to each row of 2x2 matrix
    # [1, 1] -> [elem, elem]. [elem, elem] rows will be repeated 4 times, then next elem
    X_mat[:, 0, :] = np.repeat(x_new, n_repeat)[:, None]

    # vec of pairs [y_ex_down[i], y_ex_up[i]], [y_ex_down[i], y_in_up[i]], ...
    Y_vec = np.column_stack(
        (
            np.column_stack((y_ex_down, y_in_down)).ravel().repeat(2),
            np.tile(np.column_stack((y_ex_up, y_in_up)), 2).ravel(),
        )
    )

    # now we have matrix X * b = Y, but with some "additional" rows
    # we can walk over all rows and if some of them is less than 0, we can just remove it at all
    X_mat_interval = Interval(X_mat)
    Y_vec_interval = Interval(Y_vec)
    b_vec, tol_val, num_iter, calcfg_num, exit_code = Tol.maximize(
        X_mat_interval, Y_vec_interval
    )
    to_remove = []
    if tol_val < 0:
        # if Tol value is less than 0, we must iterate over all rows and add some changes to Y_vec, so Tol became 0
        for i in range(len(Y_vec)):
            X_mat_small = Interval([X_mat[i]])
            Y_vec_small = Interval([Y_vec[i]])
            value = Tol.value(X_mat_small, Y_vec_small, b_vec)
            if value < 0:
                to_remove.append(i)

        X_mat = np.delete(X_mat, to_remove, axis=0)
        Y_vec = np.delete(Y_vec, to_remove, axis=0)

    X_mat_interval = Interval(X_mat)
    Y_vec_interval = Interval(Y_vec)
    b_vec, tol_val, num_iter, calcfg_num, exit_code = Tol.maximize(
        X_mat_interval, Y_vec_interval
    )

    b_uni_vertices, b_tol_vertices = [], []
    for b_vertices, consistency in ((b_uni_vertices, "uni"), (b_tol_vertices, "tol")):
        vertices = IntLinIncR2(X_mat_interval, Y_vec_interval, consistency=consistency)
        for v in filter(len, vertices):
            b_vertices.extend(v.tolist())

    return (
        b_vec,
        np.column_stack((y_in_down, y_in_up)),
        np.column_stack((y_ex_down, y_ex_up)),
        len(to_remove),
        np.array(b_uni_vertices),
        np.array(b_tol_vertices),
    )


def plot(
    all_data: NDArray[np.float64],
    sensor_idx: int,
    cell_idx: int,
    should_plot: bool = True,
    should_show: bool = False,
):
    data = all_data[sensor_idx, cell_idx]
    # method 1
    b_vec, rads, to_remove = regression_type_1(data)

    # method 2
    (
        b_vec2,
        y_in,
        y_ex,
        to_remove,
        b_uni_vertices,
        b_tol_vertices,
    ) = regression_type_2(data)

    str_idxs = f"{(sensor_idx, cell_idx)}"
    prec = 4
    print(
        f"{str_idxs} & 1 & {float(b_vec[0]):.{prec}f} & {float(b_vec[1]):.{prec}f} & {to_remove} \\\\"
    )
    print(
        f"{str_idxs} & 2 & {float(b_vec2[0]):.{prec}f} & {float(b_vec2[1]):.{prec}f} & {to_remove} \\\\"
    )

    if should_plot:
        pos = POS(
            should_show=should_show,
            output_dir="figs",
            subdir=f"{sensor_idx}_{cell_idx}",
            fname="",
        )
        x = np.repeat(voltages, 100)
        y = data.ravel()

        plot_calib_method_1(x, y, b_vec, str_idxs, pos)
        plot_calib_diff_method_1(x, y, b_vec, rads, str_idxs, pos)
        plot_calib_method_2(
            voltages.copy(),
            y_in,
            y_ex,
            b_vec2,
            b_uni_vertices,
            b_tol_vertices,
            str_idxs,
            pos,
        )
        plot_uni_tol_vertices(b_vec2, b_uni_vertices, b_tol_vertices, str_idxs, pos)


def find_negatives(
    all_data: NDArray[np.float64],
    sensor_idxs: Collection[int],
    cell_idxs: Collection[int],
    condition: Callable[[int], bool],
    max_results: int = 10,
    should_plot: bool = False,
    should_show: bool = True,
) -> None:
    n_results = 0
    for sensor_idx, cell_idx in tqdm(
        product(sensor_idxs, cell_idxs),
        total=len(sensor_idxs) * len(cell_idxs),
    ):
        data = all_data[sensor_idx, cell_idx]
        (
            b_vec2,
            y_in,
            y_ex,
            to_remove,
            b_uni_vertices,
            b_tol_vertices,
        ) = regression_type_2(data)
        if not condition(to_remove):
            continue

        print(f"{(sensor_idx, cell_idx)}")
        print(f"\t{to_remove}")
        print(f"\t{len(b_uni_vertices) = }, {len(b_tol_vertices) = }")

        if should_plot:
            str_idxs = f"{(sensor_idx, cell_idx)}"
            pos = POS(
                should_show=should_show,
                output_dir="figs",
                subdir=f"{sensor_idx}_{cell_idx}",
                fname="",
            )
            plot_uni_tol_vertices(b_vec2, b_uni_vertices, b_tol_vertices, str_idxs, pos)

        n_results += 1
        if n_results > max_results:
            print("Max results were achieved, quitting...")
            quit()


if __name__ == "__main__":
    data_dir = Path("sensor_data/04_10_2024_070_068")
    data = load_data(data_dir, side="a")

    find_negatives(
        data,
        sensor_idxs=range(7, 8),
        cell_idxs=range(Num.CELLS)[::-1],
        condition=lambda x: x == 0,
    )
    find_negatives(
        data,
        sensor_idxs=range(0, 2),
        cell_idxs=range(Num.CELLS),
        condition=lambda x: 10 < x < 20,
    )
    find_negatives(
        data,
        sensor_idxs=range(3, 4),
        cell_idxs=range(Num.CELLS),
        condition=lambda x: x > 30,
    )

    sensor_cell_idxs = (
        (7, 505),  # x == 0
        (0, 69),  # 10 < x < 20
        (3, 141),  # x > 30
    )

    for sensor_idx, cell_idx in sensor_cell_idxs:
        plot(data, sensor_idx, cell_idx, should_plot=True, should_show=False)
