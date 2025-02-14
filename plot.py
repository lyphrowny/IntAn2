from collections import namedtuple
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt

plt.switch_backend("wxagg")

POS = namedtuple("PlotOrSave", "should_show output_dir subdir fname")

eps = 2**-14


def _plot_or_save(pos: POS) -> None:
    if pos.should_show:
        plt.show()
        return

    odir = Path(pos.output_dir, pos.subdir)
    odir.mkdir(exist_ok=True, parents=True)
    plt.savefig(odir / pos.fname)


def plot_calib_method_1(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    b: NDArray[np.float64],
    str_idxs: str,
    pos: POS,
) -> None:
    plt.figure()
    plt.title(f"$Y(x)$ method 1 for {str_idxs}")
    plt.scatter(x, y, label="medians", s=2)
    bounds = np.array([-1, 1]) * 0.5
    plt.plot(
        bounds,
        b[1] + b[0] * bounds,
        label="argmax Tol",
    )
    plt.legend()

    pos = pos._replace(fname="calib_1.png")
    _plot_or_save(pos)


def plot_calib_diff_method_1(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    b: NDArray[np.float64],
    rads: NDArray[np.float64],
    str_idxs: str,
    pos: POS,
) -> None:
    plt.figure()
    plt.title(f"$Y(x) - b_0*x - b_1$ method 1 for {str_idxs}")

    coeff = b[1] + b[0] * x

    plt.vlines(
        np.arange(len(y)),
        ymin=y - rads - coeff,
        ymax=y + rads - coeff,
        color="lightblue",
        zorder=1,
    )
    plt.vlines(
        np.arange(len(y)),
        ymin=y - eps - coeff,
        ymax=y + eps - coeff,
        color="black",
        zorder=2,
    )
    pos = pos._replace(fname="calib_diff_1.png")
    _plot_or_save(pos)


def plot_calib_method_2(
    x: NDArray[np.float64],
    y_in: NDArray[np.float64],
    y_ex: NDArray[np.float64],
    b: NDArray[np.float64],
    b_uni_vertices: NDArray[np.float64],
    b_tol_vertices: NDArray[np.float64],
    str_idxs: str,
    pos: POS,
) -> None:
    plt.figure()
    plt.title(f"$Y(x)$ method 2 for {str_idxs}")

    plt.vlines(x, y_ex[:, 0], y_ex[:, 1], color="gray", zorder=1)
    plt.vlines(x, y_in[:, 0], y_in[:, 1], color="blue", zorder=2)

    bounds = np.array([-1, 1]) * 0.5
    plt.plot(
        bounds,
        b[1] + b[0] * bounds,
        label="argmax Tol",
        color="red",
        zorder=3,
    )

    x = np.concatenate(([-3], x, [3]))
    for x0, x1 in zip(x, x[1:]):
        _x = np.array((x0, x1))
        mid = (x0 + x1) / 2

        for vertices, color in (
            (b_uni_vertices, "lightgray"),
            (b_tol_vertices, "lightblue"),
        ):
            b0, b1 = vertices.T
            min_idx, *_, max_idx = np.argsort(b1 + b0 * mid)

            y0_low, y1_low = b1[min_idx] + b0[min_idx] * _x
            y0_hi, y1_hi = b1[max_idx] + b0[max_idx] * _x
            plt.fill(
                [x0, x1, x1, x0],
                [y0_low, y1_low, y1_hi, y0_hi],
                facecolor=color,
                linewidth=0,
            )
    plt.xlim((-0.6, 0.6))
    plt.ylim((-0.6, 0.6))

    pos = pos._replace(fname="calib_2.png")
    _plot_or_save(pos)


def plot_uni_tol_vertices(
    b: NDArray[np.float64],
    b_uni_vertices: NDArray[np.float64],
    b_tol_vertices: NDArray[np.float64],
    str_idxs: str,
    pos: POS,
) -> None:
    plt.figure()
    plt.title(f"Uni and Tol method 2 for {str_idxs}")
    plt.xlabel("b0")
    plt.ylabel("b1")

    PlotArgs = namedtuple("PlotArgs", "vertices label alpha color s")
    uni_args = PlotArgs(
        vertices=b_uni_vertices,
        label="Uni",
        alpha=0.5,
        color="gray",
        s=1,
    )
    tol_args = PlotArgs(
        vertices=b_tol_vertices,
        label="Tol",
        alpha=0.3,
        color="blue",
        s=10,
    )
    for plt_args in (uni_args, tol_args):
        x, y = plt_args.vertices.T
        plt.fill(
            x,
            y,
            linestyle="-",
            linewidth=1,
            color=plt_args.color,
            alpha=plt_args.alpha,
            label=plt_args.label,
        )
        plt.scatter(x, y, s=plt_args.s, color="black", alpha=1)

    plt.scatter(b[0], b[1], s=10, color="red", alpha=1, label="argmax Tol")
    plt.legend()
    pos = pos._replace(fname="uni_tol_2.png")
    _plot_or_save(pos)
