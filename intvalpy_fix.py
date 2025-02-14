import numpy as np
from intvalpy import lineqs, clear_zero_rows, unique
from intvalpy.linear.boundary_intervals import (
    __center_rm,
    BoundaryIntervals,
    ParticularPoints,
)


def Intervals2Path(S):
    bs, bp = S[0, :2], S[0, :2]
    P = [bp]

    while len(S) > 0:
        index = 0  # added
        for k in range(len(S)):
            if np.max(np.abs(bs - S[k, :2])) < 1e-8:
                index = k
                break
        if index >= len(S):  # added
            return np.array(P)  # added
        es = S[index, 2:4]

        if np.max(np.abs(bs - es)) > 1e-8:
            P.append(es)

            if np.max(np.abs(bp - es)) < 1e-8:
                return np.array(P)
            bs = es
        S = np.delete(S, index, axis=0)
    return np.array(P)


def lineqs(
    A,
    b,
    show=True,
    title="Solution Set",
    color="gray",
    bounds=None,
    alpha=0.5,
    s=10,
    size=(15, 15),
    save=False,
):
    """
    The function visualizes the set of solutions of a system of linear algebraic
    inequalities A x >= b with two variables by the method of boundary intervals, and
    also outputs the vertices of the set of solutions.

    If the set of solutions is unlimited, then the algorithm independently
    selects the rendering boundaries.

    Parameters:

            A: float, array_like
                Matrix of a system of linear algebraic inequalities.

            b: float, array_like
                The vector of the right part of the system of linear algebraic inequalities.

            show: bool, optional
                This parameter is responsible for whether a set of solutions will be shown.
                By default, the value is set to True, i.e. the graph is being drawn.

            title: str, optional
                The top legend of the graph.

            color: str, optional
                The color of the inner area of the set of solutions.

            bounds: array_like, optional
                Borders of the drawing area. The first element of the array is responsible for the lower faces
                on the OX and OY axes, and the second for the upper ones. Thus, in order to OX
                lay within [-2, 2], and OY within [-3, 4], it is necessary to set bounds as [[-2, -3], [2, 4]].

            alpha: float, optional
                Transparency of the graph.

            s: float, optional
                How big are the points of the vertices.

            size: tuple, optional
                The size of the drawing window.

            save: bool, optional
                If the value is True, the graph is saved.

    Returns:

            out: list
                Returns a list of ordered vertices.
                If show = True, then the graph is drawn.
    """

    A = np.asarray(A)
    b = np.asarray(b)

    n, m = A.shape
    assert m <= 2, "There should be two columns in matrix A."
    assert b.shape[0] == n, (
        "The size of the matrix A must be consistent with the size of the vector of the right part of b."
    )

    A, b, cnmty = clear_zero_rows(A, b)

    S = BoundaryIntervals(A, b)
    if len(S) == 0:
        return S

    PP, nV, binf = ParticularPoints(S, A, b)

    if (np.asarray([binf]) == True).any():
        if bounds is None:
            PP = np.array(PP)
            PPmin, PPmax = np.min(PP, axis=0), np.max(PP, axis=0)
            center = (PPmin + PPmax) / 2
            rm = max((PPmax - PPmin) / 2)
            __center_rm.append([max(abs(center) + 5 * rm)])
            A = np.append(np.append(A, np.eye(2)), -np.eye(2)).reshape((len(A) + 4, 2))
            b = np.append(np.append(b, center - 5 * rm), -(center + 5 * rm))

        else:
            A = np.append(np.append(A, np.eye(2)), -np.eye(2)).reshape((len(A) + 4, 2))
            b = np.append(
                np.append(b, [bounds[0][0], bounds[0][1]]),
                [-bounds[1][0], -bounds[1][1]],
            )

        S = BoundaryIntervals(A, b)

    vertices = Intervals2Path(S)

    return unique(vertices)


def IntLinIncR2(
    A,
    b,
    consistency="uni",
):
    ortant = [(1, 1), (-1, 1), (-1, -1), (1, -1)]
    vertices = []
    n, m = A.shape

    assert m <= 2, "There should be two columns in matrix A."
    assert b.shape[0] == n, (
        "The size of the matrix A must be consistent with the size of the vector of the right part of b."
    )

    def algo(bounds):
        for ort in range(4):
            tmp = A.copy()
            WorkListA = np.zeros((2 * n + m, m))
            WorkListb = np.zeros(2 * n + m)

            for k in range(m):
                if ortant[ort][k] == -1:
                    tmp[:, k] = tmp[:, k].dual
                WorkListA[2 * n + k, k] = -ortant[ort][k]

            if consistency == "uni":
                WorkListA[:n], WorkListA[n : 2 * n] = tmp.a, -tmp.b
                WorkListb[:n], WorkListb[n : 2 * n] = b.b, -b.a
            elif consistency == "tol":
                WorkListA[:n], WorkListA[n : 2 * n] = -tmp.a, tmp.b
                WorkListb[:n], WorkListb[n : 2 * n] = -b.a, b.b
            else:
                msg = "Неверно указан тип согласования системы! Используйте 'uni' или 'tol'."
                raise Exception(msg)

            vertices.append(lineqs(-WorkListA, -WorkListb, show=False, bounds=bounds))

    algo(None)  # added

    # Если в каком-либо ортанте множество решений неограничено, то создаём
    # новое отрисовочное окно, чтобы срез решений был одинаковым.
    global __center_rm
    if len(__center_rm) > 0:
        vertices = []
        _max = max(np.array(__center_rm))
        bounds = np.array([[-_max, -_max], [_max, _max]])
        algo(bounds)
    __center_rm = []

    return vertices
