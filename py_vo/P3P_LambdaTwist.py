import numpy as np


def P3P_LambdaTwist(Points2D: np.ndarray, Points3D: np.ndarray):
    y1 = Points2D[:, 0] / np.linalg.norm(Points2D[:, 0])
    y2 = Points2D[:, 1] / np.linalg.norm(Points2D[:, 1])
    y3 = Points2D[:, 2] / np.linalg.norm(Points2D[:, 2])

    b12 = -2 * y1 @ y2
    b13 = -2 * y1 @ y3
    b23 = -2 * y2 @ y3

    x1 = Points3D[:, 0]
    x2 = Points3D[:, 1]
    x3 = Points3D[:, 2]

    d12 = x1 - x2
    d13 = x1 - x3
    d23 = x2 - x3
    d12xd13 = np.cross(d12, d13)

    a12 = np.linalg.norm(d12) ** 2
    a13 = np.linalg.norm(d13) ** 2
    a23 = np.linalg.norm(d23) ** 2

    c31 = -0.5 * b13
    c23 = -0.5 * b23
    c12 = -0.5 * b12
    blob = c12 * c23 * c31 - 1

    s31_squared = 1 - c31 * c31
    s23_squared = 1 - c23 * c23
    s12_squared = 1 - c12 * c12

    p3 = (a13 * (a23 * s31_squared - a13 * s23_squared))
    p2 = 2.0 * blob * a23 * a13 + a13 * (2.0 * a12 + a13) * s23_squared + a23 * (a23 - a12) * s31_squared
    p1 = a23 * (a13 - a23) * s12_squared - a12 * a12 * s23_squared - 2.0 * a12 * (blob * a23 + a13 * s23_squared)
    p0 = a12 * (a12 * s23_squared - a23 * s12_squared)

    if abs(p3) >= abs(p0):
        p3inv = 1.0 / p3
        p2 *= p3inv
        p1 *= p3inv
        p0 *= p3inv
        g = cubick(p2, p1, p0)
    else:
        g = 1 / cubick(p1 / p0, p2 / p0, p3 / p0)

    A00 = a23 * (1.0 - g)
    A01 = (a23 * b12) * 0.5
    A02 = (a23 * b13 * g) * (-0.5)
    A11 = a23 - a12 + a13 * g
    A12 = b23 * (a13 * g - a12) * 0.5
    A22 = g * (a13 - a23) - a12
    A = np.array([[A00, A01, A02], [A01, A11, A12], [A02, A12, A22]])

    V, L = eigwithknown0(A)
    v = np.sqrt(max(0, -L[1] / L[0]))

    valid = 0
    Ls = np.zeros((3, 4))

    for sgn in [1, -1]:
        s = sgn * v
        w2 = 1 / (s * V[0, 1] - V[0, 0])
        w0 = (V[1, 0] - s * V[1, 1]) * w2
        w1 = (V[2, 0] - s * V[2, 1]) * w2

        a = 1 / ((a13 - a23) * w1 * w1 - a23 * b13 * w1 - a23)
        b = (a13 * b12 * w1 - a23 * b13 * w0 - 2 * w0 * w1 * (a23 - a13)) * a
        c = ((a13 - a23) * w0 * w0 + a13 * b12 * w0 + a13) * a

        disc_ok = b * b - 4 * c >= 0
        if disc_ok:
            tau1, tau2 = root2real(b, c)
            for tau in [tau1, tau2]:
                if tau > 0:
                    d = a12 / (tau * (b12 + tau) + 1)
                    l2 = np.sqrt(d)
                    l3 = tau * l2
                    l1 = w0 * l2 + w1 * l3
                    if l1 >= 0:
                        valid += 1
                        Ls[:, valid - 1] = [l1, l2, l3]

    for i in range(valid):
        Ls[:, i] = gauss_newton_refineL(Ls[:, i], a12, a13, a23, b12, b13, b23, iterations=5)

    R = np.zeros((3, 3, valid))
    t = np.zeros((3, valid))
    X = np.linalg.inv(np.column_stack([d12, d13, d12xd13]))

    for i in range(valid):
        ry1 = y1 * Ls[0, i]
        ry2 = y2 * Ls[1, i]
        ry3 = y3 * Ls[2, i]

        yd1 = ry1 - ry2
        yd2 = ry1 - ry3
        yd1xd2 = np.cross(yd1, yd2)

        Y = np.column_stack([yd1, yd2, yd1xd2])
        R[:, :, i] = Y @ X
        t[:, i] = ry1 - R[:, :, i] @ x1

    if valid == 0:
        return np.zeros((3, 3, 0)), np.zeros((3, 0))
    return R, t


def root2real(b, c):
    v = b * b - 4.0 * c
    if v < 0:
        r1 = 0.5 * b
        r2 = r1
    else:
        y = np.sqrt(v)
        if b < 0:
            r1 = 0.5 * (-b + y)
            r2 = 0.5 * (-b - y)
        else:
            r1 = 2.0 * c / (-b + y)
            r2 = 2.0 * c / (-b - y)
    return r1, r2


def cubick(b, c, d, iterations: int = 50):
    if b * b >= 3 * c:
        v = np.sqrt(b * b - 3 * c)
        t1 = (-b - v) / 3
        k = ((t1 + b) * t1 + c) * t1 + d
        if k > 0:
            r0 = t1 - np.sqrt(-k / (3 * t1 + b))
        else:
            t2 = (-b + v) / 3
            k = ((t2 + b) * t2 + c) * t2 + d
            r0 = t2 + np.sqrt(-k / (3.0 * t2 + b))
    else:
        r0 = -b / 3
        if abs((3 * r0 + 2 * b) * r0 + c) < 1e-4:
            r0 = r0 + 1

    for _ in range(iterations):
        fx = (((r0 + b) * r0 + c) * r0 + d)
        if abs(fx) > np.finfo(float).eps:
            fpx = (3 * r0 + 2 * b) * r0 + c
            r0 = r0 - fx / fpx
        else:
            break
    return r0


def eigwithknown0(x: np.ndarray):
    L = np.zeros(3)
    L[2] = 0
    v3 = np.array(
        [
            x[0, 1] * x[1, 2] - x[0, 2] * x[1, 1],
            x[0, 2] * x[1, 0] - x[0, 0] * x[1, 2],
            x[0, 0] * x[1, 1] - x[0, 1] * x[1, 0],
        ]
    )
    v3 = v3 / np.linalg.norm(v3)

    x01_squared = x[0, 1] ** 2
    b = -x[0, 0] - x[1, 1] - x[2, 2]
    c = -x01_squared - x[0, 2] ** 2 - x[1, 2] ** 2 + x[0, 0] * (x[1, 1] + x[2, 2]) + x[1, 1] * x[2, 2]
    e1, e2 = root2real(b, c)
    if abs(e1) < abs(e2):
        e1, e2 = e2, e1
    L[0] = e1
    L[1] = e2

    mx0011 = -x[0, 0] * x[1, 1]
    prec_0 = x[0, 1] * x[1, 2] - x[0, 2] * x[1, 1]
    prec_1 = x[0, 1] * x[0, 2] - x[0, 0] * x[1, 2]

    e = e1
    tmp = 1.0 / (e * (x[0, 0] + x[1, 1]) + mx0011 - e * e + x01_squared)
    a1 = -(e * x[0, 2] + prec_0) * tmp
    a2 = -(e * x[1, 2] + prec_1) * tmp
    rnorm = 1.0 / np.sqrt(a1 * a1 + a2 * a2 + 1.0)
    a1 *= rnorm
    a2 *= rnorm
    v1 = np.array([a1, a2, rnorm])

    e = e2
    tmp2 = 1.0 / (e * (x[0, 0] + x[1, 1]) + mx0011 - e * e + x01_squared)
    a21 = -(e * x[0, 2] + prec_0) * tmp2
    a22 = -(e * x[1, 2] + prec_1) * tmp2
    rnorm2 = 1.0 / np.sqrt(a21 * a21 + a22 * a22 + 1.0)
    a21 *= rnorm2
    a22 *= rnorm2
    v2 = np.array([a21, a22, rnorm2])

    E = np.column_stack([v1, v2, v3])
    return E, L


def gauss_newton_refineL(L, a12, a13, a23, b12, b13, b23, iterations=5):
    L = L.copy()
    for _ in range(iterations):
        l1, l2, l3 = L
        r1 = l1 * l1 + l2 * l2 + b12 * l1 * l2 - a12
        r2 = l1 * l1 + l3 * l3 + b13 * l1 * l3 - a13
        r3 = l2 * l2 + l3 * l3 + b23 * l2 * l3 - a23
        if abs(r1) + abs(r2) + abs(r3) < 1e-10:
            break

        dr1dl1 = 2 * l1 + b12 * l2
        dr1dl2 = 2 * l2 + b12 * l1

        dr2dl1 = 2 * l1 + b13 * l3
        dr2dl3 = 2 * l3 + b13 * l1

        dr3dl2 = 2 * l2 + b23 * l3
        dr3dl3 = 2 * l3 + b23 * l2

        r = np.array([r1, r2, r3])

        v0 = dr1dl1
        v1 = dr1dl2
        v3 = dr2dl1
        v5 = dr2dl3
        v7 = dr3dl2
        v8 = dr3dl3
        det = 1 / (-v0 * v5 * v7 - v1 * v3 * v8)
        Ji = np.array([[-v5 * v7, -v1 * v8, v1 * v5], [-v3 * v8, v0 * v8, -v0 * v5], [v3 * v7, -v0 * v7, -v1 * v3]])
        L1 = L - det * (Ji @ r)

        l1, l2, l3 = L
        r11 = l1 * l1 + l2 * l2 + b12 * l1 * l2 - a12
        r12 = l1 * l1 + l3 * l3 + b13 * l1 * l3 - a13
        r13 = l2 * l2 + l3 * l3 + b23 * l2 * l3 - a23
        if abs(r11) + abs(r12) + abs(r13) > abs(r1) + abs(r2) + abs(r3):
            break
        else:
            L = L1
    return L
