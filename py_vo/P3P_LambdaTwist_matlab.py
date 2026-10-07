from __future__ import annotations

import numpy as np


def root2real(b, c):
    value = b * b - 4.0 * c
    if value < 0:
        return 0.5 * b, 0.5 * b
    y = np.sqrt(value)
    if b < 0:
        return 0.5 * (-b + y), 0.5 * (-b - y)
    return 2.0 * c / (-b + y), 2.0 * c / (-b - y)


def cubick(b, c, d):
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
            r0 += 1
    for count in range(1, 51):
        fx = ((r0 + b) * r0 + c) * r0 + d
        if count < 7 or abs(fx) > np.finfo(float).eps:
            fpx = (3 * r0 + 2 * b) * r0 + c
            r0 -= fx / fpx
        else:
            break
    return r0


def eigwithknown0(x):
    values = np.zeros(3)
    v3 = np.array([
        x[0, 1] * x[1, 2] - x[0, 2] * x[1, 1],
        x[0, 2] * x[1, 0] - x[0, 0] * x[1, 2],
        x[0, 0] * x[1, 1] - x[0, 1] * x[1, 0],
    ])
    v3 /= np.linalg.norm(v3)
    x01_squared = x[0, 1] ** 2
    b = -x[0, 0] - x[1, 1] - x[2, 2]
    c = (-x01_squared - x[0, 2] ** 2 - x[1, 2] ** 2
         + x[0, 0] * (x[1, 1] + x[2, 2]) + x[1, 1] * x[2, 2])
    e1, e2 = root2real(b, c)
    if abs(e1) < abs(e2):
        e1, e2 = e2, e1
    values[:2] = [e1, e2]
    mx0011 = -x[0, 0] * x[1, 1]
    prec0 = x[0, 1] * x[1, 2] - x[0, 2] * x[1, 1]
    prec1 = x[0, 1] * x[0, 2] - x[0, 0] * x[1, 2]
    vectors = []
    for e in (e1, e2):
        tmp = 1.0 / (e * (x[0, 0] + x[1, 1]) + mx0011 - e * e + x01_squared)
        a1 = -(e * x[0, 2] + prec0) * tmp
        a2 = -(e * x[1, 2] + prec1) * tmp
        rnorm = 1.0 / np.sqrt(a1 * a1 + a2 * a2 + 1.0)
        vectors.append(np.array([a1 * rnorm, a2 * rnorm, rnorm]))
    return np.column_stack([vectors[0], vectors[1], v3]), values


def gauss_newton_refineL(L, a12, a13, a23, b12, b13, b23, iterations=5):
    L = L.copy()
    for _ in range(iterations):
        l1, l2, l3 = L
        r1 = l1*l1 + l2*l2 + b12*l1*l2 - a12
        r2 = l1*l1 + l3*l3 + b13*l1*l3 - a13
        r3 = l2*l2 + l3*l3 + b23*l2*l3 - a23
        if abs(r1) + abs(r2) + abs(r3) < 1e-10:
            break
        v0, v1 = 2*l1 + b12*l2, 2*l2 + b12*l1
        v3, v5 = 2*l1 + b13*l3, 2*l3 + b13*l1
        v7, v8 = 2*l2 + b23*l3, 2*l3 + b23*l2
        det = 1 / (-v0*v5*v7 - v1*v3*v8)
        ji = np.array([[-v5*v7, -v1*v8, v1*v5], [-v3*v8, v0*v8, -v0*v5], [v3*v7, -v0*v7, -v1*v3]])
        candidate = L - det * (ji @ np.array([r1, r2, r3]))
        # MATLAB source evaluates r11/r12/r13 at L, not L1. Preserve it.
        r11 = l1*l1 + l2*l2 + b12*l1*l2 - a12
        r12 = l1*l1 + l3*l3 + b13*l1*l3 - a13
        r13 = l2*l2 + l3*l3 + b23*l2*l3 - a23
        if abs(r11) + abs(r12) + abs(r13) > abs(r1) + abs(r2) + abs(r3):
            break
        L = candidate
    return L


def P3P_LambdaTwist(Points2D: np.ndarray, Points3D: np.ndarray):
    y1, y2, y3 = [Points2D[:, i] / np.linalg.norm(Points2D[:, i]) for i in range(3)]
    b12, b13, b23 = -2*y1@y2, -2*y1@y3, -2*y2@y3
    x1, x2, x3 = [Points3D[:, i] for i in range(3)]
    d12, d13, d23 = x1-x2, x1-x3, x2-x3
    d12xd13 = np.cross(d12, d13)
    a12, a13, a23 = np.linalg.norm(d12)**2, np.linalg.norm(d13)**2, np.linalg.norm(d23)**2
    c31, c23, c12 = -0.5*b13, -0.5*b23, -0.5*b12
    blob = c12*c23*c31 - 1
    s31, s23, s12 = 1-c31*c31, 1-c23*c23, 1-c12*c12
    p3 = a13*(a23*s31-a13*s23)
    p2 = 2*blob*a23*a13 + a13*(2*a12+a13)*s23 + a23*(a23-a12)*s31
    p1 = a23*(a13-a23)*s12 - a12*a12*s23 - 2*a12*(blob*a23+a13*s23)
    p0 = a12*(a12*s23-a23*s12)
    if abs(p3) >= abs(p0):
        inv = 1.0/p3
        g = cubick(inv*p2, inv*p1, inv*p0)
    else:
        g = 1/cubick(p1/p0, p2/p0, p3/p0)
    A = np.array([
        [a23*(1-g), a23*b12*.5, a23*b13*g*(-.5)],
        [a23*b12*.5, a23-a12+a13*g, b23*(a13*g-a12)*.5],
        [a23*b13*g*(-.5), b23*(a13*g-a12)*.5, g*(a13-a23)-a12],
    ])
    V, eig = eigwithknown0(A)
    v = np.sqrt(max(0, -eig[1]/eig[0]))
    lengths = []
    for s in (v, -v):
        w2 = 1/(s*V[0, 1]-V[0, 0])
        w0 = (V[1, 0]-s*V[1, 1])*w2
        w1 = (V[2, 0]-s*V[2, 1])*w2
        # These a12/a23 terms intentionally follow the current MATLAB source.
        aa = 1/((a13-a12)*w1*w1-a12*b13*w1-a12)
        bb = (a13*b12*w1-a12*b13*w0-2*w0*w1*(a12-a13))*aa
        cc = ((a13-a12)*w0*w0+a13*b12*w0+a13)*aa
        if bb*bb-4*cc >= 0:
            for tau in root2real(bb, cc):
                if tau > 0:
                    d = a23/(tau*(b23+tau)+1)
                    if d < 0 or not np.isfinite(d):
                        continue
                    l2 = np.sqrt(d)
                    l3 = tau*l2
                    l1 = w0*l2+w1*l3
                    if l1 >= 0:
                        lengths.append(np.array([l1, l2, l3]))
    lengths = [gauss_newton_refineL(L, a12, a13, a23, b12, b13, b23, 5) for L in lengths]
    rotations, translations = [], []
    try:
        X = np.linalg.inv(np.column_stack([d12, d13, d12xd13]))
    except np.linalg.LinAlgError:
        return np.zeros((3, 3, 0)), np.zeros((3, 0))
    for L in lengths:
        ry1, ry2, ry3 = y1*L[0], y2*L[1], y3*L[2]
        yd1, yd2 = ry1-ry2, ry1-ry3
        R = np.column_stack([yd1, yd2, np.cross(yd1, yd2)]) @ X
        t = ry1-R@x1
        rotations.append(R)
        translations.append(t)
    if not rotations:
        return np.zeros((3, 3, 0)), np.zeros((3, 0))
    return np.stack(rotations, axis=2), np.column_stack(translations)
