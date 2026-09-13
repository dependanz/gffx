"""Independent scalar bilinear transpose and colour-affine image objective.

No tensor framework, public sampler, shader, or automatic differentiation here.
"""
import math


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError('finite real value required')
    return float(value)


def stencil(height, width, uv):
    if any(type(x) is not int or not 1 <= x <= 32 for x in (height, width)):
        raise ValueError('bounded positive texture dimensions required')
    if len(uv) != 2: raise ValueError('UV pair required')
    x, y = (_number(uv[0]) % 1)*width-.5, (_number(uv[1]) % 1)*height-.5
    ix, iy = math.floor(x), math.floor(y)
    fx, fy = x-ix, y-iy
    return [(iy % height, ix % width, (1-fx)*(1-fy)),
            (iy % height, (ix+1) % width, fx*(1-fy)),
            ((iy+1) % height, ix % width, (1-fx)*fy),
            ((iy+1) % height, (ix+1) % width, fx*fy)]


def evaluate(texture, weights, intercepts, slopes, target):
    h = len(texture); w = len(texture[0]) if h else 0
    if not 1 <= h <= 32 or not 1 <= w <= 32: raise ValueError('bounded texture required')
    for row in texture:
        if len(row) != w: raise ValueError('rectangular texture required')
        for rgb in row:
            if len(rgb) != 3 or any(not 0 <= _number(x) <= 1 for x in rgb):
                raise ValueError('unit-domain RGB texture required')
    p = len(weights)
    if not 1 <= p <= 6536 or any(len(x) != p for x in (intercepts, slopes, target)):
        raise ValueError('nonempty matching covered-pixel fields required')
    gradient = [[[0.]*3 for _ in range(w)] for _ in range(h)]
    values = []; squared = []
    for taps, b, s, t in zip(weights, intercepts, slopes, target):
        if len(taps) != 4 or any(len(x) != 3 for x in (b, s, t)):
            raise ValueError('four taps and RGB fields required')
        for y, x, weight in taps:
            if type(y) is not int or type(x) is not int or not (0 <= y < h and 0 <= x < w):
                raise ValueError('invalid texel index')
            if not 0 <= _number(weight) <= 1: raise ValueError('invalid tap weight')
        if abs(math.fsum(t[2] for t in taps)-1.) > 1e-14: raise ValueError('tap partition failed')
        rgb = []
        for c in range(3):
            a = math.fsum(texture[y][x][c]*weight for y, x, weight in taps)
            value = _number(b[c])+_number(s[c])*a
            residual = value-_number(t[c]); squared.append(residual*residual); rgb.append(value)
            derivative = 2*residual*s[c]/(3*p)
            for y, x, weight in taps: gradient[y][x][c] += derivative*weight
        values.append(rgb)
    return dict(loss=math.fsum(squared)/(3*p), gradient=gradient, values=values)


def directions(h, w):
    return [[[[1. if c == 0 else 0. for c in range(3)] for x in range(w)] for y in range(h)],
            [[[(-1. if (x+y) % 2 else 1.) if c == 1 else 0. for c in range(3)] for x in range(w)] for y in range(h)],
            [[[(2*(x+.5)/w-1.) if c == 2 else 0. for c in range(3)] for x in range(w)] for y in range(h)]]


def finite_differences(texture, weights, intercepts, slopes, target, gradient):
    h, w = len(texture), len(texture[0]); rows = []
    for name, direction in zip(('red constant', 'green checkerboard', 'blue x ramp'), directions(h, w)):
        analytic = math.fsum(gradient[y][x][c]*direction[y][x][c]
                            for y in range(h) for x in range(w) for c in range(3))
        estimates = []
        for eps in (1e-4, 5e-5):
            losses = []
            for sign in (-1, 1):
                perturbed = [[[texture[y][x][c]+sign*eps*direction[y][x][c]
                               for c in range(3)] for x in range(w)] for y in range(h)]
                losses.append(evaluate(perturbed, weights, intercepts, slopes, target)['loss'])
            estimates.append((losses[1]-losses[0])/(2*eps))
        tolerance = 1e-10+1e-6*abs(analytic)
        passed = all(abs(x-analytic) <= tolerance for x in estimates) and abs(estimates[0]-estimates[1]) <= tolerance
        rows.append(dict(direction=name, analytic=analytic, scales=[1e-4, 5e-5], estimates=estimates, passed=passed))
    return rows
