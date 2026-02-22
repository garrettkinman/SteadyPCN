# Copyright (c) 2026 Garrett Kinman
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

# tests/test_activations.nim
import unittest, math
import steadypcn

const eps = 1e-6

suite "Sigmoid":
    let s = Sigmoid()

    test "activate: σ(0) = 0.5":
        check abs(s.activate(0.0) - 0.5) < eps

    test "activate: σ(x) bounded in (0, 1)":
        for x in [-100.0, -10.0, -1.0, 0.0, 1.0, 10.0, 100.0]:
            let y = s.activate(x)
            check y >= 0.0 and y <= 1.0

    test "activate: σ(x) is monotonically increasing":
        check s.activate(-1.0) < s.activate(0.0)
        check s.activate(0.0)  < s.activate(1.0)

    test "activate: σ(-x) = 1 - σ(x)":
        for x in [-2.0, -0.5, 0.5, 2.0]:
            check abs(s.activate(-x) - (1.0 - s.activate(x))) < eps

    test "grad: σ'(0) = 0.25":
        check abs(s.grad(0.0) - 0.25) < eps

    test "grad: always positive":
        for x in [-10.0, -1.0, 0.0, 1.0, 10.0]:
            check s.grad(x) > 0.0

    test "grad: matches numerical gradient":
        let h = 1e-5
        for x in [-2.0, -1.0, 0.0, 1.0, 2.0]:
            let numerical = (s.activate(x + h) - s.activate(x - h)) / (2.0 * h)
            check abs(s.grad(x) - numerical) < 1e-4

suite "Tanh":
    let t = Tanh()

    test "activate: tanh(0) = 0":
        check abs(t.activate(0.0)) < eps

    test "activate: bounded in (-1, 1)":
        for x in [-100.0, -10.0, -1.0, 0.0, 1.0, 10.0, 100.0]:
            let y = t.activate(x)
            check y >= -1.0 and y <= 1.0

    test "activate: odd function — tanh(-x) = -tanh(x)":
        for x in [-2.0, -0.5, 0.5, 2.0]:
            check abs(t.activate(-x) + t.activate(x)) < eps

    test "grad: tanh'(0) = 1":
        check abs(t.grad(0.0) - 1.0) < eps

    test "grad: always positive":
        for x in [-10.0, -1.0, 0.0, 1.0, 10.0]:
            check t.grad(x) > 0.0

    test "grad: matches numerical gradient":
        let h = 1e-5
        for x in [-2.0, -1.0, 0.0, 1.0, 2.0]:
            let numerical = (t.activate(x + h) - t.activate(x - h)) / (2.0 * h)
            check abs(t.grad(x) - numerical) < 1e-4

suite "ReLU":
    let r = ReLU()

    test "activate: ReLU(0) = 0":
        check r.activate(0.0) == 0.0

    test "activate: identity for positive inputs":
        for x in [0.1, 1.0, 5.0, 100.0]:
            check r.activate(x) == x

    test "activate: zero for negative inputs":
        for x in [-0.1, -1.0, -5.0, -100.0]:
            check r.activate(x) == 0.0

    test "grad: 1 for positive inputs":
        for x in [0.1, 1.0, 5.0]:
            check r.grad(x) == 1.0

    test "grad: 0 for negative inputs":
        for x in [-0.1, -1.0, -5.0]:
            check r.grad(x) == 0.0

suite "Identity":
    let id = Identity()

    test "activate: returns input unchanged":
        for x in [-100.0, -1.0, 0.0, 1.0, 100.0]:
            check id.activate(x) == x

    test "grad: always 1":
        for x in [-100.0, -1.0, 0.0, 1.0, 100.0]:
            check id.grad(x) == 1.0

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Vectorized overloads
# Each suite verifies:
#   1. Output length matches input length.
#   2. Each element matches the corresponding scalar overload.
#   3. Any activation-specific property that is meaningful at the vector level.
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

suite "Sigmoid – vectorized":
    let s = Sigmoid()
    let v = initVector[float, 5]([-2.0, -1.0, 0.0, 1.0, 2.0])

    test "activate: output length matches input":
        check s.activate(v).len == v.len

    test "activate: each element matches scalar overload":
        let r = s.activate(v)
        for i in 0 ..< v.len:
            check abs(r[i] - s.activate(v[i])) < eps

    test "activate: all outputs bounded in (0, 1)":
        let r = s.activate(v)
        for i in 0 ..< r.len:
            check r[i] > 0.0 and r[i] < 1.0

    test "grad: output length matches input":
        check s.grad(v).len == v.len

    test "grad: each element matches scalar overload":
        let r = s.grad(v)
        for i in 0 ..< v.len:
            check abs(r[i] - s.grad(v[i])) < eps

    test "grad: all outputs positive":
        let r = s.grad(v)
        for i in 0 ..< r.len:
            check r[i] > 0.0

suite "Tanh – vectorized":
    let t = Tanh()
    let v = initVector[float, 5]([-2.0, -1.0, 0.0, 1.0, 2.0])

    test "activate: output length matches input":
        check t.activate(v).len == v.len

    test "activate: each element matches scalar overload":
        let r = t.activate(v)
        for i in 0 ..< v.len:
            check abs(r[i] - t.activate(v[i])) < eps

    test "activate: odd function holds element-wise":
        # activate(-v) should equal -activate(v) for each element
        let vNeg = initVector[float, 4]([-2.0, -1.0, 1.0, 2.0])
        let vPos = initVector[float, 4]([ 2.0,  1.0,-1.0,-2.0])
        let rNeg = t.activate(vNeg)
        let rPos = t.activate(vPos)
        for i in 0 ..< rNeg.len:
            check abs(rNeg[i] + rPos[i]) < eps

    test "grad: output length matches input":
        check t.grad(v).len == v.len

    test "grad: each element matches scalar overload":
        let r = t.grad(v)
        for i in 0 ..< v.len:
            check abs(r[i] - t.grad(v[i])) < eps

    test "grad: all outputs positive":
        let r = t.grad(v)
        for i in 0 ..< r.len:
            check r[i] > 0.0

suite "ReLU – vectorized":
    let r = ReLU()
    let v = initVector[float, 6]([-3.0, -1.0, 0.0, 0.0, 1.0, 3.0])

    test "activate: output length matches input":
        check r.activate(v).len == v.len

    test "activate: each element matches scalar overload":
        let res = r.activate(v)
        for i in 0 ..< v.len:
            check res[i] == r.activate(v[i])

    test "activate: negative inputs produce zero":
        let res = r.activate(v)
        check res[0] == 0.0
        check res[1] == 0.0

    test "activate: positive inputs pass through":
        let res = r.activate(v)
        check res[4] == 1.0
        check res[5] == 3.0

    test "grad: output length matches input":
        check r.grad(v).len == v.len

    test "grad: each element matches scalar overload":
        let res = r.grad(v)
        for i in 0 ..< v.len:
            check res[i] == r.grad(v[i])

    test "grad: 0 for negative, 1 for positive":
        let res = r.grad(v)
        check res[0] == 0.0   # -3
        check res[1] == 0.0   # -1
        check res[4] == 1.0   #  1
        check res[5] == 1.0   #  3

suite "Identity – vectorized":
    let id = Identity()
    let v  = initVector[float, 4]([-5.0, 0.0, 1.0, 42.0])

    test "activate: output length matches input":
        check id.activate(v).len == v.len

    test "activate: returns vector unchanged":
        let res = id.activate(v)
        for i in 0 ..< v.len:
            check res[i] == v[i]

    test "grad: output length matches input":
        check id.grad(v).len == v.len

    test "grad: all elements are 1":
        let res = id.grad(v)
        for i in 0 ..< res.len:
            check res[i] == 1.0