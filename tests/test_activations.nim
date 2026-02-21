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