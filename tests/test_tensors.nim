# Copyright (c) 2026 Garrett Kinman
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

# Copyright (c) 2024 Garrett Kinman
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import unittest
import steadypcn
import math
import random

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# MATRIX INITIALIZATION
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

suite "Matrix Initialization":
    test "zeros":
        let m = Matrix[float, 2, 3].zeros()
        check m.data.len == 6
        for v in m.data:
            check v == 0.0

    test "ones":
        let m = Matrix[float, 2, 3].ones()
        check m.data.len == 6
        for v in m.data:
            check v == 1.0

    test "initMatrix from flat array":
        let m = initMatrix[float, 2, 3]([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        check m.data == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

    test "rand in range":
        randomize(42)
        let m = Matrix[float, 3, 3].rand(-1.0, 1.0)
        for v in m.data:
            check v >= -1.0
            check v <= 1.0

    test "rows and cols":
        let m = Matrix[float, 3, 5].zeros()
        check m.rows == 3
        check m.cols == 5
        check m.size == 15

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# VECTOR INITIALIZATION
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

suite "Vector Initialization":
    test "zeros":
        let v = Vector[float, 4].zeros()
        check v.data.len == 4
        for x in v.data:
            check x == 0.0

    test "ones":
        let v = Vector[float, 4].ones()
        for x in v.data:
            check x == 1.0

    test "initVector from array":
        let v = initVector[float, 3]([1.0, 2.0, 3.0])
        check v[0] == 1.0
        check v[1] == 2.0
        check v[2] == 3.0

    test "rand in range":
        randomize(42)
        let v = Vector[float, 5].rand(0.0, 1.0)
        for x in v.data:
            check x >= 0.0
            check x <= 1.0

    test "len":
        let v = Vector[float, 7].zeros()
        check v.len == 7

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# MATRIX INDEXING
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

suite "Matrix Indexing":
    test "row-major read":
        # Row-major: data = [row0col0, row0col1, row1col0, row1col1]
        let m = initMatrix[float, 2, 2]([1.0, 2.0, 3.0, 4.0])
        check m[0, 0] == 1.0
        check m[0, 1] == 2.0
        check m[1, 0] == 3.0
        check m[1, 1] == 4.0

    test "read/write roundtrip":
        var m = Matrix[float, 3, 3].zeros()
        m[1, 2] = 9.0
        check m[1, 2] == 9.0
        # Verify no neighbouring cells were touched
        check m[1, 1] == 0.0
        check m[2, 2] == 0.0

    test "2x3 indexing":
        let m = initMatrix[float, 2, 3]([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        check m[0, 0] == 1.0
        check m[0, 2] == 3.0
        check m[1, 0] == 4.0
        check m[1, 2] == 6.0

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# VECTOR INDEXING
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

suite "Vector Indexing":
    test "read":
        let v = initVector[float, 3]([10.0, 20.0, 30.0])
        check v[0] == 10.0
        check v[1] == 20.0
        check v[2] == 30.0

    test "write":
        var v = Vector[float, 4].zeros()
        v[2] = 5.0
        check v[2] == 5.0
        check v[1] == 0.0

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# MATRIX UTILITIES
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

suite "Matrix Utilities":
    test "copy is independent":
        var m = initMatrix[float, 2, 2]([1.0, 2.0, 3.0, 4.0])
        var c = m.copy
        c[0, 0] = 99.0
        check m[0, 0] == 1.0   # original unchanged

    test "map":
        let m = initMatrix[float, 2, 2]([1.0, 2.0, 3.0, 4.0])
        let m2 = m.map(proc(x: float): float = x * 2)
        check m2.data == [2.0, 4.0, 6.0, 8.0]

    test "zip":
        let a = initMatrix[float, 2, 2]([1.0, 2.0, 3.0, 4.0])
        let b = initMatrix[float, 2, 2]([10.0, 20.0, 30.0, 40.0])
        let c = zip(a, b, proc(x, y: float): float = x + y)
        check c.data == [11.0, 22.0, 33.0, 44.0]

    test "$ string representation":
        let m = initMatrix[float, 2, 2]([1.0, 2.0, 3.0, 4.0])
        check $m == "Matrix[2x2](1.0, 2.0; 3.0, 4.0)"

    test "toVector flattens":
        let m = initMatrix[float, 2, 3]([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        let v = m.toVector
        check v.len == 6
        check v[0] == 1.0
        check v[5] == 6.0

    test "col extraction":
        let m = initMatrix[float, 2, 3]([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        let c = m.col(1)
        check c[0] == 2.0
        check c[1] == 5.0

    test "row extraction":
        let m = initMatrix[float, 2, 3]([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        let r = m.row(1)
        check r[0] == 4.0
        check r[1] == 5.0
        check r[2] == 6.0

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# VECTOR UTILITIES
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

suite "Vector Utilities":
    test "copy is independent":
        var v = initVector[float, 3]([1.0, 2.0, 3.0])
        var c = v.copy
        c[0] = 99.0
        check v[0] == 1.0

    test "map":
        let v = initVector[float, 3]([1.0, 2.0, 3.0])
        let v2 = v.map(proc(x: float): float = x * x)
        check v2[0] == 1.0
        check v2[1] == 4.0
        check v2[2] == 9.0

    test "zip":
        let a = initVector[float, 3]([1.0, 2.0, 3.0])
        let b = initVector[float, 3]([4.0, 5.0, 6.0])
        let c = zip(a, b, proc(x, y: float): float = x * y)
        check c[0] == 4.0
        check c[1] == 10.0
        check c[2] == 18.0

    test "$ string representation":
        let v = initVector[float, 3]([1.0, 2.0, 3.0])
        check $v == "Vector[3](1.0, 2.0, 3.0)"

    test "toMatrix reshape":
        let v = initVector[float, 6]([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        let m = v.toMatrix[:float, 6, 2, 3]()
        check m.rows == 2
        check m.cols == 3
        check m[0, 0] == 1.0
        check m[1, 2] == 6.0

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# MATRIX ARITHMETIC
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

suite "Matrix Arithmetic":
    test "element-wise add":
        let a = initMatrix[float, 2, 2]([1.0, 2.0, 3.0, 4.0])
        let b = initMatrix[float, 2, 2]([10.0, 20.0, 30.0, 40.0])
        let c = a + b
        check c.data == [11.0, 22.0, 33.0, 44.0]

    test "element-wise sub":
        let a = initMatrix[float, 2, 2]([5.0, 6.0, 7.0, 8.0])
        let b = initMatrix[float, 2, 2]([1.0, 2.0, 3.0, 4.0])
        let c = a - b
        check c.data == [4.0, 4.0, 4.0, 4.0]

    test "hadamard product (.*)":
        let a = initMatrix[float, 2, 2]([1.0, 2.0, 3.0, 4.0])
        let b = initMatrix[float, 2, 2]([2.0, 3.0, 4.0, 5.0])
        let c = a .* b
        check c.data == [2.0, 6.0, 12.0, 20.0]

    test "scalar add":
        let a = initMatrix[float, 2, 2]([1.0, 2.0, 3.0, 4.0])
        let b = a + 10.0
        check b.data == [11.0, 12.0, 13.0, 14.0]

    test "scalar sub":
        let a = initMatrix[float, 2, 2]([5.0, 6.0, 7.0, 8.0])
        let b = a - 2.0
        check b.data == [3.0, 4.0, 5.0, 6.0]

    test "scalar mul (right)":
        let a = initMatrix[float, 2, 2]([1.0, 2.0, 3.0, 4.0])
        let b = a * 3.0
        check b.data == [3.0, 6.0, 9.0, 12.0]

    test "scalar mul (left)":
        let a = initMatrix[float, 2, 2]([1.0, 2.0, 3.0, 4.0])
        let b = 3.0 * a
        check b.data == [3.0, 6.0, 9.0, 12.0]

    test "scalar div":
        let a = initMatrix[float, 2, 2]([2.0, 4.0, 6.0, 8.0])
        let b = a / 2.0
        check b.data == [1.0, 2.0, 3.0, 4.0]

    test "negation":
        let a = initMatrix[float, 2, 2]([1.0, -2.0, 3.0, -4.0])
        let b = -a
        check b.data == [-1.0, 2.0, -3.0, 4.0]

    test "in-place +=":
        var a = initMatrix[float, 2, 2]([1.0, 2.0, 3.0, 4.0])
        let b = initMatrix[float, 2, 2]([1.0, 1.0, 1.0, 1.0])
        a += b
        check a.data == [2.0, 3.0, 4.0, 5.0]

    test "in-place -=":
        var a = initMatrix[float, 2, 2]([5.0, 6.0, 7.0, 8.0])
        let b = initMatrix[float, 2, 2]([1.0, 2.0, 3.0, 4.0])
        a -= b
        check a.data == [4.0, 4.0, 4.0, 4.0]

    test "in-place .*=":
        var a = initMatrix[float, 2, 2]([1.0, 2.0, 3.0, 4.0])
        let b = initMatrix[float, 2, 2]([2.0, 2.0, 2.0, 2.0])
        a .*= b
        check a.data == [2.0, 4.0, 6.0, 8.0]

    test "in-place scalar +=":
        var a = initMatrix[float, 2, 2]([1.0, 2.0, 3.0, 4.0])
        a += 5.0
        check a.data == [6.0, 7.0, 8.0, 9.0]

    test "in-place scalar *=":
        var a = initMatrix[float, 2, 2]([1.0, 2.0, 3.0, 4.0])
        a *= 2.0
        check a.data == [2.0, 4.0, 6.0, 8.0]

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# VECTOR ARITHMETIC
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

suite "Vector Arithmetic":
    test "element-wise add":
        let a = initVector[float, 3]([1.0, 2.0, 3.0])
        let b = initVector[float, 3]([4.0, 5.0, 6.0])
        let c = a + b
        check c.data == [5.0, 7.0, 9.0]

    test "element-wise sub":
        let a = initVector[float, 3]([4.0, 5.0, 6.0])
        let b = initVector[float, 3]([1.0, 2.0, 3.0])
        let c = a - b
        check c.data == [3.0, 3.0, 3.0]

    test "hadamard product (.*)":
        let a = initVector[float, 3]([1.0, 2.0, 3.0])
        let b = initVector[float, 3]([4.0, 5.0, 6.0])
        let c = a .* b
        check c.data == [4.0, 10.0, 18.0]

    test "scalar mul":
        let a = initVector[float, 3]([1.0, 2.0, 3.0])
        let b = a * 2.0
        check b.data == [2.0, 4.0, 6.0]

    test "negation":
        let a = initVector[float, 3]([1.0, -2.0, 3.0])
        let b = -a
        check b.data == [-1.0, 2.0, -3.0]

    test "in-place +=":
        var a = initVector[float, 3]([1.0, 2.0, 3.0])
        let b = initVector[float, 3]([1.0, 1.0, 1.0])
        a += b
        check a.data == [2.0, 3.0, 4.0]

    test "in-place -=":
        var a = initVector[float, 3]([5.0, 6.0, 7.0])
        let b = initVector[float, 3]([1.0, 2.0, 3.0])
        a -= b
        check a.data == [4.0, 4.0, 4.0]

    test "in-place scalar +=":
        var a = initVector[float, 3]([1.0, 2.0, 3.0])
        a += 10.0
        check a.data == [11.0, 12.0, 13.0]

    test "in-place scalar *=":
        var a = initVector[float, 3]([1.0, 2.0, 3.0])
        a *= 3.0
        check a.data == [3.0, 6.0, 9.0]

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# MATRIX MULTIPLICATION
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

suite "Matrix Multiplication":
    test "2x2 identity":
        var I = Matrix[float, 2, 2].zeros()
        I[0, 0] = 1.0; I[1, 1] = 1.0

        var A = initMatrix[float, 2, 2]([1.0, 2.0, 3.0, 4.0])

        let C = I * A
        check C[0, 0] == 1.0
        check C[0, 1] == 2.0
        check C[1, 0] == 3.0
        check C[1, 1] == 4.0

    test "2x3 × 3x2":
        # A = [[1, 2, 3],   B = [[7,  8],    C = [[58,  64],
        #      [4, 5, 6]]        [9,  10],         [139, 154]]
        #                        [11, 12]]
        let A = initMatrix[float, 2, 3]([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        let B = initMatrix[float, 3, 2]([7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
        let C = A * B

        check abs(C[0, 0] - 58.0)  < 1e-10   # 1*7 + 2*9 + 3*11
        check abs(C[0, 1] - 64.0)  < 1e-10   # 1*8 + 2*10 + 3*12
        check abs(C[1, 0] - 139.0) < 1e-10   # 4*7 + 5*9 + 6*11
        check abs(C[1, 1] - 154.0) < 1e-10   # 4*8 + 5*10 + 6*12

    test "1x1":
        let A = initMatrix[float, 1, 1]([3.0])
        let B = initMatrix[float, 1, 1]([4.0])
        let C = A * B
        check abs(C[0, 0] - 12.0) < 1e-10

    test "multiply by zero matrix":
        let A = initMatrix[float, 2, 2]([1.0, 2.0, 3.0, 4.0])
        let Z = Matrix[float, 2, 2].zeros()
        let C = A * Z
        check C[0, 0] == 0.0
        check C[0, 1] == 0.0
        check C[1, 0] == 0.0
        check C[1, 1] == 0.0

    test "non-square 3x4 × 4x2":
        # Fill A[i,j] = i+j, B[i,j] = i+j and verify associativity holds
        # against a second equivalent computation
        var A = Matrix[float, 3, 4].zeros()
        var B = Matrix[float, 4, 2].zeros()
        for i in 0..2:
            for j in 0..3:
                A[i, j] = float(i + j)
        for i in 0..3:
            for j in 0..1:
                B[i, j] = float(i + j)
        let C1 = A * B
        let C2 = matmul(A, B)
        for i in 0..2:
            for j in 0..1:
                check abs(C1[i, j] - C2[i, j]) < 1e-10

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# MATRIX-VECTOR MULTIPLICATION
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

suite "Matrix-Vector Multiplication":
    test "2x3 × (3,)":
        # W = [[1, 2, 3],   x = [1, 0, -1]
        #      [4, 5, 6]]
        # y[0] = 1*1 + 2*0 + 3*(-1) = -2
        # y[1] = 4*1 + 5*0 + 6*(-1) = -2
        let W = initMatrix[float, 2, 3]([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        let x = initVector[float, 3]([1.0, 0.0, -1.0])
        let y = W * x
        check abs(y[0] - (-2.0)) < 1e-10
        check abs(y[1] - (-2.0)) < 1e-10

    test "identity matrix × vector passes through":
        var I = Matrix[float, 3, 3].zeros()
        I[0, 0] = 1.0; I[1, 1] = 1.0; I[2, 2] = 1.0
        let x = initVector[float, 3]([3.0, 7.0, 2.0])
        let y = I * x
        check y[0] == 3.0
        check y[1] == 7.0
        check y[2] == 2.0

    test "zero matrix × vector is zero":
        let W = Matrix[float, 3, 4].zeros()
        let x = initVector[float, 4]([1.0, 2.0, 3.0, 4.0])
        let y = W * x
        for i in 0..2:
            check y[i] == 0.0

    test "operator * dispatches to mvMul":
        # Verify `*` and explicit `mvMul` give identical results
        let W = initMatrix[float, 2, 3]([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        let x = initVector[float, 3]([1.0, 1.0, 1.0])
        let y1 = W * x
        let y2 = mvMul(W, x)
        check y1[0] == y2[0]
        check y1[1] == y2[1]

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# TRANSPOSE
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

suite "TransposedMatrix":
    test "t() is zero-copy — same data":
        let A = initMatrix[float, 2, 3]([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        let At = A.t
        # The underlying data array must be identical
        check Matrix[float, 2, 3](At).data == A.data

    test "[] swaps indices":
        # A = [[1, 2, 3],   A.T = [[1, 4],
        #      [4, 5, 6]]          [2, 5],
        #                          [3, 6]]
        let A = initMatrix[float, 2, 3]([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        let At = A.t
        check At[0, 0] == 1.0
        check At[0, 1] == 4.0
        check At[1, 0] == 2.0
        check At[1, 1] == 5.0
        check At[2, 0] == 3.0
        check At[2, 1] == 6.0

    test "double transpose is identity":
        let A  = initMatrix[float, 2, 3]([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        let At = A.t
        # A.t.t should index identically to A
        for i in 0..1:
            for j in 0..2:
                check At.t[i, j] == A[i, j]

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# TRANSPOSED MATRIX MULTIPLICATION  (Aᵀ * B)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

suite "Transposed Matrix Multiplication":
    test "square: A.t * I = A.t":
        # A = [[1, 2],   A.T = [[1, 3],
        #      [3, 4]]          [2, 4]]
        let A = initMatrix[float, 2, 2]([1.0, 2.0, 3.0, 4.0])
        var I = Matrix[float, 2, 2].zeros()
        I[0, 0] = 1.0; I[1, 1] = 1.0

        let C = A.t * I
        check C[0, 0] == 1.0
        check C[0, 1] == 3.0   # was A[1,0]
        check C[1, 0] == 2.0   # was A[0,1]
        check C[1, 1] == 4.0

    test "rectangular: (3x2).t * (3x1) = (2x1)":
        # A = [[1,  10],   A.T = [[1, 2, 3],    B = [[1],
        #      [2,  20],          [10,20,30]]         [1],
        #      [3,  30]]                               [1]]
        # A.T * B = [[6], [60]]
        let A = initMatrix[float, 3, 2]([1.0, 10.0, 2.0, 20.0, 3.0, 30.0])
        let B = initMatrix[float, 3, 1]([1.0, 1.0, 1.0])
        let C = A.t * B
        check C.rows == 2
        check C.cols == 1
        check abs(C[0, 0] - 6.0)  < 1e-10
        check abs(C[1, 0] - 60.0) < 1e-10

    test "matches manual matmulT call":
        let A = initMatrix[float, 3, 2]([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        let B = initMatrix[float, 3, 2]([1.0, 0.0, 0.0, 1.0, 1.0, 0.0])
        let C1 = A.t * B
        let C2 = matmulT(A.t, B)
        for i in 0..1:
            for j in 0..1:
                check abs(C1[i, j] - C2[i, j]) < 1e-10

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# TRANSPOSED MATRIX-VECTOR MULTIPLICATION  (Wᵀ * x)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

suite "Transposed Matrix-Vector Multiplication":
    test "W.t * x basic":
        # W = [[1, 2, 3],   W.T = [[1, 4],   x = [1, 1]
        #      [4, 5, 6]]          [2, 5],
        #                          [3, 6]]
        # W.T * x = [1+4, 2+5, 3+6] = [5, 7, 9]
        let W = initMatrix[float, 2, 3]([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        let x = initVector[float, 2]([1.0, 1.0])
        let y = W.t * x
        check abs(y[0] - 5.0) < 1e-10
        check abs(y[1] - 7.0) < 1e-10
        check abs(y[2] - 9.0) < 1e-10

    test "W.t * x with identity weight":
        var I = Matrix[float, 3, 3].zeros()
        I[0, 0] = 1.0; I[1, 1] = 1.0; I[2, 2] = 1.0
        let x = initVector[float, 3]([4.0, 5.0, 6.0])
        let y = I.t * x
        check y[0] == 4.0
        check y[1] == 5.0
        check y[2] == 6.0

    test "matches manual mvMulT call":
        let W = initMatrix[float, 2, 3]([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        let x = initVector[float, 2]([2.0, 3.0])
        let y1 = W.t * x
        let y2 = mvMulT(W.t, x)
        for i in 0..2:
            check abs(y1[i] - y2[i]) < 1e-10

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# DOT AND OUTER PRODUCTS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

suite "Dot and Outer Products":
    test "dot product":
        let a = initVector[float, 3]([1.0, 2.0, 3.0])
        let b = initVector[float, 3]([4.0, 5.0, 6.0])
        check abs(dot(a, b) - 32.0) < 1e-10   # 1*4 + 2*5 + 3*6

    test "dot product with orthogonal vectors is zero":
        let a = initVector[float, 2]([1.0, 0.0])
        let b = initVector[float, 2]([0.0, 1.0])
        check abs(dot(a, b)) < 1e-10

    test "outer product shape and values":
        # a = [1, 2],  b = [3, 4, 5]
        # outer = [[1*3, 1*4, 1*5],
        #          [2*3, 2*4, 2*5]]
        let a = initVector[float, 2]([1.0, 2.0])
        let b = initVector[float, 3]([3.0, 4.0, 5.0])
        let O = outer(a, b)
        check O.rows == 2
        check O.cols == 3
        check O[0, 0] == 3.0
        check O[0, 1] == 4.0
        check O[0, 2] == 5.0
        check O[1, 0] == 6.0
        check O[1, 1] == 8.0
        check O[1, 2] == 10.0

    test "outer product is consistent with mvMul":
        # outer(a, b) * b should equal dot(b,b) * a for unit-ish vectors
        let a = initVector[float, 2]([1.0, 0.0])
        let b = initVector[float, 2]([1.0, 0.0])
        let O = outer(a, b)
        let y = O * b
        check abs(y[0] - dot(b, b)) < 1e-10
        check abs(y[1] - 0.0) < 1e-10