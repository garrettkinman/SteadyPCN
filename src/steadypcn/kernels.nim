# Copyright (c) 2026 Garrett Kinman
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import tensors

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# MATRIX MULTIPLICATION  (C = A * B)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

func matmul*[T; M, K, N: static int](
    A: Matrix[T, M, K],
    B: Matrix[T, K, N]
): Matrix[T, M, N] =
    ## General matrix multiplication: (M×K) * (K×N) → (M×N).
    ## Loop order is optimised for the active memory layout.

    when ColMajor:
        # JIK loop — streams down columns of C and A (contiguous in col-major).
        for j in 0 ..< N:
            for k in 0 ..< K:
                let bVal = B[k, j]
                for i in 0 ..< M:
                    result[i, j] = result[i, j] + A[i, k] * bVal
    else:
        # IKJ loop — streams across rows of C and B (contiguous in row-major).
        for i in 0 ..< M:
            for k in 0 ..< K:
                let aVal = A[i, k]
                for j in 0 ..< N:
                    result[i, j] = result[i, j] + aVal * B[k, j]

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# MATRIX-VECTOR MULTIPLICATION  (y = W * x)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

func mvMul*[T; M, K: static int](
    W: Matrix[T, M, K],
    x: Vector[T, K]
): Vector[T, M] =
    ## Optimised matrix-vector multiply: (M×K) * (K,) → (M,).
    ## In col-major mode, sparse inputs are cheap due to the sparsity check.

    when ColMajor:
        # Column-scaling: y += x[k] * W[:,k]  — streams down contiguous columns.
        for k in 0 ..< K:
            let xVal = x[k]
            if xVal != T(0):          # cheap sparsity short-circuit
                for i in 0 ..< M:
                    result[i] = result[i] + W[i, k] * xVal
    else:
        # Dot-product per row: streams across contiguous rows of W.
        for i in 0 ..< M:
            var s = T(0)
            for k in 0 ..< K:
                s += W[i, k] * x[k]
            result[i] = s

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# TRANSPOSED MATRIX MULTIPLICATION  (C = A.T * B)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

func matmulT*[T; M, K, N: static int](
    A: TransposedMatrix[T, K, M],   # logically K×M, stored as Matrix[T, M, K]
    B: Matrix[T, M, N]
): Matrix[T, K, N] =
    ## C = Aᵀ * B without allocating a transposed copy.
    ## A is stored as (M×K); TransposedMatrix[] handles the index swap.

    when ColMajor:
        for j in 0 ..< N:
            for i in 0 ..< K:
                var s = T(0)
                for m in 0 ..< M:
                    s += A[i, m] * B[m, j]
                result[i, j] = s
    else:
        for m in 0 ..< M:
            for i in 0 ..< K:
                let aVal = A[i, m]
                for j in 0 ..< N:
                    result[i, j] = result[i, j] + aVal * B[m, j]

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# TRANSPOSED MATRIX-VECTOR MULTIPLICATION  (y = W.T * x)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

func mvMulT*[T; M, K: static int](
    W: TransposedMatrix[T, K, M],   # logically K×M, stored as Matrix[T, M, K]
    x: Vector[T, M]
): Vector[T, K] =
    ## y = Wᵀ * x without allocating a transposed copy.

    when ColMajor:
        for k in 0 ..< K:
            var s = T(0)
            for m in 0 ..< M:
                s += W[k, m] * x[m]
            result[k] = s
    else:
        for m in 0 ..< M:
            let xVal = x[m]
            if xVal != T(0):
                for k in 0 ..< K:
                    result[k] = result[k] + W[k, m] * xVal

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ELEMENT-WISE KERNELS  (Matrix)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Out-parameter style: `res` may alias `a` or `b`, enabling in-place ops.

func add*[T; M, N: static int](a, b: Matrix[T, M, N], res: var Matrix[T, M, N]) {.inline.} =
    for i in 0 ..< M * N:
        res.data[i] = a.data[i] + b.data[i]

func sub*[T; M, N: static int](a, b: Matrix[T, M, N], res: var Matrix[T, M, N]) {.inline.} =
    for i in 0 ..< M * N:
        res.data[i] = a.data[i] - b.data[i]

func mul*[T; M, N: static int](a, b: Matrix[T, M, N], res: var Matrix[T, M, N]) {.inline.} =
    ## Element-wise (Hadamard) product.
    for i in 0 ..< M * N:
        res.data[i] = a.data[i] * b.data[i]

func scale*[T; M, N: static int](a: Matrix[T, M, N], s: T, res: var Matrix[T, M, N]) {.inline.} =
    for i in 0 ..< M * N:
        res.data[i] = a.data[i] * s

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ELEMENT-WISE KERNELS  (Vector)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

func add*[T; N: static int](a, b: Vector[T, N], res: var Vector[T, N]) {.inline.} =
    for i in 0 ..< N:
        res.data[i] = a.data[i] + b.data[i]

func sub*[T; N: static int](a, b: Vector[T, N], res: var Vector[T, N]) {.inline.} =
    for i in 0 ..< N:
        res.data[i] = a.data[i] - b.data[i]

func mul*[T; N: static int](a, b: Vector[T, N], res: var Vector[T, N]) {.inline.} =
    ## Element-wise product.
    for i in 0 ..< N:
        res.data[i] = a.data[i] * b.data[i]

func scale*[T; N: static int](a: Vector[T, N], s: T, res: var Vector[T, N]) {.inline.} =
    for i in 0 ..< N:
        res.data[i] = a.data[i] * s