# Copyright (c) 2026 Garrett Kinman
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import tensors, kernels

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# SCALAR BROADCASTING  (Matrix)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

func `+`*[T; M, N: static int](mat: Matrix[T, M, N], s: T): Matrix[T, M, N] {.inline.} =
    for i in 0 ..< M * N: result.data[i] = mat.data[i] + s

func `-`*[T; M, N: static int](mat: Matrix[T, M, N], s: T): Matrix[T, M, N] {.inline.} =
    for i in 0 ..< M * N: result.data[i] = mat.data[i] - s

func `*`*[T; M, N: static int](mat: Matrix[T, M, N], s: T): Matrix[T, M, N] {.inline.} =
    scale(mat, s, result)

func `/`*[T; M, N: static int](mat: Matrix[T, M, N], s: T): Matrix[T, M, N] {.inline.} =
    scale(mat, T(1) / s, result)

func `*`*[T; M, N: static int](s: T, mat: Matrix[T, M, N]): Matrix[T, M, N] {.inline.} =
    scale(mat, s, result)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# SCALAR BROADCASTING  (Vector)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

func `+`*[T; N: static int](v: Vector[T, N], s: T): Vector[T, N] {.inline.} =
    for i in 0 ..< N: result.data[i] = v.data[i] + s

func `-`*[T; N: static int](v: Vector[T, N], s: T): Vector[T, N] {.inline.} =
    for i in 0 ..< N: result.data[i] = v.data[i] - s

func `*`*[T; N: static int](v: Vector[T, N], s: T): Vector[T, N] {.inline.} =
    scale(v, s, result)

func `/`*[T; N: static int](v: Vector[T, N], s: T): Vector[T, N] {.inline.} =
    scale(v, T(1) / s, result)

func `*`*[T; N: static int](s: T, v: Vector[T, N]): Vector[T, N] {.inline.} =
    scale(v, s, result)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ELEMENT-WISE ARITHMETIC  (Matrix)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

func `+`*[T; M, N: static int](a, b: Matrix[T, M, N]): Matrix[T, M, N] {.inline.} =
    add(a, b, result)

func `-`*[T; M, N: static int](a, b: Matrix[T, M, N]): Matrix[T, M, N] {.inline.} =
    sub(a, b, result)

func `.*`*[T; M, N: static int](a, b: Matrix[T, M, N]): Matrix[T, M, N] {.inline.} =
    ## Hadamard (element-wise) product. Use `.*` to distinguish from matmul `*`.
    mul(a, b, result)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ELEMENT-WISE ARITHMETIC  (Vector)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

func `+`*[T; N: static int](a, b: Vector[T, N]): Vector[T, N] {.inline.} =
    add(a, b, result)

func `-`*[T; N: static int](a, b: Vector[T, N]): Vector[T, N] {.inline.} =
    sub(a, b, result)

func `.*`*[T; N: static int](a, b: Vector[T, N]): Vector[T, N] {.inline.} =
    ## Element-wise product.
    mul(a, b, result)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# IN-PLACE OPERATORS  (Matrix)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

func `+=`*[T; M, N: static int](a: var Matrix[T, M, N], b: Matrix[T, M, N]) {.inline.} =
    add(a, b, a)

func `-=`*[T; M, N: static int](a: var Matrix[T, M, N], b: Matrix[T, M, N]) {.inline.} =
    sub(a, b, a)

func `.*=`*[T; M, N: static int](a: var Matrix[T, M, N], b: Matrix[T, M, N]) {.inline.} =
    mul(a, b, a)

func `+=`*[T; M, N: static int](a: var Matrix[T, M, N], s: T) {.inline.} =
    for i in 0 ..< M * N: a.data[i] += s

func `*=`*[T; M, N: static int](a: var Matrix[T, M, N], s: T) {.inline.} =
    scale(a, s, a)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# IN-PLACE OPERATORS  (Vector)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

func `+=`*[T; N: static int](a: var Vector[T, N], b: Vector[T, N]) {.inline.} =
    add(a, b, a)

func `-=`*[T; N: static int](a: var Vector[T, N], b: Vector[T, N]) {.inline.} =
    sub(a, b, a)

func `.*=`*[T; N: static int](a: var Vector[T, N], b: Vector[T, N]) {.inline.} =
    mul(a, b, a)

func `+=`*[T; N: static int](a: var Vector[T, N], s: T) {.inline.} =
    for i in 0 ..< N: a.data[i] += s

func `*=`*[T; N: static int](a: var Vector[T, N], s: T) {.inline.} =
    scale(a, s, a)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# LINEAR ALGEBRA OPERATORS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

func `*`*[T; M, K, N: static int](A: Matrix[T, M, K], B: Matrix[T, K, N]): Matrix[T, M, N] {.inline.} =
    ## Matrix multiplication: (M×K) * (K×N) → (M×N).
    matmul(A, B)

func `*`*[T; M, K: static int](W: Matrix[T, M, K], x: Vector[T, K]): Vector[T, M] {.inline.} =
    ## Matrix-vector multiplication: (M×K) * (K,) → (M,).
    mvMul(W, x)

# Aᵀ * B  →  uses matmulT kernel
func `*`*[T; M, K, N: static int](A: TransposedMatrix[T, M, K], B: Matrix[T, K, N]): Matrix[T, M, N] {.inline.} =
    matmulT(A, B)

# Wᵀ * x  →  uses mvMulT kernel  
func `*`*[T; M, K: static int](W: TransposedMatrix[T, M, K], x: Vector[T, K]): Vector[T, M] {.inline.} =
    mvMulT(W, x)

func dot*[T; N: static int](a, b: Vector[T, N]): T =
    ## Inner (dot) product of two vectors.
    var s = T(0)
    for i in 0 ..< N:
        s += a[i] * b[i]
    s

func outer*[T; M, N: static int](a: Vector[T, M], b: Vector[T, N]): Matrix[T, M, N] =
    ## Outer product: (M,) ⊗ (N,) → (M×N).
    ## Useful for rank-1 weight updates in Hebbian / PCN learning rules.
    for i in 0 ..< M:
        for j in 0 ..< N:
            result[i, j] = a[i] * b[j]

func neg*[T; M, N: static int](mat: Matrix[T, M, N]): Matrix[T, M, N] {.inline.} =
    for i in 0 ..< M * N: result.data[i] = -mat.data[i]

func neg*[T; N: static int](v: Vector[T, N]): Vector[T, N] {.inline.} =
    for i in 0 ..< N: result.data[i] = -v.data[i]

func `-`*[T; M, N: static int](mat: Matrix[T, M, N]): Matrix[T, M, N] {.inline.} = neg(mat)
func `-`*[T; N: static int](v: Vector[T, N]): Vector[T, N] {.inline.} = neg(v)