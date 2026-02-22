# Copyright (c) 2026 Garrett Kinman
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import random

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# CONFIGURATION
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Global Layout Switch
# Pass -d:colMajor to the compiler to switch to Column-Major (Fortran/MATLAB) layout.
# Default is Row-Major (C/NumPy) layout.
const ColMajor*: bool = defined(colMajor)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# CORE TYPES
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

type
    Matrix*[T; M, N: static int] = object
        ## A 2D matrix stored as a flat array.
        ## M = rows, N = cols.
        data*: array[M * N, T]

    Vector*[T; N: static int] = object
        ## A 1D column vector stored as a flat array.
        data*: array[N, T]

    # TransposedMatrix[T, M, N] is logically N×M, but stored as Matrix[T, N, M].
    # `distinct` makes it a zero-cost reinterpretation — same bits, different type.
    TransposedMatrix*[T; M, N: static int] = distinct Matrix[T, N, M]

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# INDEXING
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

func `[]`*[T; M, N: static int](mat: Matrix[T, M, N], i, j: int): T {.inline.} =
    when ColMajor:
        mat.data[j * M + i]   # col-major: column stride = M
    else:
        mat.data[i * N + j]   # row-major: row stride = N

proc `[]=`*[T; M, N: static int](mat: var Matrix[T, M, N], i, j: int, val: T) {.inline.} =
    when ColMajor:
        mat.data[j * M + i] = val
    else:
        mat.data[i * N + j] = val

func `[]`*[T; N: static int](v: Vector[T, N], i: int): T {.inline.} =
    v.data[i]

proc `[]=`*[T; N: static int](v: var Vector[T, N], i: int, val: T) {.inline.} =
    v.data[i] = val

func `[]`*[T; M, N: static int](tr: TransposedMatrix[T, M, N], i, j: int): T {.inline.} =
    Matrix[T, N, M](tr)[j, i]   # swap indices, read from original storage

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# SHAPE / SIZE QUERIES
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

func rows*[T; M, N: static int](mat: Matrix[T, M, N]): int {.inline.} = M
func cols*[T; M, N: static int](mat: Matrix[T, M, N]): int {.inline.} = N
func size*[T; M, N: static int](mat: Matrix[T, M, N]): int {.inline.} = M * N

func len*[T; N: static int](v: Vector[T, N]): int {.inline.} = N

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# INITIALIZATION
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

func zeros*[T; M, N: static int](_: typedesc[Matrix[T, M, N]]): Matrix[T, M, N] =
    ## Returns the zero matrix. (Nim zero-initialises by default, but this is explicit.)
    Matrix[T, M, N]()

func zeros*[T; N: static int](_: typedesc[Vector[T, N]]): Vector[T, N] =
    Vector[T, N]()

func ones*[T; M, N: static int](_: typedesc[Matrix[T, M, N]]): Matrix[T, M, N] =
    for i in 0 ..< M * N:
        result.data[i] = T(1)

func ones*[T; N: static int](_: typedesc[Vector[T, N]]): Vector[T, N] =
    for i in 0 ..< N:
        result.data[i] = T(1)

func initMatrix*[T; M, N: static int](data: array[M * N, T]): Matrix[T, M, N] =
    ## Constructs a Matrix directly from a flat array (row-major source order).
    Matrix[T, M, N](data: data)

func initVector*[T; N: static int](data: array[N, T]): Vector[T, N] =
    Vector[T, N](data: data)

proc rand*[T; M, N: static int](_: typedesc[Matrix[T, M, N]], lo: T = T(0), hi: T = T(1)): Matrix[T, M, N] =
    for i in 0 ..< M * N:
        result.data[i] = rand(lo .. hi)

proc rand*[T; N: static int](_: typedesc[Vector[T, N]], lo: T = T(0), hi: T = T(1)): Vector[T, N] =
    for i in 0 ..< N:
        result.data[i] = rand(lo .. hi)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ITERATORS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

iterator items*[T; M, N: static int](mat: Matrix[T, M, N]): T =
    for v in mat.data: yield v

iterator mitems*[T; M, N: static int](mat: var Matrix[T, M, N]): var T =
    for v in mat.data.mitems: yield v

iterator items*[T; N: static int](v: Vector[T, N]): T =
    for x in v.data: yield x

iterator mitems*[T; N: static int](v: var Vector[T, N]): var T =
    for x in v.data.mitems: yield x

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# UTILITIES
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

func copy*[T; M, N: static int](mat: Matrix[T, M, N]): Matrix[T, M, N] =
    Matrix[T, M, N](data: mat.data)

func copy*[T; N: static int](v: Vector[T, N]): Vector[T, N] =
    Vector[T, N](data: v.data)

func map*[T, U; M, N: static int](mat: Matrix[T, M, N], f: proc(x: T): U {.noSideEffect.}): Matrix[U, M, N] =
    for i in 0 ..< M * N:
        result.data[i] = f(mat.data[i])

func map*[T, U; N: static int](v: Vector[T, N], f: proc(x: T): U {.noSideEffect.}): Vector[U, N] =
    for i in 0 ..< N:
        result.data[i] = f(v.data[i])

func zip*[T, U, V; M, N: static int](
    a: Matrix[T, M, N],
    b: Matrix[U, M, N],
    f: proc(x: T, y: U): V {.noSideEffect.}
): Matrix[V, M, N] =
    for i in 0 ..< M * N:
        result.data[i] = f(a.data[i], b.data[i])

func zip*[T, U, V; N: static int](
    a: Vector[T, N],
    b: Vector[U, N],
    f: proc(x: T, y: U): V {.noSideEffect.}
): Vector[V, N] =
    for i in 0 ..< N:
        result.data[i] = f(a.data[i], b.data[i])

func `$`*[T; M, N: static int](mat: Matrix[T, M, N]): string =
    result = "Matrix[" & $M & "x" & $N & "]("
    for i in 0 ..< M:
        if i > 0: result.add "; "
        for j in 0 ..< N:
            if j > 0: result.add ", "
            result.add $mat[i, j]
    result.add ")"

func `$`*[T; N: static int](v: Vector[T, N]): string =
    result = "Vector[" & $N & "]("
    for i in 0 ..< N:
        if i > 0: result.add ", "
        result.add $v[i]
    result.add ")"

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# CONVERSIONS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

func toVector*[T; M, N: static int](mat: Matrix[T, M, N]): Vector[T, M * N] =
    ## Flattens a matrix into a vector (zero-copy reinterpretation of storage).
    Vector[T, M * N](data: mat.data)

func toMatrix*[T; N: static int; M, K: static int](v: Vector[T, N]): Matrix[T, M, K] =
    ## Reshapes a vector into a matrix at compile time.
    ## Requires M * K == N (enforced statically).
    static: assert M * K == N, "Vector length must equal M * K for reshape"
    Matrix[T, M, K](data: v.data)

func t*[T; M, N: static int](mat: Matrix[T, M, N]): TransposedMatrix[T, N, M] {.inline.} =
    ## Zero-copy transpose. Just a type-level reinterpretation.
    TransposedMatrix[T, N, M](mat)

func t*[T; M, N: static int](mat: TransposedMatrix[T, M, N]): Matrix[T, N, M] =
    ## Zero‑cost “untranspose”: turns a TransposedMatrix back into a normal Matrix.
    Matrix[T, N, M](mat)

func col*[T; M, N: static int](mat: Matrix[T, M, N], j: static int): Vector[T, M] =
    ## Extracts column j as a Vector.
    for i in 0 ..< M:
        result[i] = mat[i, j]

func row*[T; M, N: static int](mat: Matrix[T, M, N], i: static int): Vector[T, N] =
    ## Extracts row i as a Vector.
    for j in 0 ..< N:
        result[j] = mat[i, j]