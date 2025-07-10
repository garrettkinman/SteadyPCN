# Copyright (c) 2024 Garrett Kinman
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

# import ../tensors
# import ../architectures/cpu_primops

type
    TensorShape[rank: static int] = array[rank, int]

# Calculate total size at compile time
func totalSize[rank: static int](shape: static TensorShape[rank]): static int =
    var size = 1
    for dim in shape:
        size *= dim
    size

type
    Tensor*[T; shape: static TensorShape] = object
        data: array[totalSize(shape), T]

func initTensor*[T; shape: static TensorShape](default: T = default(T)): Tensor[T, shape] =
    result = Tensor[T, shape](data: default(array[totalSize(shape), T]))
    for i in 0..<result.data.len:
        result.data[i] = default

func `[]`*[T; shape: static TensorShape](t: Tensor[T, shape], indices: varargs[int]): T =
    assert indices.len == shape.len, "Invalid number of indices"
    var flatIndex = 0
    var stride = 1
    for dim in countdown(shape.high, 0):
        assert indices[dim] >= 0 and indices[dim] < shape[dim], "Index out of bounds"
        flatIndex += indices[dim] * stride
        if dim > 0:
            stride *= shape[dim]
    t.data[flatIndex]

func `[]=`*[T; shape: static TensorShape](t: var Tensor[T, shape], indices: varargs[int], value: T) =
    assert indices.len == shape.len, "Invalid number of indices"
    var flatIndex = 0
    var stride = 1
    for dim in countdown(shape.high, 0):
        assert indices[dim] >= 0 and indices[dim] < shape[dim], "Index out of bounds"
        flatIndex += indices[dim] * stride
        if dim > 0:
            stride *= shape[dim]
    t.data[flatIndex] = value

func shape*[T; shape: static TensorShape](t: Tensor[T, shape]): TensorShape[shape.len] =
    shape

func size*[T; shape: static TensorShape](t: Tensor[T, shape]): int =
    totalSize(shape)

func `$`*[T; shape: static TensorShape](t: Tensor[T, shape]): string =
    result = "Tensor["
    result.add $shape
    result.add "]("
    for i, val in t.data:
        if i > 0: result.add ", "
    result.add $val
    result.add ")"

# Example operations
func map*[T, U; shape: static TensorShape](t: Tensor[T, shape], f: static proc(x: T): U {.noSideEffect.}): Tensor[U, shape] =
  result = initTensor[U, shape]()
  for i in 0..<t.data.len:
    result.data[i] = f(t.data[i])

func zip*[T, U, V; shape: static TensorShape](
  t1: Tensor[T, shape],
  t2: Tensor[U, shape],
  f: static proc(x: T, y: U): V {.noSideEffect.}
): Tensor[V, shape] =
  result = initTensor[V, shape]()
  for i in 0..<t1.data.len:
    result.data[i] = f(t1.data[i], t2.data[i])

# Example usage
when isMainModule:
  # Create a 2x3 tensor of integers
  const tensorShape: TensorShape[2] = [2, 3]
  var t = initTensor[int, tensorShape]()
  
  # Set some values
  t[0, 0] = 1
  t[0, 1] = 2
  t[0, 2] = 3
  t[1, 0] = 4
  t[1, 1] = 5
  t[1, 2] = 6
  
  # Create another tensor and perform operations
  const doubleValue = proc(x: int): float {.noSideEffect.} = x.float * 2
  var t2 = t.map(doubleValue)
  echo t2  # Will print the tensor with doubled values
  
  # Perform element-wise addition
  const addValues = proc(x, y: int): int {.noSideEffect.} = x + y
  var t3 = zip(t, t, addValues)
  echo t3  # Will print the tensor with summed values

# TODO:
# 1. constructors
# 2. operators
# 3. functions