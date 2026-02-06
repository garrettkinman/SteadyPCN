# Copyright (c) 2023 Garrett Kinman
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import math
import ../tensors

# TODO: add common activation functions for sequential operation on the cpu
# 1. sigmoid
# 2. relu
# 3. tanh
# and more

func sigmoid(x: float32): float32 =
    result = 1 / (1 + exp(-x))

proc sigmoid*[T](tensor: Tensor[T]): Tensor[T] =
    ## A generic sequential implementation of the sigmoid activation function
    result = Tensor[T].new(tensor.shape)
    for it in 0..<tensor.len:
        let x: T = tensor.data[it]
        let y: T = sigmoid(x)
        result.data[it] = y

func relu(x: float32): float32 =
    if x > 0:
        result = x
    else:
        result = 0

proc relu*[T](tensor: Tensor[T]): Tensor[T] =
    ## A generic sequential implementation of the ReLU activation function
    result = Tensor[T].new(tensor.shape)
    for it in 0..<tensor.len:
        let x: T = tensor.data[it]
        let y: T = relu(x)
        result.data[it] = y