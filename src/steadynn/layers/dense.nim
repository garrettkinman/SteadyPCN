# Copyright (c) 2023 Garrett Kinman
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import sugar
import ../tensors

type
    Dense*[T] = object
        # TODO: make these statically allocated on the stack for performance + memory safety
        weights*: Tensor[T]
        bias*: Tensor[T]
        activation*: (Tensor[T]) -> Tensor[T]

# TODO: fix this to work and look nice
proc newDense*[T](shape: openArray[int], slice: HSlice[T, T], activation: (Tensor[T]) -> Tensor[T]): Dense[T] =
    result = Dense[T](
        weights: Tensor[T].rand(shape, slice),
        bias: Tensor[T].rand([shape[0], 1], slice), # make sure this is the right shape
        activation: activation
    )

# TODO: fix this to work and look nice
proc newDense*[T](weights: Tensor[T], bias: Tensor[T], activation: (Tensor[T]) -> Tensor[T]): Dense[T] =
    result = Dense[T](
        weights: weights,
        bias: bias, # make sure this is the right shape
        activation: activation
    )

# TODO: fix this to work and look nice
proc newDense*[T](shape: openArray[int], weights: seq[T], bias: seq[T], activation: (Tensor[T]) -> Tensor[T]): Dense[T] =
    result = Dense[T](
        weights: Tensor[T].new(shape, weights),
        bias: Tensor[T].new([shape[0], 1], bias), # make sure this is the right shape
        activation: activation
    )

proc new*[T](_: typedesc[Dense[T]], shape: openArray[int], slice: HSlice[T, T], activation: (Tensor[T]) -> Tensor[T]): Dense[T] = newDense[T](shape, slice, activation)
proc new*[T](_: typedesc[Dense[T]], weights: Tensor[T], bias: Tensor[T], activation: (Tensor[T]) -> Tensor[T]): Dense[T] = newDense(weights, bias, activation)
proc new*[T](_: typedesc[Dense[T]], shape: openArray[int], weights: seq[T], bias: seq[T], activation: (Tensor[T]) -> Tensor[T]): Dense[T] = newDense(shape, weights, bias, activation)

proc forward*[T](self: Dense, x: Tensor[T]): Tensor[T] =
    result = self.activation(((self.weights * x) + self.bias))