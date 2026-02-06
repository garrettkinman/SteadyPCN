# Copyright (c) 2024 Garrett Kinman
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import unittest
import steadynn

type
    SingleLayerPerceptronModel[T] = object
        dense: Dense[float32]

    MultiLayerPerceptronModel[T] = object
        dense1: Dense[float32]
        dense2: Dense[float32]

proc forward[T](model: SingleLayerPerceptronModel[T], x: Tensor[T]): Tensor[T] =
    result = model.dense.forward(x)

proc forward[T](model: MultiLayerPerceptronModel[T], x: Tensor[T]): Tensor[T] =
    result = model.dense2.forward(model.dense1.forward(x))

let
    slp = SingleLayerPerceptronModel[float32](
        dense: Dense[float32].new(
            [1, 3],
            (1.float32)..(1.float32),
            relu
        )
    )
    mlp = MultiLayerPerceptronModel[float32](
        dense1: Dense[float32].new(
            [3, 3],
            (1.float32)..(1.float32),
            relu
        ),
        dense2: Dense[float32].new(
            [1, 3],
            (1.float32)..(1.float32),
            relu
        )
    )

test "singleLayerPerceptron":
    echo slp.forward(Tensor.new([3, 1], @[float32 1, 2, 3]))
    echo mlp.forward(Tensor.new([3, 1], @[float32 1, 2, 3]))