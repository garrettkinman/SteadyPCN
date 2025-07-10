# Copyright (c) 2024 Garrett Kinman
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import tinyNN

# declare model type + layers
type MultiLayerPerceptronModel[T] = object
        dense1: Dense[float32]
        dense2: Dense[float32]

# define forward pass
proc forward[T](model: MultiLayerPerceptronModel[T], x: Tensor[T]): Tensor[T] =
    result = model.dense2.forward(model.dense1.forward(x))

let mlp = MultiLayerPerceptronModel[float32](
    dense1: Dense[float32].new(
        [3, 3],
        (0.float32)..(1.float32),
        relu
    ),
    dense2: Dense[float32].new(
        [1, 3],
        (1.float32)..(1.float32),
        relu
    )
    )