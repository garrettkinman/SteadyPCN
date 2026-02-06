# Copyright (c) 2024 Garrett Kinman
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import math
import ../tensors

func exp(x: float32): float32 =
    result = math.exp(x)

proc exp(tensor: Tensor[float32]): Tensor[float32] =
    result = Tensor[float32].new(tensor.shape)
    for it in 0..<tensor.len:
        let x: float32 = tensor.data[it]
        let y: float32 = exp(x)
        result.data[it] = y

# TODO: Add dims parameter?
proc softmax*(tensor: Tensor[float32]): Tensor[float32] =
    result = Tensor[float32].new(tensor.shape)
    let temp: Tensor[float32] = exp(tensor)
    let temp_sum: float32 = temp.sum()
    for it in 0..<tensor.len:
        result.data[it] = temp.data[it] / temp_sum