# Copyright (c) 2023 Garrett Kinman
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import unittest
import steadynn

let
    a = Dense[float32].new([2, 3], (1.float32)..(1.float32), relu)
    b = Dense[float32].new(
        Tensor.new([2, 2], @[float32 1, 0, 0, 1]),
        Tensor.new([2, 1], @[float32 0, 0]),
        relu
    )
    c = Dense[float32].new(
        [2, 2],
        @[float32 1, 0, 0, 1],
        @[float32 0, 0],
        relu
    )

test "dense":
    check a == Dense[float32].new(
        [2, 3],
        (1.float32)..(1.float32),
        relu
    )
    check a.forward(Tensor.new([3, 1], @[float32 -1, 0, 1])) == Tensor.new([2, 1], @[float32 1, 1])
    check b == Dense[float32].new(
        Tensor.new([2, 2], @[float32 1, 0, 0, 1]),
        Tensor.new([2, 1], @[float32 0, 0]),
        relu
    )
    check b.forward(Tensor.new([2, 1], @[float32 1, 1])) == Tensor.new([2, 1], @[float32 1, 1])
    check c == Dense[float32].new(
        [2, 2],
        @[float32 1, 0, 0, 1],
        @[float32 0, 0],
        relu
    )
    check c.forward(Tensor.new([2, 1], @[float32 1, 1])) == Tensor.new([2, 1], @[float32 1, 1])
