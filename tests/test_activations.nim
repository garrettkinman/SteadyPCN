# Copyright (c) 2023 Garrett Kinman
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import unittest
import steadynn

var
    a = Tensor.new([2, 3], @[float32 -3, -2, -1, 1, 2, 3])

test "sigmoid":
    check a.sigmoid() == Tensor.new([2, 3], @[float32 0.04742587357759476, 0.1192029193043709, 0.2689414322376251, 0.7310585975646973, 0.8807970285415649, 0.9525741338729858])

test "relu":
    check a.relu() == Tensor.new([2, 3], @[float32 0, 0, 0, 1, 2, 3])

test "softmax":
    # TODO: Add approx equals for float32
    check a.softmax() == Tensor.new([2, 3], @[float32 0.001619308721274137, 0.004401737824082375, 0.01196516398340464, 0.08841126412153244, 0.2403267323970795, 0.6532757878303528])