# Copyright (c) 2026 Garrett Kinman
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

# src/steadypcn/activations.nim
import math

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ACTIVATION FUNCTIONS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

func sigmoid*[T](x: T): T =
    1.T / (1.T + exp(-x))

func sigmoidDerivative*[T](x: T): T =
    let s = sigmoid(x)
    s * (1.T - s)

func tanhDerivative*[T](x: T): T =
    let t = tanh(x)
    1.T - t * t

func relu*[T](x: T): T =
    if x > 0.T: x else: 0.T

func reluDerivative*[T](x: T): T =
    if x > 0.T: 1.T else: 0.T