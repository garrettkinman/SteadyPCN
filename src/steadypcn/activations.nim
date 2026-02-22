# Copyright (c) 2026 Garrett Kinman
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

# src/steadypcn/activations.nim
import math

type
    Activation*[T] = concept a
        activate(a, T) is T
        grad(a, T) is T

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ACTIVATION FUNCTIONS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

type Sigmoid* = object
    ## σ(x) = 1 / (1 + e^-x)
    ## Smooth, bounded in (0, 1).  Good default for PCN belief states.

func activate*[T](_: Sigmoid, x: T): T {.inline.} =
    1.T / (1.T + exp(-x))
    # TODO: call `sigmoid(x)` directly (needs to be defined somewhere)

func grad*[T](_: Sigmoid, x: T): T {.inline.} =
    ## Derivative expressed in terms of x directly (recomputes σ(x)).
    ## If you already have σ(x) available, prefer: s * (1 - s).
    let s = activate(Sigmoid(), x)
    s * (1.T - s)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

type Tanh* = object
    ## tanh(x) = (e^x - e^-x) / (e^x + e^-x)
    ## Smooth, bounded in (-1, 1).  Can be useful for zero-centered activations.

func activate*[T](_: Tanh, x: T): T {.inline.} =
    tanh(x)

func grad*[T](_: Tanh, x: T): T {.inline.} =
    ## Derivative expressed in terms of x directly (recomputes tanh(x)).
    ## If you already have tanh(x) available, prefer: 1 - t^2.
    let t = activate(Tanh(), x)
    1.T - t * t

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

type ReLU* = object
    ## ReLU(x) = max(0, x)
    ## Simple, unbounded above, zero for negative inputs.  Can help with sparse activations.

func activate*[T](_: ReLU, x: T): T {.inline.} =
    if x > 0.T: x else: 0.T

func grad*[T](_: ReLU, x: T): T {.inline.} =
    if x > 0.T: 1.T else: 0.T

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

type Identity* = object
    ## Identity(x) = x
    ## No non-linearity, useful for testing or linear layers.

func activate*[T](_: Identity, x: T): T {.inline.} =
    x

func grad*[T](_: Identity, x: T): T {.inline.} =
    1.T