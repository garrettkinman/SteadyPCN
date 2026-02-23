# Copyright (c) 2026 Garrett Kinman
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

# src/steadypcn/pcn.nim
import tensors, ops, activations

type
    # A Dense Predictive Coding Layer
    # Explanations: N, Observations: M
    Layer*[M, N: static int; T; A] = object
        # Parameters
        weights*: Matrix[T, M, N]   # Maps this layer's state [N] to prediction [M]
        bias*:    Vector[T, M]      # Added to pre-activation; shape matches prediction, not state

        # State buffers (preserved between steps)
        state*: Vector[T, N]        # state = current belief / representation
        drive*: Vector[T, M]        # drive = W * state + bias
        error*: Vector[T, N]        # error = state - prediction received from above

        # Activation (zero-size for current types, but properly typed)
        activation*: A

        # Configuration
        learningRate*:  T
        inferenceRate*: T

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# INITIALIZATION
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

proc init*[M, N: static int; T; A](
    layer: var Layer[M, N, T, A],
    activation: sink A,
    lr: T = 0.01,
    infRate: T = 0.1
) =
    static: assert A is Activation[T], $A & " does not satisfy Activation[" & $T & "]"
    layer.weights       = Matrix[T, M, N].rand(T(-0.1), T(0.1))
    layer.bias          = Vector[T, M].zeros()
    layer.state         = Vector[T, N].zeros()
    layer.drive         = Vector[T, M].zeros()
    layer.error         = Vector[T, N].zeros()
    layer.activation    = activation
    layer.learningRate  = lr
    layer.inferenceRate = infRate

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# STEP 1 — PREDICT
# Downward pass: generate this layer's prediction of what the layer below
# should look like.  Pure (no mutation); caller forwards the result to the
# layer below's updateError.
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

proc predict*[M, N: static int; T; A](layer: var Layer[M, N, T, A]): Vector[T, M] =
    ## pred = activate(act, W * state + bias)
    ## Returns [M] — the prediction sent downward.
    layer.drive = (layer.weights * layer.state) + layer.bias
    result = activate(layer.activation, layer.drive)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# STEP 2 — UPDATE ERROR
# Store the top-down prediction received from above and compute local error.
# Call after the layer above has called predict().
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

proc updateError*[M, N: static int; T; A](layer: var Layer[M, N, T, A], predFromAbove: Vector[T, N]) =
    ## e = state - pred_from_above
    ## Mutates layer.error in-place — no allocation.
    layer.error = layer.state - predFromAbove  # TODO: update to in-place version when available

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# STEP 3 — RELAX (state update / inference)
# Update this layer's state to reduce both its own error and the errors
# it causes in the layer below.
# Assumes updateError has already been called this step.
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

proc relax*[M, N: static int; T; A](layer: var Layer[M, N, T, A], errorBelow: Vector[T, M]) =
    ## dx = inferenceRate * ((W^T * errorBelow) .* act.grad(layer.drive) - layer.error)
    ##
    ## Two small stack vectors are allocated (dDrive [M], delta [N]);
    ## the update is then applied directly into layer.state.

    # Derivative of the activation at the current drive.
    # Result: [M]
    let dDrive = grad(layer.activation, layer.drive)

    # Back-project the error from below through the transposed weights.
    # layer.weights.t gives a zero-copy TransposedMatrix[T, N, M]; ops.nim
    # dispatches this to mvMulT — no copy of weights needed.
    # Result: [N]
    let delta = (layer.weights.t * (errorBelow .* dDrive)) - layer.error
    layer.state += delta * layer.inferenceRate
    # TODO: update to in-place version when available

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# STEP 4 — LEARN (weight update)
# Hebbian-like rule: weights learn to make this layer's predictions accurate.
# Assumes state and error are current for this step.
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

proc learn*[M, N: static int; T; A](layer: var Layer[M, N, T, A], errorBelow: Vector[T, M]) =
    ## dW = learningRate * (errorBelow .* act.grad(drive)) ⊗ state   (outer product [M] x [N] -> [M,N])
    ## db = learningRate * errorBelow   (bias update omitted; see note below)
    ##
    ## Rank-1 update computed via outer() — no intermediate [M,N] tensor allocated
    ## beyond the result itself.

    let dDrive = grad(layer.activation, layer.drive)
    let delta  = outer(errorBelow .* dDrive, layer.state)  # [M, N]
    layer.weights += delta * layer.learningRate
    # TODO: update to in-place version when available