# Copyright (c) 2026 Garrett Kinman
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

# src/steadypcn/pcn.nim
import steadytensor
import activations

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# SHAPE HELPERS
# Nim will not propagate the `static` attribute from M/N into a bare array
# literal like [M, N] used inside a generic type body or proc signature.
# Wrapping the construction in a static-returning func is the same trick
# used in steadytensor: the compiler evaluates the call at instantiation time
# and the result is properly treated as a static TensorShape.
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

func sh*(m, n: static int): static TensorShape[2] = [m, n]

type
    # A Dense Predictive Coding Layer
    # Explanations: N, Observations: M
    PcnDenseLayer*[M, N: static int; T; A] = object
        # Parameters
        weights*: Tensor[T, sh(M, N)]  # Maps this layer's state [N,1] to prediction [M,1]
        bias*:    Tensor[T, sh(M, 1)]     # Added to pre-activation; shape matches prediction, not state

        # State buffers (preserved between steps)
        state*: Tensor[T, sh(N, 1)]    # state = current belief / representation
        drive*: Tensor[T, sh(M, 1)]    # drive = W * state + bias
        error*: Tensor[T, sh(N, 1)]    # error = state - prediction received from above

        # Activation (zero-size for current types, but properly typed)
        activation*: A

        # Configuration
        learningRate*:  T
        inferenceRate*: T

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# INITIALIZATION
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

proc init*[M, N: static int; T; A](
    layer: var PcnDenseLayer[M, N, T, A],
    activation: sink A,
    lr: T = 0.01,
    infRate: T = 0.1
) =
    static: assert A is Activation[T], $A & " does not satisfy Activation[" & $T & "]"
    layer.weights       = rand[T, sh(M, N)](-0.1, 0.1)
    layer.bias          = zeros[T, sh(M, 1)]()
    layer.state         = zeros[T, sh(N, 1)]()
    layer.drive         = zeros[T, sh(M, 1)]()
    layer.error         = zeros[T, sh(N, 1)]()
    layer.activation    = activation
    layer.learningRate  = lr
    layer.inferenceRate = infRate

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# STEP 1 — PREDICT
# Downward pass: generate this layer's prediction of what the layer below
# should look like.  Pure (no mutation); caller forwards the result to the
# layer below's updateError.
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

proc predict*[M, N: static int; T; A](layer: var PcnDenseLayer[M, N, T, A]): Tensor[T, sh(M, 1)] =
    ## pred = act.activate(W * state + bias)
    ## Returns [M, 1] — the prediction sent downward.
    ##
    ## act.activate() is zero-cost; the lambda below is a compile-time shim that lets
    ## map() accept the concept's forward proc without allocating a closure.
    layer.drive = (layer.weights * layer.state) + layer.bias
    result = map(
        layer.drive,
        func(x: T): T = activate(layer.activation, x)
    )

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# STEP 2 — UPDATE ERROR
# Store the top-down prediction received from above and compute local error.
# Call after the layer above has called predict().
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

proc updateError*[M, N: static int; T; A](layer: var PcnDenseLayer[M, N, T, A], predFromAbove: lent Tensor[T, sh(N, 1)]) =
    ## e = state - pred_from_above
    ## Mutates layer.error in-place — no allocation.
    layer.error = layer.state - predFromAbove # TODO: update to in-place version when available

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# STEP 3 — RELAX (state update / inference)
# Update this layer's state to reduce both its own error and the errors
# it causes in the layer below.
# Assumes updateError has already been called this step.
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

proc relax*[M, N: static int; T; A](layer: var PcnDenseLayer[M, N, T, A], errorBelow: lent Tensor[T, sh(M, 1)]) =
    ## dx = inferenceRate * ((W^T * errorBelow) .* act.grad(layer.drive) - layer.error)
    ##
    ## Two small stack tensors are allocated (feedback [N,1], dState [N,1]);
    ## the update is then applied directly into layer.state — no delta tensor.

    # Derivative of the activation at the current drive.
    # Result: [M, 1]
    let dDrive = map(layer.drive, func(x: T): T = grad(layer.activation, x))

    # Back-project the error from below through our weights.
    # matmulT uses W virtually transposed — no copy of weights needed.
    # Result: [N, 1]
    let delta = matmulT(layer.weights, errorBelow .* dDrive) - layer.error
    layer.state += delta * layer.inferenceRate
    # TODO: update to in-place version when available (currently allocates several tensors for the intermediate results)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# STEP 4 — LEARN (weight update)
# Hebbian-like rule: weights learn to make this layer's predictions accurate.
# Assumes state and error are current for this step.
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

proc learn*[M, N: static int; T; A](layer: var PcnDenseLayer[M, N, T, A], errorBelow: lent Tensor[T, sh(M, 1)]) =
    ## dW = learningRate * errorBelow * state^T   (outer product [M,1] x [1,N] -> [M,N])
    ## db = learningRate * errorBelow
    ##
    ## Rank-1 update computed as a fused loop — no intermediate [M,N] tensor allocated.

    let dDrive = map(layer.drive, func(x: T): T = grad(layer.activation, x))
    let delta = matmulT(errorBelow .* dDrive, layer.state) # [M,N]
    layer.weights += delta * layer.learningRate
    # TODO: update to in-place version when available (currently allocates several tensors for the intermediate results)