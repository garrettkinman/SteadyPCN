# Copyright (c) 2026 Garrett Kinman
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

# src/steadypcn/pcn.nim
import steadytensor
import activations

type
    # A Dense Predictive Coding Layer
    # Explanations: N, Observations: M
    PcnDenseLayer*[M, N: static int; T; act: Activation[T]] = object
        # Parameters
        weights*: Tensor[T, [M, N]]  # Maps this layer's state [N,1] to prediction [M,1]
        bias*:    Tensor[T, [M, 1]]  # Added to pre-activation; shape matches prediction, not state

        # State buffers (preserved between steps)
        state*: Tensor[T, [N, 1]]    # state = current belief / representation
        drive*: Tensor[T, [M, 1]]    # drive = W * state + bias
        error*: Tensor[T, [N, 1]]    # error = state - prediction received from above

        # Configuration
        learningRate*:  T
        inferenceRate*: T

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# INITIALIZATION
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

proc initPcnLayer*[M, N: static int; T; act: Activation[T]](lr: T = 0.01, infRate: T = 0.1): PcnDenseLayer[M, N, T, act] =
    result.weights       = rand[T, [M, N]](-0.1, 0.1)
    result.bias          = zeros[T, [M, 1]]()
    result.state         = zeros[T, [N, 1]]()
    result.drive         = zeros[T, [M, 1]]()
    result.error         = zeros[T, [N, 1]]()
    result.learningRate  = lr
    result.inferenceRate = infRate

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# STEP 1 — PREDICT
# Downward pass: generate this layer's prediction of what the layer below
# should look like.  Pure (no mutation); caller forwards the result to the
# layer below's updateError.
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

proc predict*[M, N: static int; T; act: Activation[T]](layer: lent PcnDenseLayer[M, N, T, act]): Tensor[T, [M, 1]] =
    ## pred = act.activate(W * state + bias)
    ## Returns [M, 1] — the prediction sent downward.
    ##
    ## act.activate() is zero-cost; the lambda below is a compile-time shim that lets
    ## map() accept the concept's forward proc without allocating a closure.
    layer.drive = (layer.weights * layer.state) + layer.bias
    result = map(
        layer.drive,
        func(x: T): T = act.activate(x)
    )

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# STEP 2 — UPDATE ERROR
# Store the top-down prediction received from above and compute local error.
# Call after the layer above has called predict().
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

proc updateError*[M, N: static int; T; act: Activation[T]](layer: var PcnDenseLayer[M, N, T, act], predFromAbove: lent Tensor[T, [N, 1]]) =
    ## e = state - pred_from_above
    ## Mutates layer.error in-place — no allocation.
    # for i in 0..<N:
    #     layer.error[i, 0] = layer.state[i, 0] - predFromAbove[i, 0]
    layer.error = layer.state - predFromAbove # TODO: update to in-place version when available


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# STEP 3 — RELAX (state update / inference)
# Update this layer's state to reduce both its own error and the errors
# it causes in the layer below.
# Assumes updateError has already been called this step.
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

proc relax*[M, N: static int; T; act: Activation[T]](layer: var PcnDenseLayer[M, N, T, act], errorBelow: lent Tensor[T, [M, 1]]) =
    ## dx = inferenceRate * ((W^T * errorBelow) .* act.grad(layer.drive) - layer.error)
    ##
    ## Two small stack tensors are allocated (feedback [N,1], dState [N,1]);
    ## the update is then applied directly into layer.state — no delta tensor.

    # Derivative of the activation at the current drive.
    # Result: [M, 1]
    let dDrive = map(layer.drive, func(x: T): T = act.grad(x))

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

proc learn*[M, N: static int; T; act: Activation[T]](layer: var PcnDenseLayer[M, N, T, act], errorBelow: lent Tensor[T, [M, 1]]) =
    ## dW = learningRate * errorBelow * state^T   (outer product [M,1] x [1,N] -> [M,N])
    ## db = learningRate * errorBelow
    ##
    ## Rank-1 update computed as a fused loop — no intermediate [M,N] tensor allocated.

    let dDrive = map(layer.drive, func(x: T): T = act.grad(x))
    let delta = matmulT(errorBelow .* dDrive, layer.state) # [M,N]
    layer.weights += delta * layer.learningRate
    # TODO: update to in-place version when available (currently allocates several tensors for the intermediate results)