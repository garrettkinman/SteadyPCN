# Copyright (c) 2026 Garrett Kinman
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import unittest
import math
import steadypcn

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

const eps = 1e-6

func approxEq(a, b, tol: float): bool =
    abs(a - b) < tol

# ---------------------------------------------------------------------------
# Suites
# ---------------------------------------------------------------------------

suite "PcnDenseLayer – Initialization":

    test "Default shapes are correct":
        var layer: PcnDenseLayer[3, 2, float, Identity]
        layer.init(Identity(), lr = 0.01, infRate = 0.1)
        check layer.weights.rows == 3
        check layer.weights.cols == 2
        check layer.bias.len    == 3
        check layer.state.len   == 2
        check layer.drive.len   == 3
        check layer.error.len   == 2

    test "Bias and state start at zero":
        var layer: PcnDenseLayer[3, 2, float, Identity]
        layer.init(Identity(), lr = 0.01, infRate = 0.1)
        for v in layer.bias.data:  check v == 0.0
        for v in layer.state.data: check v == 0.0
        for v in layer.error.data: check v == 0.0
        for v in layer.drive.data: check v == 0.0

    test "Weights are in initialisation range (-0.1, 0.1)":
        var layer: PcnDenseLayer[4, 4, float, Identity]
        layer.init(Identity())
        for v in layer.weights.data:
            check v >= -0.1
            check v <=  0.1

    test "Learning-rate and inference-rate stored correctly":
        var layer: PcnDenseLayer[2, 2, float, Sigmoid]
        layer.init(Sigmoid(), lr = 0.05, infRate = 0.2)
        check layer.learningRate  == 0.05
        check layer.inferenceRate == 0.2


# ---------------------------------------------------------------------------

suite "PcnDenseLayer – predict()  (Identity activation)":
    # With Identity, predict(layer) = W * state + bias  exactly.

    test "Zero state → prediction equals bias":
        var layer: PcnDenseLayer[2, 2, float, Identity]
        layer.init(Identity())
        # state is already zero; set bias to something non-trivial
        layer.bias[0] = 3.0
        layer.bias[1] = -1.0
        let pred = layer.predict()
        check approxEq(pred[0],  3.0, eps)
        check approxEq(pred[1], -1.0, eps)

    test "Non-zero state with known weights":
        # W = [[2, 1], [0, 3]]   state = [1, 2]   bias = [0, 0]
        # drive = W * state = [2*1+1*2, 0*1+3*2] = [4, 6]
        var layer: PcnDenseLayer[2, 2, float, Identity]
        layer.init(Identity())
        layer.weights[0, 0] = 2.0; layer.weights[0, 1] = 1.0
        layer.weights[1, 0] = 0.0; layer.weights[1, 1] = 3.0
        layer.state[0] = 1.0;   layer.state[1] = 2.0
        let pred = layer.predict()
        check approxEq(pred[0], 4.0, eps)
        check approxEq(pred[1], 6.0, eps)

    test "Bias is added on top of W*state":
        var layer: PcnDenseLayer[2, 2, float, Identity]
        layer.init(Identity())
        layer.weights[0, 0] = 1.0; layer.weights[0, 1] = 0.0
        layer.weights[1, 0] = 0.0; layer.weights[1, 1] = 1.0  # identity matrix
        layer.state[0] = 5.0;   layer.state[1] = -3.0
        layer.bias[0]  = 2.0;   layer.bias[1]  = 10.0
        let pred = layer.predict()
        check approxEq(pred[0],  7.0, eps)  # 5 + 2
        check approxEq(pred[1],  7.0, eps)  # -3 + 10

    test "drive buffer is updated after predict()":
        var layer: PcnDenseLayer[2, 2, float, Identity]
        layer.init(Identity())
        layer.weights[0, 0] = 1.0; layer.weights[0, 1] = 0.0
        layer.weights[1, 0] = 0.0; layer.weights[1, 1] = 1.0
        layer.state[0] = 4.0;   layer.state[1] = -2.0
        discard layer.predict()
        check approxEq(layer.drive[0],  4.0, eps)
        check approxEq(layer.drive[1], -2.0, eps)

    test "Rectangular layer (M≠N)":
        # M=3, N=2: W is [3,2], state is [2], prediction is [3]
        var layer: PcnDenseLayer[3, 2, float, Identity]
        layer.init(Identity())
        layer.weights[0, 0] = 1.0; layer.weights[0, 1] = 0.0
        layer.weights[1, 0] = 0.0; layer.weights[1, 1] = 1.0
        layer.weights[2, 0] = 1.0; layer.weights[2, 1] = 1.0
        layer.state[0] = 2.0;   layer.state[1] = 3.0
        let pred = layer.predict()
        check approxEq(pred[0], 2.0, eps)
        check approxEq(pred[1], 3.0, eps)
        check approxEq(pred[2], 5.0, eps)


suite "PcnDenseLayer – predict()  (non-linear activations)":

    test "Sigmoid: output is strictly in (0, 1)":
        var layer: PcnDenseLayer[3, 3, float, Sigmoid]
        layer.init(Sigmoid())
        layer.state[0] = 100.0  # should saturate near 1
        layer.state[1] = -100.0 # should saturate near 0
        layer.state[2] = 0.0    # should be near 0.5
        # Use identity weights so drive = state
        layer.weights[0, 0] = 1.0; layer.weights[0, 1] = 0.0; layer.weights[0, 2] = 0.0
        layer.weights[1, 0] = 0.0; layer.weights[1, 1] = 1.0; layer.weights[1, 2] = 0.0
        layer.weights[2, 0] = 0.0; layer.weights[2, 1] = 0.0; layer.weights[2, 2] = 1.0
        let pred = layer.predict()
        check pred[0] > 0.999
        check pred[1] < 0.001
        check approxEq(pred[2], 0.5, 1e-4)

    test "ReLU: negative drive produces zero output":
        var layer: PcnDenseLayer[2, 2, float, ReLU]
        layer.init(ReLU())
        layer.weights[0, 0] = 1.0; layer.weights[0, 1] = 0.0
        layer.weights[1, 0] = 0.0; layer.weights[1, 1] = 1.0
        layer.state[0] =  2.0
        layer.state[1] = -3.0
        let pred = layer.predict()
        check approxEq(pred[0], 2.0, eps)
        check approxEq(pred[1], 0.0, eps)

    test "Tanh: zero drive produces zero output":
        var layer: PcnDenseLayer[2, 2, float, Tanh]
        layer.init(Tanh())
        layer.weights[0, 0] = 1.0; layer.weights[1, 1] = 1.0
        # state is zero → drive is zero → tanh(0) = 0
        let pred = layer.predict()
        check approxEq(pred[0], 0.0, eps)
        check approxEq(pred[1], 0.0, eps)

    test "Tanh: output is bounded in (-1, 1)":
        var layer: PcnDenseLayer[2, 2, float, Tanh]
        layer.init(Tanh())
        layer.weights[0, 0] = 1.0; layer.weights[1, 1] = 1.0
        layer.state[0] =  1000.0
        layer.state[1] = -1000.0
        let pred = layer.predict()
        check pred[0] >  0.999
        check pred[1] < -0.999


# ---------------------------------------------------------------------------

suite "PcnDenseLayer – updateError()":

    test "Zero state and zero prediction → zero error":
        var layer: PcnDenseLayer[2, 2, float, Identity]
        layer.init(Identity())
        var predAbove = Vector[float, 2].zeros()
        layer.updateError(predAbove)
        check approxEq(layer.error[0], 0.0, eps)
        check approxEq(layer.error[1], 0.0, eps)

    test "error = state − predFromAbove":
        var layer: PcnDenseLayer[2, 2, float, Identity]
        layer.init(Identity())
        layer.state[0] = 3.0
        layer.state[1] = -1.0
        var predAbove = Vector[float, 2].zeros()
        predAbove[0] = 1.0
        predAbove[1] = 2.0
        layer.updateError(predAbove)
        check approxEq(layer.error[0],  2.0, eps)   # 3 - 1
        check approxEq(layer.error[1], -3.0, eps)   # -1 - 2

    test "When state equals prediction, error is zero":
        var layer: PcnDenseLayer[3, 3, float, Sigmoid]
        layer.init(Sigmoid())
        layer.state[0] = 0.5
        layer.state[1] = 1.5
        layer.state[2] = -0.7
        var predAbove = Vector[float, 3].zeros()
        predAbove[0] = layer.state[0]
        predAbove[1] = layer.state[1]
        predAbove[2] = layer.state[2]
        layer.updateError(predAbove)
        for v in layer.error.data:
            check approxEq(v, 0.0, eps)

    test "updateError overwrites any previous error":
        var layer: PcnDenseLayer[2, 2, float, Identity]
        layer.init(Identity())
        # First update
        var p1 = Vector[float, 2].zeros()
        p1[0] = 10.0
        layer.updateError(p1)
        check approxEq(layer.error[0], -10.0, eps)
        # Second update with different prediction
        var p2 = Vector[float, 2].zeros()
        p2[0] = 1.0
        layer.state[0] = 3.0
        layer.updateError(p2)
        check approxEq(layer.error[0], 2.0, eps)   # 3 - 1


# ---------------------------------------------------------------------------

suite "PcnDenseLayer – relax()  (Identity activation)":
    # With Identity, act.grad = 1 everywhere, so:
    #   delta = W^T * errorBelow - self.error
    #   state += inferenceRate * delta

    test "Zero error and zero errorBelow → state unchanged":
        var layer: PcnDenseLayer[2, 2, float, Identity]
        layer.init(Identity())
        layer.state[0] = 5.0
        layer.state[1] = -2.0
        # error and errorBelow both zero
        var errorBelow = Vector[float, 2].zeros()
        layer.relax(errorBelow)
        check approxEq(layer.state[0],  5.0, eps)
        check approxEq(layer.state[1], -2.0, eps)

    test "Non-zero errorBelow moves state via W^T feedback":
        # Use identity weight matrix so W^T * errorBelow = errorBelow
        var layer: PcnDenseLayer[2, 2, float, Identity]
        layer.init(Identity(), infRate = 1.0)
        layer.weights[0, 0] = 1.0; layer.weights[0, 1] = 0.0
        layer.weights[1, 0] = 0.0; layer.weights[1, 1] = 1.0
        # errorBelow = [2, 0]
        # delta = W^T * [2, 0] - error = [2, 0] - [0, 0] = [2, 0]
        # new state = [0, 0] + 1.0 * [2, 0] = [2, 0]
        var errorBelow = Vector[float, 2].zeros()
        errorBelow[0] = 2.0
        layer.relax(errorBelow)
        check approxEq(layer.state[0], 2.0, eps)
        check approxEq(layer.state[1], 0.0, eps)

    test "inferenceRate scales the state update":
        var layerFast: PcnDenseLayer[2, 2, float, Identity]
        layerFast.init(Identity(), infRate = 0.5)
        var layerSlow: PcnDenseLayer[2, 2, float, Identity]
        layerSlow.init(Identity(), infRate = 0.1)
        # Identical identity-weight setup
        layerFast.weights[0, 0] = 1.0; layerFast.weights[1, 1] = 1.0
        layerSlow.weights[0, 0] = 1.0; layerSlow.weights[1, 1] = 1.0
        var errorBelow = Vector[float, 2].zeros()
        errorBelow[0] = 1.0; errorBelow[1] = 1.0
        layerFast.relax(errorBelow)
        layerSlow.relax(errorBelow)
        # Fast layer should move state more than slow layer
        check layerFast.state[0] > layerSlow.state[0]

    test "Self-error opposes state update":
        # Without errorBelow, delta = -error, state moves toward predFromAbove
        var layer: PcnDenseLayer[2, 2, float, Identity]
        layer.init(Identity(), infRate = 1.0)
        layer.state[0] = 3.0
        var predAbove = Vector[float, 2].zeros()
        predAbove[0] = 1.0
        layer.updateError(predAbove)  # error = 3 - 1 = 2
        var errorBelow = Vector[float, 2].zeros()
        layer.relax(errorBelow)
        # delta = W^T * 0 - error = -[2, 0] = [-2, 0]
        # new state = [3, 0] + 1.0 * [-2, 0] = [1, 0]
        check approxEq(layer.state[0], 1.0, eps)


# ---------------------------------------------------------------------------

suite "PcnDenseLayer – learn()  (Identity activation)":

    test "Weights unchanged when errorBelow is zero":
        var layer: PcnDenseLayer[2, 2, float, Identity]
        layer.init(Identity(), lr = 0.1)
        let wBefore = layer.weights
        var errorBelow = Vector[float, 2].zeros()
        layer.learn(errorBelow)
        check layer.weights == wBefore

    test "Weights change when errorBelow is non-zero":
        var layer: PcnDenseLayer[2, 2, float, Identity]
        layer.init(Identity(), lr = 0.1)
        # Drive must be set (predict() populates drive; Identity grad = 1 always)
        layer.state[0] = 1.0; layer.state[1] = 1.0
        discard layer.predict()
        var errorBelow = Vector[float, 2].zeros()
        errorBelow[0] = 1.0
        let wBefore00 = layer.weights[0, 0]
        layer.learn(errorBelow)
        check layer.weights[0, 0] != wBefore00

    test "Larger errorBelow produces larger weight update":
        var layerBig: PcnDenseLayer[2, 2, float, Identity]
        layerBig.init(Identity(), lr = 0.1)
        var layerSmall: PcnDenseLayer[2, 2, float, Identity]
        layerSmall.init(Identity(), lr = 0.1)
        # Override to known zero weights
        layerBig.weights[0, 0] = 0.0; layerBig.weights[0, 1] = 0.0
        layerBig.weights[1, 0] = 0.0; layerBig.weights[1, 1] = 0.0
        layerSmall.weights = layerBig.weights

        layerBig.state[0] = 1.0; layerSmall.state[0] = 1.0

        var errBig   = Vector[float, 2].zeros()
        var errSmall = Vector[float, 2].zeros()
        errBig[0]   = 2.0
        errSmall[0] = 0.5

        discard layerBig.predict()
        discard layerSmall.predict()

        layerBig.learn(errBig)
        layerSmall.learn(errSmall)

        check abs(layerBig.weights[0, 0]) > abs(layerSmall.weights[0, 0])

    test "Learning rate scales the weight update":
        var layerHighLr: PcnDenseLayer[2, 2, float, Identity]
        layerHighLr.init(Identity(), lr = 0.5)
        var layerLowLr: PcnDenseLayer[2, 2, float, Identity]
        layerLowLr.init(Identity(), lr = 0.01)
        # Override to known zero weights
        for w in [layerHighLr.weights.addr, layerLowLr.weights.addr]:
            w[][0, 0] = 0.0; w[][0, 1] = 0.0
            w[][1, 0] = 0.0; w[][1, 1] = 0.0
        layerHighLr.state[0] = 1.0; layerLowLr.state[0] = 1.0

        var err = Vector[float, 2].zeros()
        err[0] = 1.0

        discard layerHighLr.predict()
        discard layerLowLr.predict()
        layerHighLr.learn(err)
        layerLowLr.learn(err)

        check abs(layerHighLr.weights[0, 0]) > abs(layerLowLr.weights[0, 0])


# ---------------------------------------------------------------------------

suite "PcnDenseLayer – End-to-end convergence":

    test "Repeated relax() reduces self-error magnitude (Identity)":
        # Layer with identity weights; predFromAbove is constant.
        # Repeated relax() should drive state toward predFromAbove, reducing error.
        var layer: PcnDenseLayer[2, 2, float, Identity]
        layer.init(Identity(), infRate = 0.3)
        layer.weights[0, 0] = 1.0; layer.weights[1, 1] = 1.0
        layer.state[0] = 4.0; layer.state[1] = -2.0

        var predAbove = Vector[float, 2].zeros()
        predAbove[0] = 1.0; predAbove[1] = 0.0

        var errorBelow = Vector[float, 2].zeros()   # no signal from below

        layer.updateError(predAbove)
        let errMag0 = abs(layer.error[0]) + abs(layer.error[1])

        for _ in 0..19:
            discard layer.predict()
            layer.updateError(predAbove)
            layer.relax(errorBelow)

        let errMagFinal = abs(layer.error[0]) + abs(layer.error[1])
        check errMagFinal < errMag0

    test "Repeated learn() reduces prediction error on fixed input (Identity)":
        # Supervised-style: drive state to a fixed value, observe prediction,
        # compute errorBelow as (target - prediction), call learn().
        # Prediction should move toward the target over many steps.
        var layer: PcnDenseLayer[2, 2, float, Identity]
        layer.init(Identity(), lr = 0.05)
        layer.weights[0, 0] = 0.1; layer.weights[1, 1] = 0.1
        layer.state[0] = 1.0;   layer.state[1] = 1.0

        let target = initVector[float, 2]([2.0, 3.0])

        # Record error from the very first prediction (initial weights)
        let initPred = layer.predict()
        let firstErr = abs(target[0] - initPred[0]) +
                       abs(target[1] - initPred[1])

        var prevError = float.high
        for _ in 0..49:
            let pred = layer.predict()
            var err = Vector[float, 2].zeros()
            err[0] = target[0] - pred[0]
            err[1] = target[1] - pred[1]
            let curError = abs(err[0]) + abs(err[1])
            layer.learn(err)
            prevError = curError

        let finalPred = layer.predict()
        let finalErr = abs(target[0] - finalPred[0]) +
                       abs(target[1] - finalPred[1])
        # Error after 50 steps should be smaller than the very first error
        check finalErr < firstErr