# Copyright (c) 2026 Garrett Kinman
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

# Copyright (c) 2026 Garrett Kinman
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import unittest
import math
import steadytensor
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
        check layer.weights.shape == [3, 2]
        check layer.bias.shape    == [3, 1]
        check layer.state.shape   == [2, 1]
        check layer.drive.shape   == [3, 1]
        check layer.error.shape   == [2, 1]

#     test "Bias and state start at zero":
#         let layer = initPcnLayer[3, 2, float](Identity, lr = 0.01, infRate = 0.1)
#         for v in layer.bias.data:  check v == 0.0
#         for v in layer.state.data: check v == 0.0
#         for v in layer.error.data: check v == 0.0
#         for v in layer.drive.data: check v == 0.0

#     test "Weights are in initialisation range (-0.1, 0.1)":
#         let layer = initPcnLayer[4, 4, float](Identity)
#         for v in layer.weights.data:
#             check v >= -0.1
#             check v <=  0.1

#     test "Learning-rate and inference-rate stored correctly":
#         let layer = initPcnLayer[2, 2, float](Sigmoid, lr = 0.05, infRate = 0.2)
#         check layer.learningRate  == 0.05
#         check layer.inferenceRate == 0.2


# # ---------------------------------------------------------------------------

# suite "PcnDenseLayer – predict()  (Identity activation)":
#     # With Identity, predict(layer) = W * state + bias  exactly.

#     test "Zero state → prediction equals bias":
#         var layer = initPcnLayer[2, 2, float](Identity)
#         # state is already zero; set bias to something non-trivial
#         layer.bias[0, 0] = 3.0
#         layer.bias[1, 0] = -1.0
#         let pred = layer.predict()
#         check approxEq(pred[0, 0],  3.0, eps)
#         check approxEq(pred[1, 0], -1.0, eps)

#     test "Non-zero state with known weights":
#         # W = [[2, 1], [0, 3]]   state = [[1], [2]]   bias = [[0], [0]]
#         # drive = W * state = [[2*1+1*2], [0*1+3*2]] = [[4], [6]]
#         var layer = initPcnLayer[2, 2, float](Identity)
#         layer.weights[0, 0] = 2.0; layer.weights[0, 1] = 1.0
#         layer.weights[1, 0] = 0.0; layer.weights[1, 1] = 3.0
#         layer.state[0, 0] = 1.0;   layer.state[1, 0] = 2.0
#         let pred = layer.predict()
#         check approxEq(pred[0, 0], 4.0, eps)
#         check approxEq(pred[1, 0], 6.0, eps)

#     test "Bias is added on top of W*state":
#         var layer = initPcnLayer[2, 2, float](Identity)
#         layer.weights[0, 0] = 1.0; layer.weights[0, 1] = 0.0
#         layer.weights[1, 0] = 0.0; layer.weights[1, 1] = 1.0  # identity matrix
#         layer.state[0, 0] = 5.0;   layer.state[1, 0] = -3.0
#         layer.bias[0, 0] =  2.0;   layer.bias[1, 0] = 10.0
#         let pred = layer.predict()
#         check approxEq(pred[0, 0],  7.0, eps)  # 5 + 2
#         check approxEq(pred[1, 0],  7.0, eps)  # -3 + 10

#     test "drive buffer is updated after predict()":
#         var layer = initPcnLayer[2, 2, float](Identity)
#         layer.weights[0, 0] = 1.0; layer.weights[1, 1] = 1.0
#         layer.state[0, 0] = 4.0;   layer.state[1, 0] = -2.0
#         discard layer.predict()
#         check approxEq(layer.drive[0, 0],  4.0, eps)
#         check approxEq(layer.drive[1, 0], -2.0, eps)

#     test "Rectangular layer (M≠N)":
#         # M=3, N=2: W is [3,2], state is [2,1], prediction is [3,1]
#         var layer = initPcnLayer[3, 2, float](Identity)
#         layer.weights[0, 0] = 1.0; layer.weights[0, 1] = 0.0
#         layer.weights[1, 0] = 0.0; layer.weights[1, 1] = 1.0
#         layer.weights[2, 0] = 1.0; layer.weights[2, 1] = 1.0
#         layer.state[0, 0] = 2.0;   layer.state[1, 0] = 3.0
#         let pred = layer.predict()
#         check approxEq(pred[0, 0], 2.0, eps)
#         check approxEq(pred[1, 0], 3.0, eps)
#         check approxEq(pred[2, 0], 5.0, eps)


# suite "PcnDenseLayer – predict()  (non-linear activations)":

#     test "Sigmoid: output is strictly in (0, 1)":
#         var layer = initPcnLayer[3, 3, float](Sigmoid)
#         layer.state[0, 0] = 100.0  # should saturate near 1
#         layer.state[1, 0] = -100.0 # should saturate near 0
#         layer.state[2, 0] = 0.0    # should be near 0.5
#         # Use identity weights so drive = state
#         layer.weights[0, 0] = 1.0
#         layer.weights[1, 1] = 1.0
#         layer.weights[2, 2] = 1.0
#         let pred = layer.predict()
#         check pred[0, 0] > 0.999
#         check pred[1, 0] < 0.001
#         check approxEq(pred[2, 0], 0.5, 1e-4)

#     test "ReLU: negative drive produces zero output":
#         var layer = initPcnLayer[2, 2, float](ReLU)
#         layer.weights[0, 0] = 1.0; layer.weights[1, 1] = 1.0
#         layer.state[0, 0] =  2.0
#         layer.state[1, 0] = -3.0
#         let pred = layer.predict()
#         check approxEq(pred[0, 0], 2.0, eps)
#         check approxEq(pred[1, 0], 0.0, eps)

#     test "Tanh: zero drive produces zero output":
#         var layer = initPcnLayer[2, 2, float](Tanh)
#         layer.weights[0, 0] = 1.0; layer.weights[1, 1] = 1.0
#         # state is zero → drive is zero → tanh(0) = 0
#         let pred = layer.predict()
#         check approxEq(pred[0, 0], 0.0, eps)
#         check approxEq(pred[1, 0], 0.0, eps)

#     test "Tanh: output is bounded in (-1, 1)":
#         var layer = initPcnLayer[2, 2, float](Tanh)
#         layer.weights[0, 0] = 1.0; layer.weights[1, 1] = 1.0
#         layer.state[0, 0] =  1000.0
#         layer.state[1, 0] = -1000.0
#         let pred = layer.predict()
#         check pred[0, 0] >  0.999
#         check pred[1, 0] < -0.999


# # ---------------------------------------------------------------------------

# suite "PcnDenseLayer – updateError()":

#     test "Zero state and zero prediction → zero error":
#         var layer = initPcnLayer[2, 2, float](Identity)
#         var predAbove = zeros[float, [2, 1]]()
#         layer.updateError(predAbove)
#         check approxEq(layer.error[0, 0], 0.0, eps)
#         check approxEq(layer.error[1, 0], 0.0, eps)

#     test "error = state − predFromAbove":
#         var layer = initPcnLayer[2, 2, float](Identity)
#         layer.state[0, 0] = 3.0
#         layer.state[1, 0] = -1.0
#         var predAbove = zeros[float, [2, 1]]()
#         predAbove[0, 0] = 1.0
#         predAbove[1, 0] = 2.0
#         layer.updateError(predAbove)
#         check approxEq(layer.error[0, 0],  2.0, eps)   # 3 - 1
#         check approxEq(layer.error[1, 0], -3.0, eps)   # -1 - 2

#     test "When state equals prediction, error is zero":
#         var layer = initPcnLayer[3, 3, float](Sigmoid)
#         layer.state[0, 0] = 0.5
#         layer.state[1, 0] = 1.5
#         layer.state[2, 0] = -0.7
#         var predAbove = zeros[float, [3, 1]]()
#         predAbove[0, 0] = layer.state[0, 0]
#         predAbove[1, 0] = layer.state[1, 0]
#         predAbove[2, 0] = layer.state[2, 0]
#         layer.updateError(predAbove)
#         for v in layer.error.data:
#             check approxEq(v, 0.0, eps)

#     test "updateError overwrites any previous error":
#         var layer = initPcnLayer[2, 2, float](Identity)
#         # First update
#         var p1 = zeros[float, [2, 1]]()
#         p1[0, 0] = 10.0
#         layer.updateError(p1)
#         check approxEq(layer.error[0, 0], -10.0, eps)
#         # Second update with different prediction
#         var p2 = zeros[float, [2, 1]]()
#         p2[0, 0] = 1.0
#         layer.state[0, 0] = 3.0
#         layer.updateError(p2)
#         check approxEq(layer.error[0, 0], 2.0, eps)   # 3 - 1


# # ---------------------------------------------------------------------------

# suite "PcnDenseLayer – relax()  (Identity activation)":
#     # With Identity, act.grad = 1 everywhere, so:
#     #   delta = W^T * errorBelow - self.error
#     #   state += inferenceRate * delta

#     test "Zero error and zero errorBelow → state unchanged":
#         var layer = initPcnLayer[2, 2, float](Identity)
#         layer.state[0, 0] = 5.0
#         layer.state[1, 0] = -2.0
#         # error and errorBelow both zero
#         var errorBelow = zeros[float, [2, 1]]()
#         layer.relax(errorBelow)
#         check approxEq(layer.state[0, 0],  5.0, eps)
#         check approxEq(layer.state[1, 0], -2.0, eps)

#     test "Non-zero errorBelow moves state via W^T feedback":
#         # Use identity weight matrix so W^T * errorBelow = errorBelow
#         var layer = initPcnLayer[2, 2, float](Identity, infRate = 1.0)
#         layer.weights[0, 0] = 1.0; layer.weights[1, 1] = 1.0
#         # state = [0, 0], error = [0, 0]  (default)
#         # errorBelow = [[2], [0]]
#         # delta = W^T * [[2],[0]] - error = [[2],[0]] - [[0],[0]] = [[2],[0]]
#         # new state = [0,0] + 1.0 * [2,0] = [2, 0]
#         var errorBelow = zeros[float, [2, 1]]()
#         errorBelow[0, 0] = 2.0
#         layer.relax(errorBelow)
#         check approxEq(layer.state[0, 0], 2.0, eps)
#         check approxEq(layer.state[1, 0], 0.0, eps)

#     test "inferenceRate scales the state update":
#         var layerFast = initPcnLayer[2, 2, float](Identity, infRate = 0.5)
#         var layerSlow = initPcnLayer[2, 2, float](Identity, infRate = 0.1)
#         # Identical identity-weight setup
#         layerFast.weights[0, 0] = 1.0; layerFast.weights[1, 1] = 1.0
#         layerSlow.weights[0, 0] = 1.0; layerSlow.weights[1, 1] = 1.0
#         var errorBelow = zeros[float, [2, 1]]()
#         errorBelow[0, 0] = 1.0; errorBelow[1, 0] = 1.0
#         layerFast.relax(errorBelow)
#         layerSlow.relax(errorBelow)
#         # Fast layer should move state more than slow layer
#         check layerFast.state[0, 0] > layerSlow.state[0, 0]

#     test "Self-error opposes state update":
#         # Without errorBelow, delta = -error, state moves toward predFromAbove
#         var layer = initPcnLayer[2, 2, float](Identity, infRate = 1.0)
#         layer.state[0, 0] = 3.0
#         var predAbove = zeros[float, [2, 1]]()
#         predAbove[0, 0] = 1.0
#         layer.updateError(predAbove)  # error = 3 - 1 = 2
#         var errorBelow = zeros[float, [2, 1]]()
#         layer.relax(errorBelow)
#         # delta = W^T * 0 - error = -[2, 0] = [-2, 0]
#         # new state = [3, 0] + 1.0 * [-2, 0] = [1, 0]
#         check approxEq(layer.state[0, 0], 1.0, eps)


# # ---------------------------------------------------------------------------

# suite "PcnDenseLayer – learn()  (Identity activation)":

#     test "Weights unchanged when errorBelow is zero":
#         var layer = initPcnLayer[2, 2, float](Identity, lr = 0.1)
#         let wBefore = layer.weights
#         var errorBelow = zeros[float, [2, 1]]()
#         layer.learn(errorBelow)
#         check layer.weights == wBefore

#     test "Weights change when errorBelow is non-zero":
#         var layer = initPcnLayer[2, 2, float](Identity, lr = 0.1)
#         # Drive must be set (predict() populates drive; Identity grad = 1 always)
#         layer.state[0, 0] = 1.0; layer.state[1, 0] = 1.0
#         discard layer.predict()
#         var errorBelow = zeros[float, [2, 1]]()
#         errorBelow[0, 0] = 1.0
#         let wBefore00 = layer.weights[0, 0]
#         layer.learn(errorBelow)
#         check layer.weights[0, 0] != wBefore00

#     test "Larger errorBelow produces larger weight update":
#         var layerBig   = initPcnLayer[2, 2, float](Identity, lr = 0.1)
#         var layerSmall = initPcnLayer[2, 2, float](Identity, lr = 0.1)
#         # Identical initial weights (zero, since initPcnLayer randomises; let's
#         # override to a known value)
#         layerBig.weights[0, 0] = 0.0; layerBig.weights[0, 1] = 0.0
#         layerBig.weights[1, 0] = 0.0; layerBig.weights[1, 1] = 0.0
#         layerSmall.weights     = layerBig.weights

#         layerBig.state[0, 0] = 1.0; layerSmall.state[0, 0] = 1.0

#         var errBig   = zeros[float, [2, 1]]()
#         var errSmall = zeros[float, [2, 1]]()
#         errBig[0, 0]   = 2.0
#         errSmall[0, 0] = 0.5

#         discard layerBig.predict()
#         discard layerSmall.predict()

#         layerBig.learn(errBig)
#         layerSmall.learn(errSmall)

#         check abs(layerBig.weights[0, 0]) > abs(layerSmall.weights[0, 0])

#     test "Learning rate scales the weight update":
#         var layerHighLr = initPcnLayer[2, 2, float](Identity, lr = 0.5)
#         var layerLowLr  = initPcnLayer[2, 2, float](Identity, lr = 0.01)
#         for w in [layerHighLr.weights.addr, layerLowLr.weights.addr]:
#             w[][0, 0] = 0.0; w[][0, 1] = 0.0
#             w[][1, 0] = 0.0; w[][1, 1] = 0.0
#         layerHighLr.state[0, 0] = 1.0; layerLowLr.state[0, 0] = 1.0

#         var err = zeros[float, [2, 1]]()
#         err[0, 0] = 1.0

#         discard layerHighLr.predict()
#         discard layerLowLr.predict()
#         layerHighLr.learn(err)
#         layerLowLr.learn(err)

#         check abs(layerHighLr.weights[0, 0]) > abs(layerLowLr.weights[0, 0])


# # ---------------------------------------------------------------------------

# suite "PcnDenseLayer – End-to-end convergence":

#     test "Repeated relax() reduces self-error magnitude (Identity)":
#         # Layer with identity weights; predFromAbove is constant.
#         # Repeated relax() should drive state toward predFromAbove, reducing error.
#         var layer = initPcnLayer[2, 2, float](Identity, infRate = 0.3)
#         layer.weights[0, 0] = 1.0; layer.weights[1, 1] = 1.0
#         layer.state[0, 0] = 4.0; layer.state[1, 0] = -2.0

#         var predAbove = zeros[float, [2, 1]]()
#         predAbove[0, 0] = 1.0; predAbove[1, 0] = 0.0

#         var errorBelow = zeros[float, [2, 1]]()   # no signal from below

#         layer.updateError(predAbove)
#         let errMag0 = abs(layer.error[0, 0]) + abs(layer.error[1, 0])

#         for _ in 0..19:
#             discard layer.predict()
#             layer.updateError(predAbove)
#             layer.relax(errorBelow)

#         let errMagFinal = abs(layer.error[0, 0]) + abs(layer.error[1, 0])
#         check errMagFinal < errMag0

#     test "Repeated learn() reduces prediction error on fixed input (Identity)":
#         # Supervised-style: drive state to a fixed value, observe prediction,
#         # compute errorBelow as (target - prediction), call learn().
#         # Prediction should move toward the target over many steps.
#         var layer = initPcnLayer[2, 2, float](Identity, lr = 0.05)
#         layer.weights[0, 0] = 0.1; layer.weights[1, 1] = 0.1
#         layer.state[0, 0] = 1.0;   layer.state[1, 0] = 1.0

#         # We'll measure how much the prediction changes toward [2, 3].
#         let target = initTensor[float, [2, 1], 2]([2.0, 3.0])

#         var prevError = float.high
#         for _ in 0..49:
#             let pred = layer.predict()
#             var err = zeros[float, [2, 1]]()
#             err[0, 0] = target[0, 0] - pred[0, 0]
#             err[1, 0] = target[1, 0] - pred[1, 0]
#             let curError = abs(err[0, 0]) + abs(err[1, 0])
#             layer.learn(err)
#             prevError = curError

#         let finalPred = layer.predict()
#         let finalErr = abs(target[0, 0] - finalPred[0, 0]) +
#                        abs(target[1, 0] - finalPred[1, 0])
#         # Error after 50 steps should be smaller than the very first error
#         let firstPred = initTensor[float, [2, 1], 2](
#             [layer.weights[0,0]*1.0 + layer.weights[0,1]*1.0,
#              layer.weights[1,0]*1.0 + layer.weights[1,1]*1.0])
#         let firstErr = abs(target[0,0] - firstPred[0,0]) +
#                        abs(target[1,0] - firstPred[1,0])
#         check finalErr < firstErr