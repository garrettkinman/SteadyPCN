# Copyright (c) 2026 Garrett Kinman
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

# src/steadypcn/pcn.nim
import steadytensor
import activations

type
    # A Dense Predictive Coding Layer
    # Inputs: M, Outputs: N
    PcnDenseLayer*[M, N: static int; T] = object
        # Parameters
        weights*: Tensor[T, [M, N]]
        bias*: Tensor[T, [N, 1]]
        
        # State Buffers (Preserved between steps)
        state*: Tensor[T, [N, 1]]      # x: The current belief/activity
        prediction*: Tensor[T, [N, 1]] # mu: The prediction coming from bottom-up
        error*: Tensor[T, [N, 1]]      # e: The difference (state - prediction)
        
        # Configuration
        learningRate*: T
        inferenceRate*: T # For relaxing the state during inference
        # TODO: activation function

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# INITIALIZATION
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

proc initPcnLayer*[N, M: static int; T](lr: T = 0.01, infRate: T = 0.1): PcnDenseLayer[N, M, T] =
    result.weights = rand[T, [M, N]](-0.1, 0.1) # Initialize small random weights
    result.bias = zeros[T, [N, 1]]()
    result.state = zeros[T, [N, 1]]()
    result.prediction = zeros[T, [N, 1]]()
    result.error = zeros[T, [N, 1]]()
    result.learningRate = lr
    result.inferenceRate = infRate

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# GENERATE PREDICTION
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

func predict*[M, N: static int; T](layer: lent PcnDenseLayer[M, N, T]): Tensor[T, [M, 1]] =
    ## 1. Generates prediction: mu = f(Wx + b)
    ## 2. Calculates error: e = x - mu
    
    # Linear transform: (W * r) + b
    var preAct = (layer.weights * layer.state) + layer.biases
    
    # Apply Non-linearity (Sigmoid example)
    result = map(preAct, sigmoid) # TODO: handle activation functions better
    
    # # Calculate Local Prediction Error: e = state - prediction
    # layer.error = layer.state - layer.prediction

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# STATE UPDATE
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

func relax*[N, M, K: static int; T](
    layer: var PcnDenseLayer[N, M, T], 
    nextLayerError: Tensor[T, [K, 1]], 
    nextLayerWeights: Tensor[T, [K, N]]
) =
    ## Updates the layer's state (x) to minimize local and downstream errors.
    ## formula: dx = -local_error + (NextWeights.T * next_error) * f'(x)
    
    # 1. Top-Down Error Projection (Feedback)
    # We use matmulT to virtually transpose weights 
    # Result shape is [N, 1]
    let feedbackError = matmulT(nextLayerWeights, nextLayerError)
    
    # 2. Calculate Derivative of state (assuming Sigmoid)
    let dState = map(layer.state, sigmoidDerivative)
    
    # 3. Compute Total Gradient
    # We want to minimize Free Energy. 
    # Gradient descent on State: x += rate * (-error + feedback)
    # Note: This is a simplified F.E. update common in approximations.
    
    var delta = (feedbackError .* dState) - layer.error
    
    # 4. Apply Update
    layer.state += delta * layer.inferenceRate

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# WEIGHT UPDATE
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

func learn*[N, M: static int; T](layer: var PcnDenseLayer[N, M, T], input: Tensor[T, [M, 1]]) =
    ## Updates weights based on local prediction error.
    ## dW = learning_rate * (error * input.T)
    
    # In standard PCN: Weights learn to minimize the error of the prediction they generated.
    # We need an outer product: error [N, 1] * input.T [1, M] -> [N, M]
    
    # Since we don't have an explicit outer product kernel yet, we can loop:
    # (Or add a rank-1 update kernel to kernels.nim later for speed)
    for i in 0..<N:
        let errVal = layer.error[i, 0] * layer.learningRate
        for j in 0..<M:
            # W[i,j] += lr * e[i] * u[j]
            layer.weights[i, j] += errVal * input[j, 0]
        
    # Update biases (input is implicitly 1.0)
    layer.biases += layer.error * layer.learningRate