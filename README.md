<!--
 Copyright (c) 2023 Garrett Kinman
 
 This software is released under the MIT License.
 https://opensource.org/licenses/MIT
-->

# SteadyNN
Lightweight, performant, and dependency-free neural network inference engine for TinyML written in pure Nim. Portable to custom accelerators and WASM.

# Concept
This framework is built around a few key principles:
1. **No dynamic memory allocation.** This is to improve performance and to provide compile-time guarantees of memory usage. Currently no other TinyML frameworks (that I know of) use only static memory allocation.
2. **A small set of primitive operators.** All higher-level operations (e.g., matrix multiplication) can be broken down to some combination of these primitive operations. So long as these primitive operators (primops) are optimized for hardware, you will have reasonably performant higher-level operations.
3. **Portable to new hardware.** Porting to new hardware (including dedicated accelerators) is as easy as implementing the primops on the hardware. If that isn't efficient enough, you can implement a new tensor subtype to optimize to your heart's content.
4. **No dependencies.** Working with other TinyML frameworks can be a pain, as there are so many dependencies that can (and often do) give you problems. By avoiding dependencies, this framework is much easier to use, simpler to understand and debug, and less of a pain to set up and use.
5. **Flexibility and extensibility.** TODO

# Primitive Operators
1. Arithmetic Operations
   1. Element-wise Vector Addition
   2. Element-wise Vector Subtraction
   3. Element-wise Vector Multiplication
   4. Element-wise Vector Division
   5. Element-wise Vector Modulus (?)
2. Activation Functions
   1. Element-wise Identity
   2. Element-wise ReLu
   3. Element-wise Sigmoid
   4. Element-wise Tanh
   5. Element-wise Leaky ReLu (?)
   6. Element-wise Swish (?)
   7. Element-wise Softmax (???)
   8. More?
3. Reduction Operations
   1. Sum Reduction
   2. Mean Reduction
   3. Max Reduction
   4. Min Reduction
   5. Argmax Reduction
   6. Argmin Reduction
   7. More?
4. Logical Operations
   1. Element-wise Equal
   2. Element-wise Not Equal
   3. Element-wise Greater
   4. Element-wise Greater Equal
   5. Element-wise Less
   6. Element-wise Less Equal
5. Miscellaneous Functions
   1. Element-wise Abs
   2. Element-wise Sqrt
   3. Element-wise Pow
   4. Element-wise Ln
   5. Element-wise Exp
   6. Element-wise Sin
   7. Element-wise Cos
   8. Element-wise Negate
   9. Element-wise Floor (?)
   10. Element-wise Ceil (?)
   11. Element-wise Round (?)
   12. Element-wise Clip (?)
   13. More?
6. More?

# Planned Features
- Data types
  - `int8`
  - `float32`
  - More? (TBD)
- Tensor types
  - Regular (dynamically allocated)
  - Sparse
  - Statically allocated? (TBD)
- Standard layers
  - Dense
  - Conv2D, DepthwiseConv
  - Pooling layers
  - Output layers, e.g., softmax
  - Recurrent layers? (TBD)
  - Attention layers? (TBD)
  - More? (TBD)
- Standard activation functions
  - sigmoid
  - relu
  - tanh
  - More? (TBD)
- Statically-allocated model parameters
  - Maybe have the entire library itself have no dynamic allocations?
- Optimized CPU operations (so no BLAS dependencies)
- Built-in hardware acceleration support
  - RISC-V V extension (vector)
  - RISC-V P extension (packed SIMD)
  - More? (TBD)
- Ability to (relatively) easily accelerate on other hardware
  - Expose layers as some combination of tensor operations so that accelerating becomes a matter of accelerating the tensor operations
  - Treat activation functions as tensor operations so that they, too, can be accelerated
- Ability to easily port to WASM
- Ability to load pre-trained parameters from a standard `.tflite` file
- On-device learning? (TBD)
- Forward-only learning? (TBD)
  - Forward-Forward
  - PEPITA
  - MEMPITA
