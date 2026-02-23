<!--
 Copyright (c) 2026 Garrett Kinman
 
 This software is released under the MIT License.
 https://opensource.org/licenses/MIT
-->

# SteadyPCN

An ultra-light, ultra-flexible predictive coding framework written in pure Nim, built for microcontrollers.

---

## Key Design Principles

**No dynamic memory allocation.** All tensor shapes are resolved at compile time, giving you static guarantees on memory usage — a hard requirement in many safety-critical systems.

**Configurable memory layout.** Pass `-d:colMajor` at compile time to switch to column-major (Fortran/MATLAB) storage order; the default is row-major (C/NumPy). No code changes required.

**Configurable data types.** `Matrix` and `Vector` are generic over their element type `T`, making it easy to prototype low-bit or fixed-point architectures.

**Portable kernels.** Porting to new hardware or a dedicated accelerator only requires reimplementing the kernels in `kernels.nim`.

**Zero dependencies.** No package manager headaches, no transitive breakage — just Nim.

---

## Architecture

A predictive coding network is composed of `Layer` objects. Each layer holds:
- `weights` — a weight matrix mapping this layer's state to a prediction for the layer below
- `bias` — a bias vector added before activation
- `state` — the layer's current belief / latent representation
- `drive` — the pre-activation value `W * state + bias`
- `error` — the difference between the layer's state and the prediction it received from above

Each inference step follows four stages: **predict → updateError → relax → learn**.

---

## Quick Start

### Defining a Network

Layers are fully typed at compile time. Dimensions, data type, and activation function are all part of the type signature.

```nim
import steadypcn/[tensors, activations, layer]

# A two-layer network:
#   layer1: 8 observations, 16-dim state, float32, Sigmoid activation
#   layer2: 16 observations, 4-dim state, float32, Tanh activation
var layer1: Layer[8, 16, float32, Sigmoid]
var layer2: Layer[16, 4, float32, Tanh]

layer1.init(Sigmoid(), lr = 0.01, infRate = 0.1)
layer2.init(Tanh(),    lr = 0.01, infRate = 0.1)
```

### Seeding the Input

The bottom layer's state is set directly from your observation vector:

```nim
let observation = initVector[float32, 8]([0.1, 0.5, 0.3, 0.8,
                                          0.2, 0.7, 0.4, 0.6])
layer1.state = observation
```

### Running an Inference Step

```nim
# 1. Each layer predicts what the layer below should look like.
let pred1 = layer1.predict()   # Vector[float32, 8]  — sent downward
let pred2 = layer2.predict()   # Vector[float32, 16] — sent to layer1

# 2. Update error signals (top-down prediction vs. current state).
layer1.updateError(pred2)      # error = layer1.state - pred2
layer2.updateError(someTopDownPred)

# 3. Relax: nudge states to reduce prediction error.
layer1.relax(errorBelow = layer1.state - observation)
layer2.relax(errorBelow = layer1.error)

# 4. Learn: update weights via a Hebbian-style rule.
layer1.learn(errorBelow = layer1.state - observation)
layer2.learn(errorBelow = layer1.error)
```

For a full training loop, repeat the four stages for each sample in your dataset.

---

## Tensors

`Matrix[T, M, N]` and `Vector[T, N]` are the core data types. All shapes are static — mismatched dimensions are a compile error, not a runtime panic.

### Construction

```nim
import steadypcn/tensors

# Zero-initialised
let z = Matrix[float32, 3, 3].zeros()
let v = Vector[float32, 4].zeros()

# From literal data
let w = initMatrix[float32, 2, 3]([1.0'f32, 2.0, 3.0,
                                    4.0,     5.0, 6.0])
let b = initVector[float32, 3]([0.1'f32, 0.2, 0.3])

# Random uniform in [lo, hi]
let r = Matrix[float32, 4, 4].rand(-0.1'f32, 0.1'f32)
```

### Arithmetic

```nim
import steadypcn/[tensors, ops]

# Scalar broadcast
let scaled = w * 2.0'f32
let shifted = w + 1.0'f32

# Element-wise
let sum  = w + w
let diff = w - w
let had  = w .* w     # Hadamard (element-wise) product; `.*` to avoid matmul ambiguity

# Matrix / vector products
let y = w * b         # (2×3) * (3,) → (2,)

# Zero-copy transpose
let wt = w.t          # TransposedMatrix[float32, 3, 2] — no data copied
let z2 = wt * v       # (3×2) * (2,) → (3,) dispatches to mvMulT kernel

# Outer product (useful for rank-1 weight updates)
let delta = outer(b, b)   # (3,) ⊗ (3,) → (3×3)

# Dot product
let s = dot(b, b)     # float32
```

### Shape Queries

```nim
echo w.rows   # 2
echo w.cols   # 3
echo w.size   # 6
echo b.len    # 3
```

---

## Activations

Four activations ship out of the box. All satisfy the `Activation[T]` concept — any type implementing `activate` and `grad` over `T` and `Vector[T, N]` works as a drop-in.

| Type | Formula | Range | Notes |
|------|---------|-------|-------|
| `Sigmoid` | `1 / (1 + e^-x)` | (0, 1) | Good default for PCN belief states |
| `Tanh` | `(e^x - e^-x) / (e^x + e^-x)` | (-1, 1) | Zero-centred; useful for symmetric representations |
| `ReLU` | `max(0, x)` | [0, ∞) | Promotes sparse activations |
| `Identity` | `x` | (-∞, ∞) | Linear layers; useful for testing |

Scalar and vectorised overloads are both provided. The vectorised forms operate in-place on `Vector[T, N]` without heap allocation.

```nim
import steadypcn/[tensors, activations]

let sig = Sigmoid()
echo activate(sig, 0.0'f32)   # ≈ 0.5
echo grad(sig, 0.0'f32)       # ≈ 0.25

let v = initVector[float32, 4]([0.0'f32, 1.0, -1.0, 2.0])
echo activate(sig, v)         # element-wise sigmoid
echo grad(sig, v)             # element-wise sigmoid derivative
```

### Custom Activations

Any object type that implements scalar `activate` and `grad` overloads satisfies the `Activation[T]` concept and can be passed directly to `Layer.init`:

```nim
type Hardtanh* = object

func activate*[T](_: Hardtanh, x: T): T {.inline.} =
  if x < T(-1): T(-1) elif x > T(1): T(1) else: x

func grad*[T](_: Hardtanh, x: T): T {.inline.} =
  if x < T(-1) or x > T(1): T(0) else: T(1)

# Also add vectorised overloads (same pattern as built-ins), then:
var myLayer: Layer[8, 4, float32, Hardtanh]
myLayer.init(Hardtanh())
```

---

## Compile-Time Options

| Flag | Effect |
|------|--------|
| *(default)* | Row-major (C/NumPy) memory layout |
| `-d:colMajor` | Column-major (Fortran/MATLAB) memory layout |

Example:

```sh
nim c -d:colMajor -d:release -o:myapp src/main.nim
```

---

## License

MIT — see [LICENSE](LICENSE) for details.