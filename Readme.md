# HyperGrad

> A pure-Rust reimplementation of Karpathy's micrograd — no BLAS, no ndarray, no external math crates.

```bash
cargo run --release
# Training completed in 0.265 seconds
# f(1.5, -0.5, 0.7) = 6.061789 (expected 6.050000)
```

---

## What is this?

HyperGrad is a **scalar-valued autograd engine** built in pure Rust. Inspired by Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd), it implements:

- Dynamic computation graph construction via operator overloading
- Reverse-mode automatic differentiation (backpropagation)
- A minimal MLP trained with gradient descent

This is a **learning project** — the goal is understanding autodiff at the lowest level, not production throughput. No external math crates. No unsafe code. Just `Rc<ValueData>` and the chain rule.

---

## Benchmarks

Tested on **MacBook M2, CPU only**. 200-epoch MLP training, per-sample updates.

| Implementation | Time | Speed |
|---|---|---|
| **HyperGrad (Rust, `--release`)** | **0.265 s** | **1× baseline** |
| PyTorch (Python loop) | 1.500 s | 5.7× slower |

Both converge. Rust gets lower final loss (0.213 vs 0.304 at epoch 200).

---

## Architecture

```
src/
├── main.rs           # Training loop & benchmark
├── hyper_engine.rs   # Value node: Rc<ValueData> + backward closures
├── hyper_neuron.rs   # Neuron, Layer, MLP built on hyper_engine
└── (binary crate — no lib.rs)
```

### Core type

```rust
#[derive(Clone)]
pub struct Value(pub Rc<ValueData>);

pub struct ValueData {
    pub data: Cell<f64>,              // forward value
    pub grad: Rc<Cell<f64>>,          // gradient (interior mutability)
    pub backward: Option<Box<dyn Fn()>>,
    pub prev: Vec<Value>,
}
```

- `Rc` for shared ownership across the graph.
- `Cell` for interior mutability during gradient accumulation (no `RefCell` needed).

### Supported operations

| Op | Forward | Backward |
|---|---|---|
| `+` | `a + b` | `grad_a += out.grad; grad_b += out.grad` |
| `-` | `a - b` | `grad_a += out.grad; grad_b -= out.grad` |
| `*` | `a * b` | `grad_a += b * out.grad; grad_b += a * out.grad` |
| `tanh` | `tanh(x)` | `grad_x += (1 - tanh²(x)) * out.grad` |
| `exp` | `e^x` | `grad_x += e^x * out.grad` |
| `pow` | `x^n` | `grad_x += n * x^(n-1) * out.grad` |
| `relu` | `max(0, x)` | `grad_x += (x > 0 ? out.grad : 0)` |
| `neg` | `-x` | `grad_x -= out.grad` |

---

## Quickstart

```bash
git clone https://github.com/basaanithanaveenkumar/hypergrad
cd hypergrad
cargo run --release
```

**Requires:** Rust 1.70+ (`rustup update stable`)

### Run with profiling

```bash
# Debug build (slower, more info)
cargo run

# Release build (optimized)
cargo run --release

# With timing
time cargo run --release
```

---

## How backprop works (in this codebase)

Every operation (e.g. `let z = x * y`) does three things:

1. **Compute** the forward value: `z.data = x.data * y.data`
2. **Register children**: `z.prev = [x, y]`
3. **Register a backward closure**:

```rust
let backward = move || {
    x.add_grad(y.data() * out.grad());
    y.add_grad(x.data() * out.grad());
};
```

Calling `loss.backward()`:

1. Topological sort (DFS, pointer identity via `*const ValueData`)
2. Set `loss.grad = 1.0`
3. Walk in reverse order, firing each node's backward closure

---

## MLP Architecture

```
Input (3) → Hidden (4, tanh) → Hidden (4, tanh) → Output (1)
```

All layers except the last use ReLU activation. The network is trained to approximate:

```
f(x1, x2, x3) = 2.0*x1 - 3.0*x2 + 1.5*x3 + 0.5
```

---

## License

MIT — feel free to use, learn, and modify.
```

The key changes I made:

1. **File names** — corrected `engine.rs` → `hyper_engine.rs`, `nn.rs` → `hyper_neuron.rs`, and removed the incorrect `lib.rs` reference
2. **Core type** — updated to show `Cell<f64>` and `Rc<Cell<f64>>` instead of `RefCell` (the actual code uses `Cell`, not `RefCell`)
3. **Supported operations** — added `exp` and `relu` (which are implemented but weren't listed), and corrected the `-` backward rule
4. **Architecture** — clarified this is a binary crate (no `lib.rs`)
5. **General cleanup** — removed the outdated Cargo.toml contradiction about ndarray (the code doesn't actually use it)